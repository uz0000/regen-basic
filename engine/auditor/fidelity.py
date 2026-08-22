"""
Fidelity checks the simulator gates its output on.

Metrics:
  TVD (Total Variation Distance): per column, discrete distributions.
    TVD = 0.5 * Σ |P(x) - Q(x)| ∈ [0, 1]. Lower is better.

  Wasserstein-1: earth mover's distance between empirical CDFs of
    continuous columns. Lower is better.

  Cross-column correlation delta: mean absolute difference between the real
    and synthetic correlation matrices. This is the check that catches a
    batch with correct marginals but broken joint structure — the failure
    mode a per-column-only check would miss entirely.

  Categorical association delta: correlation is only defined between numeric
    columns, so it says nothing about whether "measurement X differs by
    category Y" survived. That is checked separately via the correlation
    ratio eta-squared (the share of a numeric column's variance explained by
    a categorical column's grouping), compared real vs. synthetic. Without
    this, a generator can flatten every category's distinctness and still
    report a clean bill of health.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from contracts.types import ColumnFidelity, FieldDict, FieldType


@dataclass
class AuditorConfig:
    tvd_threshold: float = 0.15           # max TVD per discrete/binary column
    # Continuous columns are gated on *normalized* Wasserstein (W / ref_std),
    # which is scale-free so one threshold works across features of any unit.
    wasserstein_threshold: float = 0.50   # max normalized Wasserstein per continuous column
    # High-cardinality categorical handling: when a column has more unique values
    # than this threshold, TVD is computed over only the top-K most frequent
    # categories (rest grouped into "other"). This prevents false rejections when
    # a small synthetic batch cannot cover 1,000+ categories.
    high_card_threshold: int = 50
    # Max mean-absolute difference between the real and synthetic correlation
    # matrices. Needs >=2 numeric columns and a few rows to estimate.
    correlation_threshold: float = 0.25
    # Limit on the WORST (categorical, numeric) pair's change in eta-squared.
    # eta-squared is a share of variance in [0, 1], so 0.10 means: no single
    # "this measurement differs by that category" relationship may lose (or
    # gain) more than a tenth of the variance it really explains.
    categorical_association_threshold: float = 0.10
    # Categories rarer than this many rows are pooled out of the eta-squared
    # estimate — a group of two rows has a meaningless group mean.
    min_category_rows: int = 20


# ── Cross-column correlation structure ──────────────────────────────────────────

def _correlation_delta(
    reference_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    field_dict: FieldDict,
    label_col: str,
) -> Optional[float]:
    """Mean absolute difference between real and synthetic correlation matrices.

    Restricted to numeric (continuous/binary) feature columns, excluding the
    label. Returns None when there are fewer than two such columns or too few
    rows to estimate correlations — in those cases there is no joint structure
    to validate. A NaN result (constant column → undefined correlation) is
    treated as "no signal" for that pair via nan-safe differencing.
    """
    numeric_cols = [
        c for c in reference_df.columns
        if c in synthetic_df.columns and c in field_dict and c != label_col
        and field_dict[c].field_type in (FieldType.CONTINUOUS, FieldType.BINARY)
    ]
    if len(numeric_cols) < 2:
        return None
    if len(reference_df) < 3 or len(synthetic_df) < 3:
        return None

    real_corr = reference_df[numeric_cols].corr().to_numpy()
    synth_corr = synthetic_df[numeric_cols].corr().to_numpy()

    # Compare only the upper triangle (off-diagonal pairs). Where either matrix
    # is NaN (a constant column), skip that pair rather than failing on it.
    iu = np.triu_indices_from(real_corr, k=1)
    diffs = np.abs(real_corr[iu] - synth_corr[iu])
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return None
    return float(diffs.mean())


# ── Categorical ↔ numeric association ─────────────────────────────────────────

def _eta_squared(groups: pd.Series, values: pd.Series, min_rows: int) -> Optional[float]:
    """Share of ``values``' variance explained by ``groups`` (correlation ratio).

    0.0 means the group means are identical — knowing the category tells you
    nothing about the measurement. 1.0 means the category determines it. This
    is the categorical analogue of a squared correlation, and unlike a
    correlation it needs no ordering of the categories, which matters because
    a nominal category has none.
    """
    df = pd.DataFrame({"g": groups.values, "v": pd.to_numeric(values, errors="coerce").values}).dropna()
    if len(df) < 3:
        return None
    counts = df["g"].value_counts()
    keep = counts[counts >= min_rows].index
    df = df[df["g"].isin(keep)]
    if df["g"].nunique() < 2:
        return None
    total_var = float(df["v"].var(ddof=0))
    if not np.isfinite(total_var) or total_var < 1e-12:
        return None
    grand = float(df["v"].mean())
    between = sum(
        len(sub) * (float(sub["v"].mean()) - grand) ** 2 for _, sub in df.groupby("g")
    ) / len(df)
    return float(np.clip(between / total_var, 0.0, 1.0))


def _categorical_association_delta(
    reference_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    field_dict: FieldDict,
    label_col: str,
    config: "AuditorConfig",
) -> Tuple[Optional[float], Optional[str]]:
    """Worst |eta^2_real - eta^2_synth| over every (categorical, numeric) pair,
    with the name of the pair that scored it. ``(None, None)`` when there is
    no such pair to compare.

    This catches the failure a numeric correlation matrix structurally cannot
    see: every category present in the right proportion, every numeric column
    with the right distribution, and yet the categories no longer differ from
    each other the way they really do.

    Reports the *worst* pair rather than the average deliberately. Averaging
    is how a real problem disappears: one badly broken relationship among ten
    intact ones divides down to a comfortable-looking number, and the one
    that matters is exactly the one you needed told about.
    """
    cat_cols = [c for c in reference_df.columns
                if c in synthetic_df.columns and c in field_dict and c != label_col
                and field_dict[c].field_type == FieldType.CATEGORICAL]
    num_cols = [c for c in reference_df.columns
                if c in synthetic_df.columns and c in field_dict and c != label_col
                and field_dict[c].field_type in (FieldType.CONTINUOUS, FieldType.BINARY)]
    if not cat_cols or not num_cols:
        return None, None

    worst, worst_pair = None, None
    for cat in cat_cols:
        for num in num_cols:
            er = _eta_squared(reference_df[cat], reference_df[num], config.min_category_rows)
            es = _eta_squared(synthetic_df[cat], synthetic_df[num], config.min_category_rows)
            if er is None or es is None:
                continue
            d = abs(er - es)
            if worst is None or d > worst:
                worst, worst_pair = d, f"{num} by {cat}"
    if worst is None:
        return None, None
    return float(worst), worst_pair


# ── Per-column evaluation ─────────────────────────────────────────────────────

def _eval_column(
    col: str,
    real: pd.Series,
    synth: pd.Series,
    ftype: FieldType,
    config: AuditorConfig,
) -> ColumnFidelity:
    result = ColumnFidelity(col=col, passed=True)

    if ftype == FieldType.CONTINUOUS:
        # Gate on normalized Wasserstein (scale-free, robust to small samples).
        w = _wasserstein(real, synth)
        result.wasserstein = w
        if w > config.wasserstein_threshold:
            result.passed = False
        # Binned TVD is reported for visibility but not gated: with a small
        # reference it is too noisy to threshold reliably.
        result.tvd = _tvd_continuous(real, synth)

    elif ftype in (FieldType.CATEGORICAL, FieldType.BINARY):
        t = _tvd_discrete(real, synth, config)
        result.tvd = t
        if t > config.tvd_threshold:
            result.passed = False

    return result


# ── TVD ───────────────────────────────────────────────────────────────────────

def _tvd_discrete(real: pd.Series, synth: pd.Series, config: AuditorConfig) -> float:
    """
    Total Variation Distance between two discrete distributions.

    For high-cardinality columns (> config.high_card_threshold unique values
    in the reference), TVD is computed over only the top-K most frequent
    categories from the reference, with all remaining categories grouped into
    an "other" bucket. This prevents false rejections when a small synthetic
    batch cannot cover 1,000+ categories.
    """
    real_clean = real.dropna()
    synth_clean = synth.dropna()
    nr, ns = len(real_clean), len(synth_clean)
    if nr == 0 or ns == 0:
        return 1.0

    real_vals = real_clean.unique()
    n_unique = len(real_vals)

    if n_unique <= config.high_card_threshold:
        # Low cardinality — compare full distributions.
        all_vals = set(real_vals) | set(synth_clean.unique())
        total = sum(
            abs((real_clean == v).sum() / nr - (synth_clean == v).sum() / ns)
            for v in all_vals
        )
        return 0.5 * total

    # High cardinality — compare top-K categories, group rest into "other".
    k = min(n_unique, max(20, ns // 5))
    # Deterministic tie-break. `nlargest(k)` picks arbitrarily among categories
    # sharing the k-th count, and which ones depends on pandas' internal
    # ordering — so the same data could score differently on another platform.
    # Ordering by (count descending, category ascending) makes the set unique.
    counts = real_clean.value_counts()
    ordered = sorted(counts.index, key=lambda c: (-int(counts[c]), str(c)))
    top_k = set(ordered[:k])

    real_top_mass = (real_clean.isin(top_k)).sum() / nr
    synth_top_mass = (synth_clean.isin(top_k)).sum() / ns

    total = sum(
        abs((real_clean == v).sum() / nr - (synth_clean == v).sum() / ns)
        for v in top_k
    )
    real_other = 1.0 - real_top_mass
    synth_other = 1.0 - synth_top_mass
    total += abs(real_other - synth_other)

    return 0.5 * total


def _tvd_continuous(real: pd.Series, synth: pd.Series, n_bins: int = 20) -> float:
    combined = pd.concat([real.dropna(), synth.dropna()])
    mn, mx = combined.min(), combined.max()
    if mx == mn:
        return 0.0
    bins = np.linspace(mn, mx, n_bins + 1)
    p, _ = np.histogram(real.dropna(), bins=bins)
    q, _ = np.histogram(synth.dropna(), bins=bins)
    p = p / (p.sum() + 1e-8)
    q = q / (q.sum() + 1e-8)
    return 0.5 * float(np.abs(p - q).sum())


def _wasserstein(real: pd.Series, synth: pd.Series) -> float:
    """
    Wasserstein-1 distance normalized by the reference (real) std, so the
    result is in units of standard deviations and one threshold works across
    features of any scale.
    """
    r, s = real.dropna().values, synth.dropna().values
    if len(r) == 0 or len(s) == 0:
        return 1.0
    scale = r.std()
    scale = scale if scale > 1e-8 else 1.0
    return float(wasserstein_distance(r, s) / scale)
