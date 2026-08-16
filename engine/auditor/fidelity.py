"""
Fidelity checks the basic generator gates on.

Metrics:
  TVD (Total Variation Distance): per column, discrete distributions.
    TVD = 0.5 * Σ |P(x) - Q(x)| ∈ [0, 1]. Lower is better.

  Wasserstein-1: earth mover's distance between empirical CDFs of
    continuous columns. Lower is better.

  Cross-column correlation delta: mean absolute difference between the real
    and synthetic correlation matrices. This is the check that catches a
    batch with correct marginals but broken joint structure — the failure
    mode a per-column-only check would miss entirely.
"""

from dataclasses import dataclass
from typing import Optional

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
    top_k = set(real_clean.value_counts().nlargest(k).index)

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
