"""
Simulate a table: model the joint distribution of a real table, draw new rows.

The goal is a stand-in for real data — something you can hand to a developer,
a test suite, or a collaborator in place of records they should not see, that
still behaves like the real thing.

"Behaves like the real thing" is the hard part, and it is deliberately not
just "every column has the right distribution." A generator that gets each
column right in isolation can still scramble how columns move *together*,
and everything interesting in a table lives in that joint structure. So the
model here is one joint Gaussian copula fit over the whole table —
continuous and categorical columns sharing a single latent correlation —
rather than a per-column sampler.

How far that gets you is an open question this repo tries to answer honestly
rather than assume: see certify/ for an independent check of whether a real
statistical conclusion survives the round trip, and examples/decision_check/
for what happened when that check was pointed at this generator's own output.

No LLM, no network. Deterministic given a seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from contracts.types import ColumnFidelity, FieldDict, FieldType
from engine.auditor.fidelity import (
    AuditorConfig,
    _categorical_association_delta,
    _correlation_delta,
    _eval_column,
)
from engine.ingest.loader import _build_field_dict
from engine.privacy import guard_against_duplicates
from engine.prior.grounded import (
    _copula_uniforms,
    _discrete_inverse_cdf,
    _encode_features,
    _fit_discrete_freq,
    _quantile_inverse,
)

_NO_LABEL = ""  # sentinel: this generator has no declared label/target column


@dataclass
class SimulationResult:
    """Everything about one generated table: the data plus what was checked."""
    table_name: str
    synthetic_df: pd.DataFrame
    n_real_rows: int
    n_synthetic_rows: int
    fidelity_passed: bool
    column_reports: List[ColumnFidelity] = field(default_factory=list)
    correlation_delta: Optional[float] = None
    categorical_association_delta: Optional[float] = None
    categorical_worst_pair: Optional[str] = None
    n_duplicates_guarded: int = 0
    identifier_cols: List[str] = field(default_factory=list)
    seed: int = 0

    def summary(self) -> str:
        cfg = AuditorConfig()
        lines = [
            f"{self.table_name}: {self.n_real_rows} real rows -> {self.n_synthetic_rows} synthetic rows",
            f"  fidelity: {'PASS' if self.fidelity_passed else 'FAIL'}",
        ]
        if self.correlation_delta is not None:
            ok = self.correlation_delta <= cfg.correlation_threshold
            lines.append(
                f"    numeric correlations   delta {self.correlation_delta:.3f}"
                f"  (limit {cfg.correlation_threshold})  {'ok' if ok else 'TOO FAR'}")
        if self.categorical_association_delta is not None:
            ok = self.categorical_association_delta <= cfg.categorical_association_threshold
            pair = f" [worst: {self.categorical_worst_pair}]" if self.categorical_worst_pair else ""
            lines.append(
                f"    category vs. measure   delta {self.categorical_association_delta:.3f}"
                f"  (limit {cfg.categorical_association_threshold})  {'ok' if ok else 'TOO FAR'}{pair}")
        failed_cols = [r.col for r in self.column_reports if not r.passed]
        if failed_cols:
            lines.append(f"    columns off on their own distribution: {failed_cols}")
        lines.append(
            f"  privacy: {self.n_duplicates_guarded} row(s) landed on a real record and were moved off"
            "  (no synthetic row copies a real one)")
        return "\n".join(lines)


def generate_table(
    real_df: pd.DataFrame,
    n_rows: int,
    seed: int = 42,
    table_name: str = "table",
) -> SimulationResult:
    """Generate a synthetic version of ``real_df`` with ``n_rows`` rows.

    Every column's marginal distribution is preserved (each synthetic value
    lies on the real support), and the cross-column correlation structure —
    including correlation between categorical and continuous columns — is
    preserved via a joint Gaussian copula fit on the whole table. Identifier
    columns (near-unique keys) are detected and re-minted as fresh unique
    values rather than sampled, since a copula over a near-unique column
    would just reproduce noise.

    Privacy: no synthetic row can be a near-copy of a real row, by
    construction — every value is drawn from a fitted distribution, never
    from perturbing a real anchor row — plus a checked verbatim-duplicate
    guard as a safety net.

    One privacy mechanism was deliberately tried and removed: a δ-distance
    floor, which pushes every synthetic row at least some distance away from
    every real row. It works when the real rows are sparse, but on a dense
    table there is often nowhere legal left to put a row, and its fallback
    for that case re-draws each column independently — destroying exactly
    the joint structure this generator exists to preserve. Measured, not
    assumed: on a 2000-row two-column table it turned a real correlation of
    0.91 into -0.29 in the synthetic output.

    Raises ValueError if every column looks like an identifier (nothing left
    to actually generate), if real_df is empty, or if real_df has missing
    values (v1 scope: fails loudly rather than silently corrupting the copula
    fit — impute or drop nulls before calling this).
    """
    if len(real_df) == 0:
        raise ValueError(f"{table_name}: real_df has no rows")
    if n_rows <= 0:
        raise ValueError(f"{table_name}: n_rows must be positive, got {n_rows}")
    null_cols = [c for c in real_df.columns if real_df[c].isna().any()]
    if null_cols:
        raise ValueError(
            f"{table_name}: columns with missing values are not supported yet: "
            f"{null_cols}. Impute or drop nulls before calling generate_table()."
        )

    rng = np.random.default_rng(seed)
    field_dict: FieldDict = _build_field_dict(real_df, label_col=_NO_LABEL)

    all_cols = list(real_df.columns)
    identifier_cols = [c for c in all_cols if field_dict[c].is_identifier]
    gen_cols = [c for c in all_cols if c not in identifier_cols]
    if not gen_cols:
        raise ValueError(
            f"{table_name}: every column ({all_cols}) looks like an identifier — "
            "nothing left to generate. Check for accidental ID/UUID-like columns."
        )

    synth = _generate_columns(real_df, gen_cols, field_dict, n_rows, rng)
    # Decode categorical codes to real labels BEFORE fidelity/privacy checks —
    # both compare synth values against real_df's values directly, and a
    # categorical column left as numeric codes would never match real_df's
    # string values, silently defeating the duplicate check for that column.
    synth = _decode_categoricals(synth, real_df, gen_cols, field_dict)
    synth = _mint_identifiers(synth, real_df, identifier_cols, field_dict, n_rows)
    synth = synth[all_cols]  # restore original column order

    col_reports = [
        _eval_column(c, real_df[c], synth[c], field_dict[c].field_type, AuditorConfig())
        for c in gen_cols
    ]
    cfg = AuditorConfig()
    corr_delta = _correlation_delta(real_df, synth, field_dict, _NO_LABEL)
    corr_ok = corr_delta is None or corr_delta <= cfg.correlation_threshold
    cat_delta, cat_worst_pair = _categorical_association_delta(
        real_df, synth, field_dict, _NO_LABEL, cfg)
    cat_ok = cat_delta is None or cat_delta <= cfg.categorical_association_threshold
    fidelity_passed = all(r.passed for r in col_reports) and corr_ok and cat_ok

    synth, n_dup = guard_against_duplicates(synth, real_df, field_dict, _NO_LABEL, rng)
    synth = _apply_numeric_constraints(synth, gen_cols, field_dict)
    synth = synth[all_cols]

    return SimulationResult(
        table_name=table_name,
        synthetic_df=synth,
        n_real_rows=len(real_df),
        n_synthetic_rows=n_rows,
        fidelity_passed=fidelity_passed,
        column_reports=col_reports,
        correlation_delta=corr_delta,
        categorical_association_delta=cat_delta,
        categorical_worst_pair=cat_worst_pair,
        n_duplicates_guarded=n_dup,
        identifier_cols=identifier_cols,
        seed=seed,
    )


def generate_tables(
    tables: Dict[str, pd.DataFrame],
    n_rows: int | Dict[str, int],
    seed: int = 42,
) -> Dict[str, SimulationResult]:
    """Generate a synthetic version of each table in ``tables``.

    ``n_rows`` is either one row count applied to every table, or a
    {table_name: n_rows} dict for per-table sizes. Each table gets its own
    derived seed (seed + index) so results are still reproducible but tables
    don't share identical latent draws.
    """
    results: Dict[str, SimulationResult] = {}
    for i, (name, df) in enumerate(tables.items()):
        rows = n_rows[name] if isinstance(n_rows, dict) else n_rows
        results[name] = generate_table(df, rows, seed=seed + i, table_name=name)
    return results


# ── Internals ──────────────────────────────────────────────────────────────────

def _generate_columns(
    real_df: pd.DataFrame, gen_cols: List[str], field_dict: FieldDict,
    n_rows: int, rng: np.random.Generator,
) -> pd.DataFrame:
    """Fit a joint Gaussian copula on ``real_df[gen_cols]`` and draw n_rows from it.

    Every column (continuous and discrete together) shares one latent
    correlation, so cross-column relationships survive — a per-column
    independent sampler would reproduce each marginal but erase exactly the
    structure a downstream analysis depends on.

    Binary columns are remapped to {0.0, 1.0} before encoding, regardless of
    their real values (e.g. {1, 2}, {"Y", "N"} coded numeric upstream). Found
    by testing against real data, not assumed: _fit_discrete_freq's binary
    path clips codes to [0, 1], so a raw {1, 2}-valued column collapses both
    codes into bucket 1 and the whole column comes out constant — a silent
    fidelity failure that only a table with a non-0/1 binary column exposes.
    """
    work = real_df[gen_cols].copy()
    for col in gen_cols:
        if field_dict[col].field_type == FieldType.BINARY:
            vals = sorted(pd.unique(real_df[col].dropna()))
            lo = vals[0]
            work[col] = (work[col] != lo).astype(np.float64)  # lo -> 0.0, hi -> 1.0

    X = _encode_features(work, field_dict)
    is_continuous = np.array(
        [field_dict[c].field_type == FieldType.CONTINUOUS for c in gen_cols]
    )
    disc_idx = np.where(~is_continuous)[0]
    disc_freq = _fit_discrete_freq(X, disc_idx, gen_cols, field_dict)

    U = _copula_uniforms(X.astype(np.float64), n_rows, rng)
    Xs = np.zeros((n_rows, len(gen_cols)), dtype=np.float64)
    disc_set = set(int(j) for j in disc_idx)
    Xf = X.astype(np.float64)
    for j, col in enumerate(gen_cols):
        if j in disc_set:
            freq = disc_freq.get(col)
            if freq is None or freq.size == 0:
                Xs[:, j] = Xf[0, j]
            else:
                Xs[:, j] = _discrete_inverse_cdf(freq, U[:, j])
        else:
            Xs[:, j] = _quantile_inverse(Xf[:, j], U[:, j])

    return pd.DataFrame(Xs, columns=gen_cols)


def _decode_categoricals(
    synth: pd.DataFrame, real_df: pd.DataFrame, gen_cols: List[str], field_dict: FieldDict,
) -> pd.DataFrame:
    """Map categorical/binary columns from copula-sampled numeric codes back to
    their real values. Must run before any check that compares synth values
    against real_df values directly (fidelity, duplicate guard) — codes vs.
    real labels would never match, silently no-op'ing those checks."""
    out = synth.copy()
    for col in gen_cols:
        meta = field_dict[col]
        if meta.field_type == FieldType.CATEGORICAL:
            cats = meta.categories or list(pd.Categorical(real_df[col].dropna()).categories)
            codes = out[col].round().astype(int).clip(0, max(len(cats) - 1, 0))
            out[col] = pd.Categorical.from_codes(codes, categories=cats).astype(object)
        elif meta.field_type == FieldType.BINARY:
            vals = sorted(pd.unique(real_df[col].dropna()))
            lo, hi = (vals[0], vals[-1]) if len(vals) >= 2 else (vals[0], vals[0])
            out[col] = np.where(out[col].to_numpy() >= 0.5, hi, lo)
    return out


def _apply_numeric_constraints(
    synth: pd.DataFrame, gen_cols: List[str], field_dict: FieldDict,
) -> pd.DataFrame:
    """Fold continuous columns back onto valid values: clip to the observed
    range, round integer-valued columns. Never invents a value the real data
    never showed — this only constrains the copula's real-valued output
    toward what the column can actually be. Categorical/binary columns are
    already decoded to their real values by this point (_decode_categoricals)."""
    out = synth.copy()
    for col in gen_cols:
        meta = field_dict[col]
        if meta.field_type == FieldType.CONTINUOUS:
            if meta.min_val is not None and meta.max_val is not None:
                out[col] = out[col].clip(meta.min_val, meta.max_val)
            if meta.is_integer:
                out[col] = out[col].round().astype("int64")
    return out


def _mint_identifiers(
    synth: pd.DataFrame, real_df: pd.DataFrame, identifier_cols: List[str],
    field_dict: FieldDict, n_rows: int,
) -> pd.DataFrame:
    """Replace identifier columns with fresh unique values, never sampled —
    a copula over a near-unique key would just reproduce noise, and reusing a
    real ID verbatim is the strongest re-identification signal there is."""
    out = synth.copy()
    for col in identifier_cols:
        meta = field_dict[col]
        if meta.is_integer and meta.max_val is not None:
            start = int(meta.max_val) + 1
            out[col] = np.arange(start, start + n_rows, dtype="int64")
        else:
            out[col] = [f"{col}-{i}" for i in range(1, n_rows + 1)]
    return out
