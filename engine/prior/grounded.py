"""
Mixed-data Gaussian copula — the core sampler basic/generate.py fits on a
whole table and draws new rows from.

Every feature column (continuous *and* discrete) is mapped to standard-normal
scores, one latent correlation is estimated across all of them, fresh latent
rows are drawn from that correlation, then each column is mapped back to its
own marginal — continuous columns through their empirical quantiles,
categorical/binary columns through the inverse-CDF of their frequency table.
This preserves each marginal exactly (values lie on the real support) *and*
the full correlation structure — including correlation between discrete and
continuous features — without ever copying a real row.

Sampling every column jointly (not continuous-only with discretes drawn
independently) matters: an earlier version of this sampler drew discrete
columns independently and reproduced their marginals but erased any
correlation *between* a discrete feature and the continuous ones, which
breaks exactly the joint structure a downstream analysis depends on.

No LLM, no network. All randomness flows through the passed Generator.
"""

from typing import Dict, List

import numpy as np
import pandas as pd


def _encode_features(df: pd.DataFrame, field_dict=None) -> np.ndarray:
    """Convert DataFrame to float32 array. Categorical → label-encoded.

    When field_dict is provided, categorical columns are encoded against their
    canonical category order (computed once from the full dataset), so the
    same string maps to the same code consistently. Without it, codes are
    derived per-call from whatever values are present.
    """
    out = df.copy()
    for col in out.columns:
        is_cat = out[col].dtype == object or str(out[col].dtype) == "category"
        if is_cat:
            cats = None
            if field_dict is not None and col in field_dict:
                cats = getattr(field_dict[col], "categories", None)
            if cats is not None:
                out[col] = pd.Categorical(out[col], categories=cats).codes.astype(np.float32)
            else:
                out[col] = pd.Categorical(out[col]).codes.astype(np.float32)
        else:
            out[col] = out[col].astype(np.float32)
    return out.values.astype(np.float32)


def _fit_discrete_freq(
    X: np.ndarray, disc_idx: np.ndarray, feature_cols: List[str], field_dict,
) -> Dict[str, np.ndarray]:
    """Empirical frequency table for each discrete (categorical/binary) column,
    over its canonical code range.

    Sampling from these tables (instead of copying an anchor's value) is what
    stops categorical values being reproduced verbatim. Unseen codes get a
    small epsilon so the sampler never starves a real category. Length matches
    the canonical category count where known, so sampled codes decode cleanly.
    """
    out: Dict[str, np.ndarray] = {}
    if disc_idx.size == 0:
        return out
    Xn = np.asarray(X, dtype=np.float64)
    for j in disc_idx:
        col = feature_cols[j]
        codes = np.rint(Xn[:, j]).astype(int)
        meta = field_dict.get(col) if field_dict is not None else None
        if meta is not None and getattr(meta, "categories", None) is not None:
            n_codes = len(meta.categories)
        else:
            n_codes = int(codes.max()) + 1 if codes.size else 1
        n_codes = max(n_codes, 1)
        counts = np.bincount(
            np.clip(codes, 0, n_codes - 1), minlength=n_codes,
        ).astype(np.float64)
        # Floor every category so one that never shows a value can still
        # produce it rarely (smooths the tail, avoids zero-probability traps).
        counts = counts + 0.5
        out[col] = counts / counts.sum()
    return out


def _normal_scores(x: np.ndarray) -> np.ndarray:
    """Map a 1-D sample to standard-normal scores via the empirical CDF
    (rank → uniform → Φ⁻¹). Constant columns map to 0. This is the latent-
    Gaussian representation a Gaussian copula samples from."""
    from scipy.stats import norm, rankdata
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    if np.allclose(x.std(), 0):
        return np.zeros_like(x)
    u = rankdata(x, method="average") / (x.size + 1.0)
    return norm.ppf(u)


def _quantile_inverse(real_col: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Map uniform values ``u`` back to the scale of ``real_col`` via linear
    interpolation over its sorted empirical quantiles. Preserves the marginal
    distribution exactly (output values lie on the real support)."""
    sv = np.sort(np.asarray(real_col, dtype=np.float64))
    if sv.size == 0:
        return np.zeros_like(u)
    if sv.size == 1:
        return np.full_like(u, sv[0])
    pos = np.clip(u, 0.0, 1.0) * (sv.size - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, sv.size - 1)
    frac = pos - lo
    return sv[lo] * (1.0 - frac) + sv[hi] * frac


def _copula_uniforms(
    source: np.ndarray, n: int, rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``n`` correlated uniform rows from a Gaussian copula fit on
    ``source`` (shape (m, p)), for ALL columns jointly — continuous and discrete.

    Returns an (n, p) matrix of uniforms in (0, 1); mapping each uniform back to
    its column's marginal (continuous → empirical-quantile inverse; discrete →
    inverse-CDF on the frequency table) is the caller's job. Handling every
    column jointly here is what lets a mixed continuous/discrete batch keep the
    cross-correlation between the two kinds of feature.

    The copula factors a joint distribution into (marginals) × (dependence):
      1. Map each source column to standard-normal scores (``_normal_scores``) →
         the latent Gaussian space where dependence is a plain correlation.
      2. Estimate that latent correlation and draw ``n`` fresh latent rows from
         it. With <2 rows or a degenerate/non-finite correlation, fall back to
         the identity (independent columns).
      3. Push each latent column through Φ to a uniform.

    No real row is ever emitted: the uniforms are drawn from a fitted latent,
    and the caller's marginal lookup selects sorted real values / frequency-
    table codes by an independently drawn latent rank.
    """
    from scipy.stats import norm

    m, p = source.shape
    if p == 0:
        return np.zeros((n, 0), dtype=np.float64)

    Z = np.column_stack([_normal_scores(source[:, c]) for c in range(p)])

    corr = np.eye(p)
    if m >= 2:
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.corrcoef(Z, rowvar=False)
        c = np.atleast_2d(c)
        if c.shape == (p, p) and np.all(np.isfinite(c)):
            corr = 0.999 * c + 0.001 * np.eye(p)  # pull toward identity
    L = rng.multivariate_normal(np.zeros(p), corr, size=n)

    return norm.cdf(L)


def _discrete_inverse_cdf(freq: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Map uniforms ``u`` ∈ [0,1] to discrete codes via the inverse CDF of the
    frequency table ``freq`` (probabilities over codes 0..K-1).

    ``code = min{ k : cumsum(freq)[k] ≥ u }``. This reproduces the marginal
    frequency exactly, while the *ordering* of the draw is governed by the
    shared copula latent — so a discrete column co-varies with the continuous
    ones it was correlated with, instead of being sampled in isolation.
    """
    cum = np.cumsum(np.asarray(freq, dtype=np.float64))
    if cum.size:
        cum[-1] = 1.0  # guard fp drift so u≈1 lands on the last code, not past it
    codes = np.searchsorted(cum, np.clip(u, 0.0, 1.0), side="left")
    return np.clip(codes, 0, freq.size - 1).astype(np.float64)
