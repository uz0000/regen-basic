"""
Privacy — the verbatim-duplicate guard.

The simulator never anchors a synthetic row on a real one (every value comes
from a fitted copula, sampled fresh), so near-copies are already measure-zero
by construction. This guard is the checked safety net: it finds any synthetic
row that happens to exactly reproduce a real row's full attribute set and
nudges it away.

Deliberately does NOT include a δ-distance floor — the stronger mechanism that
pushes every synthetic row a fixed distance clear of every real row. It was
tried and measured: on a dense table there is frequently nowhere legal left to
place a row, and the fallback for that case re-draws each column
independently, destroying the joint structure the simulator exists to
preserve. See simulate/generate.py's docstring for the numbers. What remains
is not Differential Privacy — it bounds record-level near-copy
re-identification, not aggregate or membership-inference attacks.

Pure Python (numpy + scipy.spatial.cKDTree). No model, no network.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from contracts.types import FieldDict, FieldType

logger = logging.getLogger(__name__)

# A released row that reproduces a real row's full non-identifier attribute set
# is a re-identification risk only when that attribute set *uniquely*
# identifies the real individual. A discrete tuple shared by >= this many real
# rows is k-anonymous — reusing it reveals nothing about any one person — so it
# is not a verbatim leak. Below this count (a singleton real record) it is.
_MIN_ANON_COUNT = 2


def guard_against_duplicates(
    synth_df: pd.DataFrame,
    real_df: pd.DataFrame,
    field_dict: FieldDict,
    label_col: str,
    rng: np.random.Generator,
    tol_sigma: float = 1e-3,
) -> Tuple[pd.DataFrame, int]:
    """Guarantee no released row duplicates a real row's full attribute set.

    A row is a "duplicate" if it matches some real row on every non-identifier
    feature: categorical/binary values equal, and continuous values within
    ``tol_sigma`` (σ-normalized) — i.e. a verbatim repro of a real individual's
    attributes. Identifiers are excluded (they are re-minted fresh and never
    match).

    Duplicate rows are nudged: a tiny σ-scaled jitter is added to their
    continuous features (categoricals re-drawn from the row's own value is a
    no-op; the jitter on continuous is enough to break the exact-attribute
    match while staying in-distribution). Deterministic via ``rng``.

    Returns:
        (guarded_df, n_duplicates): a copy with duplicates nudged, and the count.
    """
    feat = [c for c in synth_df.columns
            if c in field_dict and c != label_col
            and not getattr(field_dict[c], "is_identifier", False)]
    cont = [c for c in feat if field_dict[c].field_type == FieldType.CONTINUOUS]
    disc = [c for c in feat if field_dict[c].field_type in (FieldType.CATEGORICAL,
                                                            FieldType.BINARY)]

    out = synth_df.copy()
    if len(real_df) == 0 or len(synth_df) == 0 or not feat:
        return out, 0

    # Candidate matches via continuous proximity (tiny radius), then confirm the
    # discrete attributes match exactly. Cheap because the radius is measure-zero.
    dup_idx: list = []
    if cont:
        sigma = real_df[cont].to_numpy(dtype=np.float64).std(axis=0)
        sigma = np.where(sigma < 1e-8, 1.0, sigma)
        tree = cKDTree(real_df[cont].to_numpy(dtype=np.float64) / sigma)
        Sc = out[cont].to_numpy(dtype=np.float64) / sigma
        for i, nbrs in enumerate(tree.query_ball_point(Sc, tol_sigma)):
            if not len(nbrs):
                continue
            if disc:
                srow = out.iloc[i][disc]
                if (real_df.iloc[nbrs][disc] == srow).all(axis=1).any():
                    dup_idx.append(i)
            else:
                dup_idx.append(i)
    elif disc:
        # No continuous features — match on discrete signature only, but flag a
        # row only when it reproduces a *singleton* real tuple (a uniquely-
        # identifying record). Tuples shared by >= _MIN_ANON_COUNT real rows are
        # k-anonymous and reusing them is not a leak.
        from collections import Counter
        real_counts = Counter(map(tuple, real_df[disc].to_numpy().tolist()))
        for i in range(len(out)):
            c = real_counts.get(tuple(out.iloc[i][disc].tolist()), 0)
            if 0 < c < _MIN_ANON_COUNT:
                dup_idx.append(i)

    n = len(dup_idx)
    if n and cont:
        sigma = real_df[cont].to_numpy(dtype=np.float64).std(axis=0)
        sigma = np.where(sigma < 1e-8, 1.0, sigma)
        # Nudge just enough to break the exact-attribute match (a few × tol_sigma).
        jitter = rng.standard_normal((n, len(cont))) * sigma * (5.0 * tol_sigma)
        out.iloc[dup_idx, out.columns.get_indexer(cont)] = (
            out.iloc[dup_idx][cont].to_numpy(dtype=np.float64) + jitter
        )

    if n:
        logger.info("Privacy verbatim guard: nudged %d duplicate-attribute rows.", n)
    return out, n
