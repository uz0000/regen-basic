"""
Schema inference for the basic generator.

Infers each column's type (continuous / categorical / binary) and whether it
looks like an identifier (a near-unique key that should be re-minted rather
than sampled), from the data alone — no label column or declared target
required.
"""

import numpy as np
import pandas as pd

from contracts.types import FieldDict, FieldMeta, FieldType


def _build_field_dict(df: pd.DataFrame, label_col: str) -> FieldDict:
    field_dict: FieldDict = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            ftype = FieldType.BINARY
        elif pd.api.types.is_numeric_dtype(s):
            ftype = FieldType.BINARY if s.nunique() == 2 else FieldType.CONTINUOUS
        else:
            ftype = FieldType.CATEGORICAL
        # A continuous column whose real values are all whole numbers (counts,
        # hour, Time) must come back as integers — the generator emits floats.
        is_integer = False
        if ftype == FieldType.CONTINUOUS:
            nona = s.dropna()
            is_integer = bool(len(nona)) and bool(np.all(nona == np.floor(nona)))
        # Canonical category order from the full dataset, so encode and decode
        # share one code mapping.
        categories = (
            list(pd.Categorical(s.dropna()).categories)
            if ftype == FieldType.CATEGORICAL else None
        )
        is_identifier = _is_identifier(col, s, len(df), label_col, ftype, is_integer)
        field_dict[col] = FieldMeta(
            name=col,
            field_type=ftype,
            nullable=bool(s.isna().any()),
            cardinality=int(s.nunique()) if ftype == FieldType.CATEGORICAL else None,
            min_val=float(s.min()) if ftype == FieldType.CONTINUOUS else None,
            max_val=float(s.max()) if ftype == FieldType.CONTINUOUS else None,
            is_integer=is_integer,
            categories=categories,
            is_identifier=is_identifier,
        )
    return field_dict


# Threshold above which a non-label column is treated as an identifier key.
_IDENTIFIER_UNIQUE_RATIO = 0.99


def _is_identifier(col, s: pd.Series, n: int, label_col: str,
                   ftype: FieldType, is_integer: bool) -> bool:
    """Conservative, model-free identifier detection.

    A column is an identifier if its values are (near-)unique per row AND it
    looks like a key, not a measurement: integer-valued, or string/categorical,
    or its name hints at an id. A near-unique *float* with no name hint (e.g. a
    continuous sensor reading) is deliberately NOT flagged. Repeated foreign
    keys (e.g. user_id, which recurs across rows) are also out of scope; their
    uniqueness is too low to tell apart from a high-cardinality category.
    """
    if col == label_col or n < 20:
        return False
    ratio = s.nunique(dropna=True) / n if n else 0.0
    if ratio < _IDENTIFIER_UNIQUE_RATIO:
        return False
    name_l = str(col).strip().lower()
    name_hint = (name_l == "id" or name_l.endswith("_id") or name_l.startswith("id_")
                 or any(k in name_l for k in ("uuid", "guid", "email", "hash")))
    dtype_ok = is_integer or ftype == FieldType.CATEGORICAL
    return bool(dtype_ok or name_hint)
