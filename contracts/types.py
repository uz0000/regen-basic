"""
Shared dataclasses for the basic generator (basic/generate.py) and the
engine modules it reuses (engine/auditor/fidelity.py, engine/ingest/loader.py,
engine/privacy.py, engine/prior/grounded.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


# ── Field dictionary ──────────────────────────────────────────────────────────

class FieldType(str, Enum):
    CONTINUOUS  = "continuous"
    CATEGORICAL = "categorical"
    BINARY      = "binary"
    IDENTIFIER  = "identifier"


@dataclass
class FieldMeta:
    name: str
    field_type: FieldType
    nullable: bool = False
    cardinality: Optional[int] = None  # for categorical fields
    min_val: Optional[float] = None    # for continuous fields
    max_val: Optional[float] = None
    is_integer: bool = False           # continuous field whose real values are all integral
                                       # (counts, hour, Time) → round synthetic output back to int
    categories: Optional[List[object]] = None  # canonical category order (categorical fields),
                                                # computed from the FULL dataset so encode/decode agree
    is_identifier: bool = False        # near-unique key column (order_id, uuid, email) →
                                       # regenerate as fresh unique values, not Gaussian noise


FieldDict = Dict[str, FieldMeta]


# ── Fidelity report (auditor output) ─────────────────────────────────────────

@dataclass
class ColumnFidelity:
    col: str
    tvd: Optional[float] = None         # Total Variation Distance
    wasserstein: Optional[float] = None # Wasserstein-1 (continuous only)
    passed: bool = True
