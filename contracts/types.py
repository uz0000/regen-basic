"""
Shared dataclasses for the basic generator (basic/generate.py), the engine
modules it reuses (engine/auditor/fidelity.py, engine/ingest/loader.py,
engine/privacy.py, engine/prior/grounded.py), and the certifier (certify/).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


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


# ── Estimand (certifier input) ────────────────────────────────────────────────

@dataclass
class EstimandSpec:
    """A declared analysis whose *estimate* the synthetic data must preserve.

    An estimand is the target quantity of an analysis. v1 supports regression
    coefficients: fit ``outcome ~ predictors`` (``family`` = ols | logit) on the
    real reference to get theta_real +/- CI, fit the *same* spec on the
    synthetic data to get theta_synth, and certify preservation iff every
    coefficient-of-interest's theta_synth is statistically indistinguishable
    from theta_real. Distinct from fidelity (marginals/correlations): a batch
    can pass every fidelity check while a coefficient silently shifts.

    Empty (``outcome == ""``) means no estimand declared.
    """
    outcome: str = ""                                     # dependent-variable column
    predictors: List[str] = field(default_factory=list)   # regressor columns
    family: str = "ols"                                   # "ols" | "logit"
    # Subset of predictors whose recovery is certified; [] -> every predictor.
    coefficients_of_interest: List[str] = field(default_factory=list)
    ci_level: float = 0.95                                # confidence level
    # "consistent" (default): preserved iff theta_real and theta_synth are
    # indistinguishable beyond their combined standard error (two-sample Wald
    # test). "within_ci" (stricter): theta_synth must lie inside theta_real's CI.
    rule: str = "consistent"

    def declared(self) -> bool:
        return bool(self.outcome and self.predictors)

    def targets(self) -> List[str]:
        return list(self.coefficients_of_interest) or list(self.predictors)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outcome": self.outcome,
            "predictors": list(self.predictors),
            "family": self.family,
            "coefficients_of_interest": list(self.coefficients_of_interest),
            "ci_level": self.ci_level,
            "rule": self.rule,
        }
