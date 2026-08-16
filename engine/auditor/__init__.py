"""
Auditor — fidelity checks: per-column (TVD/Wasserstein) and cross-column
correlation structure. A batch that looks plausible per-column but breaks
the real correlation structure is worse than no data — this is the check
that catches it.
"""

from .fidelity import AuditorConfig

__all__ = ["AuditorConfig"]
