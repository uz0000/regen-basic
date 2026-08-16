"""
Contracts — shared types used by simulate/ (the generator), certify/ (the
verifier), and the engine modules they reuse. No LLM clients, no networking,
no agent framework imports.
"""

from .types import ColumnFidelity, EstimandSpec, FieldDict, FieldMeta, FieldType

__all__ = [
    "ColumnFidelity",
    "EstimandSpec",
    "FieldDict",
    "FieldMeta",
    "FieldType",
]
