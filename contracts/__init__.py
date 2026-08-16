"""
Contracts — shared types used by basic/ and the engine modules it reuses.
No LLM clients, no networking, no agent framework imports.
"""

from .types import ColumnFidelity, FieldDict, FieldMeta, FieldType

__all__ = [
    "ColumnFidelity",
    "FieldDict",
    "FieldMeta",
    "FieldType",
]
