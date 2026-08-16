"""
Ingest — infer each column's type (continuous/categorical/binary) and
whether it looks like an identifier, from the data alone. No label column
or declared target required.
"""

from .loader import _build_field_dict

__all__ = ["_build_field_dict"]
