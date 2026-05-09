"""Public error hierarchy.

The actual exception classes are defined in the Rust binding (``_native``) so that
errors raised from Rust are catchable via ``except VirtualParquetError`` without any
runtime translation layer. This module re-exports them under their stable Python names.
"""

from __future__ import annotations

from virtual_parquet._native import (
    ByteSizeMismatchError,
    NonReplayableAdapterError,
    SchemaMismatchError,
    StatisticsMismatchError,
    VirtualParquetError,
)

__all__ = [
    "ByteSizeMismatchError",
    "NonReplayableAdapterError",
    "SchemaMismatchError",
    "StatisticsMismatchError",
    "VirtualParquetError",
]
