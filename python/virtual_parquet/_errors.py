"""Public error hierarchy.

The actual exception classes are defined in the Rust binding (``_native``) so that
errors raised from Rust are catchable via ``except VirtualParquetError`` without any
runtime translation layer. This module re-exports them under their stable Python names
and attaches Python-side docstrings so ``help()`` and IDE hovers carry the contract
text from ``contracts/public-api.md``.
"""

from __future__ import annotations

from virtual_parquet._native import (
    ByteSizeMismatchError,
    NonReplayableAdapterError,
    SchemaMismatchError,
    StatisticsMismatchError,
    VirtualParquetError,
)

VirtualParquetError.__doc__ = """Base class for all library-raised exceptions.

Catch this to handle any library-domain error uniformly. More specific subclasses
exist for distinct error categories.
"""

SchemaMismatchError.__doc__ = """Adapter-yielded data does not conform to the declared schema.

Raised when an Arrow batch returned from ``Adapter.fetch`` (or ``AsyncAdapter.fetch``)
disagrees with ``Adapter.schema`` in column count, names, types, or nullability, or
when the row count does not match ``RowGroupPlan(index).rows``.
"""

StatisticsMismatchError.__doc__ = (
    """Adapter-declared statistic conflicts with declared schema or row count.

Raised, for example, when ``ColumnStatistics.null_count`` exceeds ``RowGroupPlan.rows``
or when a declared ``min``/``max`` value does not match the column's declared type.
"""
)

ByteSizeMismatchError.__doc__ = (
    """Adapter-declared byte size differs from the library-computed size.

Raised when ``RowGroupPlan.column_byte_sizes`` declares an integer for a fixed-width
column whose Plain-encoded size the library can compute deterministically, and the
declared value does not match.
"""
)

NonReplayableAdapterError.__doc__ = """Adapter signaled it cannot replay a row group fetch.

Raised by an adapter (or by the library on its behalf) to indicate that a second
``fetch(i)`` call for the same row group is not supported. Adapters that cannot
replay MUST declare ``RowGroupPlan.column_byte_sizes`` for every variable-width
column so the library can size the footer without re-fetching.
"""

__all__ = [
    "ByteSizeMismatchError",
    "NonReplayableAdapterError",
    "SchemaMismatchError",
    "StatisticsMismatchError",
    "VirtualParquetError",
]
