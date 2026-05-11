"""Integration: public API surface conformance.

Conformance check 1 from ``contracts/public-api.md`` — ``virtual_parquet.__all__``
MUST equal the list declared in the contract document exactly. Conformance check 3
— no symbol in ``__all__`` begins with an underscore.
"""

from __future__ import annotations

import virtual_parquet as vp

# This list is the verbatim copy of contracts/public-api.md §"`__all__` declaration".
# Order matters: the contract specifies a fixed order, and that order is preserved by
# `# noqa: RUF022` in __init__.py. If you change either list, change BOTH together
# and bump the contract document.
EXPECTED_ALL: tuple[str, ...] = (
    # Entry point
    "open",
    # Adapter contract
    "Adapter",
    "AsyncAdapter",
    "BaseAdapter",
    "BaseAsyncAdapter",
    # Schema declaration
    "Schema",
    "Column",
    "ColumnType",
    # Row group declaration
    "RowGroupPlan",
    "ColumnStatistics",
    # Engine-facing return type
    "VirtualParquetFile",
    # Errors
    "VirtualParquetError",
    "SchemaMismatchError",
    "StatisticsMismatchError",
    "ByteSizeMismatchError",
    "NonReplayableAdapterError",
)


def test_all_matches_contract_exactly() -> None:
    """
    GIVEN the public-api.md contract document
    WHEN virtual_parquet is imported
    THEN __all__ equals the contract's declaration exactly, including order.
    """
    assert tuple(vp.__all__) == EXPECTED_ALL


def test_no_exported_symbol_starts_with_underscore() -> None:
    """
    GIVEN the published public surface
    WHEN __all__ is enumerated
    THEN no entry begins with an underscore (sanity check that no internal module
    is accidentally exported).
    """
    underscored = [name for name in vp.__all__ if name.startswith("_")]
    assert underscored == [], (
        f"underscore-prefixed symbols are internal and must not appear in __all__: {underscored}"
    )


def test_every_exported_symbol_is_actually_present_on_the_module() -> None:
    """
    GIVEN __all__ as the contract for what the package exports
    WHEN each declared name is looked up on the package
    THEN every name resolves to an attribute (no missing exports).
    """
    missing = [name for name in vp.__all__ if not hasattr(vp, name)]
    assert missing == [], f"declared in __all__ but missing on virtual_parquet: {missing}"
