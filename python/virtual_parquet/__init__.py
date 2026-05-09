"""virtual-parquet: scaffolding for serving non-Parquet data sources to mainstream
Parquet-consuming engines via a stable Python adapter contract.

Quickstart::

    import virtual_parquet as vp
    import pyarrow.parquet as pq

    class MyAdapter(vp.BaseAdapter):
        schema = vp.Schema(columns=(vp.Column("id", vp.ColumnType.INT64),))
        row_group_count = 1

        def row_group_plan(self, index):
            return vp.RowGroupPlan(rows=3, column_stats=(None,))

        def fetch(self, index):
            import pyarrow as pa
            return pa.record_batch({"id": pa.array([1, 2, 3], type=pa.int64())})

    table = pq.read_table(vp.open(MyAdapter()))

The public surface is enumerated in ``__all__`` below and matches
``contracts/public-api.md``. Symbols outside this list (including everything in
``virtual_parquet._native``, ``virtual_parquet._bridge``, ``virtual_parquet._file``,
``virtual_parquet._types``, and ``virtual_parquet._errors``) are internal.
"""

from __future__ import annotations

from virtual_parquet._bridge import _SyncAdapterFacade
from virtual_parquet._errors import (
    ByteSizeMismatchError,
    NonReplayableAdapterError,
    SchemaMismatchError,
    StatisticsMismatchError,
    VirtualParquetError,
)
from virtual_parquet._file import VirtualParquetFile
from virtual_parquet._native import VirtualFile as _NativeVirtualFile
from virtual_parquet._types import (
    Column,
    ColumnStatistics,
    ColumnType,
    RowGroupPlan,
    Schema,
)
from virtual_parquet.adapter import (
    Adapter,
    AsyncAdapter,
    BaseAdapter,
    BaseAsyncAdapter,
)


def open(adapter: Adapter | AsyncAdapter) -> VirtualParquetFile:
    """Construct a :class:`VirtualParquetFile` over the given adapter.

    Returns immediately; the metadata pre-pass and any data fetches happen lazily on
    the first ``read()`` or ``seek()`` triggered by the consuming engine.

    Accepts either a synchronous :class:`Adapter` or an asynchronous
    :class:`AsyncAdapter`; the binding bridges async to sync transparently via
    ``anyio``.
    """
    bridge = _SyncAdapterFacade(adapter)
    native = _NativeVirtualFile(bridge)
    return VirtualParquetFile(native, bridge)


# NOTE: __all__ order is fixed by contracts/public-api.md and verified by CI.
# Do not reorder — alphabetical sorting is intentionally suppressed.
__all__ = [  # noqa: RUF022
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
]
