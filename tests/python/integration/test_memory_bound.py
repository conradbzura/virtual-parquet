"""Integration: peak memory bounded by O(1) row groups in flight.

Covers FR-009 / SC-003. The library's design (LRU cache cap 1, on-demand encoding)
enforces this structurally — but no other test exercises it behaviorally end-to-end.

Strategy: build an adapter that lazily generates each row group's batch from
parameters (no retained per-row-group state on the adapter), stream the file
through PyArrow's ``iter_batches`` (which itself does not assemble a full Table),
and track ``tracemalloc`` peak across the iteration. Assert peak is bounded by a
small constant multiple of a single row group's encoded size, not by the total
dataset size.
"""

from __future__ import annotations

import gc
import tracemalloc

import pyarrow as pa
import pyarrow.parquet as pq

import virtual_parquet as vp

_SCHEMA = vp.Schema(
    columns=(vp.Column("id", vp.ColumnType.INT64, nullable=False),)
)
_ROW_GROUPS = 200
_ROWS_PER_GROUP = 5_000
_BYTES_PER_ROW = 8  # int64 plain-encoded
_PER_GROUP_BYTES = _ROWS_PER_GROUP * _BYTES_PER_ROW  # 40 KiB
_TOTAL_BYTES = _ROW_GROUPS * _PER_GROUP_BYTES  # ~7.6 MiB


class _LazyManyRowGroupAdapter(vp.BaseAdapter):
    """Generates each row group's int64 batch on demand; retains no per-group state."""

    def __init__(self) -> None:
        self.schema = _SCHEMA
        self.row_group_count = _ROW_GROUPS

    def row_group_plan(self, index: int) -> vp.RowGroupPlan:
        return vp.RowGroupPlan(rows=_ROWS_PER_GROUP, column_stats=(None,))

    def fetch(self, index: int) -> pa.RecordBatch:
        base = index * _ROWS_PER_GROUP
        return pa.record_batch(
            {"id": pa.array(range(base, base + _ROWS_PER_GROUP), type=pa.int64())}
        )


def test_streaming_read_peak_memory_bounded_by_constant_row_groups() -> None:
    """
    GIVEN a sync adapter producing 200 row groups of 5000 int64 rows each (~7.6 MiB total)
    WHEN PyArrow streams the virtual Parquet file batch by batch via iter_batches
    THEN tracemalloc-measured peak allocation during the iteration is bounded by a
    small constant multiple of one row group's encoded size — well below the total
    dataset size — confirming the library does not retain per-row-group state.
    """
    rows_seen = 0
    expected_first_id = 0

    gc.collect()
    tracemalloc.start()
    try:
        with vp.open(_LazyManyRowGroupAdapter()) as vpf:
            pf = pq.ParquetFile(vpf)
            assert pf.metadata.num_row_groups == _ROW_GROUPS
            assert pf.metadata.num_rows == _ROW_GROUPS * _ROWS_PER_GROUP

            for batch in pf.iter_batches(batch_size=_ROWS_PER_GROUP):
                # Sanity: order is preserved.
                ids = batch.column("id")
                assert ids[0].as_py() == expected_first_id
                rows_seen += batch.num_rows
                expected_first_id += batch.num_rows
                # Drop the reference; iter_batches is itself a generator.
                del batch

        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert rows_seen == _ROW_GROUPS * _ROWS_PER_GROUP

    # If the library retained every row group's encoded bytes, peak would be at
    # least _TOTAL_BYTES (~7.6 MiB). With LRU cap 1, peak should be a small
    # multiple of one row group (~40 KiB) plus PyArrow's working set. We allow
    # a generous 25% of the total dataset size as the upper bound; in practice
    # we expect peak to be <<10% of total.
    bound = _TOTAL_BYTES // 4
    assert peak_bytes < bound, (
        f"peak allocation {peak_bytes} bytes exceeded bound {bound} bytes "
        f"(total dataset is {_TOTAL_BYTES} bytes); the library appears to retain "
        "per-row-group state, violating FR-009 / SC-003."
    )
