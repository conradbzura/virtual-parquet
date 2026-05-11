"""Conformance: engine-driven out-of-order range serving.

Covers FR-006 / SC-004 — engines issue range requests in arbitrary orders (footer
first, then column chunks; some engines re-read regions). The library must serve
all of these correctly without re-reading the adapter for the same row group.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

import virtual_parquet as vp


class _ReplayableSyncAdapter(vp.BaseAdapter):
    """Adapter whose ``fetch`` is deterministic; safe to call multiple times per index."""

    def __init__(self, batches: list[pa.RecordBatch], schema: vp.Schema) -> None:
        self.schema = schema
        self.row_group_count = len(batches)
        self._batches = batches

    def row_group_plan(self, index: int) -> vp.RowGroupPlan:
        return vp.RowGroupPlan(
            rows=self._batches[index].num_rows,
            column_stats=tuple(None for _ in self.schema.columns),
        )

    def fetch(self, index: int) -> pa.RecordBatch:
        return self._batches[index]


def _build_adapter() -> _ReplayableSyncAdapter:
    schema = vp.Schema(
        columns=(
            vp.Column("id", vp.ColumnType.INT64, nullable=False),
            vp.Column("label", vp.ColumnType.STRING, nullable=True),
        )
    )
    batches = [
        pa.record_batch(
            {
                "id": pa.array([rg * 100 + i for i in range(100)], type=pa.int64()),
                "label": pa.array(
                    [f"rg{rg}-row{i}" for i in range(100)], type=pa.string()
                ),
            }
        )
        for rg in range(3)
    ]
    return _ReplayableSyncAdapter(batches, schema)


def test_repeated_full_reads_return_identical_tables() -> None:
    """
    GIVEN a virtual Parquet file built from a replayable sync adapter
    WHEN PyArrow reads it twice via fresh sessions
    THEN both reads produce equal Tables (engines may re-read the same file).
    """
    with vp.open(_build_adapter()) as vpf_a:
        first = pq.read_table(vpf_a)
    with vp.open(_build_adapter()) as vpf_b:
        second = pq.read_table(vpf_b)

    assert first.equals(second)


def test_footer_only_then_full_read_succeeds() -> None:
    """
    GIVEN a virtual Parquet file
    WHEN PyArrow first reads only the footer (via pq.ParquetFile.metadata) and then
    reads the full Table
    THEN the metadata reflects the declared schema and the full read returns the
    expected rows.
    """
    with vp.open(_build_adapter()) as vpf_meta:
        pf = pq.ParquetFile(vpf_meta)
        meta = pf.metadata
        assert meta is not None
        assert meta.num_rows == 300
        assert meta.num_row_groups == 3

    with vp.open(_build_adapter()) as vpf_full:
        table = pq.read_table(vpf_full)

    assert table.num_rows == 300
    assert table.column("id").to_pylist() == list(range(300))


def test_random_access_row_group_reads_match_full_read() -> None:
    """
    GIVEN a virtual Parquet file with multiple row groups
    WHEN row groups are read out of order via pq.ParquetFile.read_row_group
    THEN the concatenation of out-of-order reads matches an in-order full read.
    """
    with vp.open(_build_adapter()) as vpf_full:
        full = pq.read_table(vpf_full)

    with vp.open(_build_adapter()) as vpf_random:
        pf = pq.ParquetFile(vpf_random)
        # Read row groups in reversed order, then re-read RG 0 last to exercise
        # repeated access to an already-served row group.
        reversed_rgs = [pf.read_row_group(i) for i in (2, 1, 0)]
        rg_zero_again = pf.read_row_group(0)

    in_natural_order = pa.concat_tables(list(reversed(reversed_rgs)))
    assert in_natural_order.equals(full)
    assert rg_zero_again.equals(reversed_rgs[2])
