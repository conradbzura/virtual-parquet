"""Conformance: an adapter that yields LargeUtf8 (i64 offsets) round-trips correctly.

Covers M8: LargeUtf8 must be encoded the same way as Utf8 in Parquet's BYTE_ARRAY
Plain encoding (4-byte LE length prefix + UTF-8 bytes per non-null value); only
the Arrow offset width differs.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

import virtual_parquet as vp

_SCHEMA = vp.Schema(
    columns=(vp.Column("label", vp.ColumnType.STRING, nullable=True),)
)


class _LargeStringAdapter(vp.BaseAdapter):
    """Sync adapter that returns a LargeStringArray instead of a StringArray."""

    def __init__(self, values: list[str | None]) -> None:
        self.schema = _SCHEMA
        self.row_group_count = 1
        self._values = values
        self._batch = pa.record_batch(
            {"label": pa.array(values, type=pa.large_string())}
        )

    def row_group_plan(self, index: int) -> vp.RowGroupPlan:
        return vp.RowGroupPlan(rows=len(self._values), column_stats=(None,))

    def fetch(self, index: int) -> pa.RecordBatch:
        return self._batch


def test_large_string_round_trips_through_pyarrow() -> None:
    """
    GIVEN a sync adapter yielding a LargeStringArray (i64 offsets)
    WHEN PyArrow reads the virtual Parquet object
    THEN the returned Table contains the same string values, including null entries.
    """
    values: list[str | None] = ["alpha", "beta", None, "delta"]
    with vp.open(_LargeStringAdapter(values)) as vpf:
        table = pq.read_table(vpf)
    assert table.num_rows == len(values)
    assert table.column("label").to_pylist() == values
