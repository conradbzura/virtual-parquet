"""Conformance: a non-nullable column carrying a null bitmap is rejected before
any divergent bytes flow.

Covers M5: the encoder skips invalid entries regardless of nullable; without an
explicit reject, a non-nullable column with nulls in its Arrow array would
silently produce a values block shorter than the declared row count.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import virtual_parquet as vp

_SCHEMA_NON_NULLABLE = vp.Schema(
    columns=(vp.Column("id", vp.ColumnType.INT64, nullable=False),)
)


class _AdapterReturningNullsForNonNullable(vp.BaseAdapter):
    """Adapter that declares non-nullable but returns a batch with nulls."""

    def __init__(self) -> None:
        self.schema = _SCHEMA_NON_NULLABLE
        self.row_group_count = 1

    def row_group_plan(self, index: int) -> vp.RowGroupPlan:
        return vp.RowGroupPlan(rows=3, column_stats=(None,))

    def fetch(self, index: int) -> pa.RecordBatch:
        # Arrow array with a null bitmap on a "non-nullable" column. PyArrow allows
        # constructing this; the library's job is to reject it.
        return pa.record_batch({"id": pa.array([1, None, 3], type=pa.int64())})


def test_non_nullable_with_null_entries_is_rejected() -> None:
    """
    GIVEN a sync adapter declaring a non-nullable column but yielding a batch
    whose Arrow array contains null entries
    WHEN PyArrow attempts to read the virtual Parquet object
    THEN SchemaMismatchError is raised before any divergent bytes are emitted.
    """
    with (
        pytest.raises(vp.SchemaMismatchError),
        vp.open(_AdapterReturningNullsForNonNullable()) as vpf,
    ):
        pq.read_table(vpf)
