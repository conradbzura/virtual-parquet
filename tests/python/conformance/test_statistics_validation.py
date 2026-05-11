"""Conformance: statistics-validation rules from data-model.md.

Covers:

- M3 / strict-reject: empty (0-row) and all-null row groups must not declare
  Some(min) / Some(max) — there is no real value to summarize.
- M15 / typed-value rule: adapter-declared min/max values that do not match the
  column's declared ColumnType must surface as StatisticsMismatchError.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import virtual_parquet as vp

_INT64_SCHEMA = vp.Schema(
    columns=(vp.Column("id", vp.ColumnType.INT64, nullable=True),)
)


class _AdapterWithStats(vp.BaseAdapter):
    """Sync adapter parameterized by rows, batch and stats so each test wires its own."""

    def __init__(
        self,
        schema: vp.Schema,
        rows: int,
        stats: vp.ColumnStatistics,
        batch: pa.RecordBatch,
    ) -> None:
        self.schema = schema
        self.row_group_count = 1
        self._rows = rows
        self._stats = stats
        self._batch = batch

    def row_group_plan(self, index: int) -> vp.RowGroupPlan:
        return vp.RowGroupPlan(rows=self._rows, column_stats=(self._stats,))

    def fetch(self, index: int) -> pa.RecordBatch:
        return self._batch


def test_zero_row_with_min_max_declared_is_rejected() -> None:
    """
    GIVEN a sync adapter declaring a 0-row row group with Some(min) and Some(max)
    WHEN PyArrow attempts to read the virtual Parquet object
    THEN StatisticsMismatchError is raised — there is no real value to summarize.
    """
    schema = vp.Schema(
        columns=(vp.Column("id", vp.ColumnType.INT64, nullable=False),)
    )
    empty_batch = pa.record_batch({"id": pa.array([], type=pa.int64())})
    adapter = _AdapterWithStats(
        schema=schema,
        rows=0,
        stats=vp.ColumnStatistics(min=1, max=10, null_count=0),
        batch=empty_batch,
    )
    with pytest.raises(vp.StatisticsMismatchError), vp.open(adapter) as vpf:
        pq.read_table(vpf)


def test_all_null_with_min_max_declared_is_rejected() -> None:
    """
    GIVEN a sync adapter declaring an all-null row group with Some(min) and Some(max)
    WHEN PyArrow attempts to read the virtual Parquet object
    THEN StatisticsMismatchError is raised — there is no real value to summarize.
    """
    all_null_batch = pa.record_batch(
        {"id": pa.array([None, None, None], type=pa.int64())}
    )
    adapter = _AdapterWithStats(
        schema=_INT64_SCHEMA,
        rows=3,
        stats=vp.ColumnStatistics(min=1, max=10, null_count=3),
        batch=all_null_batch,
    )
    with pytest.raises(vp.StatisticsMismatchError), vp.open(adapter) as vpf:
        pq.read_table(vpf)


def test_min_value_typed_wrong_for_int64_column_is_rejected() -> None:
    """
    GIVEN a sync adapter declaring an Int64 column whose min is a string
    WHEN PyArrow attempts to read the virtual Parquet object
    THEN StatisticsMismatchError is raised at the binding seam, not a generic TypeError.
    """
    schema = vp.Schema(
        columns=(vp.Column("id", vp.ColumnType.INT64, nullable=False),)
    )
    batch = pa.record_batch({"id": pa.array([1, 2, 3], type=pa.int64())})
    adapter = _AdapterWithStats(
        schema=schema,
        rows=3,
        stats=vp.ColumnStatistics(min="not an int", max=3, null_count=0),
        batch=batch,
    )
    with pytest.raises(vp.StatisticsMismatchError), vp.open(adapter) as vpf:
        pq.read_table(vpf)


def test_min_value_bool_for_int64_column_is_rejected() -> None:
    """
    GIVEN a sync adapter declaring an Int64 column whose min is a Python bool
    WHEN PyArrow attempts to read the virtual Parquet object
    THEN StatisticsMismatchError is raised — bool is an int subclass and would
    otherwise silently coerce True -> 1.
    """
    schema = vp.Schema(
        columns=(vp.Column("id", vp.ColumnType.INT64, nullable=False),)
    )
    batch = pa.record_batch({"id": pa.array([1, 2, 3], type=pa.int64())})
    adapter = _AdapterWithStats(
        schema=schema,
        rows=3,
        stats=vp.ColumnStatistics(min=True, max=3, null_count=0),
        batch=batch,
    )
    with pytest.raises(vp.StatisticsMismatchError), vp.open(adapter) as vpf:
        pq.read_table(vpf)
