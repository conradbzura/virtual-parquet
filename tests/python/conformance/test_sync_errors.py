"""Conformance: synchronous adapter error paths through PyArrow.

Covers the spec edge cases:

- Schema mismatch detection (wrong column count, wrong types, wrong row count) —
  ``SchemaMismatchError`` raised before any divergent bytes are emitted (T027).
- Adapter raises an exception mid-stream after byte emission has begun — PyArrow
  surfaces an error to the caller rather than returning truncated data (T050).
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import virtual_parquet as vp


class _AdapterReturning(vp.BaseAdapter):
    """Sync adapter that declares a schema and returns whatever batch the test supplies."""

    def __init__(
        self,
        schema: vp.Schema,
        declared_rows: int,
        fetched_batch: pa.RecordBatch,
    ) -> None:
        self.schema = schema
        self.row_group_count = 1
        self._declared_rows = declared_rows
        self._batch = fetched_batch

    def row_group_plan(self, index: int) -> vp.RowGroupPlan:
        return vp.RowGroupPlan(
            rows=self._declared_rows,
            column_stats=tuple(None for _ in self.schema.columns),
        )

    def fetch(self, index: int) -> pa.RecordBatch:
        return self._batch


_DECLARED_SCHEMA = vp.Schema(
    columns=(
        vp.Column("id", vp.ColumnType.INT64, nullable=False),
        vp.Column("label", vp.ColumnType.STRING, nullable=True),
    )
)


def _read_through_pyarrow(adapter: vp.Adapter) -> pa.Table:
    with vp.open(adapter) as vpf:
        return pq.read_table(vpf)


def test_wrong_column_count_raises_schema_mismatch() -> None:
    """
    GIVEN an adapter that declares a 2-column schema but yields a 1-column batch
    WHEN PyArrow attempts to read the virtual Parquet object
    THEN a SchemaMismatchError is raised before any column data flows.
    """
    bad_batch = pa.record_batch({"id": pa.array([1, 2, 3], type=pa.int64())})
    adapter = _AdapterReturning(_DECLARED_SCHEMA, declared_rows=3, fetched_batch=bad_batch)

    with pytest.raises(vp.SchemaMismatchError):
        _read_through_pyarrow(adapter)


def test_wrong_column_type_raises_schema_mismatch() -> None:
    """
    GIVEN an adapter that declares int64 + string but yields int32 + string
    WHEN PyArrow attempts to read the virtual Parquet object
    THEN a SchemaMismatchError is raised.
    """
    bad_batch = pa.record_batch(
        {
            "id": pa.array([1, 2, 3], type=pa.int32()),
            "label": pa.array(["a", "b", "c"], type=pa.string()),
        }
    )
    adapter = _AdapterReturning(_DECLARED_SCHEMA, declared_rows=3, fetched_batch=bad_batch)

    with pytest.raises(vp.SchemaMismatchError):
        _read_through_pyarrow(adapter)


def test_wrong_row_count_raises_schema_mismatch() -> None:
    """
    GIVEN an adapter that declares 5 rows in row_group_plan but yields a 3-row batch
    WHEN PyArrow attempts to read the virtual Parquet object
    THEN a SchemaMismatchError is raised.
    """
    bad_batch = pa.record_batch(
        {
            "id": pa.array([1, 2, 3], type=pa.int64()),
            "label": pa.array(["a", "b", "c"], type=pa.string()),
        }
    )
    adapter = _AdapterReturning(_DECLARED_SCHEMA, declared_rows=5, fetched_batch=bad_batch)

    with pytest.raises(vp.SchemaMismatchError):
        _read_through_pyarrow(adapter)


# ---------------------------------------------------------------------------
# T050: adapter raises mid-stream after byte emission has begun.
# ---------------------------------------------------------------------------


class _RaisesOnLaterRowGroup(vp.BaseAdapter):
    """Adapter that fetches successfully for early row groups then raises later."""

    def __init__(self, fail_at: int, total: int, schema: vp.Schema) -> None:
        self.schema = schema
        self.row_group_count = total
        self._fail_at = fail_at
        self._rows_per_group = 100
        self._total = total

    def row_group_plan(self, index: int) -> vp.RowGroupPlan:
        return vp.RowGroupPlan(
            rows=self._rows_per_group,
            column_stats=tuple(None for _ in self.schema.columns),
        )

    def fetch(self, index: int) -> pa.RecordBatch:
        if index >= self._fail_at:
            raise RuntimeError(f"adapter exploded at row group {index}")
        base = index * self._rows_per_group
        return pa.record_batch(
            {
                "id": pa.array(
                    [base + i for i in range(self._rows_per_group)], type=pa.int64()
                ),
                "label": pa.array(
                    [f"rg{index}-row{i}" for i in range(self._rows_per_group)],
                    type=pa.string(),
                ),
            }
        )


def test_mid_stream_exception_is_surfaced_not_silent_truncation() -> None:
    """
    GIVEN an adapter whose fetch succeeds for early row groups but raises on a later index
    WHEN PyArrow reads the virtual Parquet object
    THEN the read raises an error rather than silently returning a truncated Table.
    """
    adapter = _RaisesOnLaterRowGroup(fail_at=2, total=4, schema=_DECLARED_SCHEMA)

    with pytest.raises(Exception) as exc_info, vp.open(adapter) as vpf:
        pq.read_table(vpf)

    raised = exc_info.value
    # Either the underlying RuntimeError propagates, or PyArrow wraps it in a Parquet
    # decoding error. Both are acceptable per the spec edge case ("the engine's error
    # model determines what the user sees"). What is NOT acceptable is silently
    # returning a truncated Table.
    assert raised is not None
