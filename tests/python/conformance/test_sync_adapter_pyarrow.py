"""Conformance: synchronous adapter end-to-end through PyArrow.

Covers User Story 1 acceptance scenarios 1-3 from spec.md. Each test exercises the
public API only (``vp.open(...)`` -> ``pq.read_table(vpf)``) and asserts against the
``pyarrow.Table`` PyArrow returns. No mocking, patching, or assertions on internal
implementation details.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

import virtual_parquet as vp


class _SingleBatchSyncAdapter(vp.BaseAdapter):
    def __init__(self, batch: pa.RecordBatch, schema: vp.Schema) -> None:
        self.schema = schema
        self.row_group_count = 1
        self._batch = batch

    def row_group_plan(self, index: int) -> vp.RowGroupPlan:
        return vp.RowGroupPlan(
            rows=self._batch.num_rows,
            column_stats=tuple(None for _ in self.schema.columns),
        )

    def fetch(self, index: int) -> pa.RecordBatch:
        return self._batch


class _MultiBatchSyncAdapter(vp.BaseAdapter):
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


def test_single_batch_100_rows_int_and_string() -> None:
    """
    GIVEN a sync adapter declaring a single row group of 100 rows over int64 + string columns
    WHEN PyArrow reads the virtual Parquet object end-to-end
    THEN the returned Table has 100 rows whose values match what the adapter yielded.
    """
    schema = vp.Schema(
        columns=(
            vp.Column("id", vp.ColumnType.INT64, nullable=False),
            vp.Column("label", vp.ColumnType.STRING, nullable=True),
        )
    )
    ids = list(range(100))
    labels = [f"row-{i}" if i % 7 != 0 else None for i in range(100)]
    batch = pa.record_batch(
        {
            "id": pa.array(ids, type=pa.int64()),
            "label": pa.array(labels, type=pa.string()),
        }
    )
    adapter = _SingleBatchSyncAdapter(batch, schema)

    with vp.open(adapter) as vpf:
        table = pq.read_table(vpf)

    assert table.num_rows == 100
    assert table.column_names == ["id", "label"]
    assert table.column("id").to_pylist() == ids
    assert table.column("label").to_pylist() == labels


def test_zero_row_schema_round_trips() -> None:
    """
    GIVEN a sync adapter declaring a valid schema but yielding zero rows
    WHEN PyArrow reads the virtual Parquet object
    THEN it returns an empty Table with the declared schema preserved.
    """
    schema = vp.Schema(
        columns=(
            vp.Column("id", vp.ColumnType.INT64, nullable=False),
            vp.Column("label", vp.ColumnType.STRING, nullable=True),
        )
    )

    class _ZeroRowAdapter(vp.BaseAdapter):
        def __init__(self) -> None:
            self.schema = schema
            self.row_group_count = 0

        def row_group_plan(self, index: int) -> vp.RowGroupPlan:
            raise AssertionError("row_group_plan must not be called when row_group_count is 0")

        def fetch(self, index: int) -> pa.RecordBatch:
            raise AssertionError("fetch must not be called when row_group_count is 0")

    adapter = _ZeroRowAdapter()

    with vp.open(adapter) as vpf:
        table = pq.read_table(vpf)

    assert table.num_rows == 0
    assert table.column_names == ["id", "label"]
    assert pa.types.is_int64(table.schema.field("id").type)
    assert pa.types.is_string(table.schema.field("label").type)


def test_multi_batch_10x1000_preserves_order() -> None:
    """
    GIVEN a sync adapter yielding 10 row groups of 1000 rows each (10000 rows total)
    WHEN PyArrow reads the virtual Parquet object
    THEN it returns all 10000 rows in the order the adapter declared them.
    """
    schema = vp.Schema(
        columns=(
            vp.Column("id", vp.ColumnType.INT64, nullable=False),
            vp.Column("label", vp.ColumnType.STRING, nullable=True),
        )
    )

    batches: list[pa.RecordBatch] = []
    expected_ids: list[int] = []
    expected_labels: list[str | None] = []
    for rg in range(10):
        ids = [rg * 1000 + i for i in range(1000)]
        labels = [f"rg{rg}-row{i}" for i in range(1000)]
        expected_ids.extend(ids)
        expected_labels.extend(labels)
        batches.append(
            pa.record_batch(
                {
                    "id": pa.array(ids, type=pa.int64()),
                    "label": pa.array(labels, type=pa.string()),
                }
            )
        )

    adapter = _MultiBatchSyncAdapter(batches, schema)

    with vp.open(adapter) as vpf:
        table = pq.read_table(vpf)

    assert table.num_rows == 10000
    assert table.column("id").to_pylist() == expected_ids
    assert table.column("label").to_pylist() == expected_labels
