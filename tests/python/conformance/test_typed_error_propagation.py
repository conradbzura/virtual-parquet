"""Conformance: typed VirtualParquetError-subclass exceptions raised inside an
adapter propagate verbatim through the binding.

Covers M16: when a user adapter raises e.g. ``vp.NonReplayableAdapterError`` from
``fetch``, the engine must see the same typed class — not a flattened base
``VirtualParquetError`` carrying the message in its repr.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import virtual_parquet as vp

_SCHEMA = vp.Schema(
    columns=(vp.Column("id", vp.ColumnType.INT64, nullable=False),)
)


class _AdapterRaisingTyped(vp.BaseAdapter):
    """Adapter that raises a typed VirtualParquetError-subclass exception from fetch."""

    def __init__(self, exception_cls: type[BaseException]) -> None:
        self.schema = _SCHEMA
        self.row_group_count = 1
        self._exc_cls = exception_cls

    def row_group_plan(self, index: int) -> vp.RowGroupPlan:
        return vp.RowGroupPlan(rows=3, column_stats=(None,))

    def fetch(self, index: int) -> pa.RecordBatch:
        raise self._exc_cls(f"adapter chose to raise {self._exc_cls.__name__}")


def test_non_replayable_adapter_error_propagates_typed() -> None:
    """
    GIVEN an adapter whose fetch raises NonReplayableAdapterError
    WHEN PyArrow attempts to read the virtual Parquet object
    THEN the caller catches NonReplayableAdapterError specifically (not the base).
    """
    adapter = _AdapterRaisingTyped(vp.NonReplayableAdapterError)
    with pytest.raises(vp.NonReplayableAdapterError), vp.open(adapter) as vpf:
        pq.read_table(vpf)


def test_schema_mismatch_error_raised_by_adapter_propagates_typed() -> None:
    """
    GIVEN an adapter whose fetch raises SchemaMismatchError directly
    WHEN PyArrow attempts to read the virtual Parquet object
    THEN the caller catches SchemaMismatchError specifically.
    """
    adapter = _AdapterRaisingTyped(vp.SchemaMismatchError)
    with pytest.raises(vp.SchemaMismatchError), vp.open(adapter) as vpf:
        pq.read_table(vpf)


def test_arbitrary_exception_does_not_get_typed_classification() -> None:
    """
    GIVEN an adapter whose fetch raises a generic RuntimeError
    WHEN PyArrow attempts to read the virtual Parquet object
    THEN an exception propagates (the exact base may be VirtualParquetError or
    PyArrow's wrapper); critically, the read does not silently succeed.
    """
    adapter = _AdapterRaisingTyped(RuntimeError)
    with pytest.raises(Exception), vp.open(adapter) as vpf:  # noqa: B017
        pq.read_table(vpf)
