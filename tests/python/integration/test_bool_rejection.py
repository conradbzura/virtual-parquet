"""Integration: Python bool is not silently accepted in int-typed contract fields.

Covers M18. Python's ``bool`` subclasses ``int``, so without an explicit guard
``True`` would silently coerce to 1 and ``False`` to 0 in fields like
``row_group_count`` or ``RowGroupPlan.rows``.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import virtual_parquet as vp

_SCHEMA = vp.Schema(
    columns=(vp.Column("id", vp.ColumnType.INT64, nullable=False),)
)


class _AdapterWithBoolRowGroupCount(vp.BaseAdapter):
    """Adapter whose row_group_count is a bool, not an int."""

    schema = _SCHEMA
    row_group_count = True  # deliberately wrong for the test

    def row_group_plan(self, index: int) -> vp.RowGroupPlan:
        return vp.RowGroupPlan(rows=3, column_stats=(None,))

    def fetch(self, index: int) -> pa.RecordBatch:
        return pa.record_batch({"id": pa.array([1, 2, 3], type=pa.int64())})


def test_row_group_count_bool_is_rejected() -> None:
    """
    GIVEN an adapter whose row_group_count is True
    WHEN vp.open is called
    THEN TypeError is raised — bool is not a valid int for this contract field.
    """
    with pytest.raises(TypeError):
        vp.open(_AdapterWithBoolRowGroupCount())
