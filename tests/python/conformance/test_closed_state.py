"""Conformance: `closed` reflects file lifecycle.

Covers M11: the file-protocol's ``closed`` property must report the actual state
so engines that check ``closed`` before ``read`` get the truth.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import virtual_parquet as vp


class _TrivialSyncAdapter(vp.BaseAdapter):
    schema = vp.Schema(
        columns=(vp.Column("id", vp.ColumnType.INT64, nullable=False),)
    )
    row_group_count = 1

    def row_group_plan(self, index: int) -> vp.RowGroupPlan:
        return vp.RowGroupPlan(rows=3, column_stats=(None,))

    def fetch(self, index: int) -> pa.RecordBatch:
        return pa.record_batch({"id": pa.array([1, 2, 3], type=pa.int64())})


def test_closed_property_reflects_state() -> None:
    """
    GIVEN a fresh VirtualParquetFile
    WHEN close() is called
    THEN the closed property transitions from False to True.
    """
    vpf = vp.open(_TrivialSyncAdapter())
    assert vpf.closed is False
    vpf.close()
    assert vpf.closed is True


def test_read_after_close_raises() -> None:
    """
    GIVEN a closed VirtualParquetFile
    WHEN read() is called
    THEN a ValueError is raised (Python file-protocol convention).
    """
    vpf = vp.open(_TrivialSyncAdapter())
    vpf.close()
    with pytest.raises(ValueError, match="closed file"):
        vpf.read()


def test_context_manager_exit_marks_closed() -> None:
    """
    GIVEN a VirtualParquetFile used as a context manager
    WHEN the with-block exits
    THEN the closed property is True.
    """
    with vp.open(_TrivialSyncAdapter()) as vpf:
        assert vpf.closed is False
    assert vpf.closed is True
