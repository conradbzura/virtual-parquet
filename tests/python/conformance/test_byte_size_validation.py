"""Conformance: adapter-declared byte sizes for fixed-width columns must match
the library-computed value.

Covers M2: data-model.md §RowGroupPlan rule that fixed-width column_byte_sizes
declared by the adapter MUST equal the library's deterministic computation;
otherwise the footer's offsets diverge from the bytes the encoder will emit.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import virtual_parquet as vp

_SCHEMA = vp.Schema(
    columns=(vp.Column("id", vp.ColumnType.INT64, nullable=False),)
)


class _AdapterDeclaringWrongByteSize(vp.BaseAdapter):
    """Adapter that declares an incorrect byte size for a fixed-width INT64 column.

    Three rows of int64 should be exactly 24 bytes (3 * 8). The adapter declares
    99 to drive the validation failure path.
    """

    def __init__(self) -> None:
        self.schema = _SCHEMA
        self.row_group_count = 1

    def row_group_plan(self, index: int) -> vp.RowGroupPlan:
        return vp.RowGroupPlan(
            rows=3,
            column_stats=(None,),
            column_byte_sizes=(99,),
        )

    def fetch(self, index: int) -> pa.RecordBatch:
        return pa.record_batch({"id": pa.array([1, 2, 3], type=pa.int64())})


def test_wrong_fixed_width_byte_size_is_rejected_before_footer() -> None:
    """
    GIVEN a sync adapter that declares a fixed-width column_byte_sizes value disagreeing
    with the library-computed size
    WHEN PyArrow attempts to read the virtual Parquet object
    THEN ByteSizeMismatchError is raised before any footer bytes flow.
    """
    with (
        pytest.raises(vp.ByteSizeMismatchError),
        vp.open(_AdapterDeclaringWrongByteSize()) as vpf,
    ):
        pq.read_table(vpf)
