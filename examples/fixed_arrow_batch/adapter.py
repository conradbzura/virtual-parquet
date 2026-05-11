"""Reference adapter used by the quickstart and the integration smoke test.

Teaching code, NOT part of the published ``virtual_parquet`` package surface
(see ``contracts/public-api.md`` and FR-010 / SC-006).
"""

from __future__ import annotations

import pyarrow as pa

import virtual_parquet as vp


class FixedBatchAdapter(vp.BaseAdapter):
    """A trivial sync adapter: yields one row group containing a fixed Arrow batch."""

    def __init__(self) -> None:
        self.schema = vp.Schema(
            columns=(
                vp.Column(name="id", type=vp.ColumnType.INT64, nullable=False),
                vp.Column(name="label", type=vp.ColumnType.STRING, nullable=True),
            )
        )
        self.row_group_count = 1

        self._batch = pa.record_batch(
            {
                "id": pa.array([1, 2, 3, 4], type=pa.int64()),
                "label": pa.array(["alpha", "beta", None, "delta"]),
            }
        )

    def row_group_plan(self, index: int) -> vp.RowGroupPlan:
        return vp.RowGroupPlan(
            rows=4,
            column_stats=(
                vp.ColumnStatistics(min=1, max=4, null_count=0),
                None,
            ),
            column_byte_sizes=None,
        )

    def fetch(self, index: int) -> pa.RecordBatch:
        return self._batch
