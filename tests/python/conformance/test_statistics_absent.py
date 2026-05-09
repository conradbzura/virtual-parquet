"""Conformance: declined statistics are emitted as absent in the Parquet footer.

Covers FR-008: when an adapter declines to declare a statistic, the library MUST
emit the corresponding Statistics field as absent rather than fabricating a value.
PyArrow surfaces this via ``column.statistics is None`` or, when a Statistics
object is present, via ``has_null_count`` / ``has_min_max`` returning False.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

import virtual_parquet as vp

_SCHEMA = vp.Schema(
    columns=(
        vp.Column("id", vp.ColumnType.INT64, nullable=False),
        vp.Column("label", vp.ColumnType.STRING, nullable=True),
    )
)


class _StatsDeclinedAdapter(vp.BaseAdapter):
    """Sync adapter that declines stats by passing None for every column."""

    def __init__(self) -> None:
        self.schema = _SCHEMA
        self.row_group_count = 1

    def row_group_plan(self, index: int) -> vp.RowGroupPlan:
        return vp.RowGroupPlan(
            rows=3,
            column_stats=(None, None),
        )

    def fetch(self, index: int) -> pa.RecordBatch:
        return pa.record_batch(
            {
                "id": pa.array([1, 2, 3], type=pa.int64()),
                "label": pa.array(["a", "b", "c"], type=pa.string()),
            }
        )


class _StatsAllFieldsNoneAdapter(vp.BaseAdapter):
    """Sync adapter that declines stats by passing ColumnStatistics with all-None fields."""

    def __init__(self) -> None:
        self.schema = _SCHEMA
        self.row_group_count = 1

    def row_group_plan(self, index: int) -> vp.RowGroupPlan:
        return vp.RowGroupPlan(
            rows=3,
            column_stats=(
                vp.ColumnStatistics(min=None, max=None, null_count=None),
                vp.ColumnStatistics(min=None, max=None, null_count=None),
            ),
        )

    def fetch(self, index: int) -> pa.RecordBatch:
        return pa.record_batch(
            {
                "id": pa.array([1, 2, 3], type=pa.int64()),
                "label": pa.array(["a", "b", "c"], type=pa.string()),
            }
        )


class _StatsMixedAdapter(vp.BaseAdapter):
    """Sync adapter that declares stats for ``id`` but declines for ``label``."""

    def __init__(self) -> None:
        self.schema = _SCHEMA
        self.row_group_count = 1

    def row_group_plan(self, index: int) -> vp.RowGroupPlan:
        return vp.RowGroupPlan(
            rows=3,
            column_stats=(
                vp.ColumnStatistics(min=1, max=3, null_count=0),
                None,
            ),
        )

    def fetch(self, index: int) -> pa.RecordBatch:
        return pa.record_batch(
            {
                "id": pa.array([1, 2, 3], type=pa.int64()),
                "label": pa.array(["a", "b", "c"], type=pa.string()),
            }
        )


def _column_stats_absent(stats: pq.Statistics | None) -> bool:
    """True iff the column's footer Statistics declares no min/max and no null_count."""
    if stats is None:
        return True
    return not stats.has_null_count and not stats.has_min_max


def test_all_columns_declined_via_none_emit_absent_statistics() -> None:
    """
    GIVEN a sync adapter that passes None for every entry of column_stats
    WHEN PyArrow reads the virtual Parquet object's footer
    THEN every column's Statistics is absent (None) or has no min/max and no null_count.
    """
    with vp.open(_StatsDeclinedAdapter()) as vpf:
        meta = pq.ParquetFile(vpf).metadata
        rg = meta.row_group(0)
        for col_idx in range(rg.num_columns):
            stats = rg.column(col_idx).statistics
            assert _column_stats_absent(stats), (
                f"column {col_idx} stats unexpectedly present: "
                f"has_null_count={stats.has_null_count if stats else None}, "
                f"has_min_max={stats.has_min_max if stats else None}, "
                f"null_count={stats.null_count if stats else None}"
            )


def test_all_columns_declined_via_all_none_fields_emit_absent_statistics() -> None:
    """
    GIVEN a sync adapter that passes ColumnStatistics(min=None, max=None, null_count=None)
    WHEN PyArrow reads the virtual Parquet object's footer
    THEN every column's Statistics is absent or has no min/max and no null_count.
    """
    with vp.open(_StatsAllFieldsNoneAdapter()) as vpf:
        meta = pq.ParquetFile(vpf).metadata
        rg = meta.row_group(0)
        for col_idx in range(rg.num_columns):
            stats = rg.column(col_idx).statistics
            assert _column_stats_absent(stats), (
                f"column {col_idx} stats unexpectedly present despite all fields None: "
                f"has_null_count={stats.has_null_count if stats else None}, "
                f"has_min_max={stats.has_min_max if stats else None}, "
                f"null_count={stats.null_count if stats else None}"
            )


def test_mixed_declared_and_declined_only_declared_appears() -> None:
    """
    GIVEN a sync adapter that declares stats for one column and declines for another
    WHEN PyArrow reads the virtual Parquet object's footer
    THEN the declared column's stats are present with the declared values
    AND the declined column's Statistics is absent or has no min/max and no null_count.
    """
    with vp.open(_StatsMixedAdapter()) as vpf:
        meta = pq.ParquetFile(vpf).metadata
        rg = meta.row_group(0)

        id_stats = rg.column(0).statistics
        assert id_stats is not None, "declared id stats missing"
        assert id_stats.has_null_count
        assert id_stats.null_count == 0
        assert id_stats.has_min_max
        assert id_stats.min == 1
        assert id_stats.max == 3

        label_stats = rg.column(1).statistics
        assert _column_stats_absent(label_stats), (
            f"label column declined stats but Statistics surfaced as present: "
            f"has_null_count={label_stats.has_null_count if label_stats else None}, "
            f"has_min_max={label_stats.has_min_max if label_stats else None}, "
            f"null_count={label_stats.null_count if label_stats else None}"
        )
