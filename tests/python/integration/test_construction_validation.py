"""Integration: validation rules at the Python construction boundary.

Covers system-boundary validation (per CLAUDE.md) on Column, ColumnStatistics,
and RowGroupPlan dataclasses — the moment user-supplied values enter the library.
"""

from __future__ import annotations

import pytest

import virtual_parquet as vp


def test_column_type_must_be_column_type_instance() -> None:
    """
    GIVEN a string passed in place of a ColumnType enum value
    WHEN Column is constructed
    THEN TypeError is raised at the construction boundary.
    """
    with pytest.raises(TypeError):
        vp.Column("id", "int64", nullable=False)  # type: ignore[arg-type]


def test_column_type_accepts_only_column_type_enum() -> None:
    """
    GIVEN a ColumnType enum value
    WHEN Column is constructed
    THEN the construction succeeds.
    """
    col = vp.Column("id", vp.ColumnType.INT64, nullable=False)
    assert col.type is vp.ColumnType.INT64


def test_row_group_plan_rejects_null_count_exceeding_rows() -> None:
    """
    GIVEN a ColumnStatistics whose null_count is greater than the row count
    WHEN RowGroupPlan is constructed
    THEN ValueError is raised at the construction boundary.
    """
    with pytest.raises(ValueError, match="null_count"):
        vp.RowGroupPlan(
            rows=3,
            column_stats=(vp.ColumnStatistics(null_count=4),),
        )


def test_row_group_plan_accepts_null_count_equal_to_rows() -> None:
    """
    GIVEN a ColumnStatistics whose null_count equals the row count (all-null column)
    WHEN RowGroupPlan is constructed
    THEN the construction succeeds (the all-null edge case is valid; any min/max rule
    is enforced separately).
    """
    plan = vp.RowGroupPlan(
        rows=3,
        column_stats=(vp.ColumnStatistics(null_count=3),),
    )
    assert plan.column_stats[0] is not None
    assert plan.column_stats[0].null_count == 3


def test_row_group_plan_rejects_byte_sizes_length_mismatch() -> None:
    """
    GIVEN a column_byte_sizes tuple whose length differs from column_stats
    WHEN RowGroupPlan is constructed
    THEN ValueError is raised at the construction boundary.
    """
    with pytest.raises(ValueError, match="column_byte_sizes length"):
        vp.RowGroupPlan(
            rows=3,
            column_stats=(None, None),
            column_byte_sizes=(8,),
        )


def test_row_group_plan_accepts_matching_byte_sizes_length() -> None:
    """
    GIVEN a column_byte_sizes tuple whose length matches column_stats
    WHEN RowGroupPlan is constructed
    THEN the construction succeeds.
    """
    plan = vp.RowGroupPlan(
        rows=3,
        column_stats=(None, None),
        column_byte_sizes=(8, None),
    )
    assert plan.column_byte_sizes == (8, None)
