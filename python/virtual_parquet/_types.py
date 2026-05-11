"""Public dataclass types describing schema and row group plans.

These are the values an Adapter exchanges with the library. All dataclasses are frozen
so adapters can hash and cache them safely.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


def _reject_bool(value: object, label: str) -> None:
    # bool is a subclass of int; accepting it silently turns True/False into 1/0
    # for fields documented as integer counts.
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an int, not a bool")


class ColumnType(Enum):
    """Logical column types supported in v1.

    Out of scope for v1: Date32, Date64, Timestamp, Decimal, Binary (non-UTF8), List,
    Struct, Map. These will be added under future features.
    """

    INT32 = "int32"
    INT64 = "int64"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    BOOLEAN = "boolean"
    STRING = "string"


@dataclass(frozen=True)
class Column:
    """A single column declaration within a Schema."""

    name: str
    type: ColumnType
    nullable: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Column.name must be non-empty")
        if "." in self.name or "/" in self.name:
            raise ValueError(f"Column.name {self.name!r} must not contain '.' or '/'")


@dataclass(frozen=True)
class Schema:
    """The structural description of the data an adapter produces."""

    columns: tuple[Column, ...]

    def __post_init__(self) -> None:
        # Coerce list/iterable input to tuple to honor the frozen-hashability promise
        # in the module docstring; passing a list would otherwise leave a mutable
        # container in a "frozen" dataclass.
        if not isinstance(self.columns, tuple):  # pyright: ignore[reportUnnecessaryIsInstance]
            object.__setattr__(self, "columns", tuple(self.columns))
        if not self.columns:
            raise ValueError("Schema.columns must be non-empty")
        names = [c.name for c in self.columns]
        if len(set(names)) != len(names):
            raise ValueError("Schema column names must be unique")


@dataclass(frozen=True)
class ColumnStatistics:
    """Per-column, per-row-group declared statistics.

    Any field set to ``None`` is emitted as absent in the Parquet footer (never
    fabricated). The library does not validate adapter-supplied min/max against
    observed data; the adapter is responsible for fidelity.
    """

    min: object | None = None
    max: object | None = None
    null_count: int | None = None

    def __post_init__(self) -> None:
        if self.null_count is not None:
            _reject_bool(self.null_count, "ColumnStatistics.null_count")
            if self.null_count < 0:
                raise ValueError(
                    f"ColumnStatistics.null_count must be >= 0, got {self.null_count}"
                )


@dataclass(frozen=True)
class RowGroupPlan:
    """An adapter's structural declaration of one row group.

    Returned during the metadata pre-pass before any data is fetched, so the library can
    construct the Parquet footer with correct offsets and sizes ahead of time.
    """

    rows: int
    column_stats: tuple[ColumnStatistics | None, ...]
    column_byte_sizes: tuple[int | None, ...] | None = None

    def __post_init__(self) -> None:
        _reject_bool(self.rows, "RowGroupPlan.rows")
        if self.rows < 0:
            raise ValueError(f"RowGroupPlan.rows must be >= 0, got {self.rows}")
        # Coerce iterables to tuple to preserve frozen-hashability.
        if not isinstance(self.column_stats, tuple):  # pyright: ignore[reportUnnecessaryIsInstance]
            object.__setattr__(self, "column_stats", tuple(self.column_stats))
        for i, stat in enumerate(self.column_stats):
            if stat is not None and stat.null_count is not None and stat.null_count > self.rows:
                raise ValueError(
                    f"RowGroupPlan.column_stats[{i}].null_count={stat.null_count} "
                    f"exceeds rows={self.rows}"
                )
        if self.column_byte_sizes is not None:
            if not isinstance(self.column_byte_sizes, tuple):  # pyright: ignore[reportUnnecessaryIsInstance]
                object.__setattr__(
                    self, "column_byte_sizes", tuple(self.column_byte_sizes)
                )
            if len(self.column_byte_sizes) != len(self.column_stats):
                raise ValueError(
                    f"RowGroupPlan.column_byte_sizes length {len(self.column_byte_sizes)} "
                    f"must equal column_stats length {len(self.column_stats)}"
                )
            for i, sz in enumerate(self.column_byte_sizes):
                if sz is not None:
                    _reject_bool(sz, f"RowGroupPlan.column_byte_sizes[{i}]")
                    if sz < 0:
                        raise ValueError(
                            f"RowGroupPlan.column_byte_sizes[{i}] must be >= 0, got {sz}"
                        )
