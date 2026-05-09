//! Rust-side mirror of the Python adapter contract.
//!
//! The types here are the internal Rust representation of `Schema`, `Column`,
//! `RowGroupPlan`, and `ColumnStatistics`. They are constructed from the Python-side
//! types in `bindings.rs` (the only module allowed to hold PyO3 references — see
//! forward-compat discipline #2 in `plan.md`).

use std::collections::HashSet;

use crate::encoding::plain::compute_fixed_value_size;
use crate::error::Error;

/// The logical column types supported in v1.
///
/// Mapped 1:1 to Python's `ColumnType` enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ColumnType {
    Int32,
    Int64,
    Float32,
    Float64,
    Boolean,
    String,
}

impl ColumnType {
    /// Returns true for types whose Plain-encoded byte size depends only on row count
    /// and null bitmap presence (i.e., no data scan required to compute size).
    #[must_use]
    pub(crate) fn is_fixed_width(self) -> bool {
        match self {
            Self::Int32 | Self::Int64 | Self::Float32 | Self::Float64 | Self::Boolean => true,
            Self::String => false,
        }
    }
}

/// A single column declaration within a `Schema`.
///
/// Fields are crate-private so that construction always flows through `Column::new`;
/// the constructor is the single enforcement point for the adapter-protocol's name
/// rules.
#[derive(Debug, Clone)]
pub(crate) struct Column {
    pub(crate) name: String,
    pub(crate) data_type: ColumnType,
    pub(crate) nullable: bool,
}

impl Column {
    /// Construct a column, validating the name.
    pub(crate) fn new(
        name: impl Into<String>,
        data_type: ColumnType,
        nullable: bool,
    ) -> Result<Self, Error> {
        let name = name.into();
        if name.is_empty() {
            return Err(Error::InvalidSchema("Column.name must be non-empty".into()));
        }
        if name.contains('.') || name.contains('/') {
            return Err(Error::InvalidSchema(format!(
                "Column.name {name:?} must not contain '.' or '/'"
            )));
        }
        Ok(Self {
            name,
            data_type,
            nullable,
        })
    }
}

/// The structural description of the data an adapter produces.
#[derive(Debug, Clone)]
pub(crate) struct Schema {
    columns: Vec<Column>,
}

impl Schema {
    /// Build a schema, validating non-emptiness and unique column names.
    pub(crate) fn new(columns: Vec<Column>) -> Result<Self, Error> {
        if columns.is_empty() {
            return Err(Error::InvalidSchema(
                "Schema.columns must be non-empty".into(),
            ));
        }
        let mut seen = HashSet::with_capacity(columns.len());
        for c in &columns {
            if !seen.insert(c.name.as_str()) {
                return Err(Error::InvalidSchema(format!(
                    "Schema column names must be unique; duplicate: {:?}",
                    c.name
                )));
            }
        }
        Ok(Self { columns })
    }

    #[must_use]
    pub(crate) fn columns(&self) -> &[Column] {
        &self.columns
    }

    #[must_use]
    pub(crate) fn len(&self) -> usize {
        self.columns.len()
    }
}

/// A typed scalar used in adapter-declared statistics. Mirrors `ColumnType`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum StatValue {
    Int32(i32),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    Boolean(bool),
    String(String),
}

impl StatValue {
    #[must_use]
    pub(crate) fn matches_type(&self, ty: ColumnType) -> bool {
        matches!(
            (self, ty),
            (Self::Int32(_), ColumnType::Int32)
                | (Self::Int64(_), ColumnType::Int64)
                | (Self::Float32(_), ColumnType::Float32)
                | (Self::Float64(_), ColumnType::Float64)
                | (Self::Boolean(_), ColumnType::Boolean)
                | (Self::String(_), ColumnType::String)
        )
    }

    /// True if the value is a NaN floating-point — Parquet statistics MUST exclude
    /// NaN per the Parquet spec, so this is a contract-violation marker.
    #[must_use]
    pub(crate) fn is_nan(&self) -> bool {
        matches!(self, Self::Float32(v) if v.is_nan())
            || matches!(self, Self::Float64(v) if v.is_nan())
    }
}

/// Per-column, per-row-group declared statistics.
///
/// Any field set to `None` is emitted as **absent** in the Parquet footer (never
/// fabricated as zero or empty). Construct via the builder methods or the binding
/// extractor; `Default` is intentionally not derived to keep "all undeclared" an
/// explicit choice (use `None` at the `Option<ColumnStatistics>` level instead).
#[derive(Debug, Clone)]
pub(crate) struct ColumnStatistics {
    pub(crate) min: Option<StatValue>,
    pub(crate) max: Option<StatValue>,
    pub(crate) null_count: Option<i64>,
}

impl ColumnStatistics {
    pub(crate) fn new() -> Self {
        Self {
            min: None,
            max: None,
            null_count: None,
        }
    }

    /// Cross-check against a single row group's row count and column type.
    pub(crate) fn validate(&self, ty: ColumnType, rows: i64) -> Result<(), Error> {
        if let Some(nc) = self.null_count {
            if nc < 0 {
                return Err(Error::StatisticsMismatch(format!(
                    "null_count must be >= 0, got {nc}"
                )));
            }
            if nc > rows {
                return Err(Error::StatisticsMismatch(format!(
                    "null_count {nc} > row count {rows}"
                )));
            }
        }
        if let Some(min) = &self.min {
            if !min.matches_type(ty) {
                return Err(Error::StatisticsMismatch(format!(
                    "min value type does not match column type {ty:?}"
                )));
            }
            if min.is_nan() {
                return Err(Error::StatisticsMismatch(
                    "min must not be NaN (Parquet statistics exclude NaN)".into(),
                ));
            }
        }
        if let Some(max) = &self.max {
            if !max.matches_type(ty) {
                return Err(Error::StatisticsMismatch(format!(
                    "max value type does not match column type {ty:?}"
                )));
            }
            if max.is_nan() {
                return Err(Error::StatisticsMismatch(
                    "max must not be NaN (Parquet statistics exclude NaN)".into(),
                ));
            }
        }
        // Strict-reject min/max for empty / all-null columns: there is no real value
        // either bound could reflect, so emitting Some(...) would be a fabrication.
        let all_null = matches!(self.null_count, Some(nc) if nc == rows);
        if (rows == 0 || all_null) && (self.min.is_some() || self.max.is_some()) {
            return Err(Error::StatisticsMismatch(format!(
                "min/max must be absent when rows={rows} and null_count={null_count:?} \
                 (no real value to summarize)",
                null_count = self.null_count
            )));
        }
        Ok(())
    }
}

/// An adapter's structural declaration of one row group.
#[derive(Debug, Clone)]
pub(crate) struct RowGroupPlan {
    pub(crate) rows: i64,
    /// Parallel to `Schema.columns`. `None` per column means undeclared.
    pub(crate) column_stats: Vec<Option<ColumnStatistics>>,
    /// Parallel to `Schema.columns`. `None` (whole field) triggers a metadata pre-pass
    /// for every variable-width column. `Some(vec)` with `None` per column triggers
    /// the pre-pass for that column. `Some(value)` declares the byte size in
    /// manifest-mode.
    pub(crate) column_byte_sizes: Option<Vec<Option<i64>>>,
}

impl RowGroupPlan {
    /// Validate the plan against a schema. Cross-check rows >= 0, column counts match,
    /// per-column statistics against their declared column type, and adapter-declared
    /// `column_byte_sizes` for fixed-width columns against the library-computed value.
    pub(crate) fn validate(&self, schema: &Schema) -> Result<(), Error> {
        if self.rows < 0 {
            return Err(Error::InvalidPlan(format!(
                "RowGroupPlan.rows must be >= 0, got {}",
                self.rows
            )));
        }
        if self.column_stats.len() != schema.len() {
            return Err(Error::InvalidPlan(format!(
                "column_stats length {} != schema column count {}",
                self.column_stats.len(),
                schema.len()
            )));
        }
        if let Some(sizes) = &self.column_byte_sizes {
            if sizes.len() != schema.len() {
                return Err(Error::InvalidPlan(format!(
                    "column_byte_sizes length {} != schema column count {}",
                    sizes.len(),
                    schema.len()
                )));
            }
            for (i, sz) in sizes.iter().enumerate() {
                if let Some(s) = sz {
                    if *s < 0 {
                        return Err(Error::InvalidPlan(format!(
                            "column_byte_sizes[{i}] must be >= 0, got {s}"
                        )));
                    }
                    let col = &schema.columns()[i];
                    if col.data_type.is_fixed_width() {
                        // For fixed-width types we know the byte size from the row count
                        // and (when declared) the null count. An adapter-declared value
                        // must match exactly — otherwise the footer's offsets would
                        // diverge from the bytes we'd actually emit.
                        let null_count = self.column_stats[i]
                            .as_ref()
                            .and_then(|s| s.null_count)
                            .unwrap_or(0);
                        let expected =
                            compute_fixed_value_size(col.data_type, self.rows, null_count)?;
                        if expected != *s {
                            return Err(Error::ByteSizeMismatch(format!(
                                "column {:?}: declared column_byte_sizes[{i}]={s} \
                                 differs from computed {expected} for fixed-width \
                                 type {:?} with rows={} null_count={}",
                                col.name, col.data_type, self.rows, null_count
                            )));
                        }
                    }
                }
            }
        }
        for (i, stat) in self.column_stats.iter().enumerate() {
            if let Some(s) = stat {
                s.validate(schema.columns()[i].data_type, self.rows)?;
            }
        }
        Ok(())
    }
}

/// Owned Arrow data the adapter yields for a single row group.
///
/// The binding layer consumes a Python Arrow PyCapsule (any object exposing
/// `__arrow_c_array__` or `__arrow_c_stream__`) and converts it into this type.
#[derive(Debug)]
pub(crate) struct RowGroupData {
    pub(crate) batch: arrow_array::RecordBatch,
}

/// Internal Rust trait that the byte-server consumes. The PyO3 binding implements
/// this over a Python adapter object (see `bindings.rs`); the Rust core never sees
/// a Python reference.
pub(crate) trait Adapter: Send + Sync {
    fn schema(&self) -> &Schema;
    fn row_group_count(&self) -> u32;
    fn row_group_plan(&self, index: u32) -> Result<RowGroupPlan, Error>;
    fn fetch(&self, index: u32) -> Result<RowGroupData, Error>;
}
