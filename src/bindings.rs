//! `PyO3` bindings: convert Python adapter objects + dataclasses into Rust types,
//! expose `ByteServer` as `_native.VirtualFile`, and translate Rust errors into
//! the corresponding Python exception classes.
//!
//! This is the only module in the crate that holds `PyO3` references. Per
//! forward-compat discipline #2 (`plan.md`), the Rust core is purely sync and
//! Python-free.

use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::ffi_stream::ArrowArrayStreamReader;
use arrow::pyarrow::FromPyArrow;
use pyo3::exceptions::{PyException, PyOSError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes};

use crate::adapter::{
    Adapter, Column, ColumnStatistics, ColumnType, RowGroupData, RowGroupPlan, Schema, StatValue,
};
use crate::error::Error;
use crate::reader::ByteServer;

// Python io.SEEK_SET / SEEK_CUR / SEEK_END.
const SEEK_SET: i32 = 0;
const SEEK_CUR: i32 = 1;
const SEEK_END: i32 = 2;

// ---------- Error type registration ----------

pyo3::create_exception!(_native, VirtualParquetError, pyo3::exceptions::PyException);
pyo3::create_exception!(_native, SchemaMismatchError, VirtualParquetError);
pyo3::create_exception!(_native, StatisticsMismatchError, VirtualParquetError);
pyo3::create_exception!(_native, ByteSizeMismatchError, VirtualParquetError);
pyo3::create_exception!(_native, NonReplayableAdapterError, VirtualParquetError);

fn rust_error_to_pyerr(err: Error) -> PyErr {
    match err {
        Error::Python(py_err) => py_err,
        Error::SchemaMismatch(msg) | Error::InvalidSchema(msg) | Error::InvalidPlan(msg) => {
            SchemaMismatchError::new_err(msg)
        }
        Error::StatisticsMismatch(msg) => StatisticsMismatchError::new_err(msg),
        Error::ByteSizeMismatch(msg) => ByteSizeMismatchError::new_err(msg),
        Error::NonReplayableAdapter(msg) => NonReplayableAdapterError::new_err(msg),
        Error::Adapter(msg) | Error::Encoding(msg) | Error::Parquet(msg) => {
            VirtualParquetError::new_err(msg)
        }
        Error::InvalidState(msg) | Error::InvalidRange(msg) => PyValueError::new_err(msg),
        Error::Io(io) => PyOSError::new_err(io.to_string()),
    }
}

/// Wrap a Python exception that the adapter raised. Three cases:
///
/// 1. `BaseException`-only subclasses (`KeyboardInterrupt`, `SystemExit`, custom
///    cancellation signals) propagate verbatim via `Error::Python`, so Ctrl+C and
///    interpreter shutdown reach the runtime unchanged.
/// 2. `VirtualParquetError` subclasses propagate verbatim so the typed Python class
///    survives the round-trip back to the engine.
/// 3. Any other `Exception` subclass (a generic adapter-side bug) is wrapped in
///    `Error::Adapter` so the engine sees a base `VirtualParquetError` with the
///    original message.
fn classify_adapter_pyerr(py: Python<'_>, py_err: PyErr, context: String) -> Error {
    let exc_type = py_err.get_type(py);
    let exception_type = py.get_type::<PyException>();
    // Anything not derived from `Exception` (e.g. `KeyboardInterrupt`, `SystemExit`)
    // must propagate unchanged — wrapping it would silently break Ctrl+C / shutdown.
    if !exc_type.is_subclass(&exception_type).unwrap_or(false) {
        return Error::Python(py_err);
    }
    let vp_error_type = py.get_type::<VirtualParquetError>();
    if exc_type.is_subclass(&vp_error_type).unwrap_or(false) {
        Error::Python(py_err)
    } else {
        Error::Adapter(format!("{context}: {py_err}"))
    }
}

// ---------- Python → Rust conversions ----------

fn extract_column_type(value: &Bound<'_, PyAny>) -> PyResult<ColumnType> {
    let py_value: String = value.getattr("value")?.extract()?;
    match py_value.as_str() {
        "int32" => Ok(ColumnType::Int32),
        "int64" => Ok(ColumnType::Int64),
        "float32" => Ok(ColumnType::Float32),
        "float64" => Ok(ColumnType::Float64),
        "boolean" => Ok(ColumnType::Boolean),
        "string" => Ok(ColumnType::String),
        other => Err(PyValueError::new_err(format!(
            "unknown ColumnType: {other}"
        ))),
    }
}

/// Extract an integer attribute, rejecting Python `bool` (which subclasses `int` and
/// would otherwise silently coerce `True` → 1, `False` → 0).
fn extract_int_attr(parent: &Bound<'_, PyAny>, name: &str) -> PyResult<i64> {
    extract_int_attr_value(&parent.getattr(name)?, name)
}

/// True iff `value` is exactly a Python `bool` (`True` / `False`). Compares the
/// Python type identity directly, since `PyO3` 0.23's `is_instance_of::<PyBool>`
/// does not surface a positive answer under abi3 in this configuration.
// TODO(pyo3-upgrade): retest is_instance_of::<PyBool> when bumping PyO3.
fn is_python_bool(value: &Bound<'_, PyAny>) -> bool {
    let py = value.py();
    let bool_type = py.get_type::<PyBool>();
    value.get_type().is(&bool_type)
}

/// Reject Python `bool` for fields documented as integer counts.
fn reject_bool(value: &Bound<'_, PyAny>, label: &str) -> PyResult<()> {
    if is_python_bool(value) {
        return Err(PyTypeError::new_err(format!(
            "{label} must be an int, not a bool"
        )));
    }
    Ok(())
}

fn extract_column(value: &Bound<'_, PyAny>) -> PyResult<Column> {
    let name: String = value.getattr("name")?.extract()?;
    let type_obj = value.getattr("type")?;
    let column_type = extract_column_type(&type_obj)?;
    let nullable: bool = value.getattr("nullable")?.extract()?;
    Column::new(name, column_type, nullable).map_err(rust_error_to_pyerr)
}

fn extract_schema(value: &Bound<'_, PyAny>) -> PyResult<Schema> {
    let columns_obj = value.getattr("columns")?;
    let mut columns: Vec<Column> = Vec::new();
    for item in columns_obj.try_iter()? {
        let item = item?;
        columns.push(extract_column(&item)?);
    }
    Schema::new(columns).map_err(rust_error_to_pyerr)
}

fn extract_stat_value(value: &Bound<'_, PyAny>, ty: ColumnType) -> PyResult<StatValue> {
    let map_err = |ty_label: &str, e: PyErr| -> PyErr {
        StatisticsMismatchError::new_err(format!(
            "min/max value for {ty_label} column does not match column type: {e}"
        ))
    };
    // PyO3 silently coerces Python bool -> int / float (because bool subclasses
    // int). Reject for every numeric branch except the Boolean column itself.
    let bool_label = match ty {
        ColumnType::Int32 => Some("Int32"),
        ColumnType::Int64 => Some("Int64"),
        ColumnType::Float32 => Some("Float32"),
        ColumnType::Float64 => Some("Float64"),
        ColumnType::Boolean | ColumnType::String => None,
    };
    if let Some(label) = bool_label {
        if is_python_bool(value) {
            return Err(map_err(
                label,
                PyTypeError::new_err("got bool, expected numeric"),
            ));
        }
    }
    match ty {
        ColumnType::Int32 => value
            .extract()
            .map(StatValue::Int32)
            .map_err(|e| map_err("Int32", e)),
        ColumnType::Int64 => value
            .extract()
            .map(StatValue::Int64)
            .map_err(|e| map_err("Int64", e)),
        ColumnType::Float32 => value
            .extract()
            .map(StatValue::Float32)
            .map_err(|e| map_err("Float32", e)),
        ColumnType::Float64 => value
            .extract()
            .map(StatValue::Float64)
            .map_err(|e| map_err("Float64", e)),
        ColumnType::Boolean => value
            .extract()
            .map(StatValue::Boolean)
            .map_err(|e| map_err("Boolean", e)),
        ColumnType::String => value
            .extract()
            .map(StatValue::String)
            .map_err(|e| map_err("String", e)),
    }
}

fn extract_column_statistics(
    value: &Bound<'_, PyAny>,
    ty: ColumnType,
) -> PyResult<ColumnStatistics> {
    let mut stats = ColumnStatistics::new();
    let min_obj = value.getattr("min")?;
    if !min_obj.is_none() {
        stats.min = Some(extract_stat_value(&min_obj, ty)?);
    }
    let max_obj = value.getattr("max")?;
    if !max_obj.is_none() {
        stats.max = Some(extract_stat_value(&max_obj, ty)?);
    }
    let nc_obj = value.getattr("null_count")?;
    if !nc_obj.is_none() {
        stats.null_count = Some(extract_int_attr_value(&nc_obj, "null_count")?);
    }
    Ok(stats)
}

/// Like `extract_int_attr` but operates on an already-fetched bound value.
fn extract_int_attr_value(value: &Bound<'_, PyAny>, name: &str) -> PyResult<i64> {
    reject_bool(value, name)?;
    value.extract()
}

fn extract_row_group_plan(value: &Bound<'_, PyAny>, schema: &Schema) -> PyResult<RowGroupPlan> {
    let rows = extract_int_attr(value, "rows")?;
    // Collect column_stats entries; once we go past the schema's column count we
    // stop typed-extraction and push a sentinel `None` so the canonical
    // length-mismatch error from RowGroupPlan::validate is what surfaces — rather
    // than a misleading "min/max for Int64 column does not match" from a stat
    // attached to a column that doesn't exist.
    let stats_obj = value.getattr("column_stats")?;
    let mut column_stats: Vec<Option<ColumnStatistics>> = Vec::new();
    for (idx, item) in stats_obj.try_iter()?.enumerate() {
        let item = item?;
        if idx >= schema.columns().len() {
            column_stats.push(None);
            continue;
        }
        if item.is_none() {
            column_stats.push(None);
        } else {
            let col_type = schema.columns()[idx].data_type;
            column_stats.push(Some(extract_column_statistics(&item, col_type)?));
        }
    }
    let sizes_obj = value.getattr("column_byte_sizes")?;
    let column_byte_sizes: Option<Vec<Option<i64>>> = if sizes_obj.is_none() {
        None
    } else {
        let mut sizes: Vec<Option<i64>> = Vec::new();
        for item in sizes_obj.try_iter()? {
            let item = item?;
            if item.is_none() {
                sizes.push(None);
            } else {
                sizes.push(Some(extract_int_attr_value(&item, "column_byte_sizes[*]")?));
            }
        }
        Some(sizes)
    };
    Ok(RowGroupPlan {
        rows,
        column_stats,
        column_byte_sizes,
    })
}

// ---------- Python adapter implementing the Rust Adapter trait ----------

/// A Rust `Adapter` implementation that delegates to a Python adapter object via PyO3.
struct PythonAdapter {
    py_adapter: Py<PyAny>,
    schema: Schema,
    row_group_count: u32,
}

impl PythonAdapter {
    fn new(py_adapter: Py<PyAny>) -> PyResult<Self> {
        Python::with_gil(|py| {
            let bound = py_adapter.bind(py);
            let schema_obj = bound.getattr("schema")?;
            let schema = extract_schema(&schema_obj)?;
            let count = extract_int_attr(bound, "row_group_count")?;
            if count < 0 {
                return Err(PyValueError::new_err(format!(
                    "row_group_count must be >= 0, got {count}"
                )));
            }
            let row_group_count = u32::try_from(count).map_err(|_| {
                PyValueError::new_err(format!("row_group_count {count} exceeds u32::MAX"))
            })?;
            Ok(Self {
                py_adapter,
                schema,
                row_group_count,
            })
        })
    }
}

impl Adapter for PythonAdapter {
    fn schema(&self) -> &Schema {
        &self.schema
    }

    fn row_group_count(&self) -> u32 {
        self.row_group_count
    }

    fn row_group_plan(&self, index: u32) -> Result<RowGroupPlan, Error> {
        Python::with_gil(|py| {
            let bound = self.py_adapter.bind(py);
            let plan_obj = bound
                .call_method1("row_group_plan", (index,))
                .map_err(|e| {
                    classify_adapter_pyerr(py, e, format!("row_group_plan({index}) raised"))
                })?;
            extract_row_group_plan(&plan_obj, &self.schema).map_err(|e| {
                classify_adapter_pyerr(
                    py,
                    e,
                    format!("row_group_plan({index}) returned invalid plan"),
                )
            })
        })
    }

    fn fetch(&self, index: u32) -> Result<RowGroupData, Error> {
        Python::with_gil(|py| {
            let bound = self.py_adapter.bind(py);
            let result = bound
                .call_method1("fetch", (index,))
                .map_err(|e| classify_adapter_pyerr(py, e, format!("fetch({index}) raised")))?;
            let batch = pyarrow_object_to_record_batch(py, &result, index)?;
            Ok(RowGroupData { batch })
        })
    }
}

/// Convert a Python object exposing the Arrow C Data Interface to a single
/// `RecordBatch`. Tries `__arrow_c_array__` first (via `RecordBatch::from_pyarrow_bound`,
/// which also handles `pyarrow.RecordBatch` instances directly); falls back to
/// `__arrow_c_stream__` and consumes a single batch from the stream. The contract
/// in `contracts/adapter-protocol.md` admits both interfaces.
fn pyarrow_object_to_record_batch(
    py: Python<'_>,
    result: &Bound<'_, PyAny>,
    index: u32,
) -> Result<RecordBatch, Error> {
    if let Ok(batch) = RecordBatch::from_pyarrow_bound(result) {
        return Ok(batch);
    }
    if result
        .hasattr("__arrow_c_stream__")
        .map_err(|e| classify_adapter_pyerr(py, e, format!("fetch({index}) attribute check")))?
    {
        let stream = ArrowArrayStreamReader::from_pyarrow_bound(result).map_err(|e| {
            classify_adapter_pyerr(
                py,
                e,
                format!("fetch({index}) returned a non-Arrow stream object"),
            )
        })?;
        let mut iter = stream.into_iter();
        let first = iter.next().ok_or_else(|| {
            Error::SchemaMismatch(format!(
                "fetch({index}) returned an empty Arrow stream; expected exactly one row group's batch"
            ))
        })?;
        let batch =
            first.map_err(|e| Error::Adapter(format!("fetch({index}) stream error: {e}")))?;
        if iter.next().is_some() {
            return Err(Error::SchemaMismatch(format!(
                "fetch({index}) returned an Arrow stream with multiple batches; expected exactly one row group's batch"
            )));
        }
        return Ok(batch);
    }
    Err(Error::SchemaMismatch(format!(
        "fetch({index}) returned a non-Arrow object (no __arrow_c_array__ or __arrow_c_stream__)"
    )))
}

// ---------- VirtualFile PyO3 class ----------

#[pyclass(name = "VirtualFile", module = "virtual_parquet._native")]
struct VirtualFile {
    server: Arc<ByteServer>,
    position: i64,
}

#[pymethods]
impl VirtualFile {
    #[new]
    fn new(adapter: Py<PyAny>) -> PyResult<Self> {
        let py_adapter = PythonAdapter::new(adapter)?;
        let server = ByteServer::new(Arc::new(py_adapter));
        Ok(Self {
            server: Arc::new(server),
            position: 0,
        })
    }

    /// Return the total size of the virtual Parquet object in bytes (lazily triggers
    /// the metadata pre-pass on first call).
    fn size(&self, py: Python<'_>) -> PyResult<i64> {
        // Release the GIL while the byte-server runs the pre-pass — adapter callbacks
        // re-acquire the GIL via Python::with_gil internally.
        py.allow_threads(|| self.server.total_size())
            .map_err(rust_error_to_pyerr)
    }

    fn tell(&self) -> i64 {
        self.position
    }

    fn seek(&mut self, py: Python<'_>, offset: i64, whence: i32) -> PyResult<i64> {
        let total = py
            .allow_threads(|| self.server.total_size())
            .map_err(rust_error_to_pyerr)?;
        let new_pos = match whence {
            SEEK_SET => offset,
            SEEK_CUR => self.position.checked_add(offset).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "seek overflow: position={} offset={offset}",
                    self.position
                ))
            })?,
            SEEK_END => total.checked_add(offset).ok_or_else(|| {
                PyValueError::new_err(format!("seek overflow: total={total} offset={offset}"))
            })?,
            other => {
                return Err(PyValueError::new_err(format!(
                    "invalid whence value {other}; expected 0, 1, or 2"
                )));
            }
        };
        if new_pos < 0 {
            return Err(PyValueError::new_err(format!(
                "seek to negative position {new_pos}"
            )));
        }
        self.position = new_pos;
        Ok(self.position)
    }

    /// Read up to `n` bytes from the current position. `n = -1` means read to EOF.
    #[pyo3(signature = (n=-1))]
    fn read<'py>(&mut self, py: Python<'py>, n: i64) -> PyResult<Bound<'py, PyBytes>> {
        let server = self.server.clone();
        let position = self.position;
        let bytes = py
            .allow_threads(move || {
                let total = server.total_size()?;
                let remaining = (total - position).max(0);
                let to_read = if n < 0 {
                    remaining
                } else {
                    i64::min(n, remaining)
                };
                let to_read_usize = usize::try_from(to_read).unwrap_or(0);
                server.read_range(position, to_read_usize)
            })
            .map_err(rust_error_to_pyerr)?;
        let advance = i64::try_from(bytes.len())
            .map_err(|_| PyValueError::new_err("read returned > i64::MAX bytes"))?;
        self.position += advance;
        Ok(PyBytes::new(py, &bytes))
    }

    // The file-like protocol requires these as instance methods even when the
    // receiver is unused.
    #[allow(clippy::unused_self)]
    fn seekable(&self) -> bool {
        true
    }

    fn readable(&self) -> bool {
        !self.server.is_closed()
    }

    #[allow(clippy::unused_self)]
    fn writable(&self) -> bool {
        false
    }

    fn close(&self) {
        self.server.close();
    }

    #[getter]
    fn closed(&self) -> bool {
        self.server.is_closed()
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (_exc_type=None, _exc_value=None, _traceback=None))]
    fn __exit__(
        &self,
        _exc_type: Option<Bound<'_, PyAny>>,
        _exc_value: Option<Bound<'_, PyAny>>,
        _traceback: Option<Bound<'_, PyAny>>,
    ) -> bool {
        self.close();
        false
    }
}

// ---------- _native module ----------

#[pymodule]
fn _native(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VirtualFile>()?;
    m.add("VirtualParquetError", py.get_type::<VirtualParquetError>())?;
    m.add("SchemaMismatchError", py.get_type::<SchemaMismatchError>())?;
    m.add(
        "StatisticsMismatchError",
        py.get_type::<StatisticsMismatchError>(),
    )?;
    m.add(
        "ByteSizeMismatchError",
        py.get_type::<ByteSizeMismatchError>(),
    )?;
    m.add(
        "NonReplayableAdapterError",
        py.get_type::<NonReplayableAdapterError>(),
    )?;
    Ok(())
}
