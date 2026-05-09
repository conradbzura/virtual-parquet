//! Crate-wide error type. Variants map to Python exception classes by `bindings.rs`:
//!
//! | Rust variant                          | Python class                  |
//! | ------------------------------------- | ----------------------------- |
//! | `SchemaMismatch` / `InvalidSchema` /  |                               |
//! |   `InvalidPlan`                       | `SchemaMismatchError`         |
//! | `StatisticsMismatch`                  | `StatisticsMismatchError`    |
//! | `ByteSizeMismatch`                    | `ByteSizeMismatchError`       |
//! | `NonReplayableAdapter`                | `NonReplayableAdapterError`   |
//! | `Adapter`                             | `VirtualParquetError` (base)  |
//! | `Encoding` / `Parquet`                | `VirtualParquetError` (base)  |
//! | `InvalidState` / `InvalidRange`       | `PyValueError`                |
//! | `Io`                                  | `PyOSError`                   |
//! | `Python(PyErr)`                       | the contained PyErr is        |
//! |                                       | re-raised verbatim — used to  |
//! |                                       | preserve typed exception      |
//! |                                       | classes a user adapter raises |
//!
//! When a contributor adds a new variant, update the binding's error map at the
//! same time and pin the test in `_native.pyi` to surface the change in CI.

use thiserror::Error;

#[derive(Debug, Error)]
pub(crate) enum Error {
    #[error("invalid schema: {0}")]
    InvalidSchema(String),

    #[error("invalid row group plan: {0}")]
    InvalidPlan(String),

    #[error("schema mismatch: {0}")]
    SchemaMismatch(String),

    #[error("statistics mismatch: {0}")]
    StatisticsMismatch(String),

    #[error("byte size mismatch: {0}")]
    ByteSizeMismatch(String),

    /// Reserved for future use when adapter declares non-replayability and the
    /// byte server detects a second `fetch(i)` would be required. Constructor not
    /// yet wired up — kept here so the Python `NonReplayableAdapterError` surface
    /// is stable from day one.
    #[allow(dead_code)]
    #[error("non-replayable adapter: {0}")]
    NonReplayableAdapter(String),

    #[error("adapter error: {0}")]
    Adapter(String),

    #[error("invalid state: {0}")]
    InvalidState(String),

    #[error("invalid range: {0}")]
    InvalidRange(String),

    #[error("encoding error: {0}")]
    Encoding(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("parquet error: {0}")]
    Parquet(String),

    /// A Python exception captured at the binding boundary. Re-raised verbatim by
    /// `bindings::rust_error_to_pyerr` so that typed exception classes a user adapter
    /// raises (e.g. `vp.NonReplayableAdapterError`) flow through to the engine
    /// without being flattened into a string.
    #[cfg(feature = "extension-module")]
    #[error("python error: {0}")]
    Python(pyo3::PyErr),
}
