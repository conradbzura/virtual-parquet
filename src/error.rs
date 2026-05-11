//! Crate-wide error type. Variants map to Python exception classes by `bindings.rs`:
//!
//! | Rust variant                                              | Python class                  |
//! | --------------------------------------------------------- | ----------------------------- |
//! | `SchemaMismatch`, `InvalidSchema`, `InvalidPlan`          | `SchemaMismatchError`         |
//! | `StatisticsMismatch`                                      | `StatisticsMismatchError`     |
//! | `ByteSizeMismatch`                                        | `ByteSizeMismatchError`       |
//! | `NonReplayableAdapter`                                    | `NonReplayableAdapterError`   |
//! | `Adapter`, `Encoding`, `Parquet`                          | `VirtualParquetError` (base)  |
//! | `InvalidState`, `InvalidRange`                            | `PyValueError`                |
//! | `Io`                                                      | `PyOSError`                   |
//! | `Python(PyErr)` (feature-gated)                           | re-raised verbatim            |
//!
//! When a contributor adds a new variant, update [`crate::bindings::rust_error_to_pyerr`]
//! at the same time and adjust the conformance assertion in CI.

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

    /// Reserved for the library-side detection path described in
    /// `contracts/public-api.md`. Not currently constructed in Rust; a Python
    /// adapter raising `NonReplayableAdapterError` itself flows verbatim through
    /// `Error::Python` (see `bindings.rs::classify_adapter_pyerr`).
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
    ///
    /// Note: forward-compat discipline #2 (plan.md) confines PyO3 references to
    /// `bindings.rs`. This variant is feature-gated so `cargo check`/`cargo test`
    /// without `--features extension-module` compile as a pure-Rust core.
    #[cfg(feature = "extension-module")]
    #[error("python error: {0}")]
    Python(pyo3::PyErr),
}
