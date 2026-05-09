//! virtual-parquet: scaffolding for serving non-Parquet data sources to mainstream
//! Parquet-consuming engines via a stable adapter contract.
//!
//! The Rust crate is internal; the user-facing artifact is the Python wheel built by
//! maturin. All modules here are `pub(crate)` — there is no stability commitment to
//! Rust callers, and the documented public surface lives in `contracts/public-api.md`.

pub(crate) mod adapter;
pub(crate) mod encoding;
pub(crate) mod error;
pub(crate) mod footer;
pub(crate) mod reader;

#[cfg(feature = "extension-module")]
mod bindings;
