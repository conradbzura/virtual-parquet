//! Column-chunk encoding for v1: Plain encoding only.
//!
//! v1 emits uncompressed Parquet with Plain-encoded values and bit-packed-only
//! RLE_HYBRID-encoded definition levels. Repetition levels are not emitted (flat
//! schema only). Pure-RLE runs (a valid spec optimization) are unimplemented.
//!
//! Callers reach into `plain` directly: `crate::encoding::plain::encode_*`.

pub(crate) mod plain;
