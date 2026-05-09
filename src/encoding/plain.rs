//! Plain encoding for the v1 column types, plus RLE_HYBRID encoding for definition
//! levels (max def level 1, single-level flat schema).
//!
//! v1 emits a single bit-packed-only RLE_HYBRID run for the entire def-level block.
//! Pure-RLE runs are a valid optimization but unimplemented.
//!
//! Format references (Parquet spec):
//! - Plain: <https://github.com/apache/parquet-format/blob/master/Encodings.md#plain-plain--0>
//! - RLE_HYBRID: <https://github.com/apache/parquet-format/blob/master/Encodings.md#run-length-encoding--bit-packing-hybrid-rle--3>

use arrow_array::types::GenericStringType;
use arrow_array::{
    Array, BooleanArray, Float32Array, Float64Array, GenericByteArray, Int32Array, Int64Array,
    LargeStringArray, OffsetSizeTrait, StringArray,
};

use crate::adapter::ColumnType;
use crate::error::Error;

/// Compute the byte size of the Plain-encoded values portion of a column chunk for a
/// fixed-width column. Pre-data-scan: requires only the row count and the null count.
///
/// Plain encoding skips nulls — only `non_null = rows - null_count` values are written.
/// Returns `Err(Error::Encoding)` on `i64` overflow or if called with `String` (use a
/// variable-width path instead).
pub(crate) fn compute_fixed_value_size(
    ty: ColumnType,
    rows: i64,
    null_count: i64,
) -> Result<i64, Error> {
    let non_null = rows.saturating_sub(null_count);
    let mul = |w: i64| -> Result<i64, Error> {
        non_null.checked_mul(w).ok_or_else(|| {
            Error::Encoding(format!(
                "byte size overflow: rows={rows} null_count={null_count} width={w}"
            ))
        })
    };
    match ty {
        ColumnType::Int32 | ColumnType::Float32 => mul(4),
        ColumnType::Int64 | ColumnType::Float64 => mul(8),
        // Bit-packed: ceil(non_null / 8) bytes. `i64::div_ceil` is unstable on stable
        // Rust through 1.85, so spell it out with a saturating add.
        ColumnType::Boolean => Ok(non_null.saturating_add(7) / 8),
        ColumnType::String => Err(Error::Encoding(
            "compute_fixed_value_size called with String (variable-width)".into(),
        )),
    }
}

/// Compute the encoded byte size of a definition-level block for a nullable column
/// in a DataPageV1, given the row count.
///
/// Layout (DataPageV1 with max_def_level = 1):
/// - 4-byte little-endian length prefix
/// - One bit-packed RLE_HYBRID run header (varint), encoding `ceil(rows / 8)` groups of 8
/// - `ceil(rows / 8)` data bytes (one bit per def level, LSB-first within each byte)
#[must_use]
pub(crate) fn compute_def_levels_size(rows: i64) -> i64 {
    debug_assert!(rows >= 0, "rows must be non-negative");
    if rows <= 0 {
        return 4; // length prefix only; the body is empty
    }
    // `i64::div_ceil` is unstable on stable Rust through 1.85.
    let groups = (rows + 7) / 8;
    let header_size = varint_size(u64::try_from(groups).unwrap_or(u64::MAX) * 2 + 1) as i64;
    let data_size = groups; // 1 byte per group
    4 + header_size + data_size
}

/// Number of bytes needed to encode `value` as an unsigned varint (1..=10 bytes).
#[must_use]
pub(crate) fn varint_size(mut value: u64) -> usize {
    let mut n = 1;
    while value >= 0x80 {
        value >>= 7;
        n += 1;
    }
    n
}

/// Append `value` as an unsigned varint to `out`.
pub(crate) fn write_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

/// Encode `def_levels` (one byte per row, value 0 or 1, max def level = 1) into the
/// DataPageV1 def-level block: 4-byte LE length prefix + RLE_HYBRID body.
///
/// Returns `Err(Error::Encoding)` if the body would exceed `u32::MAX` bytes (the
/// length-prefix limit).
pub(crate) fn encode_def_levels(def_levels: &[u8]) -> Result<Vec<u8>, Error> {
    let n = def_levels.len();
    if n == 0 {
        // length prefix = 0, no body
        return Ok(vec![0, 0, 0, 0]);
    }
    let groups = n.div_ceil(8);
    let mut body = Vec::with_capacity(varint_size(groups as u64 * 2 + 1) + groups);

    // Bit-packed run header: ((groups << 1) | 1) as varint.
    write_varint(&mut body, (groups as u64) * 2 + 1);

    // Pack the def levels, LSB-first within each byte.
    for group in 0..groups {
        let mut byte = 0u8;
        let base = group * 8;
        for bit in 0..8 {
            let idx = base + bit;
            if idx < n && def_levels[idx] != 0 {
                byte |= 1u8 << bit;
            }
        }
        body.push(byte);
    }

    let body_len = u32::try_from(body.len()).map_err(|_| {
        Error::Encoding(format!(
            "def-level block size {} exceeds u32::MAX",
            body.len()
        ))
    })?;
    let mut out = Vec::with_capacity(4 + body.len());
    out.extend_from_slice(&body_len.to_le_bytes());
    out.extend_from_slice(&body);
    Ok(out)
}

// ---------- Plain value encoding for each type ----------

fn nullable_def_levels(array: &dyn Array) -> Vec<u8> {
    let len = array.len();
    let mut levels = Vec::with_capacity(len);
    if let Some(nulls) = array.nulls() {
        for i in 0..len {
            levels.push(u8::from(nulls.is_valid(i)));
        }
    } else {
        levels.resize(len, 1);
    }
    levels
}

/// Reject a non-nullable column carrying a null bitmap. The schema-conformance check
/// promises errors before bytes flow; this is the encoder's enforcement point for
/// nullability so a non-nullable column never silently produces a short values block.
fn check_no_nulls_when_required(array: &dyn Array, nullable: bool) -> Result<(), Error> {
    if !nullable && array.null_count() > 0 {
        return Err(Error::SchemaMismatch(format!(
            "non-nullable column has {} null entries in its Arrow array",
            array.null_count()
        )));
    }
    Ok(())
}

fn maybe_def_block(array: &dyn Array, nullable: bool) -> Result<Vec<u8>, Error> {
    if nullable {
        encode_def_levels(&nullable_def_levels(array))
    } else {
        Ok(Vec::new())
    }
}

/// Encode an Int32 column: emits 4 LE bytes per non-null value, plus the def-level
/// block if the column is nullable. Returns `(def_levels_block, values_block)`.
pub(crate) fn encode_int32_values(
    array: &Int32Array,
    nullable: bool,
) -> Result<(Vec<u8>, Vec<u8>), Error> {
    check_no_nulls_when_required(array, nullable)?;
    let non_null = array.len() - array.null_count();
    let mut values = Vec::with_capacity(non_null * 4);
    for i in 0..array.len() {
        if array.is_valid(i) {
            values.extend_from_slice(&array.value(i).to_le_bytes());
        }
    }
    Ok((maybe_def_block(array, nullable)?, values))
}

/// Encode an Int64 column.
pub(crate) fn encode_int64_values(
    array: &Int64Array,
    nullable: bool,
) -> Result<(Vec<u8>, Vec<u8>), Error> {
    check_no_nulls_when_required(array, nullable)?;
    let non_null = array.len() - array.null_count();
    let mut values = Vec::with_capacity(non_null * 8);
    for i in 0..array.len() {
        if array.is_valid(i) {
            values.extend_from_slice(&array.value(i).to_le_bytes());
        }
    }
    Ok((maybe_def_block(array, nullable)?, values))
}

/// Encode a Float32 column.
pub(crate) fn encode_float32_values(
    array: &Float32Array,
    nullable: bool,
) -> Result<(Vec<u8>, Vec<u8>), Error> {
    check_no_nulls_when_required(array, nullable)?;
    let non_null = array.len() - array.null_count();
    let mut values = Vec::with_capacity(non_null * 4);
    for i in 0..array.len() {
        if array.is_valid(i) {
            values.extend_from_slice(&array.value(i).to_le_bytes());
        }
    }
    Ok((maybe_def_block(array, nullable)?, values))
}

/// Encode a Float64 column.
pub(crate) fn encode_float64_values(
    array: &Float64Array,
    nullable: bool,
) -> Result<(Vec<u8>, Vec<u8>), Error> {
    check_no_nulls_when_required(array, nullable)?;
    let non_null = array.len() - array.null_count();
    let mut values = Vec::with_capacity(non_null * 8);
    for i in 0..array.len() {
        if array.is_valid(i) {
            values.extend_from_slice(&array.value(i).to_le_bytes());
        }
    }
    Ok((maybe_def_block(array, nullable)?, values))
}

/// Encode a Boolean column: bit-packed, LSB-first within each byte, non-null values only.
pub(crate) fn encode_boolean_values(
    array: &BooleanArray,
    nullable: bool,
) -> Result<(Vec<u8>, Vec<u8>), Error> {
    check_no_nulls_when_required(array, nullable)?;
    let non_null_count: usize = array.len() - array.null_count();
    let n_bytes = non_null_count.div_ceil(8);
    let mut values = vec![0u8; n_bytes];
    let mut written = 0usize;
    for i in 0..array.len() {
        if array.is_valid(i) {
            if array.value(i) {
                values[written / 8] |= 1u8 << (written % 8);
            }
            written += 1;
        }
    }
    Ok((maybe_def_block(array, nullable)?, values))
}

/// Encode a String column (Utf8): 4-byte LE length prefix + UTF-8 bytes per non-null value.
pub(crate) fn encode_string_values(
    array: &StringArray,
    nullable: bool,
) -> Result<(Vec<u8>, Vec<u8>), Error> {
    encode_generic_string::<i32>(array, nullable)
}

/// Encode a LargeString column (LargeUtf8): same wire format as Utf8 — Parquet's
/// `BYTE_ARRAY` Plain encoding uses a 4-byte LE length prefix per value regardless
/// of the Arrow offset width.
pub(crate) fn encode_large_string_values(
    array: &LargeStringArray,
    nullable: bool,
) -> Result<(Vec<u8>, Vec<u8>), Error> {
    encode_generic_string::<i64>(array, nullable)
}

fn encode_generic_string<O: OffsetSizeTrait>(
    array: &GenericByteArray<GenericStringType<O>>,
    nullable: bool,
) -> Result<(Vec<u8>, Vec<u8>), Error> {
    check_no_nulls_when_required(array, nullable)?;
    let mut total: usize = 0;
    for i in 0..array.len() {
        if array.is_valid(i) {
            total = total.saturating_add(4 + array.value(i).len());
        }
    }
    let mut values = Vec::with_capacity(total);
    for i in 0..array.len() {
        if array.is_valid(i) {
            let s = array.value(i);
            let len = u32::try_from(s.len()).map_err(|_| {
                Error::Encoding(format!("string length {} exceeds u32::MAX", s.len()))
            })?;
            values.extend_from_slice(&len.to_le_bytes());
            values.extend_from_slice(s.as_bytes());
        }
    }
    Ok((maybe_def_block(array, nullable)?, values))
}
