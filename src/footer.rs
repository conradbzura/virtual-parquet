//! Parquet footer (`FileMetaData`) construction and Thrift compact serialization.
//!
//! This module owns the conversion from our internal `Schema` + computed row group
//! metadata into the canonical `parquet::format::FileMetaData` Thrift struct, plus
//! the bytes-out side via `TCompactOutputProtocol`. Per Constitution Principle II,
//! the footer is the only piece of metadata materialized in memory; column-chunk
//! bytes are emitted on demand by `reader.rs`.

use parquet::format::{
    ColumnChunk, ColumnMetaData, ColumnOrder, CompressionCodec, DataPageHeader, Encoding,
    FieldRepetitionType, FileMetaData, KeyValue, PageHeader, PageType, RowGroup, SchemaElement,
    Statistics, Type, TypeDefinedOrder,
};
use parquet::thrift::TSerializable;
use thrift::protocol::TCompactOutputProtocol;

use crate::adapter::{ColumnType, Schema, StatValue};
use crate::error::Error;

/// Magic bytes that bracket every Parquet file.
pub(crate) const PARQUET_MAGIC: &[u8; 4] = b"PAR1";

/// Parquet thrift `FileMetaData.version`. v1 emits the v1 (DataPageV1) format only.
const PARQUET_FORMAT_VERSION: i32 = 1;

/// Internal metadata for one finalized column chunk, ready to be referenced from the
/// footer. Computed by `reader.rs` during the metadata pre-pass.
#[derive(Debug, Clone)]
pub(crate) struct ComputedColumnChunk {
    pub(crate) physical_type: ColumnType,
    /// Whether the column is nullable; controls whether a definition-level block is
    /// emitted and what encodings are advertised in the footer.
    pub(crate) nullable: bool,
    /// Byte offset of the column chunk's first page header in the file.
    pub(crate) file_offset: i64,
    /// Total bytes the column chunk occupies (page header + def levels block + values block).
    pub(crate) total_size: i64,
    /// Number of logical values (== row count of the row group).
    pub(crate) num_values: i64,
    /// Adapter-declared null count, if any. None means the adapter declined to
    /// declare; the footer omits the field per FR-008.
    pub(crate) null_count: Option<i64>,
    /// Pre-encoded min bytes (Plain wire format), if declared.
    pub(crate) min_value: Option<Vec<u8>>,
    /// Pre-encoded max bytes (Plain wire format), if declared.
    pub(crate) max_value: Option<Vec<u8>>,
}

#[derive(Debug, Clone)]
pub(crate) struct ComputedRowGroup {
    pub(crate) num_rows: i64,
    pub(crate) total_byte_size: i64,
    pub(crate) columns: Vec<ComputedColumnChunk>,
}

/// Serialize any parquet-rs Thrift struct to compact-protocol bytes.
pub(crate) fn serialize_thrift<T: TSerializable>(value: &T) -> Result<Vec<u8>, Error> {
    let mut buf: Vec<u8> = Vec::new();
    {
        let mut protocol = TCompactOutputProtocol::new(&mut buf);
        value
            .write_to_out_protocol(&mut protocol)
            .map_err(|e| Error::Parquet(format!("thrift serialize failed: {e}")))?;
    }
    Ok(buf)
}

/// Encode a `StatValue` to its Plain-encoded byte form for a Parquet `Statistics`
/// `min_value` / `max_value` field.
pub(crate) fn encode_stat_value(value: &StatValue) -> Vec<u8> {
    match value {
        StatValue::Int32(v) => v.to_le_bytes().to_vec(),
        StatValue::Int64(v) => v.to_le_bytes().to_vec(),
        StatValue::Float32(v) => v.to_le_bytes().to_vec(),
        StatValue::Float64(v) => v.to_le_bytes().to_vec(),
        StatValue::Boolean(v) => vec![u8::from(*v)],
        StatValue::String(s) => s.as_bytes().to_vec(),
    }
}

/// Map our `ColumnType` to the Parquet physical `Type`.
fn physical_type(ty: ColumnType) -> Type {
    match ty {
        ColumnType::Int32 => Type::INT32,
        ColumnType::Int64 => Type::INT64,
        ColumnType::Float32 => Type::FLOAT,
        ColumnType::Float64 => Type::DOUBLE,
        ColumnType::Boolean => Type::BOOLEAN,
        ColumnType::String => Type::BYTE_ARRAY,
    }
}

/// Build the schema-element list for the footer. Convention: a synthetic root element
/// (no type, num_children = N) followed by N child elements (one per declared column).
pub(crate) fn build_schema_elements(schema: &Schema) -> Result<Vec<SchemaElement>, Error> {
    let mut elements: Vec<SchemaElement> = Vec::with_capacity(schema.len() + 1);

    let num_children = i32::try_from(schema.len()).map_err(|_| {
        Error::InvalidSchema(format!(
            "schema has {} columns, exceeds `i32::MAX`",
            schema.len()
        ))
    })?;

    // Root.
    elements.push(SchemaElement {
        type_: None,
        type_length: None,
        repetition_type: None,
        name: "schema".to_string(),
        num_children: Some(num_children),
        converted_type: None,
        scale: None,
        precision: None,
        field_id: None,
        logical_type: None,
    });

    for col in schema.columns() {
        let repetition = if col.nullable {
            FieldRepetitionType::OPTIONAL
        } else {
            FieldRepetitionType::REQUIRED
        };
        let mut elem = SchemaElement {
            type_: Some(physical_type(col.data_type)),
            type_length: None,
            repetition_type: Some(repetition),
            name: col.name.clone(),
            num_children: None,
            converted_type: None,
            scale: None,
            precision: None,
            field_id: None,
            logical_type: None,
        };
        if matches!(col.data_type, ColumnType::String) {
            elem.converted_type = Some(parquet::format::ConvertedType::UTF8);
            elem.logical_type = Some(parquet::format::LogicalType::STRING(
                parquet::format::StringType {},
            ));
        }
        elements.push(elem);
    }

    Ok(elements)
}

/// Build a `ColumnMetaData` Thrift struct for a single finalized column chunk.
fn build_column_metadata(column_name: &str, chunk: &ComputedColumnChunk) -> ColumnMetaData {
    let any_declared =
        chunk.min_value.is_some() || chunk.max_value.is_some() || chunk.null_count.is_some();
    let statistics = if any_declared {
        // Per Parquet spec, the legacy `min`/`max` Statistics fields use unsigned
        // byte ordering, while the modern `min_value`/`max_value` use type-aware
        // (signed) ordering. Emitting identical Plain-LE bytes for both would
        // serialize the legacy fields incorrectly for signed numeric/float types
        // with negative values. Modern writers (parquet-mr's
        // `ParquetMetadataConverter`) populate only the typed pair; we follow that.
        // PyArrow consumers read the modern values via
        // `Statistics.min_raw` / `.max_raw`.
        Some(Statistics {
            max: None,
            min: None,
            null_count: chunk.null_count,
            distinct_count: None,
            max_value: chunk.max_value.clone(),
            min_value: chunk.min_value.clone(),
            // Adapter-declared min/max are exact (they reflect actual minima/maxima of
            // the data the adapter promises to yield), not loose bounds.
            is_max_value_exact: chunk.max_value.as_ref().map(|_| true),
            is_min_value_exact: chunk.min_value.as_ref().map(|_| true),
        })
    } else {
        None
    };

    // Only PLAIN values for non-nullable columns; nullable columns also use RLE_HYBRID
    // for the def-level block.
    let encodings = if chunk.nullable {
        vec![Encoding::PLAIN, Encoding::RLE]
    } else {
        vec![Encoding::PLAIN]
    };

    ColumnMetaData {
        type_: physical_type(chunk.physical_type),
        encodings,
        path_in_schema: vec![column_name.to_string()],
        codec: CompressionCodec::UNCOMPRESSED,
        num_values: chunk.num_values,
        total_uncompressed_size: chunk.total_size,
        // Codec is UNCOMPRESSED in v1, so compressed == uncompressed.
        total_compressed_size: chunk.total_size,
        key_value_metadata: None,
        data_page_offset: chunk.file_offset,
        index_page_offset: None,
        dictionary_page_offset: None,
        statistics,
        encoding_stats: None,
        bloom_filter_offset: None,
        bloom_filter_length: None,
        size_statistics: None,
    }
}

/// Build the complete `FileMetaData` Thrift struct from schema + computed row groups.
pub(crate) fn build_file_metadata(
    schema: &Schema,
    row_groups: Vec<ComputedRowGroup>,
) -> Result<FileMetaData, Error> {
    let schema_elements = build_schema_elements(schema)?;
    let total_rows: i64 = row_groups.iter().map(|rg| rg.num_rows).sum();

    let mut rg_thrift: Vec<RowGroup> = Vec::with_capacity(row_groups.len());
    for rg in row_groups {
        if rg.columns.len() != schema.len() {
            return Err(Error::InvalidPlan(format!(
                "computed row group has {} columns, schema has {}",
                rg.columns.len(),
                schema.len()
            )));
        }
        let mut chunks: Vec<ColumnChunk> = Vec::with_capacity(rg.columns.len());
        for (col, chunk) in schema.columns().iter().zip(rg.columns.iter()) {
            chunks.push(ColumnChunk {
                file_path: None,
                // Parquet thrift docs describe `file_offset` as the byte offset of
                // the ColumnMetaData; with `meta_data` embedded inline (as here),
                // parquet-mr/parquet-cpp convention is to point this at the data
                // page header start. We follow that convention.
                file_offset: chunk.file_offset,
                meta_data: Some(build_column_metadata(&col.name, chunk)),
                offset_index_offset: None,
                offset_index_length: None,
                column_index_offset: None,
                column_index_length: None,
                crypto_metadata: None,
                encrypted_column_metadata: None,
            });
        }
        rg_thrift.push(RowGroup {
            columns: chunks,
            total_byte_size: rg.total_byte_size,
            num_rows: rg.num_rows,
            sorting_columns: None,
            file_offset: None,
            // UNCOMPRESSED in v1 → compressed == uncompressed.
            total_compressed_size: Some(rg.total_byte_size),
            ordinal: None,
        });
    }

    let writer_id = format!("virtual-parquet {}", env!("CARGO_PKG_VERSION"));
    // `column_orders` MUST be populated for readers to honor the modern typed
    // `Statistics.min_value` / `max_value` fields (per Parquet spec). One entry
    // per column, in schema order; v1 uses the standard type-defined ordering.
    let column_orders = vec![ColumnOrder::TYPEORDER(TypeDefinedOrder {}); schema.len()];
    Ok(FileMetaData {
        version: PARQUET_FORMAT_VERSION,
        schema: schema_elements,
        num_rows: total_rows,
        row_groups: rg_thrift,
        key_value_metadata: Some(vec![KeyValue {
            key: "writer".to_string(),
            value: Some(writer_id.clone()),
        }]),
        created_by: Some(writer_id),
        column_orders: Some(column_orders),
        encryption_algorithm: None,
        footer_signing_key_metadata: None,
    })
}

/// Build a `DataPageV1` page header for the given column-chunk page parameters.
///
/// `nullable` controls whether the def-level/rep-level encoding fields advertise
/// `RLE` (used) or `PLAIN` (unused — the spec requires the field but Parquet
/// readers ignore it when the schema element is REQUIRED).
///
/// Returns `Err(Error::Encoding)` if any of the size fields exceeds `i32::MAX`
/// (Parquet wire-format limit).
pub(crate) fn build_data_page_header(
    num_values: i64,
    uncompressed_size: i64,
    nullable: bool,
) -> Result<PageHeader, Error> {
    let uncompressed_page_size =
        to_i32_or_encoding_err("uncompressed_page_size", uncompressed_size)?;
    let num_values_i32 = to_i32_or_encoding_err("num_values", num_values)?;
    let level_encoding = if nullable {
        Encoding::RLE
    } else {
        Encoding::PLAIN
    };
    Ok(PageHeader {
        type_: PageType::DATA_PAGE,
        uncompressed_page_size,
        compressed_page_size: uncompressed_page_size,
        crc: None,
        data_page_header: Some(DataPageHeader {
            num_values: num_values_i32,
            encoding: Encoding::PLAIN,
            definition_level_encoding: level_encoding,
            repetition_level_encoding: level_encoding,
            statistics: None,
        }),
        index_page_header: None,
        dictionary_page_header: None,
        data_page_header_v2: None,
    })
}

fn to_i32_or_encoding_err(name: &str, value: i64) -> Result<i32, Error> {
    i32::try_from(value)
        .map_err(|_| Error::Encoding(format!("{name} {value} exceeds i32::MAX (Parquet limit)")))
}
