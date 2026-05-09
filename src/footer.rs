//! Parquet footer (FileMetaData) construction and Thrift compact serialization.
//!
//! This module owns the conversion from our internal `Schema` + computed row group
//! metadata into the canonical `parquet::format::FileMetaData` Thrift struct, plus
//! the bytes-out side via `TCompactOutputProtocol`. Per Constitution Principle II,
//! the footer is the only piece of metadata materialized in memory; column-chunk
//! bytes are emitted on demand by `reader.rs`.

use parquet::format::{
    ColumnChunk, ColumnMetaData, CompressionCodec, DataPageHeader, Encoding, FieldRepetitionType,
    FileMetaData, KeyValue, PageHeader, PageType, RowGroup, SchemaElement, Statistics, Type,
};
use parquet::thrift::TSerializable;
use thrift::protocol::TCompactOutputProtocol;

use crate::adapter::{ColumnType, Schema, StatValue};
use crate::error::Error;

/// Magic bytes that bracket every Parquet file.
pub(crate) const PARQUET_MAGIC: &[u8; 4] = b"PAR1";

/// Internal metadata for one finalized column chunk, ready to be referenced from the
/// footer. Computed by `reader.rs` during the metadata pre-pass.
#[derive(Debug, Clone)]
pub(crate) struct ComputedColumnChunk {
    pub(crate) physical_type: ColumnType,
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
            "schema has {} columns, exceeds i32::MAX",
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
        Some(Statistics {
            // parquet-cpp via PyArrow surfaces min/max via the legacy `min`/`max` fields
            // when present; emit them alongside the typed `min_value`/`max_value` so both
            // reader generations see the bounds. The byte form is identical for our v1
            // physical types.
            max: chunk.max_value.clone(),
            min: chunk.min_value.clone(),
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

    ColumnMetaData {
        type_: physical_type(chunk.physical_type),
        encodings: vec![Encoding::PLAIN, Encoding::RLE],
        path_in_schema: vec![column_name.to_string()],
        codec: CompressionCodec::UNCOMPRESSED,
        num_values: chunk.num_values,
        total_uncompressed_size: chunk.total_size,
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
        for (i, chunk) in rg.columns.iter().enumerate() {
            let column_name = &schema.columns()[i].name;
            chunks.push(ColumnChunk {
                file_path: None,
                file_offset: chunk.file_offset,
                meta_data: Some(build_column_metadata(column_name, chunk)),
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
            total_compressed_size: Some(rg.total_byte_size),
            ordinal: None,
        });
    }

    Ok(FileMetaData {
        version: 1,
        schema: schema_elements,
        num_rows: total_rows,
        row_groups: rg_thrift,
        key_value_metadata: Some(vec![KeyValue {
            key: "writer".to_string(),
            value: Some(format!("virtual-parquet {}", env!("CARGO_PKG_VERSION"))),
        }]),
        created_by: Some(format!("virtual-parquet {}", env!("CARGO_PKG_VERSION"))),
        column_orders: None,
        encryption_algorithm: None,
        footer_signing_key_metadata: None,
    })
}

/// Build a DataPageV1 page header for the given column-chunk page parameters.
///
/// Returns `Err(Error::Encoding)` if any of the size fields exceeds `i32::MAX`
/// (Parquet wire-format limit).
pub(crate) fn build_data_page_header(
    num_values: i64,
    uncompressed_size: i64,
    _has_definition_levels: bool,
) -> Result<PageHeader, Error> {
    let uncompressed_page_size = i32::try_from(uncompressed_size).map_err(|_| {
        Error::Encoding(format!(
            "uncompressed_page_size {uncompressed_size} exceeds i32::MAX (Parquet limit)"
        ))
    })?;
    let num_values_i32 = i32::try_from(num_values).map_err(|_| {
        Error::Encoding(format!(
            "num_values {num_values} exceeds i32::MAX (Parquet limit)"
        ))
    })?;
    Ok(PageHeader {
        type_: PageType::DATA_PAGE,
        uncompressed_page_size,
        compressed_page_size: uncompressed_page_size,
        crc: None,
        data_page_header: Some(DataPageHeader {
            num_values: num_values_i32,
            encoding: Encoding::PLAIN,
            // Definition levels (when emitted) use RLE; if there are no def levels,
            // the field still requires a value but is unused. Parquet readers ignore
            // it when the schema element is REQUIRED.
            definition_level_encoding: Encoding::RLE,
            repetition_level_encoding: Encoding::RLE,
            statistics: None,
        }),
        index_page_header: None,
        dictionary_page_header: None,
        data_page_header_v2: None,
    })
}
