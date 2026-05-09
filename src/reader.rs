//! Byte-server: state machine, file layout, and range-request handler.
//!
//! Constitution Principle II is enforced by this module: at most one row group's
//! encoded bytes are materialized in memory at any time (the LRU cache, capacity 1).
//! The footer is the only other resident bytes. Forward-compat discipline #1 (thread
//! safety) is satisfied by wrapping mutable state in a `Mutex` plus a `Condvar` so
//! exactly one thread runs the metadata pre-pass while other threads wait.
//!
//! State transitions: `Created → MetadataPass → Serving → Closed`. Transitions are
//! one-way; `Closed` is terminal. The `MetadataPass` state holds no data — it's a
//! coordination marker so concurrent first-reads do not double-invoke the adapter.

use std::sync::{Arc, Condvar, Mutex};

use arrow_array::{
    Array, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, LargeStringArray,
    RecordBatch, StringArray,
};
use arrow_schema::DataType;

use crate::adapter::{Adapter, Column, ColumnType, RowGroupPlan, Schema};
use crate::encoding::plain::{
    compute_def_levels_size, compute_fixed_value_size, encode_boolean_values,
    encode_float32_values, encode_float64_values, encode_int32_values, encode_int64_values,
    encode_large_string_values, encode_string_values,
};
use crate::error::Error;
use crate::footer::{
    build_data_page_header, build_file_metadata, encode_stat_value, serialize_thrift,
    ComputedColumnChunk, ComputedRowGroup, PARQUET_MAGIC,
};

const TRAILER_SIZE: i64 = 8; // 4 bytes footer length + 4 bytes PAR1 magic

/// Layout of a single column chunk within the virtual file.
#[derive(Debug, Clone)]
struct ColumnChunkLayout {
    column_type: ColumnType,
    file_offset: i64,
    page_header_bytes: Vec<u8>,
    def_levels_size: i64,
    values_size: i64,
}

impl ColumnChunkLayout {
    fn total_size(&self) -> i64 {
        self.page_header_bytes.len() as i64 + self.def_levels_size + self.values_size
    }
}

#[derive(Debug, Clone)]
struct RowGroupLayout {
    num_rows: i64,
    columns: Vec<ColumnChunkLayout>,
    /// Total uncompressed byte size = sum of column chunks.
    total_byte_size: i64,
}

#[derive(Debug)]
struct FileLayout {
    schema: Arc<Schema>,
    row_groups: Vec<RowGroupLayout>,
    /// Adapter-declared row group plans (kept for footer statistics population).
    plans: Vec<RowGroupPlan>,
    footer_offset: i64,
    footer_bytes: Vec<u8>,
    total_size: i64,
    /// Sorted by `file_offset` for binary-search lookup. Each entry is
    /// `(start, row_group_index, column_index)`.
    chunk_index: Vec<(i64, u32, usize)>,
}

/// Cached row-group encoded bytes, LRU capacity 1.
#[derive(Debug)]
struct RowGroupCache {
    index: u32,
    column_chunks: Vec<Vec<u8>>,
}

#[derive(Debug)]
enum State {
    Created,
    /// Exactly one thread is running the metadata pre-pass; other threads wait on
    /// `cv` until the transition to `Serving` (or back to `Created` on error).
    MetadataPass,
    Serving {
        layout: FileLayout,
        cache: Option<RowGroupCache>,
    },
    Closed,
}

pub(crate) struct ByteServer {
    adapter: Arc<dyn Adapter>,
    state: Mutex<State>,
    cv: Condvar,
}

impl ByteServer {
    pub(crate) fn new(adapter: Arc<dyn Adapter>) -> Self {
        Self {
            adapter,
            state: Mutex::new(State::Created),
            cv: Condvar::new(),
        }
    }

    /// Total size of the virtual file. Lazily triggers the metadata pre-pass on first call.
    pub(crate) fn total_size(&self) -> Result<i64, Error> {
        self.ensure_serving()?;
        let state = self.state.lock().expect("byte-server lock poisoned");
        match &*state {
            State::Serving { layout, .. } => Ok(layout.total_size),
            State::Closed => Err(Error::InvalidState("byte-server is closed".into())),
            State::Created | State::MetadataPass => {
                unreachable!("ensure_serving guarantees Serving state on Ok")
            }
        }
    }

    /// Read up to `length` bytes starting at `start`. Returns fewer bytes near EOF.
    pub(crate) fn read_range(&self, start: i64, length: usize) -> Result<Vec<u8>, Error> {
        if start < 0 {
            return Err(Error::InvalidRange(format!(
                "negative start offset {start}"
            )));
        }
        self.ensure_serving()?;
        let total = self.total_size()?;
        let mut pos = start;
        let end = start
            .saturating_add(i64::try_from(length).unwrap_or(i64::MAX))
            .min(total);
        let mut out: Vec<u8> = Vec::with_capacity(usize::try_from((end - pos).max(0)).unwrap_or(0));
        while pos < end {
            let consumed = self.read_segment(&mut out, pos, end)?;
            if consumed == 0 {
                break;
            }
            pos += consumed;
        }
        Ok(out)
    }

    /// Close the byte-server: drop layout/cache, reject subsequent operations.
    pub(crate) fn close(&self) {
        let mut state = self.state.lock().expect("byte-server lock poisoned");
        *state = State::Closed;
        self.cv.notify_all();
    }

    /// True iff the byte-server has been closed.
    pub(crate) fn is_closed(&self) -> bool {
        let state = self.state.lock().expect("byte-server lock poisoned");
        matches!(&*state, State::Closed)
    }

    /// Run the metadata pre-pass (idempotent, single-flight) and transition to `Serving`.
    fn ensure_serving(&self) -> Result<(), Error> {
        let mut state = self.state.lock().expect("byte-server lock poisoned");
        loop {
            match &*state {
                State::Serving { .. } => return Ok(()),
                State::Closed => {
                    return Err(Error::InvalidState("byte-server is closed".into()));
                }
                State::MetadataPass => {
                    // Another thread is doing the pre-pass; wait for it.
                    state = self
                        .cv
                        .wait(state)
                        .expect("byte-server lock poisoned during cv.wait");
                }
                State::Created => {
                    // We win the race: claim the pre-pass slot.
                    *state = State::MetadataPass;
                    drop(state);
                    let outcome = self.compute_layout();
                    let mut state = self.state.lock().expect("byte-server lock poisoned");
                    match (&*state, outcome) {
                        // Closed in the meantime — just propagate.
                        (State::Closed, _) => {
                            self.cv.notify_all();
                            return Err(Error::InvalidState("byte-server is closed".into()));
                        }
                        (_, Ok(layout)) => {
                            *state = State::Serving {
                                layout,
                                cache: None,
                            };
                            self.cv.notify_all();
                            return Ok(());
                        }
                        (_, Err(e)) => {
                            // Reset to Created so a future caller can retry the pre-pass.
                            *state = State::Created;
                            self.cv.notify_all();
                            return Err(e);
                        }
                    }
                }
            }
        }
    }

    fn compute_layout(&self) -> Result<FileLayout, Error> {
        let schema = Arc::new(self.adapter.schema().clone());
        let rg_count = self.adapter.row_group_count();

        let mut plans: Vec<RowGroupPlan> = Vec::with_capacity(rg_count as usize);
        let mut row_groups: Vec<RowGroupLayout> = Vec::with_capacity(rg_count as usize);

        // First pass: gather plans, run pre-pass fetches if needed, compute null counts and sizes.
        for i in 0..rg_count {
            let plan = self.adapter.row_group_plan(i)?;
            plan.validate(&schema)?;
            let needs_prepass = needs_prepass(&schema, &plan);
            let observed = if needs_prepass {
                Some(self.adapter.fetch(i)?.batch)
            } else {
                None
            };
            if let Some(batch) = &observed {
                validate_batch(&schema, &plan, batch)?;
            }
            let layout = build_row_group_layout(&schema, &plan, observed.as_ref())?;
            plans.push(plan);
            row_groups.push(layout);
        }

        // Second pass: assign file offsets and serialize page headers using running offsets.
        let mut cursor: i64 = 4; // leading PAR1 magic occupies bytes 0..4
        let mut chunk_index: Vec<(i64, u32, usize)> = Vec::new();
        for (rg_idx, rg) in row_groups.iter_mut().enumerate() {
            for (col_idx, col) in rg.columns.iter_mut().enumerate() {
                col.file_offset = cursor;
                let rg_idx_u32 =
                    u32::try_from(rg_idx).expect("row group count fits in u32 by adapter contract");
                chunk_index.push((cursor, rg_idx_u32, col_idx));
                cursor += col.total_size();
            }
        }

        // Now build the footer with computed offsets.
        let mut computed_groups: Vec<ComputedRowGroup> = Vec::with_capacity(row_groups.len());
        for (rg_layout, plan) in row_groups.iter().zip(plans.iter()) {
            let mut chunks: Vec<ComputedColumnChunk> = Vec::with_capacity(rg_layout.columns.len());
            for (col_idx, col_layout) in rg_layout.columns.iter().enumerate() {
                let stats = plan.column_stats.get(col_idx).cloned().flatten();
                let (min_value, max_value, declared_null_count) = match stats {
                    Some(s) => (
                        s.min.as_ref().map(encode_stat_value),
                        s.max.as_ref().map(encode_stat_value),
                        s.null_count,
                    ),
                    None => (None, None, None),
                };
                // FR-008: pass adapter-declared null_count through unchanged. When the
                // adapter declined to declare it, leave it absent — never substitute the
                // observed value.
                chunks.push(ComputedColumnChunk {
                    physical_type: col_layout.column_type,
                    file_offset: col_layout.file_offset,
                    total_size: col_layout.total_size(),
                    num_values: rg_layout.num_rows,
                    null_count: declared_null_count,
                    min_value,
                    max_value,
                });
            }
            computed_groups.push(ComputedRowGroup {
                num_rows: rg_layout.num_rows,
                total_byte_size: rg_layout.total_byte_size,
                columns: chunks,
            });
        }

        let metadata = build_file_metadata(&schema, computed_groups)?;
        let footer_bytes = serialize_thrift(&metadata)?;
        let footer_offset = cursor;
        let footer_len_i64 = i64::try_from(footer_bytes.len()).map_err(|_| {
            Error::Encoding(format!(
                "footer size {} exceeds i64::MAX",
                footer_bytes.len()
            ))
        })?;
        let total_size = footer_offset + footer_len_i64 + TRAILER_SIZE;

        Ok(FileLayout {
            schema,
            row_groups,
            plans,
            footer_offset,
            footer_bytes,
            total_size,
            chunk_index,
        })
    }

    /// Read whatever single contiguous segment lies at `pos`, up to `end`. Append to `out`.
    fn read_segment(&self, out: &mut Vec<u8>, pos: i64, end: i64) -> Result<i64, Error> {
        let mut state = self.state.lock().expect("byte-server lock poisoned");
        let State::Serving { layout, cache } = &mut *state else {
            return Err(Error::InvalidState("byte-server is not serving".into()));
        };

        // [0, 4): leading magic
        if pos < 4 {
            let take = ((4 - pos) as usize).min((end - pos) as usize);
            out.extend_from_slice(&PARQUET_MAGIC[pos as usize..pos as usize + take]);
            return Ok(take as i64);
        }

        // [4, footer_offset): column chunks
        if pos < layout.footer_offset {
            return read_column_chunks_segment(out, pos, end, layout, cache, self.adapter.as_ref());
        }

        let footer_end = layout.footer_offset + layout.footer_bytes.len() as i64;

        // [footer_offset, footer_end): footer thrift
        if pos < footer_end {
            let local = (pos - layout.footer_offset) as usize;
            let take = (layout.footer_bytes.len() - local).min((end - pos) as usize);
            out.extend_from_slice(&layout.footer_bytes[local..local + take]);
            return Ok(take as i64);
        }

        // [footer_end, footer_end + 4): footer length (i32 LE)
        if pos < footer_end + 4 {
            let local = (pos - footer_end) as usize;
            let footer_len = i32::try_from(layout.footer_bytes.len()).map_err(|_| {
                Error::Encoding(format!(
                    "footer size {} exceeds i32::MAX (Parquet wire-format limit)",
                    layout.footer_bytes.len()
                ))
            })?;
            let bytes = footer_len.to_le_bytes();
            let take = (4 - local).min((end - pos) as usize);
            out.extend_from_slice(&bytes[local..local + take]);
            return Ok(take as i64);
        }

        // [total - 4, total): trailing magic
        if pos < layout.total_size {
            let local = (pos - (footer_end + 4)) as usize;
            let take = (4 - local).min((end - pos) as usize);
            out.extend_from_slice(&PARQUET_MAGIC[local..local + take]);
            return Ok(take as i64);
        }

        Ok(0)
    }
}

/// True if any column's size or null count is undeclared and requires a pre-pass fetch.
fn needs_prepass(schema: &Schema, plan: &RowGroupPlan) -> bool {
    for (i, col) in schema.columns().iter().enumerate() {
        let declared_size = plan
            .column_byte_sizes
            .as_ref()
            .and_then(|sizes| sizes.get(i).copied().flatten());
        let declared_null_count = plan
            .column_stats
            .get(i)
            .and_then(|s| s.as_ref())
            .and_then(|s| s.null_count);

        match col.data_type {
            // Variable-width: must have declared byte size to skip pre-pass.
            ColumnType::String => {
                if declared_size.is_none() {
                    return true;
                }
            }
            // Fixed-width nullable: needs null_count from stats to skip pre-pass.
            _ if col.nullable => {
                if declared_null_count.is_none() {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

fn validate_batch(schema: &Schema, plan: &RowGroupPlan, batch: &RecordBatch) -> Result<(), Error> {
    if batch.num_columns() != schema.len() {
        return Err(Error::SchemaMismatch(format!(
            "adapter returned batch with {} columns, schema declares {}",
            batch.num_columns(),
            schema.len()
        )));
    }
    if batch.num_rows() as i64 != plan.rows {
        return Err(Error::SchemaMismatch(format!(
            "adapter returned batch with {} rows, plan declared {}",
            batch.num_rows(),
            plan.rows
        )));
    }
    let arrow_schema = batch.schema();
    for (i, col) in schema.columns().iter().enumerate() {
        let field = arrow_schema.field(i);
        if field.name() != &col.name {
            return Err(Error::SchemaMismatch(format!(
                "column {i}: expected name {:?}, got {:?}",
                col.name,
                field.name()
            )));
        }
        if !arrow_type_matches(field.data_type(), col.data_type) {
            return Err(Error::SchemaMismatch(format!(
                "column {:?}: expected {:?}, got {:?}",
                col.name,
                col.data_type,
                field.data_type()
            )));
        }
        if !col.nullable && field.is_nullable() && batch.column(i).null_count() > 0 {
            return Err(Error::SchemaMismatch(format!(
                "column {:?} is declared non-nullable but contains nulls",
                col.name
            )));
        }
    }
    Ok(())
}

fn arrow_type_matches(arrow: &DataType, ours: ColumnType) -> bool {
    matches!(
        (arrow, ours),
        (DataType::Int32, ColumnType::Int32)
            | (DataType::Int64, ColumnType::Int64)
            | (DataType::Float32, ColumnType::Float32)
            | (DataType::Float64, ColumnType::Float64)
            | (DataType::Boolean, ColumnType::Boolean)
            | (DataType::Utf8, ColumnType::String)
            | (DataType::LargeUtf8, ColumnType::String)
    )
}

/// Compute the layout of one row group given the plan and (optionally) observed data.
fn build_row_group_layout(
    schema: &Schema,
    plan: &RowGroupPlan,
    observed: Option<&RecordBatch>,
) -> Result<RowGroupLayout, Error> {
    let mut columns: Vec<ColumnChunkLayout> = Vec::with_capacity(schema.len());
    for (i, col) in schema.columns().iter().enumerate() {
        let layout = build_column_chunk_layout(col, i, plan, observed)?;
        columns.push(layout);
    }
    let total_byte_size: i64 = columns.iter().map(ColumnChunkLayout::total_size).sum();
    Ok(RowGroupLayout {
        num_rows: plan.rows,
        columns,
        total_byte_size,
    })
}

fn build_column_chunk_layout(
    col: &Column,
    col_index: usize,
    plan: &RowGroupPlan,
    observed: Option<&RecordBatch>,
) -> Result<ColumnChunkLayout, Error> {
    let rows = plan.rows;
    let declared_null_count = plan
        .column_stats
        .get(col_index)
        .and_then(|s| s.as_ref())
        .and_then(|s| s.null_count);
    let declared_byte_size = plan
        .column_byte_sizes
        .as_ref()
        .and_then(|sizes| sizes.get(col_index).copied().flatten());

    // Determine null count.
    let null_count: i64 = if !col.nullable {
        0
    } else if let Some(nc) = declared_null_count {
        nc
    } else if let Some(batch) = observed {
        batch.column(col_index).null_count() as i64
    } else {
        unreachable!(
            "needs_prepass invariant: nullable column {:?} with no declared null_count must \
             have observed batch",
            col.name
        );
    };

    // Compute values block size.
    let values_size: i64 = match col.data_type {
        ColumnType::String => {
            if let Some(sz) = declared_byte_size {
                sz
            } else if let Some(batch) = observed {
                let array = batch.column(col_index);
                match array.data_type() {
                    DataType::Utf8 => {
                        let arr = array
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .expect("Utf8 type must downcast to StringArray");
                        sum_string_bytes(arr)
                    }
                    DataType::LargeUtf8 => {
                        let arr = array
                            .as_any()
                            .downcast_ref::<LargeStringArray>()
                            .expect("LargeUtf8 type must downcast to LargeStringArray");
                        sum_string_bytes(arr)
                    }
                    other => {
                        return Err(Error::SchemaMismatch(format!(
                            "column {:?}: expected Utf8 or LargeUtf8, got {other:?}",
                            col.name
                        )));
                    }
                }
            } else {
                return Err(Error::InvalidPlan(format!(
                    "variable-width column {:?} has no declared byte size and no observed data",
                    col.name
                )));
            }
        }
        ty => {
            let computed = compute_fixed_value_size(ty, rows, null_count)?;
            if let Some(declared) = declared_byte_size {
                if declared != computed {
                    return Err(Error::ByteSizeMismatch(format!(
                        "column {:?}: declared byte size {declared} != computed {computed}",
                        col.name
                    )));
                }
            }
            computed
        }
    };

    let def_levels_size = if col.nullable {
        compute_def_levels_size(rows)
    } else {
        0
    };
    let total_data_size = def_levels_size + values_size;

    let page_header = build_data_page_header(rows, total_data_size, col.nullable)?;
    let page_header_bytes = serialize_thrift(&page_header)?;

    Ok(ColumnChunkLayout {
        column_type: col.data_type,
        // file_offset is filled in during the second pass.
        file_offset: 0,
        page_header_bytes,
        def_levels_size,
        values_size,
    })
}

/// Sum the encoded byte size for a string array (4-byte length prefix + UTF-8 bytes
/// per non-null value). Generic over Utf8 (i32 offsets) and LargeUtf8 (i64 offsets).
fn sum_string_bytes<O: arrow_array::OffsetSizeTrait>(
    array: &arrow_array::GenericByteArray<arrow_array::types::GenericStringType<O>>,
) -> i64 {
    let mut total: i64 = 0;
    for i in 0..array.len() {
        if array.is_valid(i) {
            total += 4 + array.value(i).len() as i64;
        }
    }
    total
}

fn read_column_chunks_segment(
    out: &mut Vec<u8>,
    pos: i64,
    end: i64,
    layout: &FileLayout,
    cache: &mut Option<RowGroupCache>,
    adapter: &dyn Adapter,
) -> Result<i64, Error> {
    let (rg_idx, col_idx, local_offset) = locate_column_chunk(pos, layout)?;
    let chunk_bytes = ensure_chunk(rg_idx, col_idx, layout, cache, adapter)?;
    let chunk_remaining = chunk_bytes.len() - local_offset;
    let take = chunk_remaining.min((end - pos) as usize);
    out.extend_from_slice(&chunk_bytes[local_offset..local_offset + take]);
    Ok(take as i64)
}

/// Binary-search the chunk index for the column chunk containing `pos`.
fn locate_column_chunk(pos: i64, layout: &FileLayout) -> Result<(u32, usize, usize), Error> {
    // partition_point gives the index of the first chunk whose start > pos; subtract 1.
    let i = layout
        .chunk_index
        .partition_point(|(start, _, _)| *start <= pos);
    if i == 0 {
        return Err(Error::InvalidRange(format!(
            "offset {pos} precedes the first column chunk"
        )));
    }
    let (chunk_start, rg_idx, col_idx) = layout.chunk_index[i - 1];
    let chunk = &layout.row_groups[rg_idx as usize].columns[col_idx];
    let chunk_end = chunk_start + chunk.total_size();
    if pos < chunk_start || pos >= chunk_end {
        return Err(Error::InvalidRange(format!(
            "offset {pos} does not fall in any column chunk region"
        )));
    }
    Ok((rg_idx, col_idx, (pos - chunk_start) as usize))
}

/// Return a slice for the requested column chunk's encoded bytes, populating the cache
/// (capacity 1) if the row group is not currently cached.
fn ensure_chunk<'a>(
    rg_idx: u32,
    col_idx: usize,
    layout: &FileLayout,
    cache: &'a mut Option<RowGroupCache>,
    adapter: &dyn Adapter,
) -> Result<&'a [u8], Error> {
    let rg_layout = &layout.row_groups[rg_idx as usize];

    let needs_install = !matches!(cache, Some(c) if c.index == rg_idx);
    if needs_install {
        let batch = adapter.fetch(rg_idx)?.batch;
        let plan = &layout.plans[rg_idx as usize];
        validate_batch(&layout.schema, plan, &batch)?;

        let mut chunks: Vec<Vec<u8>> = Vec::with_capacity(rg_layout.columns.len());
        for (i, col) in layout.schema.columns().iter().enumerate() {
            let array = batch.column(i);
            let layout_col = &rg_layout.columns[i];
            let (def_block, value_bytes) = encode_column(col, array.as_ref())?;
            let mut chunk_bytes = Vec::with_capacity(
                layout_col.page_header_bytes.len() + def_block.len() + value_bytes.len(),
            );
            chunk_bytes.extend_from_slice(&layout_col.page_header_bytes);
            chunk_bytes.extend_from_slice(&def_block);
            chunk_bytes.extend_from_slice(&value_bytes);
            if chunk_bytes.len() as i64 != layout_col.total_size() {
                return Err(Error::Encoding(format!(
                    "column chunk size mismatch at row_group {rg_idx} col {i}: encoded {} bytes, expected {}",
                    chunk_bytes.len(),
                    layout_col.total_size()
                )));
            }
            chunks.push(chunk_bytes);
        }
        *cache = Some(RowGroupCache {
            index: rg_idx,
            column_chunks: chunks,
        });
    }
    Ok(&cache
        .as_ref()
        .expect("cache was just installed or matched")
        .column_chunks[col_idx])
}

fn encode_column(col: &Column, array: &dyn Array) -> Result<(Vec<u8>, Vec<u8>), Error> {
    match col.data_type {
        ColumnType::Int32 => {
            let arr = array.as_any().downcast_ref::<Int32Array>().ok_or_else(|| {
                Error::SchemaMismatch(format!("column {:?}: expected Int32Array", col.name))
            })?;
            encode_int32_values(arr, col.nullable)
        }
        ColumnType::Int64 => {
            let arr = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                Error::SchemaMismatch(format!("column {:?}: expected Int64Array", col.name))
            })?;
            encode_int64_values(arr, col.nullable)
        }
        ColumnType::Float32 => {
            let arr = array
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| {
                    Error::SchemaMismatch(format!("column {:?}: expected Float32Array", col.name))
                })?;
            encode_float32_values(arr, col.nullable)
        }
        ColumnType::Float64 => {
            let arr = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| {
                    Error::SchemaMismatch(format!("column {:?}: expected Float64Array", col.name))
                })?;
            encode_float64_values(arr, col.nullable)
        }
        ColumnType::Boolean => {
            let arr = array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| {
                    Error::SchemaMismatch(format!("column {:?}: expected BooleanArray", col.name))
                })?;
            encode_boolean_values(arr, col.nullable)
        }
        ColumnType::String => match array.data_type() {
            DataType::Utf8 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("Utf8 type must downcast to StringArray");
                encode_string_values(arr, col.nullable)
            }
            DataType::LargeUtf8 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .expect("LargeUtf8 type must downcast to LargeStringArray");
                encode_large_string_values(arr, col.nullable)
            }
            other => Err(Error::SchemaMismatch(format!(
                "column {:?}: expected Utf8 or LargeUtf8, got {other:?}",
                col.name
            ))),
        },
    }
}
