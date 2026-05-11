# virtual-parquet

Scaffolding for serving non-Parquet data sources to mainstream Parquet-consuming engines via a stable adapter contract.

The library is the framework adapters are built on; it ships zero domain-specific adapters in its public namespace. Adapter authors declare a schema, yield Arrow record batches, and optionally declare row-group statistics. The library produces engine-readable Parquet bytes on demand.

## Status

Pre-alpha. Single-file uncompressed Parquet, Plain encoding only, conformance tested against PyArrow.

## Install

```bash
pip install virtual-parquet pyarrow
```

PyArrow is not a dependency of `virtual-parquet`; install it for the read-back step.

## Quickstart

```python
import pyarrow as pa
import pyarrow.parquet as pq
import virtual_parquet as vp


class FixedBatchAdapter(vp.BaseAdapter):
    def __init__(self) -> None:
        self.schema = vp.Schema(columns=(
            vp.Column(name="id", type=vp.ColumnType.INT64, nullable=False),
            vp.Column(name="label", type=vp.ColumnType.STRING, nullable=True),
        ))
        self.row_group_count = 1
        self._batch = pa.record_batch({
            "id":    pa.array([1, 2, 3, 4], type=pa.int64()),
            "label": pa.array(["alpha", "beta", None, "delta"]),
        })

    def row_group_plan(self, index: int) -> vp.RowGroupPlan:
        return vp.RowGroupPlan(
            rows=4,
            column_stats=(
                vp.ColumnStatistics(min=1, max=4, null_count=0),
                None,
            ),
        )

    def fetch(self, index: int) -> pa.RecordBatch:
        return self._batch


vpf = vp.open(FixedBatchAdapter())
table = pq.read_table(vpf)
print(table)
```

The Parquet bytes are never persisted — they are generated on demand in response to PyArrow's range requests against the virtual file. Async adapters use the same shape with `async def` on the data-producing methods (see `vp.BaseAsyncAdapter`).

## License

MIT
