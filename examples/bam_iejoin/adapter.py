"""BAM-file adapter: stream a remote BAM via oxbow over HTTPS, expose as virtual Parquet.

Per-row-group statistics on `chrom` (rname), `pos`, and `end` let DuckDB push
down chromosome and range predicates without scanning every row group.

The adapter materializes batches in memory once. For a small region restriction
(e.g. a single chromosome) this fits comfortably; a full-genome BAM would
benefit from a streaming, lazy variant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import fsspec
import oxbow as ox
import pyarrow as pa
import pyarrow.compute as pc

import virtual_parquet as vp

if TYPE_CHECKING:
    from collections.abc import Iterable


def _to_pyarrow_with_chrom(batch: object) -> pa.RecordBatch:
    """Convert an oxbow (arro3) batch to PyArrow with rname decoded to STRING.

    BAM rname is dictionary-encoded by oxbow; we materialize the decoded values
    so virtual_parquet's STRING (BYTE_ARRAY) encoder can serialize it.
    """
    pa_batch = pa.record_batch(batch)
    rname = pa_batch.column("rname").cast(pa.string())
    return pa.record_batch(
        {
            "chrom": rname,
            "pos": pa_batch.column("pos"),
            "end": pa_batch.column("end"),
            "qname": pa_batch.column("qname"),
        }
    )


class BamAdapter(vp.BaseAdapter):
    """Expose a (regional) BAM file as a virtual Parquet object.

    `oxbow` decodes the BAM into Arrow record batches; the constructor takes a
    list of region strings (e.g. ``["chr22"]``) and pulls only those reads.

    The adapter caches one PyArrow ``RecordBatch`` per row group. Each batch
    becomes one Parquet row group with min/max stats for ``chrom``, ``pos``,
    and ``end``.
    """

    PROJECTION: tuple[str, ...] = ("chrom", "pos", "end", "qname")

    def __init__(
        self,
        url: str,
        index_url: str | None = None,
        regions: Iterable[str] | None = None,
        protocol: str = "https",
    ) -> None:
        fs = fsspec.filesystem(protocol)
        bf = ox.from_bam(
            lambda: fs.open(url, "rb"),
            index=(lambda: fs.open(index_url, "rb")) if index_url else None,
        )
        if regions is not None:
            bf = bf.regions(list(regions))
        self._batches: list[pa.RecordBatch] = [_to_pyarrow_with_chrom(b) for b in bf.batches()]

        self.schema = vp.Schema(columns=(
            vp.Column("chrom", vp.ColumnType.STRING, nullable=False),
            vp.Column("pos", vp.ColumnType.INT32, nullable=False),
            vp.Column("end", vp.ColumnType.INT32, nullable=False),
            vp.Column("qname", vp.ColumnType.STRING, nullable=False),
        ))
        self.row_group_count = len(self._batches)
        self._stats = [self._batch_stats(b) for b in self._batches]

    @staticmethod
    def _batch_stats(batch: pa.RecordBatch) -> dict[str, object]:
        return {
            "chrom_min": pc.min(batch.column("chrom")).as_py(),
            "chrom_max": pc.max(batch.column("chrom")).as_py(),
            "pos_min": pc.min(batch.column("pos")).as_py(),
            "pos_max": pc.max(batch.column("pos")).as_py(),
            "end_min": pc.min(batch.column("end")).as_py(),
            "end_max": pc.max(batch.column("end")).as_py(),
        }

    def row_group_plan(self, index: int) -> vp.RowGroupPlan:
        b = self._batches[index]
        s = self._stats[index]
        return vp.RowGroupPlan(
            rows=b.num_rows,
            column_stats=(
                vp.ColumnStatistics(min=s["chrom_min"], max=s["chrom_max"], null_count=0),
                vp.ColumnStatistics(min=s["pos_min"], max=s["pos_max"], null_count=0),
                vp.ColumnStatistics(min=s["end_min"], max=s["end_max"], null_count=0),
                None,  # qname — high-cardinality
            ),
        )

    def fetch(self, index: int) -> pa.RecordBatch:
        return self._batches[index]
