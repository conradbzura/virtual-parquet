"""BAI-aware BAM adapter: row groups planned from the index, data lazy-fetched.

The adapter downloads the .bai (a few MB) once at construction and parses each
reference's metadata pseudo-bin (bin 37450) to learn record counts without
reading any BAM bytes. Each non-empty reference becomes one Parquet row group:

  * Schema: chrom (STRING), pos (INT32), end (INT32).
  * Per-row-group ``rows`` comes from the BAI metadata pseudo-bin
    (``n_mapped + n_unmapped``).
  * Per-row-group ``chrom`` stats are exact (min == max == chromosome name).
  * Per-row-group ``column_byte_sizes`` for the STRING column are computed in
    closed form (every value in the row group is the same chromosome name, so
    the Plain-encoded size is ``rows * (4 + len(name))``); fixed-width column
    sizes are derived by virtual_parquet itself. This is "manifest mode" — the
    library never has to do a metadata pre-pass over the data.
  * BAM data for a row group is streamed from oxbow on the first ``fetch(i)``
    call, cached in the adapter so a second call returns the same RecordBatch
    (manifest-mode contract).

Memory bound: O(1) row groups in flight (one chromosome's reads at a time).
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import fsspec
import oxbow as ox
import pyarrow as pa

import virtual_parquet as vp

if TYPE_CHECKING:
    from collections.abc import Sequence


# BAI metadata pseudo-bin per the htslib spec; encodes per-reference n_mapped
# and n_unmapped without scanning the BAM.
_BAI_META_BIN = 37450


def _parse_bai_counts(data: bytes, chrom_names: Sequence[str]) -> list[tuple[str, int]]:
    """Walk a BAI buffer and return [(chrom, total_records)] for every non-empty reference."""
    if data[:4] != b"BAI\x01":
        raise ValueError("not a BAI file (missing magic)")
    p = 4
    (n_ref,) = struct.unpack_from("<i", data, p)
    p += 4
    out: list[tuple[str, int]] = []
    for ref_id in range(n_ref):
        (n_bin,) = struct.unpack_from("<i", data, p)
        p += 4
        n_mapped = n_unmapped = 0
        for _ in range(n_bin):
            bin_id, n_chunk = struct.unpack_from("<Ii", data, p)
            p += 8
            if bin_id == _BAI_META_BIN and n_chunk == 2:
                # chunk[0] brackets virtual offsets; chunk[1] holds the counts.
                _b1, _e1, n_mapped, n_unmapped = struct.unpack_from("<QQQQ", data, p)
                p += 32
            else:
                p += n_chunk * 16
        (n_intv,) = struct.unpack_from("<i", data, p)
        p += 4
        p += n_intv * 8
        total = n_mapped + n_unmapped
        if total > 0:
            out.append((chrom_names[ref_id], total))
    return out


class BamAdapter(vp.BaseAdapter):
    """Lazy, BAI-driven BAM adapter."""

    def __init__(
        self,
        url: str,
        index_url: str,
        protocol: str = "https",
        cache_storage: str | None = None,
        cache_block_size: int = 1024 * 1024,
    ) -> None:
        """Construct the adapter.

        ``cache_storage`` enables fsspec's ``blockcache`` layer in front of the
        source filesystem: byte ranges fetched from the BAM URL are cached
        locally in ``cache_storage`` at ``cache_block_size`` granularity, so
        repeated queries on the same chromosome serve subsequent reads from
        local disk. Has no effect when the source is already on a local
        filesystem (``protocol='file'``).
        """
        if cache_storage is not None and protocol != "file":
            fs = fsspec.filesystem(
                "blockcache",
                target_protocol=protocol,
                cache_storage=cache_storage,
                block_size=cache_block_size,
            )
        else:
            fs = fsspec.filesystem(protocol)
        self._open_bam = lambda: fs.open(url, "rb")
        self._open_bai = lambda: fs.open(index_url, "rb")

        bf = ox.from_bam(self._open_bam, index=self._open_bai)
        chrom_names = list(bf.chrom_names)
        with fs.open(index_url, "rb") as fh:
            bai_bytes = fh.read()
        self._row_groups: list[tuple[str, int]] = _parse_bai_counts(bai_bytes, chrom_names)

        self.schema = vp.Schema(
            columns=(
                vp.Column("chrom", vp.ColumnType.STRING, nullable=False),
                vp.Column("pos", vp.ColumnType.INT32, nullable=False),
                vp.Column("end", vp.ColumnType.INT32, nullable=False),
            )
        )
        self.row_group_count = len(self._row_groups)

        self._cached_index: int | None = None
        self._cached_batch: pa.RecordBatch | None = None

    def row_group_plan(self, index: int) -> vp.RowGroupPlan:
        chrom, rows = self._row_groups[index]
        # Plain-encoded BYTE_ARRAY: 4-byte length prefix + UTF-8 bytes per value.
        chrom_bytes = rows * (4 + len(chrom.encode("utf-8")))
        return vp.RowGroupPlan(
            rows=rows,
            column_stats=(
                vp.ColumnStatistics(min=chrom, max=chrom, null_count=0),
                None,  # pos — declined; computing stats would require a data scan
                None,  # end — same
            ),
            # Manifest-mode: declare the chrom column's size so virtual_parquet
            # does not trigger a metadata pre-pass on this row group. Fixed-width
            # pos / end sizes are derived by the library itself (rows × 4).
            column_byte_sizes=(chrom_bytes, None, None),
        )

    def fetch(self, index: int) -> pa.RecordBatch:
        if self._cached_index == index and self._cached_batch is not None:
            return self._cached_batch
        chrom, _ = self._row_groups[index]
        bf = ox.from_bam(self._open_bam, index=self._open_bai).regions([chrom])
        projected: list[pa.RecordBatch] = []
        for b in bf.batches():
            pa_b = pa.record_batch(b)
            projected.append(
                pa.record_batch(
                    {
                        "chrom": pa_b.column("rname").cast(pa.string()),
                        "pos": pa_b.column("pos"),
                        "end": pa_b.column("end"),
                    }
                )
            )
        if len(projected) == 1:
            batch = projected[0]
        else:
            batch = pa.Table.from_batches(projected).combine_chunks().to_batches()[0]
        self._cached_index = index
        self._cached_batch = batch
        return batch
