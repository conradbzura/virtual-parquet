"""BAI-aware BAM adapter: row groups planned from the index, data lazy-fetched.

The adapter downloads the .bai (a few MB) once at construction and parses each
reference's metadata pseudo-bin (bin 37450) to learn record counts without
reading any BAM bytes. Each non-empty reference becomes one Parquet row group:

  * Schema: chrom (STRING), pos (INT32), end (INT32).
  * Per-row-group ``rows`` is the BAI metadata pseudo-bin's ``n_mapped`` field
    only. ``n_unmapped`` (placed-unmapped reads with FLAG 0x4 set) is excluded
    because ``oxbow.from_bam(...).regions([chrom])`` returns only records with
    a valid alignment span, so including ``n_unmapped`` would produce a row
    count that ``fetch()`` cannot satisfy and would violate the manifest-mode
    contract (RowGroupPlan.rows must equal RecordBatch.num_rows exactly).
  * Per-row-group ``chrom`` stats are exact (min == max == chromosome name).
  * Per-row-group ``column_byte_sizes`` for the STRING column are computed in
    closed form (every value in the row group is the same chromosome name, so
    the Plain-encoded size is ``rows * (4 + len(name))``); fixed-width column
    sizes are derived by virtual_parquet itself. This is "manifest mode" — the
    library never has to do a metadata pre-pass over the data.
  * BAM data for a row group is streamed from oxbow on the first ``fetch(i)``
    call, cached in the adapter so a second call returns the same RecordBatch
    (manifest-mode contract). The cache is protected by a ``threading.Lock``
    because the same adapter instance is reachable from multiple concurrent
    ``virtual_parquet.open(adapter)`` sessions when DuckDB's parallel reader
    opens the virtual file from several threads.

Memory bound: one chromosome's records materialized at a time (``fetch()``
combines oxbow's batch stream into a single RecordBatch to satisfy the
manifest-mode "one batch per row group" rule). For human chromosomes that is
typically tens of millions of records — the example trades aggregate-fetch
memory for closed-form row-group planning.
"""

from __future__ import annotations

import struct
import threading
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
# The metadata pseudo-bin always carries exactly two chunks: chunk[0] brackets
# virtual offsets, chunk[1] holds (n_mapped, n_unmapped). A pseudo-bin with any
# other chunk count is a malformed or non-conforming BAI — we hard-fail rather
# than silently dropping the reference's row count.
_BAI_META_BIN_CHUNK_COUNT = 2


def _parse_bai_counts(data: bytes, chrom_names: Sequence[str]) -> list[tuple[str, int]]:
    """Walk a BAI buffer and return [(chrom, n_mapped)] for every non-empty reference."""
    if data[:4] != b"BAI\x01":
        raise ValueError("not a BAI file (missing magic)")
    p = 4
    (n_ref,) = struct.unpack_from("<i", data, p)
    p += 4
    out: list[tuple[str, int]] = []
    for ref_id in range(n_ref):
        (n_bin,) = struct.unpack_from("<i", data, p)
        p += 4
        n_mapped = 0
        for _ in range(n_bin):
            bin_id, n_chunk = struct.unpack_from("<Ii", data, p)
            p += 8
            if bin_id == _BAI_META_BIN:
                if n_chunk != _BAI_META_BIN_CHUNK_COUNT:
                    raise ValueError(
                        f"BAI metadata pseudo-bin for ref_id={ref_id} has n_chunk="
                        f"{n_chunk}, expected {_BAI_META_BIN_CHUNK_COUNT}"
                    )
                # chunk[0] brackets virtual offsets; chunk[1] holds the counts.
                _b1, _e1, n_mapped, _n_unmapped = struct.unpack_from("<QQQQ", data, p)
                p += 32
            else:
                p += n_chunk * 16
        (n_intv,) = struct.unpack_from("<i", data, p)
        p += 4
        p += n_intv * 8
        if n_mapped > 0:
            out.append((chrom_names[ref_id], n_mapped))
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

        def _open_bam() -> object:
            return fs.open(url, "rb")

        def _open_bai() -> object:
            return fs.open(index_url, "rb")

        # Read the BAI bytes once for the row-group planner, then construct the
        # oxbow handle that fetch() will reuse. The base handle is restricted
        # at construction time to the three SAM fields the schema declares so
        # over-the-wire BAM reads do not pull qname/cigar/seq/qual/tags.
        with fs.open(index_url, "rb") as fh:
            bai_bytes = fh.read()
        self._bam_file = ox.from_bam(
            _open_bam,
            index=_open_bai,
            fields=["rname", "pos", "end"],
        )
        chrom_names = list(self._bam_file.chrom_names)
        self._row_groups: list[tuple[str, int]] = _parse_bai_counts(bai_bytes, chrom_names)

        self.schema = vp.Schema(
            columns=(
                vp.Column("chrom", vp.ColumnType.STRING, nullable=False),
                vp.Column("pos", vp.ColumnType.INT32, nullable=False),
                vp.Column("end", vp.ColumnType.INT32, nullable=False),
            )
        )
        self.row_group_count = len(self._row_groups)

        # The cache is reachable from multiple virtual_parquet.open(adapter)
        # sessions (the byte server serializes calls per-session, not per-adapter),
        # so the cache invariant "_cached_batch matches _cached_index" must be
        # protected against concurrent writes.
        self._cache_lock = threading.Lock()
        self._cached_index: int | None = None
        self._cached_batch: pa.RecordBatch | None = None

    def row_group_plan(self, index: int) -> vp.RowGroupPlan:
        """Return the row-group plan for `index` from BAI metadata alone."""
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
            # pos / end sizes are derived by the library itself (rows * 4).
            column_byte_sizes=(chrom_bytes, None, None),
        )

    def fetch(self, index: int) -> pa.RecordBatch:
        """Stream the row group's BAM region into a single RecordBatch.

        Caching is required by the manifest-mode contract: when this method is
        called twice for the same `index`, it must return the same RecordBatch
        (the library may re-encode after a probe). The single-slot cache
        suffices because virtual_parquet promises at most one fetch per index
        when ``column_byte_sizes`` is declared, but concurrent sessions on the
        same adapter instance can interleave fetches across indices, so the
        cache check-and-update is serialized by ``self._cache_lock``.
        """
        with self._cache_lock:
            if self._cached_index == index and self._cached_batch is not None:
                return self._cached_batch
        chrom, _ = self._row_groups[index]
        scanner = self._bam_file.regions([chrom])
        # Project + cast inside the loop so each batch carries only the three
        # declared columns at INT32; oxbow emits Int32 today but we cast
        # defensively so a future widening to Int64 surfaces here instead of
        # at the virtual_parquet schema-validation step.
        projected: list[pa.RecordBatch] = []
        for b in scanner.batches():
            pa_b = pa.record_batch(b)
            projected.append(
                pa.record_batch(
                    {
                        "chrom": pa_b.column("rname").cast(pa.string()),
                        "pos": pa_b.column("pos").cast(pa.int32()),
                        "end": pa_b.column("end").cast(pa.int32()),
                    }
                )
            )
        if not projected:
            # BAI says n_mapped > 0 but oxbow returned nothing — surface this
            # as a typed adapter failure rather than letting Table.from_batches
            # raise from inside the example.
            raise RuntimeError(
                f"oxbow returned no batches for chrom={chrom!r} despite BAI declaring "
                "non-zero mapped records; index may be stale relative to the BAM"
            )
        if len(projected) == 1:
            batch = projected[0]
        else:
            # Manifest-mode contract requires a single RecordBatch per row group.
            batch = pa.Table.from_batches(projected).combine_chunks().to_batches()[0]
        with self._cache_lock:
            self._cached_index = index
            self._cached_batch = batch
        return batch
