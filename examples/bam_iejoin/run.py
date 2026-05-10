"""Drive a chr22 self-intersection on a whole-genome remote BAM through DuckDB.

End-to-end flow:
  remote BAM (HTTPS) -> BAI parsed at construction (row groups planned from
  the index alone, no BAM data read) -> oxbow lazy-fetches one chromosome's
  worth of reads only when DuckDB pulls bytes from that row group ->
  virtual_parquet emits Parquet bytes on demand -> fsspec adapter -> DuckDB.

The query is a self-intersection on read intervals with a 500 bp slop window,
shaped as the inequality predicates that DuckDB optimizes via its IE_JOIN
operator. Per-row-group chrom-only stats let DuckDB skip every row group that
doesn't contain chr22 reads, so only one chromosome's BAM bytes ever flow
through the pipeline.

Requires: oxbow, duckdb, fsspec[http], pyarrow installed alongside virtual_parquet.
"""

from __future__ import annotations

import datetime as _dt
import sys
import time
from pathlib import Path
from typing import IO, Any

import duckdb
import fsspec
import pyarrow.parquet as pq
from fsspec.spec import AbstractFileSystem

import virtual_parquet as vp

# Allow `python examples/bed_iejoin/run.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.bam_iejoin.adapter import BamAdapter  # noqa: E402

BAM_URL = (
    "https://projects.abdenlab.org/4dn2_hepdiff/freeze/2024-09/atacseq/encode/"
    "01_ESC/align/rep1/4DNFIVWIBHN4.trim.srt.nodup.no_chrM_MT.bam"
)
BAI_URL = BAM_URL + ".bai"
VFS_PATH = "atac.parquet"

SLOP = 500

# Self-exclusion is by ordering on (pos, end) rather than qname, so the schema
# can stay (chrom, pos, end) — the adapter never has to read a variable-width
# data column to size it, which keeps the BAI-only manifest mode honest.
QUERY = f"""
SELECT a.pos AS a_pos,
       a.end AS a_end,
       b.pos AS b_pos,
       b.end AS b_end
FROM   (SELECT * FROM atac WHERE chrom = 'chr22') a
JOIN   (SELECT * FROM atac WHERE chrom = 'chr22') b
  ON   a.pos - {SLOP} < b.end
  AND  a.end + {SLOP} > b.pos
  AND  (a.pos < b.pos OR (a.pos = b.pos AND a.end < b.end))
"""


class _CountingFile:
    """Pass-through file wrapper that records bytes read and read calls."""

    def __init__(self, inner: Any, counters: dict[str, int]) -> None:
        self._inner = inner
        self._c = counters

    def read(self, n: int = -1) -> bytes:
        b = self._inner.read(n)
        self._c["bytes"] += len(b)
        self._c["reads"] += 1
        return b

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._inner.seek(offset, whence)

    def tell(self) -> int:
        return self._inner.tell()

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def close(self) -> None:
        self._inner.close()

    def __enter__(self) -> _CountingFile:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    @property
    def closed(self) -> bool:
        return getattr(self._inner, "closed", False)


class VirtualParquetFS(AbstractFileSystem):
    """fsspec filesystem exposing a single virtual_parquet.VirtualParquetFile."""

    protocol = "vp"

    def __init__(
        self,
        open_vpf: Any,
        size_hint: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._open_vpf = open_vpf
        self._size_hint = size_hint
        self.counters: dict[str, int] = {"bytes": 0, "reads": 0}

    def reset_counters(self) -> None:
        self.counters = {"bytes": 0, "reads": 0}

    def _strip_protocol(self, path: str) -> str:
        return path.removeprefix("vp://")

    def info(self, path: str, **_: Any) -> dict[str, Any]:
        if self._size_hint is None:
            with self._open_vpf() as vpf:
                self._size_hint = vpf.size()
        return {
            "name": self._strip_protocol(path),
            "size": self._size_hint,
            "type": "file",
            "mtime": 0,
        }

    def modified(self, path: str) -> _dt.datetime:  # noqa: ARG002
        return _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc)

    def exists(self, path: str, **_: Any) -> bool:  # noqa: ARG002
        return True

    def ls(self, path: str, detail: bool = True, **_: Any) -> list[Any]:  # noqa: ARG002, FBT001, FBT002
        info = self.info(path)
        return [info] if detail else [info["name"]]

    def _open(
        self,
        path: str,
        mode: str = "rb",
        block_size: Any = None,
        autocommit: Any = True,
        cache_options: Any = None,
        **_: Any,
    ) -> IO[bytes]:
        if "w" in mode or "a" in mode or "+" in mode:
            raise NotImplementedError("VirtualParquetFS is read-only")
        return _CountingFile(self._open_vpf(), self.counters)  # type: ignore[return-value]


def _format_int(n: int) -> str:
    return f"{n:,}"


def main() -> None:
    print(f"Indexing BAM via BAI alone (no BAM data read yet): {Path(BAM_URL).name}", flush=True)
    t = time.time()
    adapter = BamAdapter(BAM_URL, index_url=BAI_URL)
    total_rows = sum(rows for _, rows in adapter._row_groups)
    print(
        f"  {adapter.row_group_count} row groups planned from BAI metadata, "
        f"{_format_int(total_rows)} reads total — in {time.time() - t:.2f}s"
    )

    print("\nFirst / last 5 row-group declarations (rows from BAI; pos/end stats declined):")
    rgs = list(enumerate(adapter._row_groups))
    for i, (chrom, rows) in rgs[:5] + rgs[-5:]:
        print(f"  rg {i:>3d}: chrom={chrom!r:>30s}  rows={_format_int(rows):>12s}")

    with vp.open(adapter) as vpf:
        size = vpf.size()
    print(f"\nVirtual Parquet size: {_format_int(size)} bytes")

    with vp.open(adapter) as vpf:
        pf = pq.ParquetFile(vpf)
        print(
            f"PyArrow footer view: {_format_int(pf.metadata.num_rows)} rows in "
            f"{pf.metadata.num_row_groups} row groups; "
            f"format_version={pf.metadata.format_version}"
        )

    vfs = VirtualParquetFS(open_vpf=lambda: vp.open(adapter), size_hint=size)
    fsspec.register_implementation("vp", lambda **_: vfs, clobber=True)
    src = f"read_parquet('vp://{VFS_PATH}')"

    # Measured query — fresh DuckDB session so byte counters are clean.
    print("\nRunning the chr22 self-intersection...", flush=True)
    vfs.reset_counters()
    con = duckdb.connect()
    con.register_filesystem(vfs)
    t = time.time()
    df = con.sql(QUERY.replace("atac", src)).df()
    elapsed = time.time() - t
    print(f"  {_format_int(len(df))} overlapping read pairs in {elapsed:.2f}s")
    print("\nFirst 5 pairs:")
    print(df.head(5).to_string(index=False))

    bytes_read = vfs.counters["bytes"]
    reads = vfs.counters["reads"]
    fraction = bytes_read / size * 100
    print(
        f"\nByte-range reads from the virtual parquet: {reads} requests, "
        f"{_format_int(bytes_read)} bytes ({fraction:.2f}% of the "
        f"{_format_int(size)}-byte file). Row-group pruning at work — "
        "DuckDB only pulled the column chunks for the chr22 row groups."
    )

    print("\nDuckDB EXPLAIN ANALYZE:\n")
    con2 = duckdb.connect()
    con2.register_filesystem(vfs)
    plan = con2.sql(("EXPLAIN ANALYZE " + QUERY).replace("atac", src)).fetchall()
    for row in plan:
        for cell in row:
            print(cell)


if __name__ == "__main__":
    main()
