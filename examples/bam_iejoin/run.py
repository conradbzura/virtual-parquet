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
import os
import sys
import threading
import time
from pathlib import Path
from typing import IO, Any

import duckdb
import fsspec
import pyarrow.parquet as pq
from fsspec.spec import AbstractFileSystem

import virtual_parquet as vp

# Allow `python examples/bam_iejoin/run.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.bam_iejoin.adapter import BamAdapter

BAM_URL = os.environ.get(
    "VP_EXAMPLE_BAM_URL",
    "https://projects.abdenlab.org/4dn2_hepdiff/freeze/2024-09/atacseq/encode/"
    "01_ESC/align/rep1/4DNFIVWIBHN4.trim.srt.nodup.no_chrM_MT.bam",
)
BAI_URL = os.environ.get("VP_EXAMPLE_BAI_URL", BAM_URL + ".bai")
VFS_PATH = "atac.parquet"
VFS_URL = f"vp://{VFS_PATH}"

SLOP = 500

# `{src}` is replaced with the DuckDB table-function call (read_parquet on
# the vp:// URL); the placeholder is unlikely to collide with column names.
# The deduplication clause uses a strict lex order on (pos, end, qname-proxy)
# — we synthesize a stable per-row id via DuckDB's `rowid` pseudo-column to
# break ties between reads with identical (pos, end), which would otherwise
# both be dropped under a strict (pos, end) comparison.
QUERY = """
WITH atac AS (
    SELECT *, rowid AS _rid FROM {src}
)
SELECT a.pos AS a_pos,
       a.end AS a_end,
       b.pos AS b_pos,
       b.end AS b_end
FROM   (SELECT * FROM atac WHERE chrom = 'chr22') a
JOIN   (SELECT * FROM atac WHERE chrom = 'chr22') b
  ON   a.pos - {slop} < b.end
  AND  a.end + {slop} > b.pos
  AND  (
         a.pos < b.pos
         OR (a.pos = b.pos AND a.end < b.end)
         OR (a.pos = b.pos AND a.end = b.end AND a._rid < b._rid)
       )
"""


class _CountingFile:
    """Pass-through file wrapper that records bytes read and read calls.

    The wrapper is shared across threads when DuckDB's parallel reader opens
    several handles concurrently — `_lock` makes the counter increments and
    the close-idempotence flag safe.
    """

    def __init__(self, inner: Any, counters: dict[str, int], lock: threading.Lock) -> None:
        self._inner = inner
        self._c = counters
        self._lock = lock
        self._closed_flag = False

    def read(self, n: int = -1) -> bytes:
        b = self._inner.read(n)
        with self._lock:
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
        with self._lock:
            if self._closed_flag:
                return
            self._closed_flag = True
        self._inner.close()

    def __enter__(self) -> _CountingFile:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    @property
    def closed(self) -> bool:
        return self._closed_flag or getattr(self._inner, "closed", False)


class VirtualParquetFS(AbstractFileSystem):
    """fsspec filesystem exposing a single virtual_parquet.VirtualParquetFile.

    See https://filesystem-spec.readthedocs.io/en/latest/developer.html for the
    filesystem-author contract this subclass implements.
    """

    protocol = "vp"

    def __init__(
        self,
        open_vpf: Any,
        size_hint: int | None = None,
        vfs_path: str = VFS_PATH,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._open_vpf = open_vpf
        self._size_hint = size_hint
        self._vfs_path = vfs_path
        # mtime is bumped on `reset_counters` (and at startup) so DuckDB's
        # parquet metadata cache, keyed on (path, mtime), invalidates between
        # measured queries — otherwise a second query within the same second
        # would silently reuse cached footer state.
        self._mtime = time.time()
        self._counters_lock = threading.Lock()
        self._size_hint_lock = threading.Lock()
        self.counters: dict[str, int] = {"bytes": 0, "reads": 0}

    def reset_counters(self) -> None:
        # Mutate in place so any in-flight `_CountingFile` instances (which
        # captured the dict reference at construction) keep writing to the
        # same dict the caller observes.
        with self._counters_lock:
            self.counters["bytes"] = 0
            self.counters["reads"] = 0
        self._mtime = time.time()

    @classmethod
    def _strip_protocol(cls, path: str) -> str:
        # Mirror the base classmethod signature: fsspec internals occasionally
        # call this as `cls._strip_protocol(...)` rather than `self._strip_protocol(...)`.
        stripped = path.removeprefix("vp://").rstrip("/")
        return stripped or "/"

    def info(self, path: str, **_: Any) -> dict[str, Any]:
        stripped = self._strip_protocol(path)
        if stripped != self._vfs_path:
            raise FileNotFoundError(f"{path}: VirtualParquetFS only exposes {self._vfs_path!r}")
        if self._size_hint is None:
            with self._size_hint_lock:
                if self._size_hint is None:
                    with self._open_vpf() as vpf:
                        self._size_hint = vpf.size()
        return {
            "name": stripped,
            "size": self._size_hint,
            "type": "file",
            "mtime": self._mtime,
        }

    def modified(self, path: str) -> _dt.datetime:
        _ = self.info(path)
        return _dt.datetime.fromtimestamp(self._mtime, tz=_dt.timezone.utc)

    def exists(self, path: str, **_: Any) -> bool:
        return self._strip_protocol(path) == self._vfs_path

    def ls(self, path: str, detail: bool = True, **_: Any) -> list[Any]:
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
        if self._strip_protocol(path) != self._vfs_path:
            raise FileNotFoundError(f"{path}: VirtualParquetFS only exposes {self._vfs_path!r}")
        return _CountingFile(self._open_vpf(), self.counters, self._counters_lock)  # type: ignore[return-value]


def _format_int(n: int) -> str:
    return f"{n:,}"


def _describe_plan(adapter: BamAdapter) -> int:
    total_rows = sum(adapter.row_group_plan(i).rows for i in range(adapter.row_group_count))
    print(
        f"  {adapter.row_group_count} row groups planned from BAI metadata, "
        f"{_format_int(total_rows)} reads total"
    )
    print("\nFirst / last 5 row-group declarations (rows from BAI; pos/end stats omitted):")
    indices = list(range(adapter.row_group_count))
    for i in indices[:5] + indices[-5:]:
        plan = adapter.row_group_plan(i)
        chrom_stat = plan.column_stats[0]
        if chrom_stat is None:
            raise AssertionError("chrom column was declared with exact stats")
        chrom = chrom_stat.min
        print(f"  rg {i:>3d}: chrom={chrom!r:>30s}  rows={_format_int(plan.rows):>12s}")
    return total_rows


def _run_measured_query(con: duckdb.DuckDBPyConnection, vfs: VirtualParquetFS) -> tuple[Any, float]:
    src = f"read_parquet('{VFS_URL}')"
    vfs.reset_counters()
    t = time.time()
    df = con.sql(QUERY.format(src=src, slop=SLOP)).df()
    elapsed = time.time() - t
    return df, elapsed


def _report_io(vfs: VirtualParquetFS, size: int) -> None:
    bytes_read = vfs.counters["bytes"]
    reads = vfs.counters["reads"]
    fraction = bytes_read / size * 100
    print(
        f"\nByte-range reads from the virtual parquet: {reads} requests, "
        f"{_format_int(bytes_read)} bytes ({fraction:.2f}% of the "
        f"{_format_int(size)}-byte file). Row-group pruning at work — "
        "DuckDB only pulled the column chunks for the chr22 row groups."
    )


def _explain_analyze(vfs: VirtualParquetFS) -> None:
    # Fresh connection on a fresh vfs so EXPLAIN ANALYZE's re-execution does
    # not pollute the byte counters the measured query reported above.
    explain_vfs = VirtualParquetFS(
        open_vpf=vfs._open_vpf,
        size_hint=vfs._size_hint,
    )
    fsspec.register_implementation("vp", lambda **_: explain_vfs, clobber=True)
    explain_vfs.reset_counters()
    src = f"read_parquet('{VFS_URL}')"
    print("\nDuckDB EXPLAIN ANALYZE:\n")
    with duckdb.connect() as ec:
        ec.register_filesystem(explain_vfs)
        plan = ec.sql(("EXPLAIN ANALYZE " + QUERY).format(src=src, slop=SLOP)).fetchall()
    # DuckDB returns a single row `('analyzed_plan', '<multi-line plan>')`; print
    # only the plan body to avoid emitting the literal "analyzed_plan" header.
    if plan:
        last_cell = plan[-1][-1]
        print(last_cell)


def main() -> None:
    print(f"Indexing BAM via BAI alone (no BAM data read yet): {Path(BAM_URL).name}", flush=True)
    t = time.time()
    try:
        adapter = BamAdapter(BAM_URL, index_url=BAI_URL)
    except Exception as e:
        print(f"\nCould not initialize BamAdapter from {BAM_URL!r}: {e}", file=sys.stderr)
        print("Set VP_EXAMPLE_BAM_URL / VP_EXAMPLE_BAI_URL to a reachable BAM.", file=sys.stderr)
        sys.exit(1)
    print(f"  (indexing took {time.time() - t:.2f}s)")

    _describe_plan(adapter)

    with vp.open(adapter) as vpf:
        size = vpf.size()
        pf = pq.ParquetFile(vpf)
        print(f"\nVirtual Parquet size: {_format_int(size)} bytes")
        print(
            f"PyArrow footer view: {_format_int(pf.metadata.num_rows)} rows in "
            f"{pf.metadata.num_row_groups} row groups; "
            f"format_version={pf.metadata.format_version}"
        )

    vfs = VirtualParquetFS(open_vpf=lambda: vp.open(adapter), size_hint=size)
    fsspec.register_implementation("vp", lambda **_: vfs, clobber=True)

    print("\nRunning the chr22 self-intersection...", flush=True)
    with duckdb.connect() as con:
        con.register_filesystem(vfs)
        df, elapsed = _run_measured_query(con, vfs)
    print(f"  {_format_int(len(df))} overlapping read pairs in {elapsed:.2f}s")
    print("\nFirst 5 pairs:")
    print(df.head(5).to_string(index=False))

    _report_io(vfs, size)
    _explain_analyze(vfs)


if __name__ == "__main__":
    main()
