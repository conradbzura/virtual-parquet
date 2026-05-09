"""``VirtualParquetFile``: the file-protocol wrapper engines consume.

The native ``_native.VirtualFile`` class implements the byte-serving core; this Python
wrapper adds the standard file protocol method names PyArrow / Polars / DuckDB use
when duck-typing file-like inputs, plus lifecycle management for the anyio bridge
when the underlying adapter is async.
"""

from __future__ import annotations

import contextlib
import io
from typing import TYPE_CHECKING

from virtual_parquet._native import VirtualFile as _NativeVirtualFile

if TYPE_CHECKING:
    from virtual_parquet._bridge import _SyncAdapterFacade


class VirtualParquetFile:
    """A Python file-like object presenting the adapter's data as a Parquet byte stream.

    Engines call this object's ``read``, ``seek``, ``tell``, ``seekable``, and
    ``close`` methods as if it were a regular Parquet file on disk. The underlying
    bytes are produced on demand by the Rust byte-server.
    """

    def __init__(self, native: _NativeVirtualFile, bridge: _SyncAdapterFacade) -> None:
        # `bridge` owns the anyio portal (if any) and is closed when this file is closed.
        self._native = native
        self._bridge = bridge
        self._closed = False

    def read(self, n: int = -1) -> bytes:
        if self._closed:
            raise ValueError("I/O operation on closed file")
        return self._native.read(n)

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if self._closed:
            raise ValueError("I/O operation on closed file")
        return self._native.seek(offset, whence)

    def tell(self) -> int:
        if self._closed:
            raise ValueError("I/O operation on closed file")
        return self._native.tell()

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return not self._closed

    def writable(self) -> bool:
        return False

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._native.close()
        finally:
            self._bridge.close()
            self._closed = True

    def __enter__(self) -> VirtualParquetFile:
        return self

    def __exit__(
        self,
        _exc_type: object,
        _exc_value: object,
        _traceback: object,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        # Best-effort cleanup; finalizer-time errors are unraisable warnings anyway.
        with contextlib.suppress(Exception):
            self.close()
