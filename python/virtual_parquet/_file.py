"""``VirtualParquetFile``: the file-protocol wrapper engines consume.

The native ``_native.VirtualFile`` class implements the byte-serving core; this Python
wrapper adds the standard file protocol method names PyArrow / Polars / DuckDB use
when duck-typing file-like inputs, plus lifecycle management for the anyio bridge
when the underlying adapter is async.
"""

from __future__ import annotations

import contextlib
import io
from types import TracebackType
from typing import TYPE_CHECKING

from virtual_parquet._native import VirtualFile as _NativeVirtualFile

if TYPE_CHECKING:
    from virtual_parquet._bridge import _SyncAdapterFacade


_CLOSED_MSG = "I/O operation on closed file"


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
        """Read up to ``n`` bytes from the current position. ``n=-1`` reads to EOF."""
        if self._closed:
            raise ValueError(_CLOSED_MSG)
        return self._native.read(n)

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        """Reposition the cursor; ``whence`` is one of :data:`io.SEEK_SET`/``CUR``/``END``."""
        if self._closed:
            raise ValueError(_CLOSED_MSG)
        return self._native.seek(offset, whence)

    def tell(self) -> int:
        """Return the current cursor position."""
        if self._closed:
            raise ValueError(_CLOSED_MSG)
        return self._native.tell()

    def size(self) -> int:
        """Return the total size of the virtual Parquet object in bytes.

        Triggers the metadata pre-pass on first call. Engines occasionally use this
        as a fast-path before issuing range reads.
        """
        if self._closed:
            raise ValueError(_CLOSED_MSG)
        return self._native.size()

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
        """Release native resources and the anyio portal (if started for an async adapter).

        Idempotent. ``self._closed`` is set before the bridge is torn down so that a
        portal-teardown failure still leaves subsequent I/O calls correctly raising
        :class:`ValueError`.
        """
        if self._closed:
            return
        try:
            self._native.close()
        finally:
            # Set the flag before bridge teardown so a portal-shutdown failure does
            # not leave the wrapper in a half-closed state where read/seek/tell
            # silently succeed on a closed native.
            self._closed = True
            self._bridge.close()

    def __enter__(self) -> VirtualParquetFile:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        self.close()
        return False

    def __del__(self) -> None:
        # Best-effort cleanup; finalizer-time errors are unraisable warnings anyway.
        with contextlib.suppress(Exception):
            self.close()
