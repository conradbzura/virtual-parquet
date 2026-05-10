"""anyio sync/async adapter facade.

The Rust byte-server consumes a sync adapter interface. When the user supplies an
``AsyncAdapter`` (whose ``fetch`` and ``row_group_plan`` methods are coroutines),
this facade wraps each async call as a synchronous shim that drives the coroutine
through an ``anyio.from_thread.start_blocking_portal()``. The Rust side is agnostic
to whether the underlying adapter was sync or async.

Lifetime: when the underlying adapter is async, the portal is started eagerly during
construction so failures surface at ``open()`` rather than on the first ``read()`` —
and stopped on ``close()``. Sync adapters never start a portal.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, cast

import anyio.from_thread

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from anyio.from_thread import BlockingPortal

    from virtual_parquet._types import RowGroupPlan, Schema
    from virtual_parquet.adapter import Adapter, ArrowBatchLike, AsyncAdapter


__all__ = ["_SyncAdapterFacade"]


class _SyncAdapterFacade:
    """Adapt either an Adapter or AsyncAdapter to a uniform sync interface.

    Owns the anyio blocking portal when the underlying adapter is async. The
    Rust binding consumes this facade and never sees the underlying async shape.
    """

    def __init__(self, adapter: Adapter | AsyncAdapter) -> None:
        self._adapter = adapter
        self._is_async = self._detect_async(adapter)
        self._portal_cm: Any = None
        self._portal: BlockingPortal | None = None
        if self._is_async:
            cm = anyio.from_thread.start_blocking_portal()
            # Only assign _portal_cm after __enter__ succeeds so a failed start
            # does not leave a never-entered context manager that close() would
            # later try to exit.
            self._portal = cm.__enter__()
            self._portal_cm = cm

    @staticmethod
    def _detect_async(adapter: Adapter | AsyncAdapter) -> bool:
        fetch = getattr(adapter, "fetch", None)
        plan = getattr(adapter, "row_group_plan", None)
        is_async_fetch = inspect.iscoroutinefunction(fetch)
        is_async_plan = inspect.iscoroutinefunction(plan)
        if is_async_fetch != is_async_plan:
            raise TypeError(
                "adapter is inconsistently async: fetch and row_group_plan must both "
                "be sync or both be async"
            )
        return is_async_fetch

    @property
    def schema(self) -> Schema:
        return self._adapter.schema

    @property
    def row_group_count(self) -> int:
        # No defensive int() coercion: the Rust binding rejects non-int values
        # (including bool, which subclasses int) at extraction time per the
        # adapter contract.
        return self._adapter.row_group_count

    def row_group_plan(self, index: int) -> RowGroupPlan:
        if self._portal is None:
            return cast("RowGroupPlan", self._adapter.row_group_plan(index))
        async_plan = cast(
            "Callable[[int], Awaitable[RowGroupPlan]]", self._adapter.row_group_plan
        )
        return self._portal.call(self._call_async, async_plan, index)

    def fetch(self, index: int) -> ArrowBatchLike:
        if self._portal is None:
            return self._adapter.fetch(index)
        async_fetch = cast(
            "Callable[[int], Awaitable[ArrowBatchLike]]", self._adapter.fetch
        )
        return self._portal.call(self._call_async, async_fetch, index)

    @staticmethod
    async def _call_async(
        coro_fn: Callable[..., Awaitable[Any]], *args: Any
    ) -> Any:
        return await coro_fn(*args)

    def close(self) -> None:
        """Tear down the anyio portal if one was started. Idempotent."""
        if self._portal_cm is not None:
            self._portal_cm.__exit__(None, None, None)
            self._portal = None
            self._portal_cm = None
