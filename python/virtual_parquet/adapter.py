"""Public adapter contract: ``Adapter`` / ``AsyncAdapter`` Protocols and base classes.

Adapter authors implement either ``Adapter`` (sync) or ``AsyncAdapter`` (async). Both
Protocols are runtime-checkable, so duck typing works alongside explicit subclassing.

Per Constitution Principle III, this contract is the product. Breaking changes here are
MAJOR-version events.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from virtual_parquet._types import RowGroupPlan, Schema


@runtime_checkable
class Adapter(Protocol):
    """Synchronous adapter contract.

    Calls happen synchronously from the byte-server's serving thread. Thread identity
    is NOT part of the contract; adapters MUST NOT depend on a specific calling thread.
    """

    @property
    def schema(self) -> Schema:
        """Schema of the data this adapter produces. Stable for the adapter's lifetime."""
        ...

    @property
    def row_group_count(self) -> int:
        """Number of row groups this adapter will yield. May be 0."""
        ...

    def row_group_plan(self, index: int) -> RowGroupPlan:
        """Plan for row group ``index`` (0-based, in ``[0, row_group_count)``).

        Called once per row group during the metadata pre-pass. May be called more than
        once for the same index; results MUST be consistent.
        """
        ...

    def fetch(self, index: int) -> object:
        """Row group data as an Arrow PyCapsule (any object exposing
        ``__arrow_c_array__`` or ``__arrow_c_stream__``).

        MUST conform to the declared schema and contain exactly
        ``row_group_plan(index).rows`` rows.
        """
        ...


@runtime_checkable
class AsyncAdapter(Protocol):
    """Asynchronous adapter contract.

    ``schema`` and ``row_group_count`` remain sync properties; only the data-producing
    methods are ``async``. Bridging to the sync byte-server happens via an anyio
    blocking portal at the binding boundary.
    """

    @property
    def schema(self) -> Schema: ...

    @property
    def row_group_count(self) -> int: ...

    async def row_group_plan(self, index: int) -> RowGroupPlan: ...

    async def fetch(self, index: int) -> object: ...


class BaseAdapter:
    """Convenience base for synchronous adapters.

    Stores ``schema`` and ``row_group_count`` as plain attributes; subclass and
    implement ``row_group_plan`` and ``fetch``.
    """

    schema: Schema
    row_group_count: int

    def row_group_plan(self, index: int) -> RowGroupPlan:
        raise NotImplementedError

    def fetch(self, index: int) -> object:
        raise NotImplementedError


class BaseAsyncAdapter:
    """Convenience base for asynchronous adapters."""

    schema: Schema
    row_group_count: int

    async def row_group_plan(self, index: int) -> RowGroupPlan:
        raise NotImplementedError

    async def fetch(self, index: int) -> object:
        raise NotImplementedError
