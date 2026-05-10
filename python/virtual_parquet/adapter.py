"""Public adapter contract: ``Adapter`` / ``AsyncAdapter`` Protocols and base classes.

Adapter authors implement either ``Adapter`` (sync) or ``AsyncAdapter`` (async). Both
Protocols are runtime-checkable, so duck typing works alongside explicit subclassing.

Per Constitution Principle III, this contract is the product. Breaking changes here are
MAJOR-version events.
"""

from __future__ import annotations

from typing import Protocol, TypeAlias, runtime_checkable

from virtual_parquet._types import RowGroupPlan, Schema

# Any object exposing the Arrow C Data Interface (`__arrow_c_array__` or
# `__arrow_c_stream__`). Common producers: pyarrow.RecordBatch, pyarrow.Table,
# polars.DataFrame, nanoarrow arrays, or any user-built object exposing the
# standard Arrow PyCapsule methods. The library does not take a hard dependency
# on PyArrow; the contract is the C Data Interface.
ArrowBatchLike: TypeAlias = object


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

    def fetch(self, index: int) -> ArrowBatchLike:
        """Row group data as an Arrow C Data Interface object.

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
    def schema(self) -> Schema:
        """Schema of the data this adapter produces. Stable for the adapter's lifetime."""
        ...

    @property
    def row_group_count(self) -> int:
        """Number of row groups this adapter will yield. May be 0."""
        ...

    async def row_group_plan(self, index: int) -> RowGroupPlan:
        """Plan for row group ``index`` (0-based, in ``[0, row_group_count)``).

        Called once per row group during the metadata pre-pass. May be called more than
        once for the same index; results MUST be consistent.
        """
        ...

    async def fetch(self, index: int) -> ArrowBatchLike:
        """Row group data as an Arrow C Data Interface object.

        MUST conform to the declared schema and contain exactly
        ``row_group_plan(index).rows`` rows.
        """
        ...


class BaseAdapter:
    """Convenience base for synchronous adapters.

    Subclasses MUST set :attr:`schema` and :attr:`row_group_count` (either as class
    attributes or in their own ``__init__``) before any library call, and override
    :meth:`row_group_plan` and :meth:`fetch`.
    """

    schema: Schema
    row_group_count: int

    def __init__(self, schema: Schema | None = None, row_group_count: int | None = None) -> None:
        # Allow either constructor-arg or class-attribute initialization. Subclasses
        # that override __init__ are free to set the attributes themselves and need
        # not call super().__init__.
        if schema is not None:
            self.schema = schema
        if row_group_count is not None:
            self.row_group_count = row_group_count

    def row_group_plan(self, index: int) -> RowGroupPlan:
        """Subclasses MUST override; see :meth:`Adapter.row_group_plan`."""
        raise NotImplementedError

    def fetch(self, index: int) -> ArrowBatchLike:
        """Subclasses MUST override; see :meth:`Adapter.fetch`."""
        raise NotImplementedError


class BaseAsyncAdapter:
    """Convenience base for asynchronous adapters.

    Subclasses MUST set :attr:`schema` and :attr:`row_group_count` and override
    :meth:`row_group_plan` and :meth:`fetch`.
    """

    schema: Schema
    row_group_count: int

    def __init__(self, schema: Schema | None = None, row_group_count: int | None = None) -> None:
        if schema is not None:
            self.schema = schema
        if row_group_count is not None:
            self.row_group_count = row_group_count

    async def row_group_plan(self, index: int) -> RowGroupPlan:
        """Subclasses MUST override; see :meth:`AsyncAdapter.row_group_plan`."""
        raise NotImplementedError

    async def fetch(self, index: int) -> ArrowBatchLike:
        """Subclasses MUST override; see :meth:`AsyncAdapter.fetch`."""
        raise NotImplementedError
