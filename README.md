# virtual-parquet

Scaffolding for serving non-Parquet data sources to mainstream Parquet-consuming engines via a stable adapter contract.

The library is the framework adapters are built on; it ships zero domain-specific adapters in its public namespace. Adapter authors declare a schema, yield Arrow record batches, and optionally declare row-group statistics. The library produces engine-readable Parquet bytes on demand.

## Status

Pre-alpha. The current development context lives at:

- `.specify/memory/constitution.md` — project principles
- `specs/001-core-adapter-library/` — the v1 feature specification, plan, contracts, and task list

## License

MIT
