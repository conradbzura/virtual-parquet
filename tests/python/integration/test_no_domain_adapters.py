"""Integration: scaffolding-only public surface.

Conformance check 2 from ``contracts/public-api.md`` (FR-010 / SC-006) — the public
namespace MUST contain zero domain-specific adapters. Adapter-author scaffolding
(Protocols, base classes, dataclasses, the entry point, errors) is acceptable;
anything whose name suggests a specific data-source format is not.
"""

from __future__ import annotations

import virtual_parquet as vp

DOMAIN_TOKENS: tuple[str, ...] = (
    "csv",
    "json",
    "http",
    "sql",
    "kafka",
    "iceberg",
    "parquet",
    "hdf5",
    "avro",
    "orc",
    "delta",
    "arrow_flight",
    "s3",
    "gcs",
    "azure",
    "minio",
    "snowflake",
    "bigquery",
    "redshift",
    "postgres",
    "mysql",
    "sqlite",
    "duckdb",
    "polars",
    "pandas",
    "spark",
    "elasticsearch",
    "mongodb",
    "redis",
)


def test_no_symbol_in_all_suggests_a_specific_data_source() -> None:
    """
    GIVEN the published __all__ surface
    WHEN each exported name is checked for tokens identifying a specific data-source format
    THEN no exported name contains any such token (case-insensitive).
    """
    # The project's own namespace ("VirtualParquet*") legitimately contains "parquet"
    # in its structural type names. The check is for *adapter* names that suggest a
    # data-source format (e.g., "ParquetAdapter", "IcebergReader"); strip the project
    # prefix before scanning so the project's own types are not flagged.
    offenders: list[tuple[str, str]] = []
    for symbol in vp.__all__:
        scanned = symbol
        if scanned.startswith("VirtualParquet"):
            scanned = scanned[len("VirtualParquet") :]
        lowered = scanned.lower()
        for token in DOMAIN_TOKENS:
            if token in lowered:
                offenders.append((symbol, token))

    assert offenders == [], (
        "domain-specific symbols leaked into the public surface — virtual-parquet is "
        f"scaffolding, not adapters: {offenders}"
    )
