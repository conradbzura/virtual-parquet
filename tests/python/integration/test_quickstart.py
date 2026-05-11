"""Integration: end-to-end smoke test of the documented quickstart.

Imports the ``examples/fixed_arrow_batch/adapter.py`` reference adapter (kept outside
the published surface per FR-010 / SC-006) and runs the same flow shown in
``specs/001-core-adapter-library/quickstart.md`` §"A minimal synchronous adapter" /
§"Read it back through PyArrow". The adapter file is loaded with importlib so the
test does not rely on ``examples/`` being importable as a package.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

import virtual_parquet as vp

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXAMPLE_PATH = _REPO_ROOT / "examples" / "fixed_arrow_batch" / "adapter.py"


@pytest.fixture(scope="module")
def fixed_batch_adapter_cls() -> type:
    if not _EXAMPLE_PATH.is_file():
        pytest.fail(f"example adapter missing at {_EXAMPLE_PATH}")

    spec = importlib.util.spec_from_file_location(
        "vp_examples_fixed_arrow_batch_adapter", _EXAMPLE_PATH
    )
    if spec is None or spec.loader is None:
        pytest.fail(f"could not load spec for {_EXAMPLE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.FixedBatchAdapter


def test_quickstart_flow_returns_documented_table(fixed_batch_adapter_cls: type) -> None:
    """
    GIVEN the FixedBatchAdapter from the quickstart example
    WHEN the quickstart's read flow runs (vp.open + pq.read_table)
    THEN the returned Table matches the documented expected output exactly.
    """
    adapter = fixed_batch_adapter_cls()
    with vp.open(adapter) as vpf:
        table = pq.read_table(vpf)

    assert table.num_rows == 4
    assert table.column_names == ["id", "label"]
    assert table.column("id").to_pylist() == [1, 2, 3, 4]
    assert table.column("label").to_pylist() == ["alpha", "beta", None, "delta"]
