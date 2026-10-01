from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"
CORE_PATH = SCRIPTS / "check_stage4_discop_core_conformance.py"
AUTHOR_PATH = SCRIPTS / "run_stage4_discop_author_cython_conformance.py"
PROBE_PATH = SCRIPTS / "probe_stage4_discop_characteristic.py"
CONFIG_PATH = REPO_ROOT / "configs" / "reproducibility" / "stage4_discop_conformance.json"
EXPECTED_COMMIT = "3c3a10099a242eae405b49cc4d09fba1abb148ad"


def _load(path: Path, name: str):
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_discop_source_oracle_matches_normalized_suite() -> None:
    module = _load(CORE_PATH, "stage4_discop_core_check")
    result = module.evaluate(random_cases=32)
    assert result["status"] == "pass"
    assert result["failure_count"] == 0
    assert result["fixture_count"] == 37
    assert result["reference_commit"] == EXPECTED_COMMIT
    assert "does not execute" in result["limitation"]


def test_discop_conformance_config_freezes_table_ii_targets() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert payload["reference_commit"] == EXPECTED_COMMIT
    assert payload["paper"]["top_p"] == [0.8, 0.92, 0.95, 0.98, 1.0]
    assert payload["paper"]["reported_discop"]["utilization"] == [0.92, 0.94, 0.94, 0.95, 0.95]
    assert payload["paper"]["reported_discop"]["average_kld_bits_per_token"] == [0, 0, 0, 0, 0]


def test_author_runner_requires_pinned_checkout_and_original_cython_api() -> None:
    source = AUTHOR_PATH.read_text(encoding="utf-8")
    assert EXPECTED_COMMIT in source or "REFERENCE_COMMIT" in source
    assert 'import_module("stega_cy")' in source
    assert "stega.encode_step" in source
    assert "stega.decode_step" in source
    assert '"diff", "--name-only"' in source
    assert "sys.dont_write_bytecode = True" in source


def test_characteristic_probe_is_explicitly_not_table_ii_numerical_reproduction() -> None:
    source = PROBE_PATH.read_text(encoding="utf-8")
    assert "normalized-characteristic-not-table-ii-numerical-reproduction" in source
    assert "0.80" in source and "1.00" in source
    assert "paper_table_ii_gpt2_reference" in source
