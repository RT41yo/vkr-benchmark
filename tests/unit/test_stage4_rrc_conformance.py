from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"
CORE_PATH = SCRIPTS / "check_stage4_rrc_core_conformance.py"
AUTHOR_PATH = SCRIPTS / "run_stage4_rrc_author_conformance.py"
CONFIG_PATH = REPO_ROOT / "configs" / "reproducibility" / "stage4_rrc_conformance.json"
EXPECTED_COMMIT = "dae326259e4fca8bc4fcf460dafdbc0e88a0a71a"


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


def test_rrc_source_oracle_matches_normalized_suite() -> None:
    module = _load(CORE_PATH, "stage4_rrc_core_check")
    result = module.evaluate(random_cases=32)
    assert result["status"] == "pass"
    assert result["failure_count"] == 0
    assert result["fixture_count"] == 36
    assert result["reference_commit"] == EXPECTED_COMMIT
    assert "does not execute" in result["limitation"]


def test_rrc_conformance_config_freezes_current_author_revision_and_boundaries() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert payload["reference_commit"] == EXPECTED_COMMIT
    assert payload["prior_analysis_commit"].startswith("8b6a86a")
    assert payload["direct_author_conformance"]["requires_clean_pinned_checkout"] is True
    assert payload["characteristic"]["payload_bits"] == 128
    assert payload["characteristic"]["verify_termination"] is True
    assert payload["explicit_q"]["tolerances"]["tvd"] == 1e-13
    assert payload["explicit_q"]["real_model_tolerances"]["kl_bits"] == 1e-12
    assert payload["explicit_q"]["real_model_tolerances"]["tvd"] == 1e-9


def test_rrc_author_runner_executes_original_python_core_without_model() -> None:
    source = AUTHOR_PATH.read_text(encoding="utf-8")
    assert '"rev-parse", "HEAD"' in source
    assert '"diff", "--name-only"' in source
    assert 'reference_dir / "rrc_core.py"' in source
    assert "embed_message(" in source
    assert "extract_message(" in source
    assert "AuthorFixtureStepper" in source
    assert "AutoModel" not in source
