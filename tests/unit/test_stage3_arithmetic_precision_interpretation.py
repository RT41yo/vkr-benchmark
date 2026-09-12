from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT = REPO_ROOT / "results" / "stage3" / "paper_reproduction" / "arithmetic_precision_probe" / "result.json"
INTERPRETATION = REPO_ROOT / "results" / "stage3" / "paper_reproduction" / "arithmetic_precision_probe" / "interpretation.json"
ANALYZER = REPO_ROOT / "scripts" / "analyze_stage3_arithmetic_precision_result.py"
RUNNER = REPO_ROOT / "scripts" / "run_stage3_arithmetic_precision_probe.py"
CHECKER = REPO_ROOT / "scripts" / "check_stage3_arithmetic_precision_probe.py"
EXPECTED_RESULT_SHA = "01bd84b283b9a685e4251d54259c2b0997e55531ea9c710678e835dfd3f90536"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_analyzer():
    spec = importlib.util.spec_from_file_location("analyze_stage3_arithmetic_precision_result", ANALYZER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_completed_precision_probe_result_is_frozen() -> None:
    assert RESULT.is_file()
    assert _sha(RESULT) == EXPECTED_RESULT_SHA
    data = json.loads(RESULT.read_text(encoding="utf-8"))
    assert data["status"] == "ok"
    assert data["gate"]["mirror_parity_passed"] is True
    assert data["gate"]["reference_worktree_unchanged"] is True
    assert data["gate"]["actual_run_count"] == 32


def test_interpretation_separates_full_run_and_zero_padding_free_kl() -> None:
    analysis = json.loads(INTERPRETATION.read_text(encoding="utf-8"))
    assert analysis["source_result"]["sha256"] == EXPECTED_RESULT_SHA
    p26 = analysis["by_precision"]["26"]
    p40 = analysis["by_precision"]["40"]
    assert p26["mean_full_run_author_kl_bits"] > p26["mean_zero_padding_free_author_kl_bits"] * 10
    assert p40["mean_full_run_author_kl_bits"] > p40["mean_zero_padding_free_author_kl_bits"] * 1000
    assert p26["implicit_zero_padding_share_of_summed_step_kl"] > 0.98
    assert p40["implicit_zero_padding_share_of_summed_step_kl"] > 0.999


def test_interpretation_localizes_finite_precision_without_claiming_exact_paper_reproduction() -> None:
    analysis = json.loads(INTERPRETATION.read_text(encoding="utf-8"))
    findings = analysis["derived_findings"]
    assert findings["finite_precision_effect_confirmed"] is True
    assert findings["pilot_payload_shorter_than_precision"] is True
    assert findings["precision_26_clean_kl_still_far_above_paper_anchor"] is True
    assert findings["precision_40_clean_kl_nats_is_same_order_as_paper_anchor"] is True
    assert findings["paper_claim_status"] == "not_numerically_reproduced_at_pinned_executable_precision_26"
    assert findings["full_figure3_sweep_ready"] is False


def test_analyzer_recomputes_committed_interpretation() -> None:
    module = _load_analyzer()
    recomputed = module.analyze(
        RESULT,
        REPO_ROOT / "configs" / "reproducibility" / "paper_reproduction_matrix.json",
    )
    committed = json.loads(INTERPRETATION.read_text(encoding="utf-8"))
    assert recomputed == committed


def test_human_facing_precision_labels_are_unambiguous() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    checker = CHECKER.read_text(encoding="utf-8")
    for source in (runner, checker):
        assert "full-run mean author KL" in source
        assert "zero-padding-free author KL" in source
