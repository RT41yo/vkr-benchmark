from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "reproducibility" / "arithmetic_precision_probe.json"
RUNNER_PATH = REPO_ROOT / "scripts" / "run_stage3_arithmetic_precision_probe.py"
CHECKER_PATH = REPO_ROOT / "scripts" / "check_stage3_arithmetic_precision_probe.py"
AUDIT_PATH = REPO_ROOT / "scripts" / "analyze_stage3_pilot_sentence_shape.py"
PILOT_RESULT_PATH = REPO_ROOT / "results" / "stage3" / "paper_reproduction" / "gpt2_medium_pilot" / "result.json"
MATRIX_PATH = REPO_ROOT / "configs" / "reproducibility" / "paper_reproduction_matrix.json"
EXPECTED_PILOT_SHA = "ee669fde58f880cca961baebca88e8c5b383bd2716c6775afb13b5a2d475c47d"
EXPECTED_MATRIX_SHA = "67b376590c7bb5a61dfb7565fd4699567880978c0a5b5bc4d741dafde5ff200b"


def _config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_stage3_arithmetic_precision_probe", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_precision_probe_config_is_predeclared_and_diagnostic_only() -> None:
    config = _config()
    assert config["schema_version"] == "stage3.arithmetic_precision_probe.v1"
    assert config["status"] == "frozen_before_run"
    assert config["reference_commit"] == "14e982564aeaf9a33f7b4de440deda2184d17f12"
    assert config["historical_reference_commit"] == "05a3944f2bc71eb252ef600c5b80a15f98fea425"
    assert config["model_id"] == "gpt2-medium"
    assert config["pilot_context_count"] == 8
    assert config["arithmetic_point"]["temperature"] == 1.0
    assert config["arithmetic_point"]["topk"] == 50256
    assert config["arithmetic_point"]["precision_values"] == [26, 32, 40, 48]
    assert config["arithmetic_point"]["finish_sent"] is False
    assert config["payload"]["bit_count"] == 256
    assert config["execution"]["expected_run_count"] == 32
    assert config["interpretation_policy"]["full_sweep_remains_blocked_until_review"] is True


def test_probe_pins_exact_pilot_result_and_does_not_modify_frozen_matrix() -> None:
    config = _config()
    assert _sha(PILOT_RESULT_PATH) == EXPECTED_PILOT_SHA
    assert config["pilot_result_sha256"] == EXPECTED_PILOT_SHA
    assert _sha(MATRIX_PATH) == EXPECTED_MATRIX_SHA


def test_probe_bitstream_is_deterministic_context_specific_and_long() -> None:
    module = _load_runner()
    first = module.probe_bits(0, seed=1234, bit_count=256)
    again = module.probe_bits(0, seed=1234, bit_count=256)
    other = module.probe_bits(1, seed=1234, bit_count=256)
    assert first == again
    assert first != other
    assert len(first) == 256
    assert set(first) <= {0, 1}


def test_probe_instruments_exact_author_fill_and_counterfactuals_without_using_them_for_selection() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "cum_author = cum_pre + residual" in source
    assert "cum_terminal[-1] += residual" in source
    assert "q_trunc = probs_temp[:k_after]" in source
    assert "selection = int((cum_abs > message_idx).nonzero()[0].item())" in source
    assert "terminal_fill_counterfactual_kl_bits" in source
    assert "truncation_only_kl_bits" in source
    assert "rounding_residual_fraction" in source
    assert "implicit_zero_lookahead_bits" in source


def test_probe_requires_sentinel_parity_before_diagnostic_sweep() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "arithmetic.encode_arithmetic" in source
    assert "exact_generated_token_ids" in source
    assert "instrumented arithmetic mirror failed pinned-reference parity check" in source
    assert "full_sweep_remains_blocked_until_review" in source


def test_checker_does_not_require_monotonic_kl_or_paper_matching() -> None:
    source = CHECKER_PATH.read_text(encoding="utf-8")
    assert "mirror parity" in source
    assert "full Figure-3 sweep remains blocked pending interpretation" in source
    assert "monotonic" not in source.lower()
    assert "4e-8" not in source


def test_sentence_shape_audit_is_derived_from_committed_pilot_and_requires_exact_sha() -> None:
    source = AUDIT_PATH.read_text(encoding="utf-8")
    assert EXPECTED_PILOT_SHA in source
    assert "early_sentence_finish_run_count" in source
    assert "first_sentence_finish_position_zero_based" in source
    assert "future_paper_driver_requirement" in source
    assert "sufficiently long uniform bitstream" in source


def test_committed_sentence_shape_analysis_records_the_five_observed_early_boundaries() -> None:
    analysis_path = REPO_ROOT / "results" / "stage3" / "paper_reproduction" / "gpt2_medium_pilot" / "sentence_shape_analysis.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    assert analysis["source_result_sha256"] == EXPECTED_PILOT_SHA
    assert analysis["total_run_count"] == 32
    assert analysis["single_sentence_shape_count"] == 27
    assert analysis["early_sentence_finish_run_count"] == 5
    assert analysis["early_sentence_finish_count_by_point"] == {
        "arithmetic_t0.9_k300": 2,
        "arithmetic_t1.0_k50256": 2,
        "huffman_e3": 1,
    }
