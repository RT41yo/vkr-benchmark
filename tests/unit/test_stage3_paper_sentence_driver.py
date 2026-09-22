from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "reproducibility" / "paper_sentence_pilot.json"
RUNNER_PATH = REPO_ROOT / "scripts" / "run_stage3_paper_sentence_pilot.py"
CORE_PATH = REPO_ROOT / "scripts" / "stage3_paper_sentence_core.py"
CHECKER_PATH = REPO_ROOT / "scripts" / "check_stage3_paper_sentence_pilot.py"
DOC_PATH = REPO_ROOT / "docs" / "stages" / "stage3" / "paper_sentence_driver.md"
MATRIX_PATH = REPO_ROOT / "configs" / "reproducibility" / "paper_reproduction_matrix.json"
EXPECTED_MATRIX_SHA = "67b376590c7bb5a61dfb7565fd4699567880978c0a5b5bc4d741dafde5ff200b"
EXPECTED_COMMIT = "14e982564aeaf9a33f7b4de440deda2184d17f12"


def _config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_stage3_paper_sentence_pilot", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(RUNNER_PATH.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(RUNNER_PATH.parent))
    return module


def test_paper_sentence_config_is_frozen_and_first_boundary_based() -> None:
    config = _config()
    assert config["schema_version"] == "stage3.paper_sentence_pilot.v1"
    assert config["status"] == "frozen_before_run"
    assert config["reference_commit"] == EXPECTED_COMMIT
    assert config["model_id"] == "gpt2-medium"
    assert config["pilot_context_count"] == 8
    assert config["secret_stream"]["bit_count"] == 16384
    assert config["sentence_stop"]["stop_at_first_boundary"] is True
    assert config["sentence_stop"]["predicate"] == "pinned_reference_utils.is_sent_finish"
    assert config["sentence_stop"]["max_generated_tokens"] == 256
    assert config["execution"]["expected_run_count"] == 32


def test_paper_sentence_pilot_keeps_the_frozen_figure3_matrix_unchanged() -> None:
    assert _sha(MATRIX_PATH) == EXPECTED_MATRIX_SHA


def test_paper_sentence_bitstream_is_deterministic_paired_and_long() -> None:
    module = _load_runner()
    first = module.paper_sentence_bits(0, seed=1234, replicate=0, bit_count=16384)
    again = module.paper_sentence_bits(0, seed=1234, replicate=0, bit_count=16384)
    other_context = module.paper_sentence_bits(1, seed=1234, replicate=0, bit_count=16384)
    other_replicate = module.paper_sentence_bits(0, seed=1234, replicate=1, bit_count=16384)
    assert first == again
    assert first != other_context
    assert first != other_replicate
    assert len(first) == 16384
    assert set(first) <= {0, 1}


def test_pilot_uses_same_representative_four_points_as_step37() -> None:
    points = {point["id"]: point for point in _config()["pilot_points"]}
    assert set(points) == {
        "bins_b3",
        "huffman_e3",
        "arithmetic_t0.9_k300",
        "arithmetic_t1.0_k50256",
    }
    assert points["bins_b3"]["block_size_bits"] == 3
    assert points["huffman_e3"]["candidate_pool_exponent"] == 3
    assert points["arithmetic_t0.9_k300"]["precision"] == 26
    assert points["arithmetic_t1.0_k50256"]["topk"] == 50256


def test_core_stops_immediately_on_first_author_sentence_boundary() -> None:
    source = CORE_PATH.read_text(encoding="utf-8")
    assert source.count('terminal_reason = "sentence_boundary"') >= 3
    assert source.count("utils.is_sent_finish") >= 3
    assert "stop_at_first_sentence" in source
    assert "max_generated_tokens" in source
    # The measurement path must never silently pad Arithmetic look-ahead with zeros.
    assert 'if stop_at_first_sentence:\n                    terminal_reason = "bitstream_exhausted"' in source
    assert 'message_bits = message_bits + [0] * missing' in source  # parity-only legacy behavior remains explicit


def test_runner_requires_fixed_message_parity_before_first_boundary_measurement() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "block_baseline.encode_block" in source
    assert "huffman_baseline.encode_huffman" in source
    assert "arithmetic.encode_arithmetic" in source
    assert "paper-sentence Bins/Huffman mirrors failed pinned-reference parity" in source
    assert "paper-sentence Arithmetic mirror failed pinned-reference parity" in source
    assert "exact_generated_token_ids" in source
    assert "len(parity) == 4" in source


def test_runner_separates_confirmed_payload_from_arithmetic_lookahead() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "payload_bits_confirmed" in source
    assert "secret_bits_read" in source
    assert "used_implicit_zero_lookahead" in source
    assert "exact_confirmed_payload_prefix_recovery" in source
    assert "decoded_extra_bit_count_after_confirmed_payload" in source


def test_checker_blocks_full_runner_until_sentence_pilot_is_clean() -> None:
    source = CHECKER_PATH.read_text(encoding="utf-8")
    assert "first_boundary_stop" in source
    assert "no_zero_padding" in source
    assert "no_stream_exhaustion" in source
    assert "no_safety_cap" in source
    assert "READY FOR FULL FIGURE-3 RUNNER IMPLEMENTATION" in source


def test_documentation_explicitly_calls_rule_an_operationalization_not_recovered_history() -> None:
    source = DOC_PATH.read_text(encoding="utf-8")
    assert "Stage-3 operationalization" in source
    assert "Оригинального Figure-3 batch/MC driver" in source
    assert "16384" in source
    assert "первой boundary" in source
    assert "Нули автоматически не дописываются" in source
