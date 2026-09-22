from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "reproducibility" / "gpt2_medium_pilot.json"
MATRIX_PATH = REPO_ROOT / "configs" / "reproducibility" / "paper_reproduction_matrix.json"
MANIFEST_PATH = REPO_ROOT / "results" / "stage3" / "paper_reproduction" / "cnndm_context_manifest.json"
RUNNER_PATH = REPO_ROOT / "scripts" / "run_stage3_gpt2_medium_pilot.py"
CHECKER_PATH = REPO_ROOT / "scripts" / "check_stage3_gpt2_medium_pilot.py"
EXPECTED_COMMIT = "14e982564aeaf9a33f7b4de440deda2184d17f12"
EXPECTED_MATRIX_SHA256 = "67b376590c7bb5a61dfb7565fd4699567880978c0a5b5bc4d741dafde5ff200b"
EXPECTED_MANIFEST_SHA256 = "cc25b492b26e102496137bc2f8eb78320c7f7768d83ff7e10af9b2e04a14aebc"


def _config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("run_stage3_gpt2_medium_pilot", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # The runner imports a sibling script when executed normally.
    import sys
    sys.path.insert(0, str(RUNNER_PATH.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(RUNNER_PATH.parent))
    return module


def test_gpt2_medium_pilot_config_is_frozen_and_author_compatible() -> None:
    config = _config()
    assert config["schema_version"] == "stage3.gpt2_medium_pilot.v1"
    assert config["status"] == "frozen_before_run"
    assert config["reference_commit"] == EXPECTED_COMMIT
    assert config["model_id"] == "gpt2-medium"
    assert config["seed"] == 1234
    assert config["pilot_context_count"] == 8
    assert config["finish_sent"] is True
    assert config["payload"]["bit_count"] == 24
    assert config["execution"]["replicates"] == 1
    assert config["execution"]["expected_run_count"] == 32


def test_pilot_uses_predeclared_representative_points() -> None:
    points = {point["id"]: point for point in _config()["pilot_points"]}
    assert set(points) == {
        "bins_b3",
        "huffman_e3",
        "arithmetic_t0.9_k300",
        "arithmetic_t1.0_k50256",
    }
    assert points["bins_b3"]["block_size_bits"] == 3
    assert points["huffman_e3"]["candidate_pool_exponent"] == 3
    assert points["arithmetic_t0.9_k300"]["temperature"] == 0.9
    assert points["arithmetic_t0.9_k300"]["topk"] == 300
    assert points["arithmetic_t1.0_k50256"]["temperature"] == 1.0
    assert points["arithmetic_t1.0_k50256"]["topk"] == 50256


def test_frozen_matrix_and_context_manifest_are_not_redefined_by_pilot() -> None:
    config = _config()
    assert _sha256(MATRIX_PATH) == EXPECTED_MATRIX_SHA256
    assert _sha256(MANIFEST_PATH) == EXPECTED_MANIFEST_SHA256
    assert config["context_manifest_sha256"] == EXPECTED_MANIFEST_SHA256


def test_pilot_secret_stream_is_deterministic_and_context_specific() -> None:
    module = _load_runner_module()
    first = module.pilot_secret_bits(0, seed=1234, bit_count=24)
    again = module.pilot_secret_bits(0, seed=1234, bit_count=24)
    second_context = module.pilot_secret_bits(1, seed=1234, bit_count=24)
    assert first == again
    assert first != second_context
    assert len(first) == 24
    assert set(first) <= {0, 1}


def test_runner_separates_legacy_and_native_arithmetic_paths() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "force_reference_slow_tokenizer(utils)" in source
    assert "LegacyCausalLMAdapter(raw_model)" in source
    assert "block_baseline.encode_block" in source
    assert "huffman_baseline.encode_huffman" in source
    assert "arithmetic.encode_arithmetic" in source
    assert "Loading raw reference GPT-2 Medium for Arithmetic" in source
    assert "finish_sent=True" in source


def test_runner_records_author_metrics_and_full_sentence_diagnostics_separately() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "bits_per_word_author_stats_prefix" in source
    assert "useful_payload_bits_per_total_generated_token" in source
    assert "final_token_finishes_sentence" in source
    assert "has_early_sentence_finish" in source
    assert "not a final Figure-3 reproduction measurement" in source


def test_runner_requires_exact_payload_prefix_but_not_exact_token_roundtrip() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "exact_payload_prefix_recovery" in source
    assert "text_token_roundtrip_exact" in source
    assert "payload_ok" in source
    assert "record_status_ok" in source
    # Roundtrip is diagnostic because pinned reference decoders include BPE repair.
    assert "text_token_roundtrip_exact and" not in source


def test_pilot_checker_requires_full_four_by_eight_cartesian_product() -> None:
    source = CHECKER_PATH.read_text(encoding="utf-8")
    assert "len(records) == 32" in source
    assert "for rank in range(8)" in source
    assert "point_context_cartesian_product" in source
    assert "Stage 3 GPT-2 Medium pilot gate: READY" in source
