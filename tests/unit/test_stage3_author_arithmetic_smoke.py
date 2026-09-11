from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "reproducibility" / "author_arithmetic_gpt2_smoke.json"
BINS_CONFIG_PATH = REPO_ROOT / "configs" / "reproducibility" / "author_bins_gpt2_smoke.json"
HUFFMAN_CONFIG_PATH = REPO_ROOT / "configs" / "reproducibility" / "author_huffman_gpt2_smoke.json"
RUNNER_PATH = REPO_ROOT / "scripts" / "run_stage3_author_arithmetic_smoke.py"
EXPECTED_COMMIT = "14e982564aeaf9a33f7b4de440deda2184d17f12"


def _config(path: Path = CONFIG_PATH) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_author_arithmetic_smoke_config_is_pinned_to_reference_defaults() -> None:
    config = _config()
    assert config["mode"] == "author-compatible"
    assert config["method"] == "arithmetic"
    assert config["reference_commit"] == EXPECTED_COMMIT
    assert config["model_id"] == "gpt2"
    assert config["seed"] == 1234
    assert config["temperature"] == 0.9
    assert config["precision"] == 26
    assert config["topk"] == 300
    assert config["finish_sent"] is False
    assert config["compatibility_profile"] == "hf_4_52_arithmetic_native_dynamic_cache"


def test_author_arithmetic_smoke_reuses_bins_and_huffman_inputs() -> None:
    arithmetic = _config()
    bins = _config(BINS_CONFIG_PATH)
    huffman = _config(HUFFMAN_CONFIG_PATH)
    assert arithmetic["context"] == bins["context"] == huffman["context"]
    assert arithmetic["secret_bits"] == bins["secret_bits"] == huffman["secret_bits"]
    assert len(str(arithmetic["secret_bits"])) == 24


def test_author_arithmetic_runner_calls_pinned_reference_without_legacy_adapter() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert 'REQUIRED_REFERENCE_FILES = ("utils.py", "arithmetic.py")' in source
    assert 'importlib.import_module("arithmetic")' in source
    assert "encode_arithmetic" in source
    assert "decode_arithmetic" in source
    assert "LegacyCausalLMAdapter" not in source
    assert "force_reference_slow_tokenizer" not in source
    assert "enc, model = utils.get_model" in source


def test_author_arithmetic_runner_records_directional_author_metrics() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "kl_q_stego_to_p_lm_untempered_bits_author" in source
    assert "avg_entropy_p_tau_bits_author_helper" in source
    assert "payload_bits_per_token_smoke" in source
    assert '"kl_bits"' not in source


def test_author_arithmetic_runner_records_lookahead_and_final_flush_separately() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "author_bits_consumed" in source
    assert "implicit_zero_lookahead_bits" in source
    assert "recovered_lookahead_padding_bits" in source
    assert "decoder_flush_extra_bits" in source
    assert "decoder_flush_extra_bit_count" in source
    assert "decoder_flush_within_precision_bound" in source


def test_author_arithmetic_runner_preserves_reference_worktree() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "sys.dont_write_bytecode = True" in source
    assert '"status", "--porcelain"' in source
    assert '"worktree_unchanged"' in source
