from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "reproducibility" / "author_huffman_gpt2_smoke.json"
BINS_CONFIG_PATH = REPO_ROOT / "configs" / "reproducibility" / "author_bins_gpt2_smoke.json"
RUNNER_PATH = REPO_ROOT / "scripts" / "run_stage3_author_huffman_smoke.py"
RUNTIME_PATH = REPO_ROOT / "scripts" / "stage3_author_runtime.py"
EXPECTED_COMMIT = "14e982564aeaf9a33f7b4de440deda2184d17f12"


def _config(path: Path = CONFIG_PATH) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_author_huffman_smoke_config_is_pinned_and_small() -> None:
    config = _config()
    assert config["mode"] == "author-compatible"
    assert config["method"] == "huffman"
    assert config["reference_commit"] == EXPECTED_COMMIT
    assert config["model_id"] == "gpt2"
    assert config["seed"] == 1234
    assert config["bits_per_word"] == 3
    assert 2 ** int(config["bits_per_word"]) == 8
    assert config["finish_sent"] is False
    assert config["compatibility_profile"] == "hf_4_52_legacy_api"


def test_author_huffman_smoke_reuses_bins_context_and_payload() -> None:
    huffman = _config()
    bins = _config(BINS_CONFIG_PATH)
    assert huffman["context"] == bins["context"]
    assert huffman["secret_bits"] == bins["secret_bits"]
    assert len(str(huffman["secret_bits"])) == 24


def test_author_huffman_runner_uses_pinned_reference_and_directional_kl() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert 'REQUIRED_REFERENCE_FILES = ("utils.py", "huffman.py", "huffman_baseline.py")' in source
    assert 'importlib.import_module("huffman_baseline")' in source
    assert "encode_huffman" in source
    assert "decode_huffman" in source
    assert "kl_q_stego_to_p_lm_bits_author" in source
    assert '"kl_bits"' not in source


def test_author_huffman_runner_records_variable_rate_end_padding() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "author_bits_consumed" in source
    assert "implicit_zero_padding_bits" in source
    assert "recovered_extra_bits" in source
    assert "trailing_padding_is_zero" in source
    assert "payload_bits_per_token_smoke" in source


def test_author_huffman_runner_preserves_reference_worktree() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "sys.dont_write_bytecode = True" in source
    assert '"status", "--porcelain"' in source
    assert '"worktree_unchanged"' in source


def _load_runtime_module():
    spec = importlib.util.spec_from_file_location("stage3_author_runtime", RUNTIME_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shared_legacy_runtime_translates_cache_without_changing_values() -> None:
    import torch

    module = _load_runtime_module()
    key = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
    value = key + 100
    historical = (torch.stack((key, value), dim=0),)

    modern = module.LegacyCausalLMAdapter.legacy_cache_to_modern(historical)
    restored = module.LegacyCausalLMAdapter.modern_cache_to_legacy(modern)

    assert isinstance(modern, tuple)
    assert torch.equal(modern[0][0], key)
    assert torch.equal(modern[0][1], value)
    assert tuple(restored[0].shape) == (2, 1, 2, 3, 4)
    assert torch.equal(restored[0][0], key)
    assert torch.equal(restored[0][1], value)
