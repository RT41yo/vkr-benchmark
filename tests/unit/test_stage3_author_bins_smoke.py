from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "reproducibility" / "author_bins_gpt2_smoke.json"
RUNNER_PATH = REPO_ROOT / "scripts" / "run_stage3_author_bins_smoke.py"
EXPECTED_COMMIT = "14e982564aeaf9a33f7b4de440deda2184d17f12"


def _config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _bits2int_reference_order(bits: str) -> int:
    return sum(int(bit) * (2**i) for i, bit in enumerate(bits))


def test_author_bins_smoke_config_is_pinned_and_small() -> None:
    config = _config()
    assert config["mode"] == "author-compatible"
    assert config["method"] == "bins"
    assert config["reference_commit"] == EXPECTED_COMMIT
    assert config["model_id"] == "gpt2"
    assert config["seed"] == 1234
    assert config["block_size"] == 3
    assert config["finish_sent"] is False
    assert config["compatibility_profile"] == "hf_4_52_legacy_api"


def test_author_bins_smoke_secret_exercises_all_eight_bins_once() -> None:
    config = _config()
    secret = str(config["secret_bits"])
    block_size = int(config["block_size"])
    chunks = [secret[i : i + block_size] for i in range(0, len(secret), block_size)]
    assert len(secret) == 24
    assert len(chunks) == 8
    assert [_bits2int_reference_order(chunk) for chunk in chunks] == list(range(8))


def test_author_bins_smoke_uses_directional_kl_field_name() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "kl_q_stego_to_p_lm_bits_author" in source
    assert '"kl_bits"' not in source


def test_author_bins_smoke_prevents_reference_bytecode_writes() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "sys.dont_write_bytecode = True" in source
    assert '"status", "--porcelain"' in source
    assert '"worktree_unchanged"' in source


def test_author_bins_smoke_uses_scoped_legacy_api_compatibility() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert 'kwargs["use_fast"] = False' in source
    assert 'past_key_values=modern_past' in source
    assert 'return_dict=False' in source
    assert 'model_call_translation' in source


def test_author_bins_smoke_preserves_first_raw_runtime_failure() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert 'raw_reference_failure.json' in source
    assert '_preserve_previous_failure' in source


def _load_runner_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("stage3_bins_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_model_adapter_returns_historical_stacked_cache() -> None:
    import torch

    module = _load_runner_module()
    key = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
    value = key + 100

    class FakeModel:
        def __init__(self) -> None:
            self.kwargs = None

        def __call__(self, input_ids, **kwargs):
            self.kwargs = kwargs
            return ("logits", ((key, value),))

        def parameters(self, *args, **kwargs):
            return iter(())

    raw_model = FakeModel()
    adapter = module._LegacyCausalLMAdapter(raw_model)
    logits, cache = adapter("ids", past=None)

    assert logits == "logits"
    assert raw_model.kwargs == {
        "past_key_values": None,
        "use_cache": True,
        "return_dict": False,
    }
    assert isinstance(cache, tuple)
    assert tuple(cache[0].shape) == (2, 1, 2, 3, 4)
    assert cache[0].shape[3] == 3  # exact access used by pinned decode_block
    assert torch.equal(cache[0][0], key)
    assert torch.equal(cache[0][1], value)


def test_legacy_model_adapter_translates_historical_cache_back_before_call() -> None:
    import torch

    module = _load_runner_module()
    old_key = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
    old_value = old_key + 100
    historical_past = (torch.stack((old_key, old_value), dim=0),)
    new_key = old_key + 200
    new_value = old_value + 200

    class FakeModel:
        def __init__(self) -> None:
            self.kwargs = None

        def __call__(self, input_ids, **kwargs):
            self.kwargs = kwargs
            return ("logits", ((new_key, new_value),))

        def parameters(self, *args, **kwargs):
            return iter(())

    raw_model = FakeModel()
    adapter = module._LegacyCausalLMAdapter(raw_model)
    _, new_historical_cache = adapter("ids", past=historical_past)

    modern_past = raw_model.kwargs["past_key_values"]
    assert isinstance(modern_past, tuple)
    assert torch.equal(modern_past[0][0], old_key)
    assert torch.equal(modern_past[0][1], old_value)
    assert tuple(new_historical_cache[0].shape) == (2, 1, 2, 3, 4)
    assert torch.equal(new_historical_cache[0][0], new_key)
    assert torch.equal(new_historical_cache[0][1], new_value)


def test_author_bins_smoke_preserves_cache_shape_failure() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert 'compat_cache_shape_failure.json' in source
    assert "'tuple' object has no attribute 'shape'" in source
    assert 'raw_reference_failure.json' in source
