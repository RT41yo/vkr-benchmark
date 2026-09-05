from __future__ import annotations

import json
from pathlib import Path

import pytest

from vkr_benchmark.config import (
    ExperimentConfig,
    GenerationRunConfig,
    MethodRunConfig,
    TerminationConfig,
)
from vkr_benchmark.errors import ConfigurationError


def _raw_config() -> dict:
    return {
        "benchmark_version": "0.1",
        "run_kind": "normalized",
        "model_config": "configs/models/test.json",
        "prompt_id": "p000001",
        "method": {
            "id": "bins",
            "implementation_revision": "test",
            "params": {"block_size": 2},
            "random_seed": 12345,
        },
        "generation": {
            "temperature": 1.0,
            "top_k": None,
            "top_p": 1.0,
            "exclude_special_tokens": True,
            "kv_cache": True,
        },
        "secret_id": "000001",
        "termination": {
            "mode": "fixed_carrier_tokens",
            "target_carrier_tokens": 16,
        },
    }


def test_experiment_config_loads_and_resolves_model_path(tmp_path) -> None:
    root = tmp_path / "project"
    config_dir = root / "configs" / "experiments"
    config_dir.mkdir(parents=True)
    path = config_dir / "run.json"
    path.write_text(json.dumps(_raw_config()), encoding="utf-8")

    config = ExperimentConfig.from_json(path)

    assert config.model_config_path == (root / "configs/models/test.json").resolve()
    assert config.prompt_id == "p000001"
    assert config.method.method_id == "bins"
    assert dict(config.method.params) == {"block_size": 2}
    assert config.method.random_seed == 12345
    assert config.generation.to_policy().temperature == 1.0
    assert config.termination.target_carrier_tokens == 16


def test_method_params_are_defensively_copied() -> None:
    params = {"block_size": 2}
    config = MethodRunConfig("bins", params, random_seed=1)
    params["block_size"] = 9
    assert dict(config.params) == {"block_size": 2}


def test_generation_config_rejects_disabled_kv_cache() -> None:
    with pytest.raises(ConfigurationError, match="kv_cache=true"):
        GenerationRunConfig(kv_cache=False)


def test_generation_config_reuses_reference_policy_validation() -> None:
    with pytest.raises(ConfigurationError, match="temperature"):
        GenerationRunConfig(temperature=0.0)


def test_termination_rejects_unknown_mode() -> None:
    with pytest.raises(ConfigurationError, match="fixed_carrier_tokens"):
        TerminationConfig(mode="fixed_payload_bits", target_carrier_tokens=16)


def test_termination_rejects_nonpositive_target() -> None:
    with pytest.raises(ConfigurationError, match="positive integer"):
        TerminationConfig(mode="fixed_carrier_tokens", target_carrier_tokens=0)


def test_experiment_config_rejects_wrong_benchmark_version(tmp_path) -> None:
    raw = _raw_config()
    raw["benchmark_version"] = "1.0"
    path = tmp_path / "run.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="benchmark_version=0.1"):
        ExperimentConfig.from_json(path, project_root=tmp_path)


def test_experiment_config_rejects_non_normalized_run(tmp_path) -> None:
    raw = _raw_config()
    raw["run_kind"] = "author_conformance"
    path = tmp_path / "run.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="run_kind=normalized"):
        ExperimentConfig.from_json(path, project_root=tmp_path)


def test_experiment_config_requires_method_params(tmp_path) -> None:
    raw = _raw_config()
    del raw["method"]["params"]
    path = tmp_path / "run.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="id and params"):
        ExperimentConfig.from_json(path, project_root=tmp_path)
