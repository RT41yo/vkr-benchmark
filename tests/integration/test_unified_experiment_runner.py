from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vkr_benchmark.config import (
    ExperimentConfig,
    GenerationRunConfig,
    MethodRunConfig,
    TerminationConfig,
)
from vkr_benchmark.errors import ConfigurationError
from vkr_benchmark.inputs import PromptRecord, PromptRegistry
from vkr_benchmark.lm import LMAdapter, LMState, TokenSpace
from vkr_benchmark.runner import create_method, create_method_runtime, run_experiment


class _ExperimentFakeLM(LMAdapter):
    def __init__(self) -> None:
        self._space = TokenSpace(
            output_vocab_size=9,
            tokenizer_vocab_size=9,
            tokenizer_token_ids=frozenset(range(9)),
            special_token_ids=frozenset({8}),
        )

    @property
    def model_id(self) -> str:
        return "fake-causal"

    @property
    def revision(self) -> str:
        return "test"

    @property
    def token_space(self) -> TokenSpace:
        return self._space

    def encode_prompt(self, text: str) -> tuple[int, ...]:
        assert text == "prompt"
        return (0, 1)

    def encode_text(self, text: str, *, add_special_tokens: bool) -> tuple[int, ...]:
        assert add_special_tokens is False
        return tuple(int(part) for part in text.split()) if text else ()

    def decode_tokens(self, token_ids: tuple[int, ...] | list[int]) -> str:
        return " ".join(str(int(x)) for x in token_ids)

    @staticmethod
    def _logits(sequence_length: int) -> np.ndarray:
        base = np.array([0.2, 1.4, 0.5, 2.0, 0.9, 1.1, 0.7, 1.6, -3.0], dtype=np.float32)
        return np.roll(base, sequence_length % 8)

    def prefill(self, prompt_token_ids: tuple[int, ...] | list[int]) -> LMState:
        n = len(prompt_token_ids)
        return LMState(
            backend_state=tuple(prompt_token_ids),
            pending_logits=self._logits(n),
            sequence_length=n,
        )

    def next_logits(self, state: LMState, previous_token_id: int | None = None):
        if previous_token_id is None:
            return state.pending_logits, state
        n = state.sequence_length + 1
        updated = LMState(
            backend_state=(*state.backend_state, int(previous_token_id)),
            pending_logits=self._logits(n),
            sequence_length=n,
        )
        return updated.pending_logits, updated


def _config(method_id: str, params: dict, *, random_seed=None) -> ExperimentConfig:
    return ExperimentConfig(
        benchmark_version="0.1",
        run_kind="normalized",
        model_config_path=Path("unused.json"),
        prompt_id="p000001",
        method=MethodRunConfig(
            method_id=method_id,
            params=params,
            implementation_revision="test",
            random_seed=random_seed,
        ),
        generation=GenerationRunConfig(),
        secret_id="000001",
        termination=TerminationConfig(
            mode="fixed_carrier_tokens",
            target_carrier_tokens=8,
        ),
    )


def _prompts() -> PromptRegistry:
    return PromptRegistry(
        [PromptRecord("p000001", "unit", "r1", "prompt", "en")]
    )


@pytest.mark.parametrize(
    ("method_id", "params", "random_seed"),
    [
        ("bins", {"block_size": 2}, 12345),
        ("huffman", {"bits_per_word": 2}, None),
        ("arithmetic_coding", {"precision": 8, "top_k": 8}, None),
    ],
)
def test_all_three_baselines_run_through_one_experiment_entrypoint(
    method_id: str,
    params: dict,
    random_seed,
) -> None:
    execution = run_experiment(
        config=_config(method_id, params, random_seed=random_seed),
        prompt_registry=_prompts(),
        lm_adapter=_ExperimentFakeLM(),
    )

    assert execution.method_id == method_id
    assert execution.prompt.prompt_id == "p000001"
    assert execution.roundtrip.encode.carrier_tokens == 8
    assert len(execution.roundtrip.encode.step_reference_entropy_bits) == 8
    assert all(value > 0.0 for value in execution.roundtrip.encode.step_reference_entropy_bits)

    metrics = execution.capacity_entropy_metrics
    assert metrics.payload_bits == execution.roundtrip.encode.payload_bits
    assert metrics.carrier_tokens == 8
    assert metrics.bits_per_token == pytest.approx(metrics.payload_bits / 8)
    assert metrics.reference_entropy_sum_bits == pytest.approx(
        sum(execution.roundtrip.encode.step_reference_entropy_bits)
    )
    assert metrics.reference_entropy_mean_bits == pytest.approx(
        metrics.reference_entropy_sum_bits / 8
    )
    assert metrics.entropy_utilization == pytest.approx(
        metrics.payload_bits / metrics.reference_entropy_sum_bits
    )
    assert metrics.entropy_utilization_percent == pytest.approx(
        100.0 * metrics.entropy_utilization
    )

    distortion = execution.distribution_distortion_metrics
    assert distortion.q_mode.value == "analytic_exact"
    assert len(execution.roundtrip.encode.step_distribution_distortion) == 8
    assert distortion.kl_infinite_steps + distortion.kl_finite_steps == 8
    assert distortion.tvd_mean is not None
    assert 0.0 <= distortion.tvd_mean <= 1.0
    assert distortion.tvd_median is not None
    assert distortion.tvd_p95 is not None
    assert distortion.tvd_max is not None
    assert 0.0 <= distortion.tvd_max <= 1.0

    assert execution.roundtrip.roundtrip_exact is True
    assert execution.roundtrip.transport.token_sequence_roundtrip_exact is True


def test_method_factory_resolves_all_baseline_ids() -> None:
    assert create_method("bins").method_id == "bins"
    assert create_method("huffman").method_id == "huffman"
    assert create_method("arithmetic_coding").method_id == "arithmetic_coding"


def test_method_factory_rejects_unknown_method() -> None:
    with pytest.raises(ConfigurationError, match="unknown method id"):
        create_method("not-a-method")


def test_bins_runtime_requires_reproducible_random_seed() -> None:
    with pytest.raises(ConfigurationError, match="random_seed"):
        create_method_runtime(_config("bins", {"block_size": 2}, random_seed=None))


def test_bins_sender_receiver_rng_are_independent_but_reproducible() -> None:
    runtime = create_method_runtime(
        _config("bins", {"block_size": 2}, random_seed=12345)
    )
    assert runtime.encoder_random_source is not runtime.decoder_random_source
    assert runtime.encoder_random_source is not None
    assert runtime.decoder_random_source is not None
    assert [runtime.encoder_random_source.randbelow(1000) for _ in range(5)] == [
        runtime.decoder_random_source.randbelow(1000) for _ in range(5)
    ]


def test_unified_runner_resolves_prompt_by_id() -> None:
    config = _config("huffman", {"bits_per_word": 2})
    missing = PromptRegistry([PromptRecord("other", "unit", "r1", "prompt", "en")])
    with pytest.raises(ConfigurationError, match="unknown prompt_id"):
        run_experiment(
            config=config,
            prompt_registry=missing,
            lm_adapter=_ExperimentFakeLM(),
        )
