"""Unified Stage-2 experiment orchestration above the streaming runner."""

from __future__ import annotations

from dataclasses import dataclass

from vkr_benchmark.config import ExperimentConfig
from vkr_benchmark.distributions import ReferenceDistributionBuilder
from vkr_benchmark.errors import ConfigurationError
from vkr_benchmark.inputs import PromptRecord, PromptRegistry, create_secret_source
from vkr_benchmark.lm import LMAdapter
from vkr_benchmark.metrics import (
    CapacityEntropyMetrics,
    DistributionDistortionMetrics,
    PerformanceMetrics,
    RawLMQualityMetrics,
    ReliabilityMetrics,
    compute_capacity_entropy_metrics,
    compute_distribution_distortion_metrics,
    compute_performance_metrics,
    compute_raw_lm_quality_metrics,
    compute_reliability_metrics,
)
from vkr_benchmark.methods import ArithmeticMethod, BinsMethod, HuffmanMethod, StegoMethod
from vkr_benchmark.randomness import MethodRandomSource, RandomSource
from vkr_benchmark.runner.streaming import (
    StreamingTextRoundtripResult,
    method_environment_from_builder,
    run_streaming_text_roundtrip,
    warm_up_streaming_path,
)


@dataclass(frozen=True, slots=True)
class MethodRuntime:
    """Concrete method adapter plus isolated sender/receiver RNG instances."""

    method: StegoMethod
    encoder_random_source: RandomSource | None
    decoder_random_source: RandomSource | None


@dataclass(frozen=True, slots=True)
class ExperimentExecution:
    """Result of one configured run with Stage-2 normalized metrics."""

    config: ExperimentConfig
    prompt: PromptRecord
    roundtrip: StreamingTextRoundtripResult
    capacity_entropy_metrics: CapacityEntropyMetrics
    distribution_distortion_metrics: DistributionDistortionMetrics
    quality_metrics: RawLMQualityMetrics
    reliability_metrics: ReliabilityMetrics
    performance_metrics: PerformanceMetrics

    @property
    def method_id(self) -> str:
        return self.config.method.method_id


_METHODS: dict[str, type[StegoMethod]] = {
    "bins": BinsMethod,
    "huffman": HuffmanMethod,
    "arithmetic_coding": ArithmeticMethod,
}


def create_method(method_id: str) -> StegoMethod:
    """Resolve the normalized method adapter by stable method id."""

    try:
        method_type = _METHODS[method_id]
    except KeyError as exc:
        supported = ", ".join(sorted(_METHODS))
        raise ConfigurationError(
            f"unknown method id {method_id!r}; supported: {supported}"
        ) from exc
    return method_type()


def create_method_runtime(config: ExperimentConfig) -> MethodRuntime:
    """Create method and reproducible per-side method RNG streams."""

    method = create_method(config.method.method_id)
    seed = config.method.random_seed

    if method.method_id == "bins" and seed is None:
        raise ConfigurationError(
            "normalized Bins requires method.random_seed so sender and receiver "
            "can reconstruct the same fixed partition"
        )

    if seed is None:
        encoder_rng = None
        decoder_rng = None
    else:
        # Independent objects with the same seed: sender and receiver reproduce
        # the same method randomness without sharing mutable RNG state.
        encoder_rng = MethodRandomSource(seed)
        decoder_rng = MethodRandomSource(seed)

    return MethodRuntime(
        method=method,
        encoder_random_source=encoder_rng,
        decoder_random_source=decoder_rng,
    )


def run_experiment(
    *,
    config: ExperimentConfig,
    prompt_registry: PromptRegistry,
    lm_adapter: LMAdapter,
) -> ExperimentExecution:
    """Execute one normalized Bins/Huffman/Arithmetic run from one config."""

    prompt = prompt_registry.get(config.prompt_id)
    builder = ReferenceDistributionBuilder(
        token_space=lm_adapter.token_space,
        policy=config.generation.to_policy(),
    )
    environment = method_environment_from_builder(builder)
    runtime = create_method_runtime(config)

    # FIXED v0.1 timing rule: warm-up is completed before measured sections.
    warm_up_streaming_path(
        lm_adapter=lm_adapter,
        reference_builder=builder,
        environment=environment,
        prompt_text=prompt.text,
    )

    roundtrip = run_streaming_text_roundtrip(
        lm_adapter=lm_adapter,
        reference_builder=builder,
        method=runtime.method,
        method_config=config.method.params,
        environment=environment,
        prompt_text=prompt.text,
        carrier_tokens=config.termination.target_carrier_tokens,
        secret_source=create_secret_source(config.secret_id),
        encoder_random_source=runtime.encoder_random_source,
        decoder_random_source=runtime.decoder_random_source,
        key=config.method.key,
    )

    capacity_entropy_metrics = compute_capacity_entropy_metrics(
        payload_bits=roundtrip.encode.payload_bits,
        reference_entropies_bits=roundtrip.encode.step_reference_entropy_bits,
    )
    distribution_distortion_metrics = compute_distribution_distortion_metrics(
        roundtrip.encode.step_distribution_distortion
    )
    quality_metrics = compute_raw_lm_quality_metrics(
        roundtrip.encode.step_raw_lm_nll_nats
    )
    reliability_metrics = compute_reliability_metrics(
        expected_bits=roundtrip.encode.payload_secret_bits,
        recovered_bits=roundtrip.decode.incremental_recovered_bits,
        roundtrip_exact=roundtrip.roundtrip_exact,
        first_mismatch_bit=roundtrip.first_bit_mismatch,
        recovered_extra_bits=roundtrip.recovered_extra_bits,
        token_sequence_roundtrip_exact=(
            roundtrip.transport.token_sequence_roundtrip_exact
        ),
        first_token_roundtrip_mismatch=roundtrip.transport.first_token_mismatch,
    )
    performance_metrics = compute_performance_metrics(
        payload_bits=roundtrip.encode.payload_bits,
        encode_tokens=roundtrip.encode.carrier_tokens,
        decode_tokens=len(roundtrip.decode.observed_token_ids),
        encode_timing=roundtrip.encode.timing,
        decode_timing=roundtrip.decode.timing,
    )

    return ExperimentExecution(
        config=config,
        prompt=prompt,
        roundtrip=roundtrip,
        capacity_entropy_metrics=capacity_entropy_metrics,
        distribution_distortion_metrics=distribution_distortion_metrics,
        quality_metrics=quality_metrics,
        reliability_metrics=reliability_metrics,
        performance_metrics=performance_metrics,
    )
