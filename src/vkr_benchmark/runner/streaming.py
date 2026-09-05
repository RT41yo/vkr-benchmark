"""Fixed-carrier streaming runner with Stage-2 metric instrumentation."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Mapping, TypeVar

import numpy as np

from vkr_benchmark.distributions import ReferenceDistributionBuilder, StepContext
from vkr_benchmark.errors import ContractError
from vkr_benchmark.lm import LMAdapter
from vkr_benchmark.metrics import (
    StepDistributionDistortion,
    TimingBreakdown,
    distribution_distortion_step,
    raw_lm_token_nll_nats,
    reference_entropy_bits,
)
from vkr_benchmark.methods import (
    DecoderFinalization,
    EncoderFinalization,
    MethodEnvironment,
    StegoMethod,
)
from vkr_benchmark.methods.base import KeyMaterial
from vkr_benchmark.randomness import RandomSource, RecordingSecretSource, SecretSource
from vkr_benchmark.transport import TextChannel, TextTransportResult

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class StreamingEncodeResult:
    """Output of one fixed-carrier streaming encode pass."""

    prompt_token_ids: tuple[int, ...]
    carrier_token_ids: tuple[int, ...]
    read_secret_bits: tuple[int, ...]
    payload_secret_bits: tuple[int, ...]
    step_bits_consumed: tuple[int | None, ...]
    step_reference_entropy_bits: tuple[float, ...]
    step_distribution_distortion: tuple[StepDistributionDistortion, ...]
    step_raw_lm_nll_nats: tuple[float, ...]
    timing: TimingBreakdown
    finalization: EncoderFinalization

    @property
    def payload_bits(self) -> int:
        return self.finalization.payload_bits

    @property
    def secret_bits_read(self) -> int:
        """Number of bits fetched from SecretSource, including method look-ahead."""

        return len(self.read_secret_bits)

    @property
    def consumed_secret_bits(self) -> tuple[int, ...]:
        """Backward-compatible alias for the confirmed useful payload bits."""

        return self.payload_secret_bits

    @property
    def carrier_tokens(self) -> int:
        return len(self.carrier_token_ids)


@dataclass(frozen=True, slots=True)
class StreamingDecodeResult:
    """Output of one streaming decode pass over receiver-side token ids."""

    prompt_token_ids: tuple[int, ...]
    observed_token_ids: tuple[int, ...]
    incremental_recovered_bits: tuple[int, ...]
    timing: TimingBreakdown
    finalization: DecoderFinalization

    @property
    def recovered_bits(self) -> tuple[int, ...]:
        return self.finalization.recovered_bits


@dataclass(frozen=True, slots=True)
class StreamingTextRoundtripResult:
    """End-to-end reliability diagnostics after ordinary-text transport."""

    encode: StreamingEncodeResult
    transport: TextTransportResult
    decode: StreamingDecodeResult
    roundtrip_exact: bool
    first_bit_mismatch: int | None
    recovered_extra_bits: int
    recovered_length_bits: int
    expected_length_bits: int

    @property
    def length_delta_bits(self) -> int:
        return self.recovered_length_bits - self.expected_length_bits


def method_environment_from_builder(
    builder: ReferenceDistributionBuilder,
) -> MethodEnvironment:
    """Create method-facing V_allowed from the common reference builder."""

    allowed_ids = tuple(
        int(token_id) for token_id in np.flatnonzero(builder.allowed_mask)
    )
    return MethodEnvironment(
        output_vocab_size=builder.token_space.output_vocab_size,
        allowed_token_ids=allowed_ids,
    )


def _first_bit_mismatch(
    expected: tuple[int, ...],
    recovered: tuple[int, ...],
) -> int | None:
    for index, (left, right) in enumerate(zip(expected, recovered, strict=False)):
        if left != right:
            return index
    if len(expected) != len(recovered):
        return min(len(expected), len(recovered))
    return None


def _cuda_synchronize_if_needed(lm_adapter: LMAdapter) -> None:
    """Synchronize the configured CUDA device before/after timed GPU work."""

    device = str(getattr(lm_adapter, "device", "cpu"))
    if not device.startswith("cuda"):
        return

    try:
        import torch
    except ImportError:  # pragma: no cover - CUDA adapter requires torch
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _measure_ms(lm_adapter: LMAdapter, operation: Callable[[], _T]) -> tuple[_T, float]:
    _cuda_synchronize_if_needed(lm_adapter)
    started = perf_counter()
    result = operation()
    _cuda_synchronize_if_needed(lm_adapter)
    elapsed_ms = (perf_counter() - started) * 1000.0
    return result, float(elapsed_ms)


def warm_up_streaming_path(
    *,
    lm_adapter: LMAdapter,
    reference_builder: ReferenceDistributionBuilder,
    environment: MethodEnvironment,
    prompt_text: str,
) -> None:
    """Warm prefill, canonical P_reference, and one cached LM advance.

    This warm-up is deliberately executed before measured encode/decode paths.
    It does not consume secret bits or initialize a steganographic method.
    """

    prompt_token_ids = lm_adapter.encode_prompt(prompt_text)
    state = lm_adapter.prefill(prompt_token_ids)
    raw_logits, state = lm_adapter.next_logits(state)
    reference_builder.build(raw_logits)
    if environment.allowed_token_ids:
        lm_adapter.next_logits(state, int(environment.allowed_token_ids[0]))
    _cuda_synchronize_if_needed(lm_adapter)


def encode_fixed_carrier_tokens(
    *,
    lm_adapter: LMAdapter,
    reference_builder: ReferenceDistributionBuilder,
    method: StegoMethod,
    method_config: Mapping[str, Any],
    environment: MethodEnvironment,
    prompt_text: str,
    carrier_tokens: int,
    secret_source: SecretSource,
    method_random_source: RandomSource | None,
    key: KeyMaterial = None,
) -> StreamingEncodeResult:
    """Generate at most ``carrier_tokens`` through one streaming method session.

    Timing includes prompt prefill, canonical distribution processing, method
    session creation/steps/finalization and cached LM advances. Metric
    calculations themselves are intentionally outside the timed components.
    """

    if carrier_tokens <= 0:
        raise ValueError("carrier_tokens must be positive")

    lm_ms = 0.0
    distribution_ms = 0.0
    stego_ms = 0.0

    prompt_token_ids = lm_adapter.encode_prompt(prompt_text)
    state, elapsed = _measure_ms(
        lm_adapter, lambda: lm_adapter.prefill(prompt_token_ids)
    )
    lm_ms += elapsed

    recorded_secret = RecordingSecretSource(secret_source)
    encoder, elapsed = _measure_ms(
        lm_adapter,
        lambda: method.create_encoder(
            config=method_config,
            environment=environment,
            secret_source=recorded_secret,
            random_source=method_random_source,
            key=key,
        ),
    )
    stego_ms += elapsed

    generated: list[int] = []
    step_bits: list[int | None] = []
    step_entropies: list[float] = []
    step_distortion: list[StepDistributionDistortion] = []
    step_raw_nll: list[float] = []

    for step_index in range(carrier_tokens):
        (raw_logits, state), elapsed = _measure_ms(
            lm_adapter, lambda: lm_adapter.next_logits(state)
        )
        lm_ms += elapsed

        reference, elapsed = _measure_ms(
            lm_adapter, lambda: reference_builder.build(raw_logits)
        )
        distribution_ms += elapsed

        # Diagnostic metric calculations are excluded from performance timing.
        step_entropies.append(reference_entropy_bits(reference))
        decision, elapsed = _measure_ms(
            lm_adapter,
            lambda: encoder.step(
                StepContext(step_index=step_index, reference=reference)
            ),
        )
        stego_ms += elapsed
        step_distortion.append(
            distribution_distortion_step(reference, decision.distribution_info)
        )

        token_id = int(decision.token_id)
        if not environment.is_allowed(token_id):
            raise ContractError(
                f"method emitted token {token_id}, which is outside V_allowed"
            )
        step_raw_nll.append(raw_lm_token_nll_nats(raw_logits, token_id))

        generated.append(token_id)
        step_bits.append(decision.bits_consumed)

        if encoder.done or step_index + 1 >= carrier_tokens:
            break

        (_, state), elapsed = _measure_ms(
            lm_adapter, lambda: lm_adapter.next_logits(state, token_id)
        )
        lm_ms += elapsed

    finalization, elapsed = _measure_ms(lm_adapter, encoder.finalize)
    stego_ms += elapsed
    read_secret_bits = recorded_secret.consumed_bits
    if finalization.payload_bits > len(read_secret_bits):
        raise ContractError(
            "encoder reports more useful payload bits than were read from SecretSource"
        )

    # Useful payload is the confirmed prefix. Arithmetic Coding can hold an
    # additional precision-bit look-ahead suffix that is not embedded payload.
    payload_secret_bits = read_secret_bits[: finalization.payload_bits]

    if all(bits is not None for bits in step_bits):
        step_payload = sum(int(bits) for bits in step_bits)
        if step_payload != finalization.payload_bits:
            raise ContractError(
                "per-step payload accounting disagrees with encoder finalization"
            )

    return StreamingEncodeResult(
        prompt_token_ids=prompt_token_ids,
        carrier_token_ids=tuple(generated),
        read_secret_bits=read_secret_bits,
        payload_secret_bits=payload_secret_bits,
        step_bits_consumed=tuple(step_bits),
        step_reference_entropy_bits=tuple(step_entropies),
        step_distribution_distortion=tuple(step_distortion),
        step_raw_lm_nll_nats=tuple(step_raw_nll),
        timing=TimingBreakdown(
            lm_forward_ms=lm_ms,
            distribution_processing_ms=distribution_ms,
            stego_algorithm_ms=stego_ms,
        ),
        finalization=finalization,
    )


def decode_streaming_tokens(
    *,
    lm_adapter: LMAdapter,
    reference_builder: ReferenceDistributionBuilder,
    method: StegoMethod,
    method_config: Mapping[str, Any],
    environment: MethodEnvironment,
    prompt_text: str,
    observed_token_ids: tuple[int, ...] | list[int],
    method_random_source: RandomSource | None,
    expected_payload_bits: int,
    key: KeyMaterial = None,
) -> StreamingDecodeResult:
    """Decode receiver-side tokens while reconstructing the same LM prefixes."""

    if expected_payload_bits < 0:
        raise ValueError("expected_payload_bits must be non-negative")

    lm_ms = 0.0
    distribution_ms = 0.0
    stego_ms = 0.0

    observed = tuple(int(token_id) for token_id in observed_token_ids)
    prompt_token_ids = lm_adapter.encode_prompt(prompt_text)
    state, elapsed = _measure_ms(
        lm_adapter, lambda: lm_adapter.prefill(prompt_token_ids)
    )
    lm_ms += elapsed

    decoder, elapsed = _measure_ms(
        lm_adapter,
        lambda: method.create_decoder(
            config=method_config,
            environment=environment,
            random_source=method_random_source,
            key=key,
            expected_payload_bits=expected_payload_bits,
        ),
    )
    stego_ms += elapsed

    incremental: list[int] = []
    for step_index, token_id in enumerate(observed):
        (raw_logits, state), elapsed = _measure_ms(
            lm_adapter, lambda: lm_adapter.next_logits(state)
        )
        lm_ms += elapsed

        reference, elapsed = _measure_ms(
            lm_adapter, lambda: reference_builder.build(raw_logits)
        )
        distribution_ms += elapsed

        progress, elapsed = _measure_ms(
            lm_adapter,
            lambda: decoder.observe(
                StepContext(step_index=step_index, reference=reference),
                token_id,
            ),
        )
        stego_ms += elapsed
        incremental.extend(progress.recovered_bits)

        if step_index + 1 < len(observed):
            (_, state), elapsed = _measure_ms(
                lm_adapter, lambda: lm_adapter.next_logits(state, token_id)
            )
            lm_ms += elapsed

    finalization, elapsed = _measure_ms(lm_adapter, decoder.finalize)
    stego_ms += elapsed
    return StreamingDecodeResult(
        prompt_token_ids=prompt_token_ids,
        observed_token_ids=observed,
        incremental_recovered_bits=tuple(incremental),
        timing=TimingBreakdown(
            lm_forward_ms=lm_ms,
            distribution_processing_ms=distribution_ms,
            stego_algorithm_ms=stego_ms,
        ),
        finalization=finalization,
    )


def run_streaming_text_roundtrip(
    *,
    lm_adapter: LMAdapter,
    reference_builder: ReferenceDistributionBuilder,
    method: StegoMethod,
    method_config: Mapping[str, Any],
    environment: MethodEnvironment,
    prompt_text: str,
    carrier_tokens: int,
    secret_source: SecretSource,
    encoder_random_source: RandomSource | None,
    decoder_random_source: RandomSource | None,
    key: KeyMaterial = None,
) -> StreamingTextRoundtripResult:
    """Encode, transmit only ordinary text, retokenize, then decode."""

    encoded = encode_fixed_carrier_tokens(
        lm_adapter=lm_adapter,
        reference_builder=reference_builder,
        method=method,
        method_config=method_config,
        environment=environment,
        prompt_text=prompt_text,
        carrier_tokens=carrier_tokens,
        secret_source=secret_source,
        method_random_source=encoder_random_source,
        key=key,
    )

    # Text transport is part of reliability validation but intentionally not
    # included in encode/decode computational-efficiency timing.
    transport = TextChannel(lm_adapter).transmit(encoded.carrier_token_ids)

    decoded = decode_streaming_tokens(
        lm_adapter=lm_adapter,
        reference_builder=reference_builder,
        method=method,
        method_config=method_config,
        environment=environment,
        prompt_text=prompt_text,
        observed_token_ids=transport.receiver_token_ids,
        method_random_source=decoder_random_source,
        expected_payload_bits=encoded.payload_bits,
        key=key,
    )

    expected = encoded.payload_secret_bits
    recovered = decoded.recovered_bits
    mismatch = _first_bit_mismatch(expected, recovered)
    raw_recovered_length = len(decoded.incremental_recovered_bits)
    recovered_extra_bits = max(0, raw_recovered_length - len(expected))
    exact = (
        mismatch is None
        and raw_recovered_length == len(expected)
        and decoded.finalization.complete
    )

    return StreamingTextRoundtripResult(
        encode=encoded,
        transport=transport,
        decode=decoded,
        roundtrip_exact=exact,
        first_bit_mismatch=mismatch,
        recovered_extra_bits=recovered_extra_bits,
        recovered_length_bits=raw_recovered_length,
        expected_length_bits=len(expected),
    )
