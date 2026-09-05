"""Minimal fixed-carrier streaming runner used during Stage-2 integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from vkr_benchmark.distributions import ReferenceDistributionBuilder, StepContext
from vkr_benchmark.errors import ContractError
from vkr_benchmark.lm import LMAdapter
from vkr_benchmark.methods import (
    DecoderFinalization,
    EncoderFinalization,
    MethodEnvironment,
    StegoMethod,
)
from vkr_benchmark.methods.base import KeyMaterial
from vkr_benchmark.randomness import RandomSource, RecordingSecretSource, SecretSource
from vkr_benchmark.transport import TextChannel, TextTransportResult


@dataclass(frozen=True, slots=True)
class StreamingEncodeResult:
    """Output of one fixed-carrier streaming encode pass."""

    prompt_token_ids: tuple[int, ...]
    carrier_token_ids: tuple[int, ...]
    read_secret_bits: tuple[int, ...]
    payload_secret_bits: tuple[int, ...]
    step_bits_consumed: tuple[int | None, ...]
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
        """Backward-compatible alias for the confirmed useful payload bits.

        Earlier Stage-2 streaming methods consumed exactly the useful payload,
        so this field originally meant both "read" and "embedded". Arithmetic
        Coding proves those concepts must be separate because it keeps a
        precision-bit look-ahead window.
        """

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

    The first carrier token is chosen from logits already produced by prefill;
    therefore N carrier tokens require only N-1 cached LM advances, matching
    the Stage-1 inference path.
    """

    if carrier_tokens <= 0:
        raise ValueError("carrier_tokens must be positive")

    prompt_token_ids = lm_adapter.encode_prompt(prompt_text)
    state = lm_adapter.prefill(prompt_token_ids)
    recorded_secret = RecordingSecretSource(secret_source)
    encoder = method.create_encoder(
        config=method_config,
        environment=environment,
        secret_source=recorded_secret,
        random_source=method_random_source,
        key=key,
    )

    generated: list[int] = []
    step_bits: list[int | None] = []

    for step_index in range(carrier_tokens):
        raw_logits, state = lm_adapter.next_logits(state)
        reference = reference_builder.build(raw_logits)
        decision = encoder.step(StepContext(step_index=step_index, reference=reference))

        token_id = int(decision.token_id)
        if not environment.is_allowed(token_id):
            raise ContractError(
                f"method emitted token {token_id}, which is outside V_allowed"
            )

        generated.append(token_id)
        step_bits.append(decision.bits_consumed)

        if encoder.done or step_index + 1 >= carrier_tokens:
            break

        _, state = lm_adapter.next_logits(state, token_id)

    finalization = encoder.finalize()
    read_secret_bits = recorded_secret.consumed_bits
    if finalization.payload_bits > len(read_secret_bits):
        raise ContractError(
            "encoder reports more useful payload bits than were read from SecretSource"
        )

    # For streaming prefix methods used in Stage 2, useful payload is the
    # confirmed prefix of the secret stream. Arithmetic Coding may have read an
    # additional precision-bit look-ahead suffix that is *not* useful payload.
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

    observed = tuple(int(token_id) for token_id in observed_token_ids)
    prompt_token_ids = lm_adapter.encode_prompt(prompt_text)
    state = lm_adapter.prefill(prompt_token_ids)
    decoder = method.create_decoder(
        config=method_config,
        environment=environment,
        random_source=method_random_source,
        key=key,
        expected_payload_bits=expected_payload_bits,
    )

    incremental: list[int] = []
    for step_index, token_id in enumerate(observed):
        raw_logits, state = lm_adapter.next_logits(state)
        reference = reference_builder.build(raw_logits)
        progress = decoder.observe(
            StepContext(step_index=step_index, reference=reference),
            token_id,
        )
        incremental.extend(progress.recovered_bits)

        if step_index + 1 < len(observed):
            _, state = lm_adapter.next_logits(state, token_id)

    finalization = decoder.finalize()
    return StreamingDecodeResult(
        prompt_token_ids=prompt_token_ids,
        observed_token_ids=observed,
        incremental_recovered_bits=tuple(incremental),
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
