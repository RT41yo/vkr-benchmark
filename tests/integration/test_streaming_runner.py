from __future__ import annotations

import numpy as np

from vkr_benchmark.distributions import GenerationPolicy, ReferenceDistributionBuilder
from vkr_benchmark.lm import LMAdapter, LMState, TokenSpace
from vkr_benchmark.methods import BinsMethod
from vkr_benchmark.randomness import MethodRandomSource, Shake256SecretSource
from vkr_benchmark.runner import method_environment_from_builder, run_streaming_text_roundtrip


class _DeterministicFakeLM(LMAdapter):
    """Small causal LM whose text transport is exactly reversible."""

    def __init__(self) -> None:
        self._space = TokenSpace(
            output_vocab_size=9,
            tokenizer_vocab_size=9,
            tokenizer_token_ids=frozenset(range(9)),
            special_token_ids=frozenset({8}),
        )
        self.prefill_calls = 0
        self.advance_calls = 0

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
        # Vary ranking across positions while keeping every allowed token in support.
        base = np.array([0.2, 1.4, 0.5, 2.0, 0.9, 1.1, 0.7, 1.6, -3.0], dtype=np.float32)
        return np.roll(base, sequence_length % 8)

    def prefill(self, prompt_token_ids: tuple[int, ...] | list[int]) -> LMState:
        self.prefill_calls += 1
        n = len(prompt_token_ids)
        return LMState(
            backend_state=tuple(prompt_token_ids),
            pending_logits=self._logits(n),
            sequence_length=n,
        )

    def next_logits(self, state: LMState, previous_token_id: int | None = None):
        if previous_token_id is None:
            return state.pending_logits, state
        self.advance_calls += 1
        n = state.sequence_length + 1
        updated = LMState(
            backend_state=(*state.backend_state, int(previous_token_id)),
            pending_logits=self._logits(n),
            sequence_length=n,
        )
        return updated.pending_logits, updated


def test_bins_full_text_roundtrip_through_common_runner() -> None:
    lm = _DeterministicFakeLM()
    builder = ReferenceDistributionBuilder(
        token_space=lm.token_space,
        policy=GenerationPolicy(),
    )
    environment = method_environment_from_builder(builder)

    result = run_streaming_text_roundtrip(
        lm_adapter=lm,
        reference_builder=builder,
        method=BinsMethod(),
        method_config={"block_size": 2},
        environment=environment,
        prompt_text="prompt",
        carrier_tokens=6,
        secret_source=Shake256SecretSource("000001"),
        encoder_random_source=MethodRandomSource(12345),
        decoder_random_source=MethodRandomSource(12345),
    )

    assert result.encode.carrier_tokens == 6
    assert result.encode.payload_bits == 12
    assert len(result.encode.consumed_secret_bits) == 12
    assert result.transport.token_sequence_roundtrip_exact is True
    assert result.transport.receiver_token_ids == result.encode.carrier_token_ids
    assert result.decode.recovered_bits == result.encode.consumed_secret_bits
    assert result.roundtrip_exact is True
    assert result.first_bit_mismatch is None
    assert result.recovered_extra_bits == 0
    assert result.expected_length_bits == 12
    assert result.recovered_length_bits == 12
    assert result.length_delta_bits == 0

    # Encode and decode each prefill once. Six carrier tokens require five
    # cached advances on each path, not six.
    assert lm.prefill_calls == 2
    assert lm.advance_calls == 10


def test_method_environment_is_derived_from_common_allowed_mask() -> None:
    lm = _DeterministicFakeLM()
    builder = ReferenceDistributionBuilder(
        token_space=lm.token_space,
        policy=GenerationPolicy(),
    )

    environment = method_environment_from_builder(builder)

    assert environment.output_vocab_size == 9
    assert environment.allowed_token_ids == tuple(range(8))

class _DuplicateLastTokenFakeLM(_DeterministicFakeLM):
    def encode_text(self, text: str, *, add_special_tokens: bool) -> tuple[int, ...]:
        ids = super().encode_text(text, add_special_tokens=add_special_tokens)
        return (*ids, ids[-1]) if ids else ids


def test_extra_recovered_bits_force_roundtrip_failure() -> None:
    lm = _DuplicateLastTokenFakeLM()
    builder = ReferenceDistributionBuilder(
        token_space=lm.token_space,
        policy=GenerationPolicy(),
    )
    environment = method_environment_from_builder(builder)

    result = run_streaming_text_roundtrip(
        lm_adapter=lm,
        reference_builder=builder,
        method=BinsMethod(),
        method_config={"block_size": 2},
        environment=environment,
        prompt_text="prompt",
        carrier_tokens=4,
        secret_source=Shake256SecretSource("000001"),
        encoder_random_source=MethodRandomSource(12345),
        decoder_random_source=MethodRandomSource(12345),
    )

    # Decoder finalization truncates to the expected useful payload, but the
    # benchmark still records extra recovered bits and roundtrip must be false.
    assert result.decode.recovered_bits == result.encode.consumed_secret_bits
    assert result.recovered_extra_bits == 2
    assert result.recovered_length_bits == 10
    assert result.expected_length_bits == 8
    assert result.length_delta_bits == 2
    assert result.roundtrip_exact is False


def test_huffman_full_text_roundtrip_through_common_runner() -> None:
    from vkr_benchmark.methods import HuffmanMethod

    lm = _DeterministicFakeLM()
    builder = ReferenceDistributionBuilder(
        token_space=lm.token_space,
        policy=GenerationPolicy(),
    )
    environment = method_environment_from_builder(builder)

    result = run_streaming_text_roundtrip(
        lm_adapter=lm,
        reference_builder=builder,
        method=HuffmanMethod(),
        method_config={"bits_per_word": 2},
        environment=environment,
        prompt_text="prompt",
        carrier_tokens=8,
        secret_source=Shake256SecretSource("000001"),
        encoder_random_source=None,
        decoder_random_source=None,
    )

    assert result.encode.carrier_tokens == 8
    assert result.encode.payload_bits == len(result.encode.consumed_secret_bits)
    assert result.encode.payload_bits == sum(
        int(bits) for bits in result.encode.step_bits_consumed if bits is not None
    )
    assert all(bits is not None and bits > 0 for bits in result.encode.step_bits_consumed)
    # Unlike Bins, Huffman payload is determined by per-step code lengths rather
    # than a fixed block size. This deterministic run exercises that contract.
    assert len(set(result.encode.step_bits_consumed)) >= 2

    assert result.transport.token_sequence_roundtrip_exact is True
    assert result.transport.receiver_token_ids == result.encode.carrier_token_ids
    assert result.decode.recovered_bits == result.encode.consumed_secret_bits
    assert result.roundtrip_exact is True
    assert result.first_bit_mismatch is None
    assert result.recovered_extra_bits == 0
    assert result.recovered_length_bits == result.expected_length_bits
    assert result.length_delta_bits == 0
    assert result.decode.finalization.complete is True

    # Eight carrier tokens require seven cached advances on encode and seven on
    # decode. The same runner is reused without Huffman-specific LM logic.
    assert lm.prefill_calls == 2
    assert lm.advance_calls == 14


def test_huffman_common_runner_requires_no_method_rng() -> None:
    from vkr_benchmark.methods import HuffmanMethod

    lm = _DeterministicFakeLM()
    builder = ReferenceDistributionBuilder(
        token_space=lm.token_space,
        policy=GenerationPolicy(),
    )
    environment = method_environment_from_builder(builder)

    result = run_streaming_text_roundtrip(
        lm_adapter=lm,
        reference_builder=builder,
        method=HuffmanMethod(),
        method_config={"bits_per_word": 1},
        environment=environment,
        prompt_text="prompt",
        carrier_tokens=4,
        secret_source=Shake256SecretSource("huffman-no-rng"),
        encoder_random_source=None,
        decoder_random_source=None,
    )

    assert result.roundtrip_exact is True
    assert result.encode.payload_bits == result.expected_length_bits


def test_arithmetic_full_text_roundtrip_through_common_runner() -> None:
    from vkr_benchmark.methods import ArithmeticMethod

    lm = _DeterministicFakeLM()
    builder = ReferenceDistributionBuilder(
        token_space=lm.token_space,
        policy=GenerationPolicy(),
    )
    environment = method_environment_from_builder(builder)

    result = run_streaming_text_roundtrip(
        lm_adapter=lm,
        reference_builder=builder,
        method=ArithmeticMethod(),
        method_config={"precision": 8, "top_k": 8},
        environment=environment,
        prompt_text="prompt",
        carrier_tokens=8,
        secret_source=Shake256SecretSource("arithmetic-e2e"),
        encoder_random_source=None,
        decoder_random_source=None,
    )

    assert result.encode.carrier_tokens == 8
    assert result.encode.payload_bits == sum(
        int(bits) for bits in result.encode.step_bits_consumed if bits is not None
    )
    assert result.encode.secret_bits_read == 8 + result.encode.payload_bits
    assert len(result.encode.read_secret_bits) == result.encode.secret_bits_read
    assert len(result.encode.payload_secret_bits) == result.encode.payload_bits
    assert result.encode.payload_secret_bits == result.encode.read_secret_bits[: result.encode.payload_bits]
    assert result.encode.secret_bits_read > result.encode.payload_bits
    # Backward-compatible name now means useful payload, not all look-ahead reads.
    assert result.encode.consumed_secret_bits == result.encode.payload_secret_bits

    assert result.transport.token_sequence_roundtrip_exact is True
    assert result.transport.receiver_token_ids == result.encode.carrier_token_ids
    assert result.decode.recovered_bits == result.encode.payload_secret_bits
    assert result.roundtrip_exact is True
    assert result.first_bit_mismatch is None
    assert result.recovered_extra_bits == 0
    assert result.recovered_length_bits == result.expected_length_bits
    assert result.expected_length_bits == result.encode.payload_bits
    assert result.length_delta_bits == 0
    assert result.decode.finalization.complete is True

    # Eight carrier tokens require seven cached advances on encode and seven on decode.
    assert lm.prefill_calls == 2
    assert lm.advance_calls == 14


def test_streaming_runner_keeps_read_and_payload_bits_equal_for_bins() -> None:
    lm = _DeterministicFakeLM()
    builder = ReferenceDistributionBuilder(
        token_space=lm.token_space,
        policy=GenerationPolicy(),
    )
    environment = method_environment_from_builder(builder)

    result = run_streaming_text_roundtrip(
        lm_adapter=lm,
        reference_builder=builder,
        method=BinsMethod(),
        method_config={"block_size": 2},
        environment=environment,
        prompt_text="prompt",
        carrier_tokens=4,
        secret_source=Shake256SecretSource("bins-accounting-regression"),
        encoder_random_source=MethodRandomSource(12345),
        decoder_random_source=MethodRandomSource(12345),
    )

    assert result.encode.read_secret_bits == result.encode.payload_secret_bits
    assert result.encode.secret_bits_read == result.encode.payload_bits
    assert result.encode.consumed_secret_bits == result.encode.payload_secret_bits
    assert result.roundtrip_exact is True


def test_finalize_only_decoder_is_supported_by_fixed_payload_runner() -> None:
    from typing import Any, Mapping

    from vkr_benchmark.methods import (
        DecodeProgress,
        DecoderFinalization,
        DecoderSession,
        EncodeDecision,
        EncoderFinalization,
        EncoderSession,
        MethodEnvironment,
        StegoMethod,
    )
    from vkr_benchmark.randomness import RandomSource, SecretSource

    class _Encoder(EncoderSession):
        def __init__(self, source: SecretSource, target: int) -> None:
            self.bits = source.read_bits(target)
            self.target = target
            self.steps = 0
            self._finalized = False

        @property
        def done(self) -> bool:
            return self.steps >= 2

        def step(self, context):
            self.steps += 1
            return EncodeDecision(
                token_id=int(context.reference.token_order[0]),
                bits_consumed=None,
            )

        def finalize(self):
            self._finalized = True
            return EncoderFinalization(
                payload_bits=self.target,
                termination_reason="fixed_payload_bits",
            )

    class _Decoder(DecoderSession):
        def __init__(self, expected: int) -> None:
            self.expected = expected
            self.steps = 0
            self._finalized = False

        @property
        def done(self) -> bool:
            return self.steps >= 2

        def observe(self, context, observed_token_id):
            del context, observed_token_id
            self.steps += 1
            # RRC-like behavior: no bits are recovered until finalize().
            return DecodeProgress()

        def finalize(self):
            self._finalized = True
            return DecoderFinalization(
                recovered_bits=tuple(expected_bits),
                complete=True,
            )

    class _Method(StegoMethod):
        method_id = "synthetic_finalize_only"

        def create_encoder(
            self,
            *,
            config: Mapping[str, Any],
            environment: MethodEnvironment,
            secret_source: SecretSource,
            random_source: RandomSource | None = None,
            key=None,
            target_payload_bits: int | None = None,
        ) -> EncoderSession:
            del config, environment, random_source, key
            assert target_payload_bits is not None
            return _Encoder(secret_source, target_payload_bits)

        def create_decoder(
            self,
            *,
            config: Mapping[str, Any],
            environment: MethodEnvironment,
            random_source: RandomSource | None = None,
            key=None,
            expected_payload_bits: int | None = None,
        ) -> DecoderSession:
            del config, environment, random_source, key
            assert expected_payload_bits is not None
            return _Decoder(expected_payload_bits)

    lm = _DeterministicFakeLM()
    builder = ReferenceDistributionBuilder(
        token_space=lm.token_space,
        policy=GenerationPolicy(),
    )
    environment = method_environment_from_builder(builder)
    secret = Shake256SecretSource("finalize-only")
    # Capture the expected prefix independently, then create a fresh source for
    # the actual run so the encoder reads the same bits from position zero.
    expected_bits = secret.read_bits(8)

    result = run_streaming_text_roundtrip(
        lm_adapter=lm,
        reference_builder=builder,
        method=_Method(),
        method_config={},
        environment=environment,
        prompt_text="prompt",
        target_payload_bits=8,
        max_carrier_tokens=4,
        secret_source=Shake256SecretSource("finalize-only"),
        encoder_random_source=None,
        decoder_random_source=None,
    )

    assert result.encode.carrier_tokens == 2
    assert result.encode.payload_bits == 8
    assert result.decode.incremental_recovered_bits == ()
    assert result.decode.recovered_bits == expected_bits
    assert result.recovered_length_bits == 8
    assert result.recovered_extra_bits == 0
    assert result.roundtrip_exact is True
