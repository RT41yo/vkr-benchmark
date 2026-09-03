from __future__ import annotations

import numpy as np
import pytest

from vkr_benchmark.distributions import (
    QMode,
    QRepresentation,
    QSource,
    ReferenceDistribution,
    StepContext,
)
from vkr_benchmark.errors import (
    ConfigurationError,
    MethodError,
    SessionStateError,
    UnsupportedConfigurationError,
)
from vkr_benchmark.methods import HuffmanConfig, HuffmanMethod, MethodEnvironment
from vkr_benchmark.randomness import RecordingSecretSource, SecretSource, Shake256SecretSource


class FixedSecretSource(SecretSource):
    def __init__(self, bits: tuple[int, ...]) -> None:
        self._bits = bits
        self._position = 0

    @property
    def position(self) -> int:
        return self._position

    def read_bits(self, count: int) -> tuple[int, ...]:
        if count < 0:
            raise ValueError("count must be non-negative")
        stop = self._position + count
        if stop > len(self._bits):
            raise RuntimeError("test secret exhausted")
        out = self._bits[self._position:stop]
        self._position = stop
        return out


def env4() -> MethodEnvironment:
    return MethodEnvironment(output_vocab_size=4, allowed_token_ids=(0, 1, 2, 3))


def ref4_unequal() -> ReferenceDistribution:
    return ReferenceDistribution(
        np.array([0.5, 0.25, 0.125, 0.125], dtype=np.float32)
    )


def ref6() -> ReferenceDistribution:
    return ReferenceDistribution(
        np.array([0.31, 0.07, 0.24, 0.05, 0.19, 0.14], dtype=np.float32)
    )


def test_config_candidate_count() -> None:
    config = HuffmanConfig(bits_per_word=3)
    assert config.candidate_count == 8


def test_config_rejects_missing_unknown_non_integer_and_non_positive_fields() -> None:
    with pytest.raises(ConfigurationError, match="requires bits_per_word"):
        HuffmanConfig.from_mapping({})
    with pytest.raises(ConfigurationError, match="unknown Huffman"):
        HuffmanConfig.from_mapping({"bits_per_word": 2, "x": 1})
    with pytest.raises(ConfigurationError, match="must be an integer"):
        HuffmanConfig.from_mapping({"bits_per_word": True})
    with pytest.raises(ConfigurationError, match="positive integer"):
        HuffmanConfig.from_mapping({"bits_per_word": 0})


def test_candidate_set_cannot_exceed_allowed_vocabulary() -> None:
    with pytest.raises(UnsupportedConfigurationError, match="exceeds V_allowed"):
        HuffmanMethod().create_encoder(
            config={"bits_per_word": 3},
            environment=env4(),
            secret_source=FixedSecretSource((0,)),
        )


def test_huffman_does_not_require_method_rng() -> None:
    encoder = HuffmanMethod().create_encoder(
        config={"bits_per_word": 2},
        environment=env4(),
        secret_source=FixedSecretSource((0,)),
    )
    decision = encoder.step(StepContext(0, ref4_unequal()))
    assert decision.token_id == 0


def test_known_tree_has_expected_variable_length_codes() -> None:
    # Deterministic tree for [0.5, 0.25, 0.125, 0.125]:
    # token 0 -> 0, token 1 -> 10, token 2 -> 110, token 3 -> 111.
    cases = [
        ((0,), 0, 1),
        ((1, 0), 1, 2),
        ((1, 1, 0), 2, 3),
        ((1, 1, 1), 3, 3),
    ]
    for secret_bits, expected_token, expected_length in cases:
        encoder = HuffmanMethod().create_encoder(
            config={"bits_per_word": 2},
            environment=env4(),
            secret_source=FixedSecretSource(secret_bits),
        )
        decision = encoder.step(StepContext(0, ref4_unequal()))
        assert decision.token_id == expected_token
        assert decision.bits_consumed == expected_length
        assert decision.method_trace["selected_code"] == secret_bits
        assert decision.method_trace["code_length"] == expected_length


def test_equal_weights_use_deterministic_token_id_tie_break() -> None:
    reference = ReferenceDistribution(np.full(4, 0.25, dtype=np.float32))
    expected = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 2,
        (1, 1): 3,
    }
    for code, token_id in expected.items():
        encoder = HuffmanMethod().create_encoder(
            config={"bits_per_word": 2},
            environment=env4(),
            secret_source=FixedSecretSource(code),
        )
        decision = encoder.step(StepContext(0, reference))
        assert decision.token_id == token_id
        assert decision.method_trace["selected_code"] == code


def test_candidate_set_is_top_power_of_two_from_canonical_token_order() -> None:
    environment = MethodEnvironment(output_vocab_size=6, allowed_token_ids=tuple(range(6)))
    encoder = HuffmanMethod().create_encoder(
        config={"bits_per_word": 2},
        environment=environment,
        secret_source=FixedSecretSource((0, 0, 0, 0)),
    )
    reference = ref6()
    decision = encoder.step(StepContext(0, reference))
    q = decision.distribution_info.probabilities
    assert q is not None
    # Top four probabilities belong to token IDs 0, 2, 4, 5.
    assert set(np.flatnonzero(q)) == {0, 2, 4, 5}


def test_q_stego_is_exact_dyadic_distribution_from_code_lengths() -> None:
    encoder = HuffmanMethod().create_encoder(
        config={"bits_per_word": 2},
        environment=env4(),
        secret_source=FixedSecretSource((0,)),
    )
    decision = encoder.step(StepContext(0, ref4_unequal()))
    info = decision.distribution_info

    assert info.mode == QMode.ANALYTIC_EXACT
    assert info.representation == QRepresentation.EXPLICIT_PROBABILITIES
    assert info.source == QSource.ADAPTER_EXACT
    assert info.probabilities is not None
    assert np.array_equal(
        info.probabilities,
        np.array([0.5, 0.25, 0.125, 0.125], dtype=np.float64),
    )
    assert info.probabilities.flags.writeable is False
    assert np.isclose(info.probabilities.sum(), 1.0)


def test_q_stego_can_differ_from_reference() -> None:
    reference = ReferenceDistribution(
        np.array([0.46, 0.27, 0.16, 0.11], dtype=np.float32)
    )
    encoder = HuffmanMethod().create_encoder(
        config={"bits_per_word": 2},
        environment=env4(),
        secret_source=FixedSecretSource((0,)),
    )
    q = encoder.step(StepContext(0, reference)).distribution_info.probabilities
    assert q is not None
    assert not np.allclose(q, reference.probabilities, rtol=0.0, atol=1e-12)


def test_decoder_recovers_huffman_code_for_observed_candidate() -> None:
    decoder = HuffmanMethod().create_decoder(
        config={"bits_per_word": 2},
        environment=env4(),
    )
    progress = decoder.observe(StepContext(0, ref4_unequal()), 2)
    assert progress.recovered_bits == (1, 1, 0)
    assert progress.method_trace["code_length"] == 3


def test_decoder_rejects_token_outside_current_candidate_set() -> None:
    environment = MethodEnvironment(output_vocab_size=6, allowed_token_ids=tuple(range(6)))
    decoder = HuffmanMethod().create_decoder(
        config={"bits_per_word": 2},
        environment=environment,
    )
    # token 3 is not in top-4 for ref6().
    with pytest.raises(MethodError, match="outside the current Huffman candidate set"):
        decoder.observe(StepContext(0, ref6()), 3)


def test_dynamic_support_failure_happens_before_secret_consumption() -> None:
    environment = MethodEnvironment(output_vocab_size=6, allowed_token_ids=tuple(range(6)))
    secret = FixedSecretSource((1, 0, 1, 0))
    encoder = HuffmanMethod().create_encoder(
        config={"bits_per_word": 2},
        environment=environment,
        secret_source=secret,
    )
    probabilities = np.array([0.6, 0.3, 0.1, 0.0, 0.0, 0.0], dtype=np.float32)
    with pytest.raises(UnsupportedConfigurationError, match="fewer positive-probability"):
        encoder.step(StepContext(0, ReferenceDistribution(probabilities)))
    assert secret.position == 0


def test_candidate_outside_method_environment_is_rejected() -> None:
    environment = MethodEnvironment(output_vocab_size=5, allowed_token_ids=(0, 1, 2, 3))
    # P_reference itself is syntactically valid, but token 4 has the highest
    # mass and is not in V_allowed. Common infrastructure must normally prevent
    # this; the adapter still rejects a violated contract explicitly.
    reference = ReferenceDistribution(
        np.array([0.15, 0.15, 0.15, 0.15, 0.4], dtype=np.float32)
    )
    encoder = HuffmanMethod().create_encoder(
        config={"bits_per_word": 2},
        environment=environment,
        secret_source=FixedSecretSource((0, 0, 0)),
    )
    with pytest.raises(MethodError, match="outside V_allowed"):
        encoder.step(StepContext(0, reference))


def test_encoder_does_not_mutate_reference_distribution() -> None:
    reference = ref4_unequal()
    before = reference.probabilities.copy()
    encoder = HuffmanMethod().create_encoder(
        config={"bits_per_word": 2},
        environment=env4(),
        secret_source=FixedSecretSource((1, 0)),
    )
    encoder.step(StepContext(0, reference))
    assert np.array_equal(reference.probabilities, before)
    assert reference.probabilities.flags.writeable is False


def test_multi_step_roundtrip_and_payload_accounting_are_variable() -> None:
    # Use the deterministic benchmark secret source so the encoder can read as
    # many variable-length codes as needed without test-side padding rules.
    recorded = RecordingSecretSource(Shake256SecretSource("huffman-unit"))
    method = HuffmanMethod()
    encoder = method.create_encoder(
        config={"bits_per_word": 2},
        environment=env4(),
        secret_source=recorded,
    )

    references = [
        ReferenceDistribution(np.array([0.5, 0.25, 0.125, 0.125], dtype=np.float32)),
        ReferenceDistribution(np.array([0.125, 0.5, 0.125, 0.25], dtype=np.float32)),
        ReferenceDistribution(np.array([0.25, 0.125, 0.5, 0.125], dtype=np.float32)),
        ReferenceDistribution(np.array([0.125, 0.125, 0.25, 0.5], dtype=np.float32)),
        ReferenceDistribution(np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32)),
        ReferenceDistribution(np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)),
    ]
    contexts = [StepContext(i, ref) for i, ref in enumerate(references)]
    decisions = [encoder.step(context) for context in contexts]
    encoded = encoder.finalize()

    assert encoded.payload_bits == len(recorded.consumed_bits)
    assert encoded.payload_bits == sum(int(d.bits_consumed) for d in decisions)
    # Huffman is not a fixed-BPT method; at least two code lengths should occur
    # in this deterministic six-step synthetic run.
    assert len({d.bits_consumed for d in decisions}) >= 2

    decoder = method.create_decoder(
        config={"bits_per_word": 2},
        environment=env4(),
        expected_payload_bits=encoded.payload_bits,
    )
    incremental: list[int] = []
    for context, decision in zip(contexts, decisions):
        incremental.extend(
            decoder.observe(context, decision.token_id).recovered_bits
        )
    decoded = decoder.finalize()

    assert tuple(incremental) == recorded.consumed_bits
    assert decoded.recovered_bits == recorded.consumed_bits
    assert decoded.complete is True


def test_sessions_cannot_be_used_after_finalize() -> None:
    encoder = HuffmanMethod().create_encoder(
        config={"bits_per_word": 1},
        environment=env4(),
        secret_source=FixedSecretSource((0,)),
    )
    encoder.finalize()
    assert encoder.done is True
    with pytest.raises(SessionStateError):
        encoder.step(StepContext(0, ref4_unequal()))

    decoder = HuffmanMethod().create_decoder(
        config={"bits_per_word": 1},
        environment=env4(),
    )
    decoder.finalize()
    assert decoder.done is True
    with pytest.raises(SessionStateError):
        decoder.observe(StepContext(0, ref4_unequal()), 0)
