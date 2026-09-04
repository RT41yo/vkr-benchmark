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
from vkr_benchmark.methods import ArithmeticConfig, ArithmeticMethod, MethodEnvironment
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


def ref4_dyadic() -> ReferenceDistribution:
    return ReferenceDistribution(
        np.array([0.5, 0.25, 0.125, 0.125], dtype=np.float32)
    )


def test_config_defaults_and_validation() -> None:
    config = ArithmeticConfig.from_mapping({"precision": 16})
    assert config.precision == 16
    assert config.top_k == 50_000
    assert config.max_value == 65_536

    with pytest.raises(ConfigurationError, match="requires precision"):
        ArithmeticConfig.from_mapping({})
    with pytest.raises(ConfigurationError, match="unknown Arithmetic"):
        ArithmeticConfig.from_mapping({"precision": 16, "temperature": 1.0})
    with pytest.raises(ConfigurationError, match="must be an integer"):
        ArithmeticConfig.from_mapping({"precision": True})
    with pytest.raises(ConfigurationError, match=r"\[2, 52\]"):
        ArithmeticConfig.from_mapping({"precision": 1})
    with pytest.raises(ConfigurationError, match="top_k must be >= 2"):
        ArithmeticConfig.from_mapping({"precision": 8, "top_k": 1})


def test_arithmetic_does_not_require_method_rng() -> None:
    # precision=4 causes an initial 4-bit look-ahead read.
    encoder = ArithmeticMethod().create_encoder(
        config={"precision": 4},
        environment=env4(),
        secret_source=FixedSecretSource((0, 0, 0, 0, 0, 0, 0, 0)),
    )
    decision = encoder.step(StepContext(0, ref4_dyadic()))
    assert decision.token_id == 0


def test_known_dyadic_partition_selects_point_and_confirms_prefix() -> None:
    # precision=4 => [0,16).  The dyadic reference maps the four tokens to
    # [0,8), [8,12), [12,14), [14,16).  Secret look-ahead 1100 = 12 selects
    # token 2 and fixes the common prefix 110 (three bits).
    secret = FixedSecretSource((1, 1, 0, 0, 1, 0, 1, 1, 0, 0))
    encoder = ArithmeticMethod().create_encoder(
        config={"precision": 4},
        environment=env4(),
        secret_source=secret,
    )
    decision = encoder.step(StepContext(0, ref4_dyadic()))

    assert decision.token_id == 2
    assert decision.bits_consumed == 3
    assert decision.method_trace["selected_rank"] == 2
    assert decision.method_trace["selected_interval"] == (12, 14)
    assert decision.method_trace["confirmed_prefix"] == (1, 1, 0)
    assert decision.method_trace["interval_after"] == (0, 16)


def test_q_stego_is_exact_integer_width_distribution() -> None:
    secret = FixedSecretSource((0,) * 12)
    encoder = ArithmeticMethod().create_encoder(
        config={"precision": 4},
        environment=env4(),
        secret_source=secret,
    )
    decision = encoder.step(StepContext(0, ref4_dyadic()))
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


def test_finite_precision_rounding_can_change_q() -> None:
    reference = ReferenceDistribution(
        np.array([0.60, 0.25, 0.10, 0.05], dtype=np.float32)
    )
    encoder = ArithmeticMethod().create_encoder(
        config={"precision": 4},
        environment=env4(),
        secret_source=FixedSecretSource((0,) * 12),
    )
    q = encoder.step(StepContext(0, reference)).distribution_info.probabilities
    assert q is not None
    # threshold=1/16 excludes token 3; the first three probabilities are
    # renormalized and rounded to widths 10,4,2.
    assert np.array_equal(q, np.array([0.625, 0.25, 0.125, 0.0]))
    assert not np.allclose(q, reference.probabilities, rtol=0.0, atol=1e-12)


def test_internal_top_k_caps_candidates_after_p_reference() -> None:
    reference = ReferenceDistribution(
        np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32)
    )
    encoder = ArithmeticMethod().create_encoder(
        config={"precision": 4, "top_k": 2},
        environment=env4(),
        secret_source=FixedSecretSource((0,) * 12),
    )
    decision = encoder.step(StepContext(0, reference))
    q = decision.distribution_info.probabilities
    assert q is not None
    assert set(np.flatnonzero(q)) == {0, 1}
    assert decision.method_trace["candidate_count"] == 2
    assert np.isclose(q.sum(), 1.0)


def test_rounding_overfill_removes_bottom_candidate_like_reference() -> None:
    environment = MethodEnvironment(output_vocab_size=3, allowed_token_ids=(0, 1, 2))
    reference = ReferenceDistribution(
        np.array([0.34, 0.33, 0.33], dtype=np.float32)
    )
    encoder = ArithmeticMethod().create_encoder(
        config={"precision": 3},  # interval width 8
        environment=environment,
        secret_source=FixedSecretSource((0,) * 12),
    )
    decision = encoder.step(StepContext(0, reference))
    q = decision.distribution_info.probabilities
    assert q is not None
    # Rounded widths are initially 3,3,3 => cumulative 9 > 8 at candidate 3.
    # The reference truncates before the overfill and assigns residual 2 to the
    # top candidate, yielding widths 5,3.
    assert np.array_equal(q, np.array([0.625, 0.375, 0.0]))
    assert decision.method_trace["candidate_count"] == 2


def test_decoder_recovers_same_confirmed_prefix() -> None:
    decoder = ArithmeticMethod().create_decoder(
        config={"precision": 4},
        environment=env4(),
    )
    progress = decoder.observe(StepContext(0, ref4_dyadic()), 2)
    assert progress.recovered_bits == (1, 1, 0)
    assert progress.method_trace["selected_interval"] == (12, 14)
    assert progress.method_trace["interval_after"] == (0, 16)


def test_state_is_carried_between_steps_when_no_prefix_is_fixed() -> None:
    # With these finite-precision masses token 1 occupies an interval crossing
    # the leading-bit boundary.  Selecting it fixes zero bits, so the narrowed
    # interval itself must survive into the next step.
    reference = ReferenceDistribution(
        np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32)
    )
    # First four bits 0111 = 7 select the second interval.
    secret = FixedSecretSource((0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1))
    encoder = ArithmeticMethod().create_encoder(
        config={"precision": 4},
        environment=env4(),
        secret_source=secret,
    )
    first = encoder.step(StepContext(0, reference))
    assert first.token_id == 1
    assert first.bits_consumed == 0
    assert first.method_trace["interval_before"] == (0, 16)
    narrowed = first.method_trace["interval_after"]
    assert narrowed != (0, 16)

    second = encoder.step(StepContext(1, reference))
    assert second.method_trace["interval_before"] == narrowed


def test_secret_lookahead_is_not_counted_as_payload() -> None:
    recorded = RecordingSecretSource(Shake256SecretSource("arithmetic-lookahead"))
    encoder = ArithmeticMethod().create_encoder(
        config={"precision": 8},
        environment=env4(),
        secret_source=recorded,
    )
    decisions = [
        encoder.step(StepContext(i, ref4_dyadic()))
        for i in range(4)
    ]
    finalization = encoder.finalize()

    payload = sum(int(decision.bits_consumed) for decision in decisions)
    assert finalization.payload_bits == payload
    assert finalization.metadata["secret_bits_read"] == 8 + payload
    assert len(recorded.consumed_bits) == 8 + payload
    assert finalization.payload_bits < len(recorded.consumed_bits)


def test_multi_step_encoder_decoder_roundtrip_uses_confirmed_payload_prefix() -> None:
    recorded = RecordingSecretSource(Shake256SecretSource("arithmetic-roundtrip"))
    method = ArithmeticMethod()
    encoder = method.create_encoder(
        config={"precision": 8, "top_k": 4},
        environment=env4(),
        secret_source=recorded,
    )

    references = [
        ReferenceDistribution(np.array([0.50, 0.25, 0.125, 0.125], dtype=np.float32)),
        ReferenceDistribution(np.array([0.10, 0.40, 0.30, 0.20], dtype=np.float32)),
        ReferenceDistribution(np.array([0.25, 0.25, 0.30, 0.20], dtype=np.float32)),
        ReferenceDistribution(np.array([0.15, 0.20, 0.25, 0.40], dtype=np.float32)),
        ReferenceDistribution(np.array([0.35, 0.15, 0.30, 0.20], dtype=np.float32)),
        ReferenceDistribution(np.array([0.20, 0.35, 0.15, 0.30], dtype=np.float32)),
    ]
    contexts = [StepContext(i, ref) for i, ref in enumerate(references)]
    decisions = [encoder.step(context) for context in contexts]
    encoded = encoder.finalize()

    expected = recorded.consumed_bits[: encoded.payload_bits]
    assert encoded.payload_bits == sum(int(d.bits_consumed) for d in decisions)

    decoder = method.create_decoder(
        config={"precision": 8, "top_k": 4},
        environment=env4(),
        expected_payload_bits=encoded.payload_bits,
    )
    incremental: list[int] = []
    for context, decision in zip(contexts, decisions, strict=True):
        incremental.extend(
            decoder.observe(context, decision.token_id).recovered_bits
        )
    decoded = decoder.finalize()

    assert tuple(incremental) == expected
    assert decoded.recovered_bits == expected
    assert decoded.complete is True
    assert decoder.interval == encoder.interval


def test_candidate_outside_method_environment_is_rejected() -> None:
    environment = MethodEnvironment(output_vocab_size=5, allowed_token_ids=(0, 1, 2, 3))
    reference = ReferenceDistribution(
        np.array([0.15, 0.15, 0.15, 0.15, 0.40], dtype=np.float32)
    )
    encoder = ArithmeticMethod().create_encoder(
        config={"precision": 4},
        environment=environment,
        secret_source=FixedSecretSource((0,) * 12),
    )
    with pytest.raises(MethodError, match="outside Arithmetic V_allowed"):
        encoder.step(StepContext(0, reference))


def test_requires_at_least_two_positive_probability_tokens() -> None:
    probabilities = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    encoder = ArithmeticMethod().create_encoder(
        config={"precision": 4},
        environment=env4(),
        secret_source=FixedSecretSource((0,) * 12),
    )
    with pytest.raises(UnsupportedConfigurationError, match="at least two"):
        encoder.step(StepContext(0, ReferenceDistribution(probabilities)))


def test_decoder_rejects_token_outside_current_candidate_set() -> None:
    reference = ReferenceDistribution(
        np.array([0.60, 0.25, 0.10, 0.05], dtype=np.float32)
    )
    decoder = ArithmeticMethod().create_decoder(
        config={"precision": 4},
        environment=env4(),
    )
    # token 3 is below the finite-precision threshold at width 16.
    with pytest.raises(MethodError, match="outside the current Arithmetic candidate set"):
        decoder.observe(StepContext(0, reference), 3)


def test_encoder_does_not_mutate_reference_distribution() -> None:
    reference = ref4_dyadic()
    before = reference.probabilities.copy()
    encoder = ArithmeticMethod().create_encoder(
        config={"precision": 4},
        environment=env4(),
        secret_source=FixedSecretSource((0,) * 12),
    )
    encoder.step(StepContext(0, reference))
    assert np.array_equal(reference.probabilities, before)
    assert reference.probabilities.flags.writeable is False


def test_sessions_cannot_be_used_after_finalize() -> None:
    encoder = ArithmeticMethod().create_encoder(
        config={"precision": 4},
        environment=env4(),
        secret_source=FixedSecretSource((0,) * 12),
    )
    encoder.finalize()
    assert encoder.done is True
    with pytest.raises(SessionStateError):
        encoder.step(StepContext(0, ref4_dyadic()))

    decoder = ArithmeticMethod().create_decoder(
        config={"precision": 4},
        environment=env4(),
    )
    decoder.finalize()
    assert decoder.done is True
    with pytest.raises(SessionStateError):
        decoder.observe(StepContext(0, ref4_dyadic()), 0)
