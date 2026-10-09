from __future__ import annotations

import numpy as np
import pytest

from vkr_benchmark.distributions import (
    QMode,
    QRepresentation,
    QSource,
    ReferenceDistribution,
    StepContext,
    explicit_rrc_q,
)
from vkr_benchmark.errors import ConfigurationError, SessionStateError
from vkr_benchmark.methods import MethodEnvironment, RRCMethod, rrc_required_precision
from vkr_benchmark.randomness import MethodRandomSource, SecretSource


class _Bits(SecretSource):
    def __init__(self, bits: tuple[int, ...]) -> None:
        self._bits = bits
        self._position = 0

    @property
    def position(self) -> int:
        return self._position

    def read_bits(self, count: int) -> tuple[int, ...]:
        stop = self._position + count
        if stop > len(self._bits):
            raise AssertionError("test secret exhausted")
        out = self._bits[self._position:stop]
        self._position = stop
        return out


def _fixture():
    reference = ReferenceDistribution(
        np.asarray([0.4, 0.3, 0.2, 0.1], dtype=np.float32)
    )
    environment = MethodEnvironment(
        output_vocab_size=4,
        allowed_token_ids=(0, 1, 2, 3),
    )
    return reference, environment


def test_rrc_precision_scales_with_payload_length() -> None:
    assert rrc_required_precision(128, 40) == 79
    assert rrc_required_precision(256, 40) == 118
    assert rrc_required_precision(8192, 40) == 2507


def test_rrc_explicit_q_is_interval_measure_not_equality_certificate() -> None:
    reference, _ = _fixture()
    q = explicit_rrc_q(reference)
    p = reference.probabilities.astype(np.float64)
    p /= p.sum(dtype=np.float64)
    assert q.sum(dtype=np.float64) == pytest.approx(1.0, abs=1e-15)
    assert np.allclose(q, p, rtol=0.0, atol=1e-15)


def test_rrc_fixed_message_roundtrip_on_repeated_distribution() -> None:
    reference, environment = _fixture()
    bits = (1, 0, 0, 0, 0, 0, 0, 1)
    encoder = RRCMethod().create_encoder(
        config={"precision_margin": 40, "verify_termination": True},
        environment=environment,
        secret_source=_Bits(bits),
        random_source=MethodRandomSource(42),
        target_payload_bits=len(bits),
    )

    tokens: list[int] = []
    decisions = []
    for step in range(32):
        context = StepContext(step, reference)
        decision = encoder.step(context)
        tokens.append(decision.token_id)
        decisions.append((context, decision))
        if encoder.done:
            break

    assert encoder.done is True
    assert tokens == [0, 0, 0, 2]
    assert all(decision.bits_consumed is None for _, decision in decisions)
    final = encoder.finalize()
    assert final.payload_bits == len(bits)
    assert final.termination_reason == "fixed_payload_bits"
    assert final.metadata["precision_digits"] == 43
    assert final.metadata["termination_retries"] == 0

    info = encoder.distribution_info(*decisions[0])
    assert info.representation == QRepresentation.EXPLICIT_PROBABILITIES
    assert info.mode == QMode.ANALYTIC_EXACT
    assert info.source == QSource.INDEPENDENT_ANALYTIC

    decoder = RRCMethod().create_decoder(
        config={"precision_margin": 40, "verify_termination": True},
        environment=environment,
        random_source=MethodRandomSource(42),
        expected_payload_bits=len(bits),
    )
    incremental = []
    for step, token_id in enumerate(tokens):
        progress = decoder.observe(StepContext(step, reference), token_id)
        incremental.extend(progress.recovered_bits)
    assert incremental == []
    recovered = decoder.finalize()
    assert recovered.complete is True
    assert recovered.recovered_bits == bits


def test_rrc_requires_fixed_payload_and_synchronized_randomness() -> None:
    _, environment = _fixture()
    with pytest.raises(ConfigurationError, match="target_payload_bits"):
        RRCMethod().create_encoder(
            config={},
            environment=environment,
            secret_source=_Bits((0, 1)),
            random_source=MethodRandomSource(1),
        )
    with pytest.raises(ConfigurationError, match="synchronized method randomness"):
        RRCMethod().create_encoder(
            config={},
            environment=environment,
            secret_source=_Bits((0, 1)),
            target_payload_bits=2,
        )


def test_rrc_rejects_method_internal_top_k() -> None:
    _, environment = _fixture()
    with pytest.raises(ConfigurationError, match="unknown RRC"):
        RRCMethod().create_encoder(
            config={"top_k": 100},
            environment=environment,
            secret_source=_Bits((0, 1)),
            random_source=MethodRandomSource(1),
            target_payload_bits=2,
        )


def test_rrc_decoder_reports_text_transport_support_failure_without_crashing() -> None:
    reference, environment = _fixture()
    decoder = RRCMethod().create_decoder(
        config={},
        environment=environment,
        random_source=MethodRandomSource(42),
        expected_payload_bits=8,
    )
    # Token 4 is within MethodEnvironment vocabulary only if the environment is
    # changed; here it is deliberately outside the current P_reference support.
    progress = decoder.observe(StepContext(0, reference), 99)
    assert progress.recovered_bits == ()
    final = decoder.finalize()
    assert final.complete is False
    assert final.recovered_bits == ()
    assert final.metadata["failure_step"] == 0


def test_rrc_encoder_cannot_finalize_before_termination() -> None:
    reference, environment = _fixture()
    encoder = RRCMethod().create_encoder(
        config={},
        environment=environment,
        secret_source=_Bits((1,) * 16),
        random_source=MethodRandomSource(7),
        target_payload_bits=16,
    )
    encoder.step(StepContext(0, reference))
    if not encoder.done:
        with pytest.raises(SessionStateError, match="before termination"):
            encoder.finalize()


def test_rrc_explicit_q_matches_random_interval_partitions() -> None:
    rng = np.random.default_rng(20261009)
    max_abs = 0.0
    for vocab_size in (2, 3, 7, 31):
        for _ in range(32):
            raw = rng.random(vocab_size)
            # Include exact support zeros in some fixtures to exercise the same
            # support semantics used after top-k/top-p.
            if vocab_size > 3:
                raw[rng.integers(0, vocab_size)] = 0.0
            raw /= raw.sum()
            reference = ReferenceDistribution(raw.astype(np.float32))
            q = explicit_rrc_q(reference)
            p = reference.probabilities.astype(np.float64)
            p /= p.sum(dtype=np.float64)
            max_abs = max(max_abs, float(np.max(np.abs(q - p))))
            assert np.array_equal(q > 0.0, p > 0.0)
            assert np.allclose(q, p, rtol=0.0, atol=2e-15)
    assert max_abs < 2e-15
