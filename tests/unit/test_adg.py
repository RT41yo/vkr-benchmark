from __future__ import annotations

import numpy as np
import pytest

from vkr_benchmark.distributions import (
    QMode,
    QRepresentation,
    QSource,
    ReferenceDistribution,
    StepContext,
    explicit_adg_q,
)
from vkr_benchmark.errors import ConfigurationError
from vkr_benchmark.methods import ADGMethod, MethodEnvironment
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
        np.asarray([0.34, 0.22, 0.18, 0.12, 0.08, 0.06], dtype=np.float32)
    )
    environment = MethodEnvironment(
        output_vocab_size=6,
        allowed_token_ids=(0, 1, 2, 3, 4, 5),
    )
    return reference, environment


def test_adg_roundtrip_recovers_little_endian_group_bits() -> None:
    reference, environment = _fixture()
    secret = (1, 0, 1, 1, 0, 0, 1, 0)
    encoder = ADGMethod().create_encoder(
        config={},
        environment=environment,
        secret_source=_Bits(secret),
        random_source=MethodRandomSource(42),
    )
    decision = encoder.step(StepContext(0, reference))

    assert decision.bits_consumed == 2
    assert decision.method_trace["selected_secret_bits"] == (1, 0)
    assert decision.method_trace["group_path"] == (1, 0)

    decoder = ADGMethod().create_decoder(
        config={},
        environment=environment,
        expected_payload_bits=decision.bits_consumed,
    )
    progress = decoder.observe(StepContext(0, reference), decision.token_id)
    assert progress.recovered_bits == (1, 0)
    final = decoder.finalize()
    assert final.complete is True
    assert final.recovered_bits == (1, 0)


def test_adg_explicit_q_is_independently_enumerated_and_nontrivial() -> None:
    reference, environment = _fixture()
    q = explicit_adg_q(reference)
    p = reference.probabilities.astype(np.float64)
    p /= p.sum(dtype=np.float64)

    assert q.sum(dtype=np.float64) == pytest.approx(1.0, abs=1e-12)
    assert np.array_equal(q > 0.0, p > 0.0)
    assert np.max(np.abs(q - p)) > 1e-3

    encoder = ADGMethod().create_encoder(
        config={},
        environment=environment,
        secret_source=_Bits((0,) * 16),
        random_source=MethodRandomSource(7),
    )
    context = StepContext(0, reference)
    decision = encoder.step(context)
    info = encoder.distribution_info(context, decision)
    assert info.representation == QRepresentation.EXPLICIT_PROBABILITIES
    assert info.mode == QMode.EXACT_ENUMERATION
    assert info.source == QSource.INDEPENDENT_ENUMERATION
    assert np.allclose(info.probabilities, q, rtol=0.0, atol=1e-15)


def test_adg_terminal_sampling_consumes_no_bits_when_pmax_above_half() -> None:
    reference = ReferenceDistribution(
        np.asarray([0.6, 0.25, 0.15], dtype=np.float32)
    )
    environment = MethodEnvironment(
        output_vocab_size=3,
        allowed_token_ids=(0, 1, 2),
    )
    encoder = ADGMethod().create_encoder(
        config={},
        environment=environment,
        secret_source=_Bits((1, 0, 1)),
        random_source=MethodRandomSource(1),
    )
    decision = encoder.step(StepContext(0, reference))
    assert decision.bits_consumed == 0
    assert decision.method_trace["recursion_depth"] == 0

    q = explicit_adg_q(reference)
    p = reference.probabilities.astype(np.float64)
    p /= p.sum(dtype=np.float64)
    assert np.allclose(q, p, rtol=0.0, atol=1e-15)


def test_adg_requires_method_random_source_and_rejects_method_params() -> None:
    _, environment = _fixture()
    with pytest.raises(ConfigurationError, match="MethodRandomSource"):
        ADGMethod().create_encoder(
            config={},
            environment=environment,
            secret_source=_Bits((0, 1)),
        )
    with pytest.raises(ConfigurationError, match="unknown ADG"):
        ADGMethod().create_encoder(
            config={"top_k": 100},
            environment=environment,
            secret_source=_Bits((0, 1)),
            random_source=MethodRandomSource(1),
        )


def test_adg_rejects_fixed_payload_lifecycle() -> None:
    _, environment = _fixture()
    with pytest.raises(ConfigurationError, match="fixed_carrier_tokens"):
        ADGMethod().create_encoder(
            config={},
            environment=environment,
            secret_source=_Bits((0, 1)),
            random_source=MethodRandomSource(1),
            target_payload_bits=2,
        )
