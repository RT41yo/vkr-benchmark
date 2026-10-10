from __future__ import annotations

import numpy as np
import pytest

from vkr_benchmark.distributions import (
    QMode,
    QRepresentation,
    QSource,
    ReferenceDistribution,
    StepContext,
    allocate_dair_intervals,
    explicit_dair_q,
)
from vkr_benchmark.errors import ConfigurationError
from vkr_benchmark.methods import DAIRMethod, MethodEnvironment
from vkr_benchmark.randomness import SecretSource


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


def _env(size: int) -> MethodEnvironment:
    return MethodEnvironment(output_vocab_size=size, allowed_token_ids=tuple(range(size)))


def test_dair_known_roulette_and_singleton_prefix_roundtrip() -> None:
    reference = ReferenceDistribution(
        np.asarray([0.5, 0.25, 0.125, 0.125], dtype=np.float32)
    )
    context = StepContext(0, reference)
    intervals = allocate_dair_intervals(reference, alpha=3, beta=1.0, top_k=4)
    assert [(x.token_id, x.start, x.stop) for x in intervals] == [
        (0, 0, 4),
        (1, 4, 6),
        (2, 6, 7),
        (3, 7, 8),
    ]

    # 110 selects the singleton interval [6, 7). Publication semantics recover
    # the full three-bit common prefix. The pinned helper returns only two bits
    # for this special case; direct author conformance characterizes that later.
    secret = _Bits((1, 1, 0, 0, 1, 1, 1, 0))
    encoder = DAIRMethod().create_encoder(
        config={"alpha": 3, "beta": 1.0, "top_k": 4},
        environment=_env(4),
        secret_source=secret,
    )
    decision = encoder.step(context)
    assert decision.token_id == 2
    assert decision.bits_consumed == 3
    assert decision.method_trace["common_prefix_bits"] == (1, 1, 0)

    decoder = DAIRMethod().create_decoder(
        config={"alpha": 3, "beta": 1.0, "top_k": 4},
        environment=_env(4),
        expected_payload_bits=3,
    )
    progress = decoder.observe(context, decision.token_id)
    assert progress.recovered_bits == (1, 1, 0)
    assert decoder.finalize().recovered_bits == (1, 1, 0)


def test_dair_overlapping_window_state_changes_exact_conditional_q() -> None:
    reference = ReferenceDistribution(
        np.asarray([0.375, 0.375, 0.25], dtype=np.float32)
    )
    q0 = explicit_dair_q(reference, alpha=3, beta=1.0, top_k=3)
    assert q0 == pytest.approx([3 / 8, 3 / 8, 2 / 8], abs=1e-15)

    # If the first emitted token selected [0, 3), its common prefix length is
    # one bit. Reusing the two-bit suffix and appending one fresh bit makes the
    # next alpha-bit window uniform on [0, 5], not on the whole [0, 7].
    q1 = explicit_dair_q(
        reference,
        alpha=3,
        beta=1.0,
        top_k=3,
        window_lower=0,
        window_upper=5,
    )
    assert q1 == pytest.approx([0.5, 0.5, 0.0], abs=1e-15)


def test_dair_distribution_info_is_independent_analytic_and_state_aware() -> None:
    reference = ReferenceDistribution(
        np.asarray([0.375, 0.375, 0.25], dtype=np.float32)
    )
    environment = _env(3)
    encoder = DAIRMethod().create_encoder(
        config={"alpha": 3, "beta": 1.0, "top_k": 3},
        environment=environment,
        secret_source=_Bits((0, 0, 1, 1, 0, 1, 1, 1, 0)),
    )

    context0 = StepContext(0, reference)
    decision0 = encoder.step(context0)
    assert decision0.token_id == 0
    assert decision0.bits_consumed == 1
    info0 = encoder.distribution_info(context0, decision0)
    assert info0.representation == QRepresentation.EXPLICIT_PROBABILITIES
    assert info0.mode == QMode.ANALYTIC_EXACT
    assert info0.source == QSource.INDEPENDENT_ANALYTIC
    assert info0.probabilities == pytest.approx([3 / 8, 3 / 8, 2 / 8], abs=1e-15)

    context1 = StepContext(1, reference)
    decision1 = encoder.step(context1)
    info1 = encoder.distribution_info(context1, decision1)
    assert info1.probabilities == pytest.approx([0.5, 0.5, 0.0], abs=1e-15)


def test_dair_rejects_invalid_params_randomness_and_fixed_payload() -> None:
    environment = _env(4)
    secret = _Bits((0,) * 32)
    method = DAIRMethod()

    with pytest.raises(ConfigurationError, match="missing DAIRstega"):
        method.create_encoder(config={}, environment=environment, secret_source=secret)
    with pytest.raises(ConfigurationError, match="beta"):
        method.create_encoder(
            config={"alpha": 8, "beta": 0.0, "top_k": 4},
            environment=environment,
            secret_source=secret,
        )
    with pytest.raises(ConfigurationError, match="random_seed"):
        method.create_encoder(
            config={"alpha": 8, "beta": 1.0, "top_k": 4},
            environment=environment,
            secret_source=secret,
            random_source=object(),
        )
    with pytest.raises(ConfigurationError, match="fixed_carrier_tokens"):
        method.create_encoder(
            config={"alpha": 8, "beta": 1.0, "top_k": 4},
            environment=environment,
            secret_source=secret,
            target_payload_bits=8,
        )
