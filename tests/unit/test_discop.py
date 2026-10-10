from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from vkr_benchmark.distributions import (
    QMode,
    QRepresentation,
    QSource,
    ReferenceDistribution,
    StepContext,
)
from vkr_benchmark.errors import ConfigurationError, MethodError
from vkr_benchmark.methods import DiscopMethod, MethodEnvironment
from vkr_benchmark.randomness import RandomSource, SecretSource


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


class _ScriptedRandom(RandomSource):
    def __init__(self, values: tuple[float, ...]) -> None:
        self._values = deque(values)
        self.draws = 0

    def random(self) -> float:
        if not self._values:
            raise AssertionError("scripted RNG exhausted")
        self.draws += 1
        return self._values.popleft()

    def randbelow(self, upper: int) -> int:
        raise AssertionError(f"randbelow({upper}) is not used by Discop")


def _context(probabilities: list[float]) -> tuple[StepContext, MethodEnvironment]:
    probs = np.asarray(probabilities, dtype=np.float32)
    reference = ReferenceDistribution(probs)
    environment = MethodEnvironment(
        output_vocab_size=len(probabilities),
        allowed_token_ids=tuple(range(len(probabilities))),
    )
    return StepContext(step_index=0, reference=reference), environment




def test_explicit_q_is_metric_instrumentation_not_part_of_encoder_step(monkeypatch) -> None:
    context, environment = _context([0.4, 0.3, 0.2, 0.1])
    encoder = DiscopMethod().create_encoder(
        config={},
        environment=environment,
        secret_source=_Bits((1, 0)),
        random_source=_ScriptedRandom((0.1, 0.7)),
    )

    import vkr_benchmark.methods.discop as discop_module

    def fail_if_called(_reference):
        raise AssertionError("explicit Q must not be constructed inside encoder.step()")

    monkeypatch.setattr(discop_module, "explicit_discop_q", fail_if_called)
    decision = encoder.step(context)
    assert decision.token_id == 1
    with pytest.raises(AssertionError, match="explicit Q"):
        encoder.distribution_info(context, decision)

def test_discop_matches_pinned_reference_step_fixture() -> None:
    # Pinned create_huffman_tree for [0.4, 0.3, 0.2, 0.1] builds:
    # root: token0 | ((token3, token2), token1).
    # r=0.1 makes copy 0 go left and copy 1 go right -> secret bit 1.
    # At the right child r=0.7 makes copy 0 go right and copy 1 go left
    # -> secret bit 0 selects token1.  The pinned encoder therefore consumes
    # exactly "10" and emits token 1.
    context, environment = _context([0.4, 0.3, 0.2, 0.1])
    encoder_rng = _ScriptedRandom((0.1, 0.7))
    encoder = DiscopMethod().create_encoder(
        config={},
        environment=environment,
        secret_source=_Bits((1, 0)),
        random_source=encoder_rng,
    )

    decision = encoder.step(context)
    assert decision.token_id == 1
    assert decision.bits_consumed == 2
    assert encoder_rng.draws == 2
    info = encoder.distribution_info(context, decision)
    assert info.representation == QRepresentation.EXPLICIT_PROBABILITIES
    assert info.mode == QMode.EXACT_ENUMERATION
    assert info.source == QSource.INDEPENDENT_ENUMERATION
    assert info.probabilities is not None
    expected_q = context.reference.probabilities.astype(np.float64)
    expected_q /= expected_q.sum(dtype=np.float64)
    assert np.allclose(
        info.probabilities, expected_q, rtol=0.0, atol=1e-14
    )

    decoder_rng = _ScriptedRandom((0.1, 0.7))
    decoder = DiscopMethod().create_decoder(
        config={},
        environment=environment,
        random_source=decoder_rng,
        expected_payload_bits=2,
    )
    progress = decoder.observe(context, decision.token_id)
    assert progress.recovered_bits == (1, 0)
    assert decoder_rng.draws == 2
    assert decoder.finalize().recovered_bits == (1, 0)


def test_discop_can_emit_a_token_without_consuming_payload() -> None:
    context, environment = _context([0.9, 0.1])
    encoder = DiscopMethod().create_encoder(
        config={},
        environment=environment,
        secret_source=_Bits(()),
        random_source=_ScriptedRandom((0.2,)),
    )
    decision = encoder.step(context)
    assert decision.token_id == 0
    assert decision.bits_consumed == 0
    assert encoder.finalize().payload_bits == 0

    decoder = DiscopMethod().create_decoder(
        config={},
        environment=environment,
        random_source=_ScriptedRandom((0.2,)),
        expected_payload_bits=0,
    )
    assert decoder.observe(context, 0).recovered_bits == ()
    final = decoder.finalize()
    assert final.recovered_bits == ()
    assert final.complete is True


def test_discop_rejects_missing_rng_and_unknown_params() -> None:
    _, environment = _context([0.6, 0.4])
    with pytest.raises(ConfigurationError, match="synchronized method randomness"):
        DiscopMethod().create_encoder(
            config={}, environment=environment, secret_source=_Bits((0,))
        )
    with pytest.raises(ConfigurationError, match="unknown Discop"):
        DiscopMethod().create_encoder(
            config={"unexpected": 1},
            environment=environment,
            secret_source=_Bits((0,)),
            random_source=_ScriptedRandom((0.1,)),
        )


def test_discop_decoder_rejects_impossible_observed_branch() -> None:
    context, environment = _context([0.9, 0.1])
    decoder = DiscopMethod().create_decoder(
        config={},
        environment=environment,
        random_source=_ScriptedRandom((0.2,)),
        expected_payload_bits=0,
    )
    # Both copies select the high-probability left branch for r=0.2, so seeing
    # token 1 means sender and receiver are not synchronized.
    with pytest.raises(MethodError, match="state mismatch"):
        decoder.observe(context, 1)


def test_discop_empirical_distribution_matches_reference_independently() -> None:
    from vkr_benchmark.randomness import MethodRandomSource, Shake256SecretSource

    context, environment = _context([0.4, 0.3, 0.2, 0.1])
    encoder = DiscopMethod().create_encoder(
        config={},
        environment=environment,
        secret_source=Shake256SecretSource("stage4-discop-q-validation"),
        random_source=MethodRandomSource(1234),
    )

    counts = np.zeros(4, dtype=np.int64)
    samples = 20_000
    for step_index in range(samples):
        decision = encoder.step(
            StepContext(step_index=step_index, reference=context.reference)
        )
        counts[decision.token_id] += 1

    empirical = counts.astype(np.float64) / samples
    target = context.reference.probabilities.astype(np.float64)
    tvd = 0.5 * float(np.abs(empirical - target).sum())

    # This Monte Carlo check is independent of the exact Q_stego calculator.
    # The fixed seeds make the test non-flaky; the observed TVD is ~0.002 for
    # this fixture.
    assert tvd < 0.01
