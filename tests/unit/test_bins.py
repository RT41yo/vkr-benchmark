from __future__ import annotations

from dataclasses import dataclass

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
from vkr_benchmark.methods import BinsConfig, BinsMethod, MethodEnvironment
from vkr_benchmark.randomness import MethodRandomSource, SecretSource


class FixedSecretSource(SecretSource):
    def __init__(self, bits: tuple[int, ...]) -> None:
        self._bits = bits
        self._position = 0

    @property
    def position(self) -> int:
        return self._position

    def read_bits(self, count: int) -> tuple[int, ...]:
        stop = self._position + count
        if stop > len(self._bits):
            raise RuntimeError("test secret exhausted")
        out = self._bits[self._position:stop]
        self._position = stop
        return out


def env8() -> MethodEnvironment:
    return MethodEnvironment(output_vocab_size=8, allowed_token_ids=tuple(range(8)))


def reference8() -> ReferenceDistribution:
    return ReferenceDistribution(
        np.array([0.05, 0.10, 0.20, 0.15, 0.12, 0.08, 0.18, 0.12], dtype=np.float32)
    )


def test_bins_config_validation() -> None:
    assert BinsConfig.from_mapping({"block_size": 2}).num_bins == 4
    with pytest.raises(ConfigurationError):
        BinsConfig.from_mapping({})
    with pytest.raises(ConfigurationError):
        BinsConfig.from_mapping({"block_size": 0})
    with pytest.raises(ConfigurationError):
        BinsConfig.from_mapping({"block_size": True})
    with pytest.raises(ConfigurationError):
        BinsConfig.from_mapping({"block_size": 2, "extra": 1})


def test_encoder_and_decoder_build_identical_partition_from_matching_rng() -> None:
    method = BinsMethod()
    secret = FixedSecretSource((0, 0))
    encoder = method.create_encoder(
        config={"block_size": 2},
        environment=env8(),
        secret_source=secret,
        random_source=MethodRandomSource(123),
    )
    decoder = method.create_decoder(
        config={"block_size": 2},
        environment=env8(),
        random_source=MethodRandomSource(123),
    )
    assert encoder.partition.bins == decoder.partition.bins
    assert np.array_equal(encoder.partition.token_to_bin, decoder.partition.token_to_bin)


def test_different_rng_seed_changes_partition() -> None:
    method = BinsMethod()
    a = method.create_encoder(
        config={"block_size": 2},
        environment=env8(),
        secret_source=FixedSecretSource((0, 0)),
        random_source=MethodRandomSource(1),
    )
    b = method.create_encoder(
        config={"block_size": 2},
        environment=env8(),
        secret_source=FixedSecretSource((0, 0)),
        random_source=MethodRandomSource(2),
    )
    assert a.partition.bins != b.partition.bins


def test_partition_uses_harvard_floor_slice_shape() -> None:
    env = MethodEnvironment(output_vocab_size=10, allowed_token_ids=tuple(range(10)))
    encoder = BinsMethod().create_encoder(
        config={"block_size": 2},
        environment=env,
        secret_source=FixedSecretSource((0, 0)),
        random_source=MethodRandomSource(7),
    )
    assert [len(b) for b in encoder.partition.bins] == [2, 3, 2, 3]
    assert sorted(token for b in encoder.partition.bins for token in b) == list(range(10))


def test_bins_rejects_more_bins_than_allowed_tokens() -> None:
    with pytest.raises(UnsupportedConfigurationError, match="at least one allowed token"):
        BinsMethod().create_encoder(
            config={"block_size": 4},
            environment=env8(),
            secret_source=FixedSecretSource((0, 0, 0, 0)),
            random_source=MethodRandomSource(1),
        )


def test_bins_requires_injected_method_rng() -> None:
    with pytest.raises(ConfigurationError, match="MethodRandomSource"):
        BinsMethod().create_encoder(
            config={"block_size": 2},
            environment=env8(),
            secret_source=FixedSecretSource((0, 0)),
        )


def test_secret_bits_select_bin_and_highest_probability_member() -> None:
    method = BinsMethod()
    secret_bits = (1, 0)  # bin 2, MSB first
    encoder = method.create_encoder(
        config={"block_size": 2},
        environment=env8(),
        secret_source=FixedSecretSource(secret_bits),
        random_source=MethodRandomSource(123),
    )
    ref = reference8()
    decision = encoder.step(StepContext(0, ref))

    selected_bin = 2
    members = set(encoder.partition.bins[selected_bin])
    expected = next(int(t) for t in ref.token_order if int(t) in members)

    assert decision.token_id == expected
    assert decision.bits_consumed == 2
    assert decision.method_trace["selected_bin"] == 2
    assert decision.method_trace["secret_bits"] == secret_bits


def test_equal_probability_tie_uses_lower_token_id() -> None:
    method = BinsMethod()
    encoder = method.create_encoder(
        config={"block_size": 1},
        environment=MethodEnvironment(output_vocab_size=4, allowed_token_ids=(0, 1, 2, 3)),
        secret_source=FixedSecretSource((0,)),
        random_source=MethodRandomSource(11),
    )
    selected_members = encoder.partition.bins[0]
    # Give both members of selected bin the same highest probability. The
    # canonical token_order must then choose the lower token ID.
    probs = np.full(4, 0.1, dtype=np.float32)
    for token_id in selected_members:
        probs[token_id] = 0.4
    probs = probs / probs.sum(dtype=np.float64)
    ref = ReferenceDistribution(probs.astype(np.float32))
    decision = encoder.step(StepContext(0, ref))
    assert decision.token_id == min(selected_members)


def test_q_stego_is_exact_uniform_mass_over_bin_representatives() -> None:
    method = BinsMethod()
    encoder = method.create_encoder(
        config={"block_size": 2},
        environment=env8(),
        secret_source=FixedSecretSource((0, 1)),
        random_source=MethodRandomSource(123),
    )
    decision = encoder.step(StepContext(0, reference8()))
    info = decision.distribution_info

    assert info.mode == QMode.ANALYTIC_EXACT
    assert info.representation == QRepresentation.EXPLICIT_PROBABILITIES
    assert info.source == QSource.ADAPTER_EXACT
    assert info.probabilities is not None
    assert np.isclose(info.probabilities.sum(), 1.0)

    nz = np.flatnonzero(info.probabilities)
    assert len(nz) == 4
    assert np.all(info.probabilities[nz] == 0.25)
    for bin_id, members in enumerate(encoder.partition.bins):
        representatives = set(nz) & set(members)
        assert len(representatives) == 1, f"bin {bin_id} must contribute one representative"


def test_encoder_does_not_mutate_reference_distribution() -> None:
    ref = reference8()
    before = ref.probabilities.copy()
    encoder = BinsMethod().create_encoder(
        config={"block_size": 2},
        environment=env8(),
        secret_source=FixedSecretSource((1, 1)),
        random_source=MethodRandomSource(5),
    )
    encoder.step(StepContext(0, ref))
    assert np.array_equal(ref.probabilities, before)
    assert ref.probabilities.flags.writeable is False


def test_missing_positive_support_in_a_bin_fails_before_consuming_secret() -> None:
    secret = FixedSecretSource((1, 0))
    encoder = BinsMethod().create_encoder(
        config={"block_size": 2},
        environment=env8(),
        secret_source=secret,
        random_source=MethodRandomSource(123),
    )
    # Only one token has positive mass, so three of four fixed bins lack a
    # representative. This can happen if an overly aggressive common top-k is
    # combined with Bins.
    probs = np.zeros(8, dtype=np.float32)
    probs[0] = 1.0
    with pytest.raises(UnsupportedConfigurationError, match="no positive-probability"):
        encoder.step(StepContext(0, ReferenceDistribution(probs)))
    assert secret.position == 0


def test_decoder_maps_observed_token_back_to_fixed_width_bits() -> None:
    method = BinsMethod()
    decoder = method.create_decoder(
        config={"block_size": 2},
        environment=env8(),
        random_source=MethodRandomSource(123),
    )
    token = decoder.partition.bins[3][0]
    progress = decoder.observe(StepContext(0, reference8()), token)
    assert progress.recovered_bits == (1, 1)


def test_decoder_rejects_token_outside_allowed_set() -> None:
    env = MethodEnvironment(output_vocab_size=5, allowed_token_ids=(0, 1, 2, 3))
    decoder = BinsMethod().create_decoder(
        config={"block_size": 1},
        environment=env,
        random_source=MethodRandomSource(3),
    )
    with pytest.raises(MethodError, match="not in V_allowed"):
        decoder.observe(
            StepContext(0, ReferenceDistribution(np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32))),
            4,
        )


def test_synthetic_multi_step_roundtrip_and_payload_accounting() -> None:
    secret_bits = (0, 0, 1, 0, 1, 1, 0, 1)
    method = BinsMethod()
    encoder = method.create_encoder(
        config={"block_size": 2},
        environment=env8(),
        secret_source=FixedSecretSource(secret_bits),
        random_source=MethodRandomSource("shared-partition"),
    )
    contexts = [StepContext(i, reference8()) for i in range(4)]
    decisions = [encoder.step(ctx) for ctx in contexts]
    final_encode = encoder.finalize()

    decoder = method.create_decoder(
        config={"block_size": 2},
        environment=env8(),
        random_source=MethodRandomSource("shared-partition"),
        expected_payload_bits=final_encode.payload_bits,
    )
    incremental = []
    for ctx, decision in zip(contexts, decisions):
        incremental.extend(decoder.observe(ctx, decision.token_id).recovered_bits)
    final_decode = decoder.finalize()

    assert final_encode.payload_bits == 8
    assert tuple(incremental) == secret_bits
    assert final_decode.recovered_bits == secret_bits
    assert final_decode.complete is True


def test_session_cannot_be_used_after_finalize() -> None:
    encoder = BinsMethod().create_encoder(
        config={"block_size": 1},
        environment=env8(),
        secret_source=FixedSecretSource((0,)),
        random_source=MethodRandomSource(1),
    )
    encoder.finalize()
    assert encoder.done is True
    with pytest.raises(SessionStateError):
        encoder.step(StepContext(0, reference8()))
