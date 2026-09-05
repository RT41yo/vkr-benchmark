from __future__ import annotations

import math

import numpy as np
import pytest

from vkr_benchmark.distributions import ReferenceDistribution
from vkr_benchmark.errors import MetricError
from vkr_benchmark.metrics import (
    compute_capacity_entropy_metrics,
    reference_entropy_bits,
)


def _reference(values: list[float]) -> ReferenceDistribution:
    return ReferenceDistribution(np.asarray(values, dtype=np.float32))


def test_uniform_four_token_reference_has_two_bits_entropy() -> None:
    assert reference_entropy_bits(_reference([0.25, 0.25, 0.25, 0.25])) == pytest.approx(2.0)


def test_zero_probability_entries_do_not_affect_entropy() -> None:
    assert reference_entropy_bits(_reference([0.5, 0.5, 0.0, 0.0])) == pytest.approx(1.0)


def test_manual_reference_entropy_example() -> None:
    assert reference_entropy_bits(_reference([0.5, 0.25, 0.25])) == pytest.approx(1.5)


def test_capacity_and_entropy_utilization_match_manual_example() -> None:
    metrics = compute_capacity_entropy_metrics(
        payload_bits=20,
        reference_entropies_bits=[2.5] * 10,
    )

    assert metrics.payload_bits == 20
    assert metrics.carrier_tokens == 10
    assert metrics.bits_per_token == pytest.approx(2.0)
    assert metrics.reference_entropy_mean_bits == pytest.approx(2.5)
    assert metrics.reference_entropy_sum_bits == pytest.approx(25.0)
    assert metrics.entropy_utilization == pytest.approx(0.8)
    assert metrics.entropy_utilization_percent == pytest.approx(80.0)


def test_zero_payload_is_valid_when_reference_entropy_is_positive() -> None:
    metrics = compute_capacity_entropy_metrics(
        payload_bits=0,
        reference_entropies_bits=[1.0, 2.0],
    )
    assert metrics.bits_per_token == 0.0
    assert metrics.entropy_utilization == 0.0
    assert metrics.entropy_utilization_percent == 0.0


def test_entropy_utilization_is_not_clipped_above_one() -> None:
    metrics = compute_capacity_entropy_metrics(
        payload_bits=4,
        reference_entropies_bits=[1.0, 1.0],
    )
    assert metrics.entropy_utilization == pytest.approx(2.0)
    assert metrics.entropy_utilization_percent == pytest.approx(200.0)


def test_metric_aggregation_uses_all_carrier_steps() -> None:
    metrics = compute_capacity_entropy_metrics(
        payload_bits=3,
        reference_entropies_bits=[0.5, 1.25, 2.25],
    )
    assert metrics.carrier_tokens == 3
    assert metrics.reference_entropy_sum_bits == pytest.approx(4.0)
    assert metrics.reference_entropy_mean_bits == pytest.approx(4.0 / 3.0)


def test_negative_payload_is_rejected() -> None:
    with pytest.raises(MetricError, match="payload_bits"):
        compute_capacity_entropy_metrics(
            payload_bits=-1,
            reference_entropies_bits=[1.0],
        )


def test_empty_entropy_sequence_is_rejected() -> None:
    with pytest.raises(MetricError, match="at least one"):
        compute_capacity_entropy_metrics(
            payload_bits=0,
            reference_entropies_bits=[],
        )


@pytest.mark.parametrize("bad_value", [-0.1, math.inf, -math.inf, math.nan])
def test_invalid_step_entropy_is_rejected(bad_value: float) -> None:
    with pytest.raises(MetricError):
        compute_capacity_entropy_metrics(
            payload_bits=1,
            reference_entropies_bits=[1.0, bad_value],
        )


def test_zero_total_reference_entropy_is_explicitly_rejected() -> None:
    with pytest.raises(MetricError, match="total reference entropy is zero"):
        compute_capacity_entropy_metrics(
            payload_bits=0,
            reference_entropies_bits=[0.0, 0.0],
        )
