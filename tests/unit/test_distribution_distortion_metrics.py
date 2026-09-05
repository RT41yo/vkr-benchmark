from __future__ import annotations

import math

import numpy as np
import pytest

from vkr_benchmark.distributions import (
    DistributionInfo,
    QMode,
    QSource,
    ReferenceDistribution,
)
from vkr_benchmark.errors import MetricError
from vkr_benchmark.metrics import (
    StepDistributionDistortion,
    compute_distribution_distortion_metrics,
    distribution_distortion_step,
)


def _reference(values: list[float]) -> ReferenceDistribution:
    return ReferenceDistribution(np.asarray(values, dtype=np.float32))


def _explicit(values: list[float], *, mode: QMode = QMode.ANALYTIC_EXACT) -> DistributionInfo:
    return DistributionInfo.explicit(
        np.asarray(values, dtype=np.float64),
        mode=mode,
        source=QSource.ADAPTER_EXACT,
    )


def _step(kl: float, tvd: float, *, mode: QMode = QMode.ANALYTIC_EXACT) -> StepDistributionDistortion:
    return StepDistributionDistortion(
        q_mode=mode,
        q_source=QSource.ADAPTER_EXACT,
        kl_bits=kl,
        tvd=tvd,
    )


def test_identical_explicit_distributions_have_zero_kl_and_tvd() -> None:
    reference = _reference([0.5, 0.25, 0.25])
    result = distribution_distortion_step(reference, _explicit([0.5, 0.25, 0.25]))

    assert result.q_mode == QMode.ANALYTIC_EXACT
    assert result.kl_bits == pytest.approx(0.0)
    assert result.tvd == pytest.approx(0.0)


def test_manual_finite_kl_and_tvd_example() -> None:
    reference = _reference([0.5, 0.25, 0.25])
    result = distribution_distortion_step(reference, _explicit([0.5, 0.4, 0.1]))

    expected_kl = 0.25 * math.log2(0.25 / 0.4) + 0.25 * math.log2(0.25 / 0.1)
    assert result.kl_bits == pytest.approx(expected_kl)
    assert result.tvd == pytest.approx(0.15)


def test_p_positive_q_zero_makes_kl_infinite_without_epsilon_smoothing() -> None:
    reference = _reference([0.5, 0.25, 0.25])
    result = distribution_distortion_step(reference, _explicit([0.5, 0.5, 0.0]))

    assert result.kl_bits == math.inf
    assert result.tvd == pytest.approx(0.25)


def test_q_zero_where_p_is_zero_does_not_force_infinite_kl() -> None:
    reference = _reference([0.5, 0.5, 0.0])
    result = distribution_distortion_step(reference, _explicit([0.6, 0.4, 0.0]))

    assert result.kl_bits is not None
    assert math.isfinite(result.kl_bits)
    assert result.tvd == pytest.approx(0.1)


def test_reference_equality_certificate_is_exact_zero_distortion() -> None:
    result = distribution_distortion_step(
        _reference([0.7, 0.2, 0.1]),
        DistributionInfo.reference_equality(),
    )

    assert result.kl_bits == 0.0
    assert result.tvd == 0.0
    assert result.q_source == QSource.ANALYTIC_THEORY


def test_unavailable_q_is_preserved_as_unavailable_not_surrogated() -> None:
    result = distribution_distortion_step(
        _reference([0.5, 0.5]),
        DistributionInfo.unavailable(),
    )

    assert result.q_mode == QMode.UNAVAILABLE
    assert result.available is False
    assert result.kl_bits is None
    assert result.tvd is None


def test_explicit_q_must_match_reference_vocabulary_size() -> None:
    reference = _reference([0.5, 0.5])
    with pytest.raises(MetricError, match="vocabulary size"):
        distribution_distortion_step(
            reference,
            _explicit([0.25, 0.25, 0.5]),
        )


def test_finite_run_aggregation_matches_manual_statistics() -> None:
    metrics = compute_distribution_distortion_metrics(
        [
            _step(0.1, 0.05),
            _step(0.2, 0.10),
            _step(0.3, 0.15),
            _step(0.4, 0.20),
        ]
    )

    assert metrics.q_mode == QMode.ANALYTIC_EXACT
    assert metrics.kl_mean_bits == pytest.approx(0.25)
    assert metrics.kl_median_bits == pytest.approx(0.25)
    # v0.2 implementation uses empirical nearest-rank p95; for n=4 this is max.
    assert metrics.kl_p95_bits == pytest.approx(0.4)
    assert metrics.kl_max_bits == pytest.approx(0.4)
    assert metrics.kl_infinite_steps == 0
    assert metrics.kl_finite_steps == 4
    assert metrics.tvd_mean == pytest.approx(0.125)
    assert metrics.tvd_median == pytest.approx(0.125)
    assert metrics.tvd_p95 == pytest.approx(0.20)
    assert metrics.tvd_max == pytest.approx(0.20)


def test_any_infinite_step_makes_full_kl_mean_infinite() -> None:
    metrics = compute_distribution_distortion_metrics(
        [_step(0.1, 0.1), _step(0.2, 0.2), _step(math.inf, 0.3)]
    )

    assert metrics.kl_mean_bits == math.inf
    assert metrics.kl_median_bits == pytest.approx(0.2)
    assert metrics.kl_p95_bits == math.inf
    assert metrics.kl_max_bits == math.inf
    assert metrics.kl_infinite_steps == 1
    assert metrics.kl_finite_steps == 2


def test_all_infinite_kl_steps_aggregate_without_nan_percentiles() -> None:
    metrics = compute_distribution_distortion_metrics(
        [_step(math.inf, 0.2), _step(math.inf, 0.3)]
    )

    assert metrics.kl_mean_bits == math.inf
    assert metrics.kl_median_bits == math.inf
    assert metrics.kl_p95_bits == math.inf
    assert metrics.kl_max_bits == math.inf
    assert metrics.kl_infinite_steps == 2
    assert metrics.kl_finite_steps == 0


def test_unavailable_run_has_none_distortion_metrics() -> None:
    unavailable = StepDistributionDistortion(
        q_mode=QMode.UNAVAILABLE,
        q_source=QSource.UNAVAILABLE,
        kl_bits=None,
        tvd=None,
    )
    metrics = compute_distribution_distortion_metrics([unavailable, unavailable])

    assert metrics.q_mode == QMode.UNAVAILABLE
    assert metrics.kl_mean_bits is None
    assert metrics.kl_median_bits is None
    assert metrics.kl_p95_bits is None
    assert metrics.kl_max_bits is None
    assert metrics.kl_infinite_steps == 0
    assert metrics.kl_finite_steps == 0
    assert metrics.tvd_mean is None
    assert metrics.tvd_median is None
    assert metrics.tvd_p95 is None
    assert metrics.tvd_max is None


def test_mixed_q_modes_are_rejected_by_v01_singular_run_field() -> None:
    with pytest.raises(MetricError, match="mixed q_mode"):
        compute_distribution_distortion_metrics(
            [
                _step(0.1, 0.1, mode=QMode.ANALYTIC_EXACT),
                _step(0.1, 0.1, mode=QMode.EXACT_ENUMERATION),
            ]
        )


def test_available_q_mode_cannot_have_missing_step_metrics() -> None:
    broken = StepDistributionDistortion(
        q_mode=QMode.ANALYTIC_EXACT,
        q_source=QSource.ADAPTER_EXACT,
        kl_bits=None,
        tvd=None,
    )
    with pytest.raises(MetricError, match="requires KL/TVD"):
        compute_distribution_distortion_metrics([broken])


def test_invalid_tvd_in_aggregation_is_rejected() -> None:
    with pytest.raises(MetricError, match="TVD"):
        compute_distribution_distortion_metrics([_step(0.1, 1.01)])


def test_empty_distribution_distortion_sequence_is_rejected() -> None:
    with pytest.raises(MetricError, match="at least one"):
        compute_distribution_distortion_metrics([])
