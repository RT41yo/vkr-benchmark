"""KL/TVD metrics between canonical P_reference and method-induced Q_stego."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, fsum, inf, isfinite
from typing import Iterable

import numpy as np

from vkr_benchmark.distributions import (
    DistributionInfo,
    QMode,
    QRepresentation,
    QSource,
    ReferenceDistribution,
)
from vkr_benchmark.errors import MetricError

# Canonical P_reference is stored in FP32, so its sum may differ from 1 by a
# few ulps after the final common renormalization. TVD is nevertheless bounded
# conceptually by [0, 1]; this tolerance is used only to absorb that storage
# roundoff at the endpoints, never to smooth Q or replace zero probabilities.
_TVD_BOUND_ATOL = 1e-6
_KL_NEGATIVE_ATOL = 1e-10


@dataclass(frozen=True, slots=True)
class StepDistributionDistortion:
    """Distribution-distortion metrics for one generated carrier token."""

    q_mode: QMode
    q_source: QSource
    kl_bits: float | None
    tvd: float | None

    @property
    def available(self) -> bool:
        return self.kl_bits is not None and self.tvd is not None


@dataclass(frozen=True, slots=True)
class DistributionDistortionMetrics:
    """Run-level aggregation required by benchmark specification v0.1."""

    q_mode: QMode
    kl_mean_bits: float | None
    kl_median_bits: float | None
    kl_p95_bits: float | None
    kl_max_bits: float | None
    kl_infinite_steps: int
    kl_finite_steps: int
    tvd_mean: float | None
    tvd_median: float | None
    tvd_p95: float | None
    tvd_max: float | None


def _explicit_q(
    *,
    reference: ReferenceDistribution,
    distribution_info: DistributionInfo,
) -> np.ndarray:
    q = distribution_info.probabilities
    if q is None:
        raise MetricError("explicit Q_stego representation contains no probabilities")
    if q.ndim != 1 or q.size != reference.vocab_size:
        raise MetricError(
            "Q_stego vocabulary size does not match canonical P_reference: "
            f"{q.size} != {reference.vocab_size}"
        )
    return q.astype(np.float64, copy=False)


def _kl_ref_to_stego_bits(p: np.ndarray, q: np.ndarray) -> float:
    positive_p = p > 0.0
    if not np.any(positive_p):
        raise MetricError("P_reference has no positive-probability support")

    p_support = p[positive_p]
    q_support = q[positive_p]

    # FIXED v0.1: no epsilon smoothing. Any P>0, Q=0 event makes KL infinite.
    if np.any(q_support == 0.0):
        return inf

    terms = p_support * (np.log2(p_support) - np.log2(q_support))
    value = float(np.sum(terms, dtype=np.float64))
    if not isfinite(value):
        raise MetricError(f"finite-support KL computation produced {value!r}")

    # Gibbs' inequality gives KL >= 0. A tiny negative value can only come from
    # finite-precision reduction and is normalized to mathematical zero.
    if value < 0.0:
        if value >= -_KL_NEGATIVE_ATOL:
            return 0.0
        raise MetricError(f"KL(P_reference || Q_stego) became negative: {value!r}")
    return value


def _tvd(p: np.ndarray, q: np.ndarray) -> float:
    value = float(0.5 * np.sum(np.abs(p - q), dtype=np.float64))
    if not isfinite(value):
        raise MetricError(f"TVD computation produced {value!r}")
    if value < -_TVD_BOUND_ATOL or value > 1.0 + _TVD_BOUND_ATOL:
        raise MetricError(f"TVD lies outside [0, 1]: {value!r}")
    # Endpoint-only roundoff normalization; this is not probability smoothing.
    return min(1.0, max(0.0, value))


def distribution_distortion_step(
    reference: ReferenceDistribution,
    distribution_info: DistributionInfo,
) -> StepDistributionDistortion:
    """Compute one-step KL(P_reference || Q_stego) and TVD.

    ``q_mode=unavailable`` is represented explicitly by ``None`` metric values;
    unavailable Q is never replaced by the selected-token probability or by a
    smoothed surrogate distribution.
    """

    if distribution_info.mode == QMode.UNAVAILABLE:
        return StepDistributionDistortion(
            q_mode=distribution_info.mode,
            q_source=distribution_info.source,
            kl_bits=None,
            tvd=None,
        )

    if distribution_info.representation == QRepresentation.REFERENCE_EQUALITY_CERTIFICATE:
        return StepDistributionDistortion(
            q_mode=distribution_info.mode,
            q_source=distribution_info.source,
            kl_bits=0.0,
            tvd=0.0,
        )

    if distribution_info.representation != QRepresentation.EXPLICIT_PROBABILITIES:
        raise MetricError(
            "available Q_stego must provide either explicit probabilities or "
            "a reference-equality certificate"
        )

    p = reference.probabilities.astype(np.float64, copy=False)
    q = _explicit_q(reference=reference, distribution_info=distribution_info)
    return StepDistributionDistortion(
        q_mode=distribution_info.mode,
        q_source=distribution_info.source,
        kl_bits=_kl_ref_to_stego_bits(p, q),
        tvd=_tvd(p, q),
    )


def _median_sorted(values: list[float]) -> float:
    n = len(values)
    middle = n // 2
    if n % 2:
        return values[middle]
    left = values[middle - 1]
    right = values[middle]
    if left == inf or right == inf:
        return inf
    return 0.5 * (left + right)


def _nearest_rank_percentile_sorted(values: list[float], percentile: float) -> float:
    """Empirical nearest-rank percentile that preserves +inf observations."""

    if not 0.0 < percentile <= 100.0:
        raise ValueError("percentile must lie in (0, 100]")
    rank = max(1, ceil((percentile / 100.0) * len(values)))
    return values[rank - 1]


def compute_distribution_distortion_metrics(
    step_metrics: Iterable[StepDistributionDistortion],
) -> DistributionDistortionMetrics:
    """Aggregate one-step KL/TVD values into the v0.1 run-level fields."""

    steps = tuple(step_metrics)
    if not steps:
        raise MetricError("at least one distribution-distortion step is required")

    q_modes = {step.q_mode for step in steps}
    if len(q_modes) != 1:
        raise MetricError(
            "mixed q_mode values cannot be represented by the singular v0.1 q_mode field"
        )
    q_mode = steps[0].q_mode

    if q_mode == QMode.UNAVAILABLE:
        if any(step.available for step in steps):
            raise MetricError("q_mode=unavailable cannot carry KL/TVD values")
        return DistributionDistortionMetrics(
            q_mode=q_mode,
            kl_mean_bits=None,
            kl_median_bits=None,
            kl_p95_bits=None,
            kl_max_bits=None,
            kl_infinite_steps=0,
            kl_finite_steps=0,
            tvd_mean=None,
            tvd_median=None,
            tvd_p95=None,
            tvd_max=None,
        )

    if any(not step.available for step in steps):
        raise MetricError("available q_mode requires KL/TVD values on every carrier step")

    kl_values = [float(step.kl_bits) for step in steps if step.kl_bits is not None]
    tvd_values = [float(step.tvd) for step in steps if step.tvd is not None]

    if len(kl_values) != len(steps) or len(tvd_values) != len(steps):
        raise MetricError("incomplete KL/TVD step sequence")
    if any(np.isnan(value) or value == -inf or value < 0.0 for value in kl_values):
        raise MetricError("KL values must be non-negative finite values or +inf")
    if any(not isfinite(value) or value < 0.0 or value > 1.0 for value in tvd_values):
        raise MetricError("TVD values must be finite and lie in [0, 1]")

    kl_sorted = sorted(kl_values)
    tvd_sorted = sorted(tvd_values)
    infinite_steps = sum(value == inf for value in kl_values)
    finite_steps = len(kl_values) - infinite_steps

    if infinite_steps:
        kl_mean = inf
    else:
        kl_mean = float(fsum(kl_values) / len(kl_values))

    return DistributionDistortionMetrics(
        q_mode=q_mode,
        kl_mean_bits=kl_mean,
        kl_median_bits=float(_median_sorted(kl_sorted)),
        kl_p95_bits=float(_nearest_rank_percentile_sorted(kl_sorted, 95.0)),
        kl_max_bits=float(kl_sorted[-1]),
        kl_infinite_steps=infinite_steps,
        kl_finite_steps=finite_steps,
        tvd_mean=float(fsum(tvd_values) / len(tvd_values)),
        tvd_median=float(_median_sorted(tvd_sorted)),
        tvd_p95=float(_nearest_rank_percentile_sorted(tvd_sorted, 95.0)),
        tvd_max=float(tvd_sorted[-1]),
    )
