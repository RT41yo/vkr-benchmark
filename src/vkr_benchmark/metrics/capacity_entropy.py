"""Capacity and reference-entropy metrics for normalized benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite
from typing import Iterable

import numpy as np

from vkr_benchmark.distributions import ReferenceDistribution
from vkr_benchmark.errors import MetricError


@dataclass(frozen=True, slots=True)
class CapacityEntropyMetrics:
    """Run-level capacity and entropy-utilization metrics.

    All quantities follow benchmark specification v0.1:

    * ``payload_bits`` is the number of useful secret bits actually embedded;
    * ``carrier_tokens`` is the number of generated carrier tokens;
    * ``bits_per_token = payload_bits / carrier_tokens``;
    * ``reference_entropy_sum_bits`` is ``sum_t H(P_reference,t)``;
    * ``entropy_utilization = payload_bits / reference_entropy_sum_bits``.

    The utilization is deliberately not clipped to ``[0, 1]``. A value above
    one is diagnostically meaningful and must remain visible to later analysis.
    """

    payload_bits: int
    carrier_tokens: int
    bits_per_token: float
    reference_entropy_mean_bits: float
    reference_entropy_sum_bits: float
    entropy_utilization: float
    entropy_utilization_percent: float


def reference_entropy_bits(reference: ReferenceDistribution) -> float:
    """Return Shannon entropy H(P_reference) in bits for one generation step.

    The canonical probabilities remain FP32 as required by the benchmark. The
    entropy reduction is evaluated in FP64 from those canonical values to avoid
    unnecessary summation error. Zero-probability entries contribute exactly
    zero and are omitted before ``log2``.
    """

    probabilities = reference.probabilities.astype(np.float64, copy=False)
    positive = probabilities > 0.0
    if not np.any(positive):
        raise MetricError("P_reference has no positive-probability support")

    values = probabilities[positive]
    entropy = float(-np.sum(values * np.log2(values), dtype=np.float64))
    if not isfinite(entropy) or entropy < 0.0:
        raise MetricError(f"invalid reference entropy value: {entropy!r}")
    return entropy


def compute_capacity_entropy_metrics(
    *,
    payload_bits: int,
    reference_entropies_bits: Iterable[float],
) -> CapacityEntropyMetrics:
    """Aggregate run-level capacity and entropy-utilization metrics."""

    payload = int(payload_bits)
    if payload < 0:
        raise MetricError("payload_bits must be non-negative")

    entropies = tuple(float(value) for value in reference_entropies_bits)
    carrier_tokens = len(entropies)
    if carrier_tokens == 0:
        raise MetricError("at least one carrier-token entropy is required")

    for value in entropies:
        if not isfinite(value):
            raise MetricError("reference entropies must be finite")
        if value < 0.0:
            raise MetricError("reference entropies must be non-negative")

    entropy_sum = float(fsum(entropies))
    if entropy_sum <= 0.0:
        raise MetricError(
            "entropy utilization is undefined because total reference entropy is zero"
        )

    bits_per_token = payload / carrier_tokens
    entropy_mean = entropy_sum / carrier_tokens
    utilization = payload / entropy_sum

    return CapacityEntropyMetrics(
        payload_bits=payload,
        carrier_tokens=carrier_tokens,
        bits_per_token=float(bits_per_token),
        reference_entropy_mean_bits=float(entropy_mean),
        reference_entropy_sum_bits=float(entropy_sum),
        entropy_utilization=float(utilization),
        entropy_utilization_percent=float(100.0 * utilization),
    )
