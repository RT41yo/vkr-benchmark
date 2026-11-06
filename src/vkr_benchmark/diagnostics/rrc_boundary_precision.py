"""Numerical diagnostics for RRC FP64 cumulative boundaries.

The production RRC implementation is intentionally not modified here.  This
module only inspects a canonical ``P_reference`` and quantifies probability
mass that disappears when positive support probabilities are converted into
sequential FP64 cumulative boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vkr_benchmark.distributions import ReferenceDistribution
from vkr_benchmark.metrics import (
    StepDistributionDistortion,
    distribution_distortion_explicit_arrays,
)


@dataclass(frozen=True, slots=True)
class RRCLostInterval:
    """One positive reference event whose FP64 interval width became zero."""

    rank: int
    token_id: int
    reference_probability: float
    raw_left_boundary: float
    raw_right_boundary: float
    normalized_left_boundary: float
    normalized_right_boundary: float
    raw_boundary_spacing: float
    probability_to_spacing_ratio: float | None
    collapse_stage: str


@dataclass(frozen=True, slots=True)
class RRCBoundaryPrecisionAnalysis:
    """Support-loss summary for one RRC reference distribution."""

    support_size: int
    reference_sum: float
    fp64_interval_sum: float
    zero_width_count: int
    lost_reference_mass: float
    max_lost_reference_probability: float
    min_lost_reference_probability: float
    first_lost_rank: int | None
    current_distortion: StepDistributionDistortion
    ideal_distortion: StepDistributionDistortion
    lost_intervals: tuple[RRCLostInterval, ...]

    @property
    def support_loss(self) -> bool:
        return self.zero_width_count > 0


def analyze_rrc_boundary_precision(
    reference: ReferenceDistribution,
    *,
    max_examples: int = 8,
) -> RRCBoundaryPrecisionAnalysis:
    """Compare current FP64 interval widths with exact-support ideal widths.

    ``ideal`` here does not propose a new production algorithm.  It is the
    mathematical interval measure implied by the stored positive
    ``P_reference`` itself: each positive token keeps width ``p_i / sum(p)``.
    Comparing it with the current sequential FP64 cumulative partition isolates
    support loss caused by the boundary representation.
    """

    if max_examples < 0:
        raise ValueError("max_examples must be non-negative")

    order = np.asarray(reference.token_order, dtype=np.int64)
    support = reference.probabilities[order].astype(np.float64, copy=True)
    if order.ndim != 1 or order.size == 0:
        raise ValueError("RRC boundary analysis requires non-empty support")

    reference_sum = float(np.sum(support, dtype=np.float64))
    if not np.isfinite(reference_sum) or reference_sum <= 0.0:
        raise ValueError("reference support sum is invalid")

    raw_boundaries = np.empty(order.size + 1, dtype=np.float64)
    raw_boundaries[0] = 0.0
    np.cumsum(support, dtype=np.float64, out=raw_boundaries[1:])
    raw_total = float(raw_boundaries[-1])
    if not np.isfinite(raw_total) or raw_total <= 0.0:
        raise ValueError("FP64 cumulative total is invalid")

    normalized_boundaries = raw_boundaries / raw_total
    current_widths = np.diff(normalized_boundaries)
    raw_widths = np.diff(raw_boundaries)

    if np.any(current_widths < 0.0) or not np.all(np.isfinite(current_widths)):
        raise ValueError("FP64 interval widths are invalid")

    current_q = np.zeros(reference.vocab_size, dtype=np.float64)
    current_q[order] = current_widths
    current_q /= float(np.sum(current_q, dtype=np.float64))

    ideal_support = support / reference_sum
    ideal_q = np.zeros(reference.vocab_size, dtype=np.float64)
    ideal_q[order] = ideal_support
    ideal_q /= float(np.sum(ideal_q, dtype=np.float64))

    # The benchmark metric layer normalizes the stored FP32 P_reference in
    # FP64 before evaluating explicit Q, so use the same P here.
    p = reference.probabilities.astype(np.float64, copy=True)
    p /= float(np.sum(p, dtype=np.float64))

    current_distortion = distribution_distortion_explicit_arrays(p, current_q)
    ideal_distortion = distribution_distortion_explicit_arrays(p, ideal_q)

    lost_indices = np.flatnonzero((support > 0.0) & (current_widths == 0.0))
    lost_mass = float(np.sum(ideal_support[lost_indices], dtype=np.float64))

    examples: list[RRCLostInterval] = []
    for index in lost_indices[:max_examples]:
        i = int(index)
        raw_left = float(raw_boundaries[i])
        raw_right = float(raw_boundaries[i + 1])
        norm_left = float(normalized_boundaries[i])
        norm_right = float(normalized_boundaries[i + 1])
        spacing = float(np.spacing(raw_left)) if raw_left != 0.0 else float(np.spacing(0.0))
        probability = float(support[i])
        if spacing > 0.0 and np.isfinite(spacing):
            ratio: float | None = probability / spacing
        else:
            ratio = None

        if raw_widths[i] == 0.0:
            collapse_stage = "fp64_cumsum"
        elif norm_left == norm_right:
            collapse_stage = "fp64_normalization"
        else:  # Defensive: lost_indices already requires zero normalized width.
            collapse_stage = "unknown"

        examples.append(
            RRCLostInterval(
                rank=i,
                token_id=int(order[i]),
                reference_probability=probability,
                raw_left_boundary=raw_left,
                raw_right_boundary=raw_right,
                normalized_left_boundary=norm_left,
                normalized_right_boundary=norm_right,
                raw_boundary_spacing=spacing,
                probability_to_spacing_ratio=ratio,
                collapse_stage=collapse_stage,
            )
        )

    if lost_indices.size:
        lost_probabilities = support[lost_indices]
        max_lost = float(np.max(lost_probabilities))
        min_lost = float(np.min(lost_probabilities))
        first_lost_rank: int | None = int(lost_indices[0])
    else:
        max_lost = 0.0
        min_lost = 0.0
        first_lost_rank = None

    return RRCBoundaryPrecisionAnalysis(
        support_size=int(order.size),
        reference_sum=reference_sum,
        fp64_interval_sum=float(np.sum(current_widths, dtype=np.float64)),
        zero_width_count=int(lost_indices.size),
        lost_reference_mass=lost_mass,
        max_lost_reference_probability=max_lost,
        min_lost_reference_probability=min_lost,
        first_lost_rank=first_lost_rank,
        current_distortion=current_distortion,
        ideal_distortion=ideal_distortion,
        lost_intervals=tuple(examples),
    )


__all__ = [
    "RRCBoundaryPrecisionAnalysis",
    "RRCLostInterval",
    "analyze_rrc_boundary_precision",
]
