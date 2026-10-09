"""Independent explicit-Q validation helpers for normalized RRC."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vkr_benchmark.distributions import (
    QMode,
    QSource,
    ReferenceDistribution,
    explicit_rrc_q,
)
from vkr_benchmark.metrics import (
    StepDistributionDistortion,
    distribution_distortion_explicit_arrays,
)


@dataclass(frozen=True, slots=True)
class RRCExplicitQValidation:
    q_probabilities: np.ndarray
    normalized_reference: np.ndarray
    distortion: StepDistributionDistortion
    reference_storage_sum: float
    q_sum: float
    max_abs_difference: float
    l1_difference: float


def normalized_rrc_reference(reference: ReferenceDistribution) -> np.ndarray:
    p = reference.probabilities.astype(np.float64, copy=True)
    total = float(np.sum(p, dtype=np.float64))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("P_reference normalization total is invalid")
    p /= total
    p.setflags(write=False)
    return p


def evaluate_explicit_rrc_q(reference: ReferenceDistribution) -> RRCExplicitQValidation:
    """Build interval-measure Q and evaluate it with common KL/TVD formulas."""

    q = explicit_rrc_q(reference)
    p = normalized_rrc_reference(reference)
    distortion = distribution_distortion_explicit_arrays(
        p,
        q,
        q_mode=QMode.ANALYTIC_EXACT,
        q_source=QSource.INDEPENDENT_ANALYTIC,
    )
    delta = np.abs(p - q)
    return RRCExplicitQValidation(
        q_probabilities=q,
        normalized_reference=p,
        distortion=distortion,
        reference_storage_sum=float(
            np.sum(reference.probabilities.astype(np.float64), dtype=np.float64)
        ),
        q_sum=float(np.sum(q, dtype=np.float64)),
        max_abs_difference=float(np.max(delta)) if delta.size else 0.0,
        l1_difference=float(np.sum(delta, dtype=np.float64)),
    )


__all__ = [
    "RRCExplicitQValidation",
    "evaluate_explicit_rrc_q",
    "normalized_rrc_reference",
]
