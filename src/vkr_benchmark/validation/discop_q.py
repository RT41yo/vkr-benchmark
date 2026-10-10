"""Independent explicit-Q validation helpers for Stage-4 Discop.

The canonical explicit Q_stego calculator lives in
``vkr_benchmark.distributions.discop_q`` and is also used by ordinary normalized
Discop benchmark runs.  This module adds validation-only normalization and
standard KL/TVD diagnostics around that explicit distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vkr_benchmark.distributions import (
    QMode,
    QSource,
    ReferenceDistribution,
    branch_selection_probabilities,
    explicit_discop_q,
)
from vkr_benchmark.metrics import (
    StepDistributionDistortion,
    distribution_distortion_explicit_arrays,
)


def normalized_reference_probabilities(reference: ReferenceDistribution) -> np.ndarray:
    """Return a FP64 unit-sum copy of canonical FP32 P_reference for validation.

    This removes only the storage-level unit-sum residual before the dedicated
    Stage-4 equality check. Ordinary benchmark runs do not use this helper: they
    pass the explicit Q_stego from the method into the standard metric layer in
    exactly the same way as other methods.
    """

    p = reference.probabilities.astype(np.float64, copy=True)
    total = float(np.sum(p, dtype=np.float64))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("P_reference normalization total is invalid")
    p /= total
    p.setflags(write=False)
    return p


@dataclass(frozen=True, slots=True)
class ExplicitQValidation:
    q_probabilities: np.ndarray
    normalized_reference: np.ndarray
    distortion: StepDistributionDistortion
    reference_storage_sum: float
    q_sum: float
    max_abs_difference: float
    l1_difference: float


def evaluate_explicit_discop_q(reference: ReferenceDistribution) -> ExplicitQValidation:
    """Build explicit Q and evaluate it with the benchmark KL/TVD formulas."""

    q = explicit_discop_q(reference)
    p = normalized_reference_probabilities(reference)
    distortion = distribution_distortion_explicit_arrays(
        p,
        q,
        q_mode=QMode.EXACT_ENUMERATION,
        q_source=QSource.INDEPENDENT_ENUMERATION,
    )
    delta = np.abs(p - q)
    return ExplicitQValidation(
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
    "ExplicitQValidation",
    "branch_selection_probabilities",
    "evaluate_explicit_discop_q",
    "explicit_discop_q",
    "normalized_reference_probabilities",
]
