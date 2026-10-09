"""Independent explicit-Q construction for normalized RRC.

RRC rotates the current secret point by a fresh uniform offset over the full
current range before selecting a token interval.  Conditional on the current
range, that makes the rotated point uniform over the range, so the probability
of selecting a token is the relative length of the interval assigned to that
token.

This module reconstructs those interval lengths independently of the RRC
encoder.  It intentionally does not import ``vkr_benchmark.methods.rrc``.
"""

from __future__ import annotations

import numpy as np

from vkr_benchmark.distributions.types import ReferenceDistribution
from vkr_benchmark.errors import DistributionError


def explicit_rrc_q(reference: ReferenceDistribution) -> np.ndarray:
    """Return the explicit RRC-induced token distribution for one step.

    The construction mirrors the numerical partition used by the pinned RRC
    implementation: positive-probability candidates are taken in deterministic
    descending order, accumulated in FP64, and the cumulative boundaries are
    normalized by their final total.  The induced probability of a token is
    the width of its normalized interval.
    """

    order = np.asarray(reference.token_order, dtype=np.int64)
    if order.ndim != 1 or order.size == 0:
        raise DistributionError("RRC requires non-empty P_reference support")

    support = reference.probabilities[order].astype(np.float64, copy=True)
    total = float(np.sum(support, dtype=np.float64))
    if not np.isfinite(total) or total <= 0.0:
        raise DistributionError(f"RRC support probability total is invalid: {total!r}")

    boundaries = np.empty(order.size + 1, dtype=np.float64)
    boundaries[0] = 0.0
    np.cumsum(support, dtype=np.float64, out=boundaries[1:])
    boundaries[1:] /= boundaries[-1]

    widths = np.diff(boundaries)
    if np.any(widths < 0.0) or not np.all(np.isfinite(widths)):
        raise DistributionError("RRC interval widths are not a valid distribution")

    q = np.zeros(reference.vocab_size, dtype=np.float64)
    q[order] = widths
    q_total = float(np.sum(q, dtype=np.float64))
    if not np.isfinite(q_total) or q_total <= 0.0:
        raise DistributionError(f"RRC explicit Q total is invalid: {q_total!r}")
    # Remove only cumulative floating-point residue from the explicit interval
    # widths.  This does not alter support.
    q /= q_total
    return q
