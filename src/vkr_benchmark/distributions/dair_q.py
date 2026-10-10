"""Independent exact Q construction for normalized DAIRstega.

DAIRstega maps the current P_reference to a method-local candidate pool,
allocates an integer roulette interval to each retained token, and selects the
token containing the current ``alpha``-bit secret window.  Because DAIRstega
advances only the common prefix of the selected interval, consecutive secret
windows overlap.  Therefore the exact conditional Q at a later step is not, in
general, simply ``interval_width / 2**alpha``.

For an IID uniform secret stream, conditioned on the already emitted carrier
history, the current alpha-bit window remains uniform on one contiguous integer
range.  ``explicit_dair_q`` accepts that hidden-state range and returns the
exact token-selection distribution by intersecting it with the independently
reconstructed roulette intervals.

This module deliberately does not import ``vkr_benchmark.methods.dair``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from vkr_benchmark.distributions.types import ReferenceDistribution
from vkr_benchmark.errors import DistributionError


@dataclass(frozen=True, slots=True)
class DAIRInterval:
    token_id: int
    start: int
    stop: int  # exclusive

    @property
    def width(self) -> int:
        return self.stop - self.start


def _validate_params(*, alpha: int, beta: float, top_k: int) -> None:
    if not isinstance(alpha, int) or isinstance(alpha, bool) or not 1 <= alpha <= 52:
        raise DistributionError("DAIRstega alpha must be an integer in [1, 52]")
    if not math.isfinite(beta) or not 0.0 < beta <= 1.0:
        raise DistributionError("DAIRstega beta must satisfy 0 < beta <= 1")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 2:
        raise DistributionError("DAIRstega top_k must be an integer >= 2")


def _candidate_ids_and_probabilities(
    reference: ReferenceDistribution, *, alpha: int, top_k: int
) -> tuple[np.ndarray, np.ndarray]:
    token_ids = np.asarray(reference.token_order, dtype=np.int64)
    if token_ids.size < 2:
        raise DistributionError("DAIRstega requires at least two positive-probability tokens")
    probabilities = reference.probabilities[token_ids].astype(np.float64, copy=True)

    # Pinned PyTorch implementation first removes the tail whose original
    # probability is below one alpha-bit code cell, while retaining at least
    # two candidates, then applies its method-local top_k restriction.
    threshold = 1.0 / float(1 << alpha)
    below = np.flatnonzero(probabilities < threshold)
    precision_count = token_ids.size
    if below.size:
        precision_count = max(2, int(below[0]))
    count = min(int(token_ids.size), precision_count, top_k)
    if count < 2:
        raise DistributionError("DAIRstega candidate pool unexpectedly has fewer than two tokens")
    return token_ids[:count].copy(), probabilities[:count].copy()


def allocate_dair_intervals(
    reference: ReferenceDistribution, *, alpha: int, beta: float, top_k: int
) -> tuple[DAIRInterval, ...]:
    """Independently reconstruct the normalized DAIRstega roulette.

    The executable author implementation realizes the paper's proportional
    relation ``n_j ∝ p_j**beta`` by power-transforming candidate probabilities,
    renormalizing those weights, multiplying by ``2**alpha``, rounding to the
    nearest integer, truncating at the first cumulative overfill, and assigning
    any residual cells to the first (highest-probability) token.  The benchmark
    freezes those deterministic allocation semantics because the paper leaves
    integer remainder handling underspecified.
    """

    _validate_params(alpha=alpha, beta=beta, top_k=top_k)
    token_ids, probabilities = _candidate_ids_and_probabilities(
        reference, alpha=alpha, top_k=top_k
    )
    total_codes = 1 << alpha

    weights = np.power(probabilities, beta, dtype=np.float64)
    weight_sum = float(np.sum(weights, dtype=np.float64))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        raise DistributionError("DAIRstega transformed candidate weights are invalid")
    weights /= weight_sum

    rounded = np.rint(weights * float(total_codes)).astype(object)
    counts = [int(value) for value in rounded]
    cumulative = 0
    retained: list[tuple[int, int]] = []
    for token_id, count in zip(token_ids.tolist(), counts, strict=True):
        next_cumulative = cumulative + count
        if next_cumulative > total_codes:
            break
        retained.append((int(token_id), int(count)))
        cumulative = next_cumulative

    if not retained:
        raise DistributionError("DAIRstega integer allocation retained no candidates")

    # The reference code shifts all cumulative endpoints by the remaining
    # amount.  In interval-width form this is exactly equivalent to assigning
    # the entire residual to the first token.
    residual = total_codes - cumulative
    first_token, first_count = retained[0]
    retained[0] = (first_token, first_count + residual)

    intervals: list[DAIRInterval] = []
    cursor = 0
    for token_id, count in retained:
        if count < 0:
            raise DistributionError("DAIRstega produced a negative interval width")
        if count == 0:
            continue
        stop = cursor + count
        intervals.append(DAIRInterval(token_id=token_id, start=cursor, stop=stop))
        cursor = stop

    if not intervals or cursor != total_codes:
        raise DistributionError(
            f"DAIRstega intervals must exactly cover [0, 2**alpha); covered {cursor}/{total_codes}"
        )
    return tuple(intervals)


def explicit_dair_q(
    reference: ReferenceDistribution,
    *,
    alpha: int,
    beta: float,
    top_k: int,
    window_lower: int = 0,
    window_upper: int | None = None,
) -> np.ndarray:
    """Return exact conditional Q for one normalized DAIRstega step.

    ``window_lower`` and ``window_upper`` are the inclusive support bounds of
    the current alpha-bit secret window after conditioning on earlier emitted
    carrier tokens.  The window is uniform on that range.
    """

    _validate_params(alpha=alpha, beta=beta, top_k=top_k)
    total_codes = 1 << alpha
    if window_upper is None:
        window_upper = total_codes - 1
    if not (0 <= window_lower <= window_upper < total_codes):
        raise DistributionError("DAIRstega secret-window support lies outside code space")

    q = np.zeros(reference.vocab_size, dtype=np.float64)
    support_size = window_upper - window_lower + 1
    for interval in allocate_dair_intervals(
        reference, alpha=alpha, beta=beta, top_k=top_k
    ):
        lo = max(window_lower, interval.start)
        hi = min(window_upper + 1, interval.stop)
        width = max(0, hi - lo)
        if width:
            q[interval.token_id] = width / float(support_size)

    total = float(np.sum(q, dtype=np.float64))
    if not np.isclose(total, 1.0, rtol=0.0, atol=1e-12):
        raise DistributionError(
            f"DAIRstega exact conditional Q failed normalization: {total!r}"
        )
    return q
