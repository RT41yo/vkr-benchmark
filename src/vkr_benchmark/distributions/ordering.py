"""Deterministic token ordering used by steganographic methods.

The benchmark specification fixes the order as:
1. decreasing probability;
2. exact probability ties by increasing token id.
"""

from __future__ import annotations

import numpy as np

from vkr_benchmark.errors import DistributionError


def deterministic_token_order(
    probabilities: np.ndarray,
    *,
    include_zero_probability: bool = False,
) -> np.ndarray:
    """Return token ids sorted by ``(-probability, token_id)``.

    By default only the positive-probability support is returned. Masked and
    otherwise impossible tokens therefore cannot accidentally become method
    candidates merely because they appear at the end of the ordering.
    """

    probs = np.asarray(probabilities)
    if probs.ndim != 1:
        raise DistributionError("probabilities must be a one-dimensional array")
    if not np.all(np.isfinite(probs)):
        raise DistributionError("probabilities must be finite")
    if np.any(probs < 0):
        raise DistributionError("probabilities must be non-negative")

    token_ids = np.arange(probs.size, dtype=np.int64)
    if not include_zero_probability:
        mask = probs > 0
        token_ids = token_ids[mask]
        probs = probs[mask]

    # lexsort uses the last key as primary. Therefore -probs is primary and
    # token id is the deterministic secondary key for exact ties.
    order = np.lexsort((token_ids, -probs))
    result = token_ids[order]
    result.setflags(write=False)
    return result
