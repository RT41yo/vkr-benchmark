"""Independent explicit-Q construction for normalized ADG.

The normalized ADG adapter follows Algorithm 1 of Zhang et al. (2021): at
any recursive node whose largest normalized token probability is at most 0.5,
tokens are divided into ``2**b`` approximately equal-mass groups.  Uniform
secret bits select one group, and the operation recurses only inside that
group.  Once the selected node is indivisible (largest normalized probability
> 0.5), the final token is sampled proportionally to the original LM
probabilities restricted to that node.

This module reconstructs the *entire* recursion tree independently of the ADG
encoder.  It therefore returns the exact distribution induced by uniformly
random secret bits and the terminal categorical sample.  It intentionally does
not import ``vkr_benchmark.methods.adg``.

The paper says to choose the token whose probability is nearest to the current
residual mass (Algorithm 1, line 10).  The public reference repository contains
a likely typo in ``near`` (``anum - up`` instead of ``anum - alist[up]``).
Normalized benchmark semantics follow the publication's nearest-probability
rule; the repository discrepancy is tracked separately by Stage-4 conformance
checks.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from vkr_benchmark.distributions.types import ReferenceDistribution
from vkr_benchmark.errors import DistributionError


@dataclass(frozen=True, slots=True)
class _Group:
    token_ids: np.ndarray
    probabilities: np.ndarray


def _bit_count(max_probability: float) -> int:
    if not np.isfinite(max_probability) or max_probability <= 0.0:
        raise DistributionError("ADG maximum probability must be finite and positive")
    if max_probability > 0.5:
        return 0
    bit = 1
    while (1.0 / (2 ** (bit + 1))) > max_probability:
        bit += 1
    return bit


def _nearest_index_desc(values: list[float], target: float) -> int:
    """Nearest value in a descending list; ties choose the lower probability.

    The tie convention matches the intended structure of the reference
    ``near`` helper (its final ``else`` chooses the upper index, which is the
    lower-probability neighbour in a descending array) after correcting the
    apparent ``anum - up`` typo.
    """

    if not values:
        raise DistributionError("ADG nearest search received an empty list")
    if len(values) == 1:
        return 0
    if target >= values[0]:
        return 0
    if target <= values[-1]:
        return len(values) - 1

    bottom = 0
    up = len(values) - 1
    while up - bottom > 1:
        index = (up + bottom) // 2
        value = values[index]
        if value < target:
            up = index
        elif value > target:
            bottom = index
        else:
            # Return the first equal value in canonical current order.
            while index > 0 and values[index - 1] == target:
                index -= 1
            return index

    lower_distance = values[bottom] - target
    upper_distance = target - values[up]
    if lower_distance < upper_distance:
        return bottom
    return up


def _sort_group(token_ids: np.ndarray, probabilities: np.ndarray) -> _Group:
    total = float(np.sum(probabilities, dtype=np.float64))
    if not np.isfinite(total) or total <= 0.0:
        raise DistributionError(f"ADG group probability total is invalid: {total!r}")
    normalized = probabilities.astype(np.float64, copy=True) / total
    tokens = token_ids.astype(np.int64, copy=True)
    # Deterministic normalized tie semantics: descending probability, then
    # increasing token id.  This is the same convention as P_reference.
    order = np.lexsort((tokens, -normalized))
    return _Group(token_ids=tokens[order], probabilities=normalized[order])


def _partition(group: _Group) -> tuple[_Group, ...] | None:
    bit = _bit_count(float(group.probabilities[0]))
    if bit == 0:
        return None

    group_count = 1 << bit
    if len(group.token_ids) < group_count:
        raise DistributionError(
            "ADG grouping requires at least as many positive-probability tokens "
            "as groups"
        )

    remaining_probs = [float(value) for value in group.probabilities]
    remaining_tokens = [int(value) for value in group.token_ids]
    mean = 1.0 / group_count
    result: list[_Group] = []

    for i in range(group_count - 1):
        selected_probs = [remaining_probs.pop(0)]
        selected_tokens = [remaining_tokens.pop(0)]
        selected_mass = selected_probs[0]

        while selected_mass < mean and remaining_probs:
            delta = mean - selected_mass
            index = _nearest_index_desc(remaining_probs, delta)
            candidate_probability = remaining_probs[index]
            # Algorithm 1, lines 11-17: append iff the overshoot introduced by
            # this token is smaller than the current residual itself.
            if candidate_probability - delta < delta:
                selected_mass += candidate_probability
                selected_probs.append(remaining_probs.pop(index))
                selected_tokens.append(remaining_tokens.pop(index))
            else:
                break

        result.append(
            _sort_group(
                np.asarray(selected_tokens, dtype=np.int64),
                np.asarray(selected_probs, dtype=np.float64),
            )
        )
        groups_left = group_count - i - 1
        if groups_left <= 0:
            raise DistributionError("ADG internal grouping count became invalid")
        remaining_mass = float(math.fsum(remaining_probs))
        mean = remaining_mass / groups_left

    result.append(
        _sort_group(
            np.asarray(remaining_tokens, dtype=np.int64),
            np.asarray(remaining_probs, dtype=np.float64),
        )
    )
    return tuple(result)


def explicit_adg_q(reference: ReferenceDistribution) -> np.ndarray:
    """Return the full token distribution induced by normalized ADG.

    Every recursive group choice has probability ``1 / number_of_groups``
    because the secret bit stream is modeled as uniform.  At each terminal
    node ADG samples a token proportionally to P_reference restricted to that
    node.  Enumerating all terminal nodes therefore yields an exact explicit
    Q_stego without using the encoder's sampled secret path or RNG draw.
    """

    token_ids = np.asarray(reference.token_order, dtype=np.int64)
    if token_ids.ndim != 1 or token_ids.size == 0:
        raise DistributionError("ADG requires non-empty P_reference support")
    probabilities = reference.probabilities[token_ids].astype(np.float64, copy=True)
    root = _sort_group(token_ids, probabilities)

    q = np.zeros(reference.vocab_size, dtype=np.float64)
    stack: list[tuple[_Group, float]] = [(root, 1.0)]

    while stack:
        group, path_mass = stack.pop()
        children = _partition(group)
        if children is None:
            q[group.token_ids] += path_mass * group.probabilities
            continue
        child_mass = path_mass / len(children)
        for child in children:
            stack.append((child, child_mass))

    total = float(np.sum(q, dtype=np.float64))
    if not np.isfinite(total) or total <= 0.0:
        raise DistributionError(f"ADG explicit Q total is invalid: {total!r}")
    q /= total
    if not np.isclose(float(np.sum(q, dtype=np.float64)), 1.0, rtol=0.0, atol=1e-12):
        raise DistributionError("ADG explicit Q failed normalization")
    return q
