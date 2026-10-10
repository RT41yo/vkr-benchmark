"""Independent Stage-4 conformance helpers for ADG.

This module intentionally re-implements the publication grouping rule and the
literal public-repository ``near`` helper without importing the normalized ADG
method.  It is diagnostic tooling, not benchmark runtime code.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

REFERENCE_REPOSITORY = "https://github.com/Mhzzzzz/ADG-steganography"
REFERENCE_COMMIT = "b4a7e802a97bc3a3b84d07e63f24627b16c51a10"


@dataclass(frozen=True, slots=True)
class ADGFixture:
    fixture_id: str
    probabilities: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class Group:
    token_ids: tuple[int, ...]
    probabilities: tuple[float, ...]


def _sort_group(token_ids: list[int] | tuple[int, ...], probabilities: list[float] | tuple[float, ...]) -> Group:
    ids = np.asarray(token_ids, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    total = float(np.sum(probs, dtype=np.float64))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("invalid ADG fixture probability total")
    probs = probs / total
    order = np.lexsort((ids, -probs))
    return Group(
        token_ids=tuple(int(x) for x in ids[order]),
        probabilities=tuple(float(x) for x in probs[order]),
    )


def bit_count(max_probability: float) -> int:
    if max_probability > 0.5:
        return 0
    bit = 1
    while (1.0 / (2 ** (bit + 1))) > max_probability:
        bit += 1
    return bit


def paper_near(values: list[float], target: float) -> int:
    """Algorithm-1 nearest-probability rule for descending ``values``."""
    if not values:
        raise ValueError("empty nearest-neighbour list")
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
        if values[index] < target:
            up = index
        elif values[index] > target:
            bottom = index
        else:
            return index
    return bottom if values[bottom] - target < target - values[up] else up


def repository_near(values: list[float], target: float) -> int:
    """Literal transcription of pinned public ``near`` (including ``anum-up``)."""
    if not values:
        raise ValueError("empty nearest-neighbour list")
    up = len(values) - 1
    if up == 0:
        return 0
    bottom = 0
    while up - bottom > 1:
        index = int((up + bottom) / 2)
        if values[index] < target:
            up = index
        elif values[index] > target:
            bottom = index
        else:
            return index
    if values[bottom] - target < target - up:
        return bottom
    return up


def partition(group: Group, *, repository_semantics: bool) -> tuple[Group, ...] | None:
    group = _sort_group(list(group.token_ids), list(group.probabilities))
    bit = bit_count(group.probabilities[0])
    if bit == 0:
        return None
    group_count = 1 << bit
    remaining_probs = list(group.probabilities)
    remaining_tokens = list(group.token_ids)
    mean = 1.0 / group_count
    output: list[Group] = []
    near = repository_near if repository_semantics else paper_near

    for i in range(group_count - 1):
        selected_probs = [remaining_probs.pop(0)]
        selected_tokens = [remaining_tokens.pop(0)]
        selected_mass = selected_probs[0]
        while selected_mass < mean and remaining_probs:
            delta = mean - selected_mass
            index = near(remaining_probs, delta)
            candidate = remaining_probs[index]
            if candidate - delta < delta:
                selected_mass += candidate
                selected_probs.append(remaining_probs.pop(index))
                selected_tokens.append(remaining_tokens.pop(index))
            else:
                break
        output.append(_sort_group(selected_tokens, selected_probs))
        mean = math.fsum(remaining_probs) / (group_count - i - 1)

    output.append(_sort_group(remaining_tokens, remaining_probs))
    return tuple(output)


def explicit_q(probabilities: tuple[float, ...], *, repository_semantics: bool) -> np.ndarray:
    root = _sort_group(list(range(len(probabilities))), list(probabilities))
    q = np.zeros(len(probabilities), dtype=np.float64)
    stack: list[tuple[Group, float]] = [(root, 1.0)]
    while stack:
        group, path_mass = stack.pop()
        children = partition(group, repository_semantics=repository_semantics)
        if children is None:
            ids = np.asarray(group.token_ids, dtype=np.int64)
            probs = np.asarray(group.probabilities, dtype=np.float64)
            q[ids] += path_mass * probs
            continue
        child_mass = path_mass / len(children)
        for child in children:
            stack.append((child, child_mass))
    q /= float(np.sum(q, dtype=np.float64))
    return q


def fixture_suite(random_cases: int = 128) -> tuple[ADGFixture, ...]:
    fixtures = [
        ADGFixture("handcrafted-near-witness", (0.24, 0.18, 0.16, 0.11, 0.10, 0.08, 0.07, 0.06)),
        ADGFixture("handcrafted-balanced", (0.25, 0.25, 0.125, 0.125, 0.0625, 0.0625, 0.0625, 0.0625)),
        ADGFixture("handcrafted-terminal", (0.61, 0.12, 0.09, 0.07, 0.05, 0.03, 0.02, 0.01)),
        ADGFixture("handcrafted-recursive", (0.19, 0.17, 0.14, 0.12, 0.10, 0.09, 0.07, 0.05, 0.04, 0.03)),
    ]
    rng = np.random.default_rng(20261010)
    sizes = (8, 16, 32, 64)
    alphas = (0.2, 0.5, 1.0, 2.0, 5.0)
    for i in range(random_cases):
        size = sizes[i % len(sizes)]
        alpha = alphas[(i // len(sizes)) % len(alphas)]
        probs = rng.dirichlet(np.full(size, alpha, dtype=np.float64))
        fixtures.append(
            ADGFixture(
                f"random-{i:04d}-n{size}-a{alpha:g}",
                tuple(float(x) for x in probs),
            )
        )
    return tuple(fixtures)


def kl_bits(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0.0
    if np.any((q <= 0.0) & mask):
        return math.inf
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask]), dtype=np.float64))


def tvd(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.sum(np.abs(p - q), dtype=np.float64))
