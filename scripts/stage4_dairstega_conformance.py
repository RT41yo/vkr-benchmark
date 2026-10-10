"""Independent Stage-4 conformance helpers for DAIRstega.

The helper intentionally does not import the normalized DAIRstega runtime.
It models two semantics from frozen sources:

* ``paper_semantics``: normalized scientific rule n_j ∝ p_j**beta, exact
  maximal common prefix, with the pinned implementation's executable
  candidate-floor / integer residual mechanics retained as frozen in ADR-0025;
* ``repository_semantics``: literal public ``DAIRstega_t.py`` behavior for the
  published ``linear`` (beta=1) and ``sqrt`` (beta=0.5) mappings, including
  four-decimal sqrt rounding and the singleton-prefix under-consumption.

This module is diagnostic tooling, not benchmark runtime code.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

REFERENCE_REPOSITORY = "https://github.com/WangYH-BUPT/DAIRstega"
REFERENCE_COMMIT = "8d85edf98d48c3efa827a125b6d4e90f88141ea2"


@dataclass(frozen=True, slots=True)
class DAIRFixture:
    fixture_id: str
    probabilities: tuple[float, ...]
    alpha: int
    beta: float
    top_k: int


@dataclass(frozen=True, slots=True)
class Interval:
    token_id: int
    start: int
    stop: int

    @property
    def width(self) -> int:
        return self.stop - self.start


def canonical_fixture_probabilities(values: tuple[float, ...]) -> np.ndarray:
    """Return the exact FP32 probability row supplied to the normalized method."""
    raw = np.asarray(values, dtype=np.float64)
    if raw.ndim != 1 or raw.size < 2:
        raise ValueError("fixture requires at least two probabilities")
    if np.any(raw <= 0.0) or not np.all(np.isfinite(raw)):
        raise ValueError("fixture probabilities must be positive and finite")
    raw = raw / float(np.sum(raw, dtype=np.float64))
    return raw.astype(np.float32)


def _ordered_probs_and_ids(values: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    probs32 = canonical_fixture_probabilities(values)
    probs = probs32.astype(np.float64)
    ids = np.arange(probs.size, dtype=np.int64)
    order = np.lexsort((ids, -probs))
    return probs[order], ids[order]


def _candidate_probs(fixture: DAIRFixture) -> tuple[np.ndarray, np.ndarray]:
    probs, token_ids = _ordered_probs_and_ids(fixture.probabilities)
    total_codes = 1 << fixture.alpha
    threshold = 1.0 / float(total_codes)
    below = np.flatnonzero(probs < threshold)
    precision_count = probs.size
    if below.size:
        precision_count = max(2, int(below[0]))
    count = min(int(probs.size), precision_count, fixture.top_k)
    return probs[:count], token_ids[:count]


def _intervals_from_counts(counts: np.ndarray, token_ids: np.ndarray, total_codes: int) -> tuple[Interval, ...]:
    retained: list[tuple[int, int]] = []
    cumulative = 0
    for token_id, interval_count in zip(token_ids.tolist(), counts.tolist(), strict=True):
        interval_count = int(interval_count)
        next_cumulative = cumulative + interval_count
        if next_cumulative > total_codes:
            break
        retained.append((token_id, interval_count))
        cumulative = next_cumulative
    if not retained:
        raise ValueError("integer allocation retained no candidates")
    residual = total_codes - cumulative
    first_token, first_count = retained[0]
    retained[0] = (first_token, first_count + residual)

    cursor = 0
    intervals: list[Interval] = []
    for token_id, width in retained:
        if width <= 0:
            continue
        intervals.append(Interval(token_id=token_id, start=cursor, stop=cursor + width))
        cursor += width
    if cursor != total_codes:
        raise ValueError("intervals do not cover the code space")
    return tuple(intervals)


def paper_allocate(fixture: DAIRFixture) -> tuple[Interval, ...]:
    """ADR-0025 / publication-faithful power allocation."""
    probs, token_ids = _candidate_probs(fixture)
    total_codes = 1 << fixture.alpha
    weights = np.power(probs, fixture.beta, dtype=np.float64)
    weights /= float(np.sum(weights, dtype=np.float64))
    counts = np.rint(weights * float(total_codes)).astype(np.int64)
    return _intervals_from_counts(counts, token_ids, total_codes)


def repository_allocate(fixture: DAIRFixture) -> tuple[Interval, ...]:
    """Literal public PyTorch allocation for linear/sqrt configurations.

    ``sqrt`` in the pinned source rounds transformed probabilities to four
    decimals before re-normalization; ``linear`` leaves them unchanged.
    """
    if not math.isclose(fixture.beta, 1.0) and not math.isclose(fixture.beta, 0.5):
        raise ValueError("literal repository comparison supports beta=1 or beta=0.5")
    probs, token_ids = _candidate_probs(fixture)
    total_codes = 1 << fixture.alpha
    if math.isclose(fixture.beta, 0.5):
        weights = np.round(np.sqrt(probs), decimals=4)
    else:
        weights = probs.copy()
    weights = weights / float(np.sum(weights, dtype=np.float64))
    counts = np.rint(weights * float(total_codes)).astype(np.int64)
    return _intervals_from_counts(counts, token_ids, total_codes)


def point_selection(intervals: tuple[Interval, ...], point: int) -> int:
    for interval in intervals:
        if interval.start <= point < interval.stop:
            return interval.token_id
    raise ValueError("point outside interval partition")


def prefix_length_paper(interval: Interval, alpha: int) -> int:
    lower = interval.start
    upper = interval.stop - 1
    xor = lower ^ upper
    return alpha if xor == 0 else alpha - xor.bit_length()


def _int2bits_little(value: int, width: int) -> list[int]:
    if width == 0:
        return []
    return [int(c) for c in reversed(f"{value:0{width}b}")]


def repository_num_same_from_beg(bits1: list[int], bits2: list[int]) -> int:
    """Literal pinned helper, including the all-equal off-by-one behavior."""
    if len(bits1) != len(bits2):
        raise ValueError("bit arrays must have equal length")
    if not bits1:
        return 0
    for i in range(len(bits1)):
        if bits1[i] != bits2[i]:
            break
    return i


def prefix_length_repository(interval: Interval, alpha: int) -> int:
    lower_bits_msb = list(reversed(_int2bits_little(interval.start, alpha)))
    upper_bits_msb = list(reversed(_int2bits_little(interval.stop - 1, alpha)))
    return repository_num_same_from_beg(lower_bits_msb, upper_bits_msb)


def fixture_suite(random_cases: int = 128) -> tuple[DAIRFixture, ...]:
    fixtures: list[DAIRFixture] = [
        DAIRFixture("linear-small-a8", (0.44, 0.24, 0.14, 0.08, 0.05, 0.03, 0.02), 8, 1.0, 100),
        DAIRFixture("sqrt-small-a8", (0.44, 0.24, 0.14, 0.08, 0.05, 0.03, 0.02), 8, 0.5, 100),
        DAIRFixture("linear-floor-a8", (0.70, 0.15, 0.08, 0.04, 0.02, 0.008, 0.002), 8, 1.0, 100),
        DAIRFixture("sqrt-topk-a8", tuple([1.0 / (i + 2) for i in range(32)]), 8, 0.5, 8),
        DAIRFixture("linear-a12", (0.31, 0.22, 0.17, 0.12, 0.08, 0.05, 0.03, 0.02), 12, 1.0, 100),
        DAIRFixture("sqrt-a12", (0.31, 0.22, 0.17, 0.12, 0.08, 0.05, 0.03, 0.02), 12, 0.5, 100),
    ]
    rng = np.random.default_rng(20261010)
    sizes = (8, 16, 32, 64)
    alphas = (8, 12, 16)
    betas = (1.0, 0.5)
    dirichlet_alpha = (0.35, 0.8, 1.5, 3.0)
    for i in range(random_cases):
        size = sizes[i % len(sizes)]
        alpha = alphas[(i // len(sizes)) % len(alphas)]
        beta = betas[(i // (len(sizes) * len(alphas))) % len(betas)]
        concentration = dirichlet_alpha[(i // (len(sizes) * len(alphas) * len(betas))) % len(dirichlet_alpha)]
        probs = rng.dirichlet(np.full(size, concentration, dtype=np.float64))
        fixtures.append(
            DAIRFixture(
                fixture_id=f"random-{i:04d}-n{size}-a{alpha}-b{beta:g}",
                probabilities=tuple(float(x) for x in probs),
                alpha=alpha,
                beta=beta,
                top_k=min(100, size),
            )
        )
    return tuple(fixtures)
