"""Reusable Stage-4 Discop conformance fixtures and reference-style oracle.

The oracle is a literal, independent Python translation of the algorithmic
parts of ``comydream/Discop@3c3a100.../src/stega_cy.pyx`` used by
``cy_encode_step``/``cy_decode_step``.  It is intentionally kept outside the
normalized benchmark package.  Passing this oracle is source-level conformance,
not evidence that the original Cython extension itself was executed.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Iterable

import numpy as np

from vkr_benchmark.distributions import ReferenceDistribution, StepContext
from vkr_benchmark.methods import DiscopMethod, MethodEnvironment
from vkr_benchmark.randomness import MethodRandomSource, SecretSource

REFERENCE_REPOSITORY = "https://github.com/comydream/Discop"
REFERENCE_COMMIT = "3c3a10099a242eae405b49cc4d09fba1abb148ad"


@dataclass(slots=True)
class ReferenceNode:
    probability: float
    token_id: int = -1
    left: "ReferenceNode | None" = None
    right: "ReferenceNode | None" = None
    search_path: int = 9

    @property
    def is_leaf(self) -> bool:
        return self.token_id != -1


@dataclass(frozen=True, slots=True)
class StepFixture:
    fixture_id: str
    probabilities: tuple[float, ...]
    secret_bits: tuple[int, ...]
    seed: int


@dataclass(frozen=True, slots=True)
class StepOutcome:
    token_id: int
    bits: tuple[int, ...]
    rng_draws: int


class FixedSecretSource(SecretSource):
    def __init__(self, bits: Iterable[int]) -> None:
        self._bits = tuple(int(bit) for bit in bits)
        if any(bit not in (0, 1) for bit in self._bits):
            raise ValueError("secret bits must be binary")
        self._position = 0

    @property
    def position(self) -> int:
        return self._position

    def read_bits(self, count: int) -> tuple[int, ...]:
        stop = self._position + count
        if stop > len(self._bits):
            raise RuntimeError("conformance fixture secret exhausted")
        out = self._bits[self._position:stop]
        self._position = stop
        return out


def _reference_pop(q1: deque[ReferenceNode], q2: deque[ReferenceNode]) -> ReferenceNode:
    # Exact queue-selection semantics from the pinned create_huffman_tree:
    # choose q1 only when q1.front.prob < q2.front.prob; ties prefer q2.
    if q1 and q2:
        if q1[0].probability < q2[0].probability:
            return q1.popleft()
        return q2.popleft()
    if q1:
        return q1.popleft()
    if q2:
        return q2.popleft()
    raise RuntimeError("reference Huffman queues are empty")


def build_reference_tree(
    indices_desc: tuple[int, ...],
    probabilities_desc: tuple[float, ...],
    *,
    search_for: int = -1,
) -> ReferenceNode:
    if len(indices_desc) != len(probabilities_desc) or not indices_desc:
        raise ValueError("indices/probabilities must be non-empty and aligned")

    q1: deque[ReferenceNode] = deque()
    q2: deque[ReferenceNode] = deque()
    for token_id, probability in reversed(tuple(zip(indices_desc, probabilities_desc))):
        q1.append(
            ReferenceNode(
                probability=float(probability),
                token_id=int(token_id),
                search_path=0 if int(token_id) == int(search_for) else 9,
            )
        )

    while len(q1) + len(q2) > 1:
        first = _reference_pop(q1, q2)
        second = _reference_pop(q1, q2)
        search_path = 9
        if first.search_path != 9:
            search_path = -1
        elif second.search_path != 9:
            search_path = 1
        q2.append(
            ReferenceNode(
                probability=first.probability + second.probability,
                left=first,
                right=second,
                search_path=search_path,
            )
        )

    return q2[0] if q2 else q1[0]


def _reference_branches(node: ReferenceNode, ptr: float) -> tuple[int, int]:
    if node.left is None or node.right is None:
        raise RuntimeError("reference branch query requires an internal node")
    prob_sum = node.probability
    ptr_0 = float(ptr) * prob_sum
    ptr_1 = (float(ptr) + 0.5) * prob_sum
    # Intentionally strict >, matching the pinned Cython source.
    if ptr_1 > prob_sum:
        ptr_1 -= prob_sum
    partition = node.left.probability
    path_0 = -1 if ptr_0 < partition else 1
    path_1 = -1 if ptr_1 < partition else 1
    return path_0, path_1


def reference_encode_step(fixture: StepFixture) -> StepOutcome:
    probabilities = np.asarray(fixture.probabilities, dtype=np.float32)
    reference = ReferenceDistribution(probabilities)
    indices = tuple(int(x) for x in reference.token_order)
    probs = tuple(float(reference.probabilities[index]) for index in indices)
    node = build_reference_tree(indices, probs)
    rng = random.Random(fixture.seed)
    consumed: list[int] = []
    draws = 0

    while not node.is_leaf:
        path_0, path_1 = _reference_branches(node, rng.random())
        draws += 1
        if path_0 != path_1:
            if len(consumed) >= len(fixture.secret_bits):
                raise RuntimeError("reference oracle secret exhausted")
            bit = int(fixture.secret_bits[len(consumed)])
            consumed.append(bit)
            path = path_0 if bit == 0 else path_1
        else:
            path = path_0
        node = node.left if path == -1 else node.right
        if node is None:
            raise RuntimeError("reference encoder entered an invalid branch")

    return StepOutcome(int(node.token_id), tuple(consumed), draws)


def reference_decode_step(fixture: StepFixture, token_id: int) -> StepOutcome:
    probabilities = np.asarray(fixture.probabilities, dtype=np.float32)
    reference = ReferenceDistribution(probabilities)
    indices = tuple(int(x) for x in reference.token_order)
    probs = tuple(float(reference.probabilities[index]) for index in indices)
    node = build_reference_tree(indices, probs, search_for=int(token_id))
    rng = random.Random(fixture.seed)
    recovered: list[int] = []
    draws = 0

    while not node.is_leaf:
        path_0, path_1 = _reference_branches(node, rng.random())
        draws += 1
        if path_0 != path_1:
            if node.search_path == 9:
                raise RuntimeError("reference decoder cannot locate observed token")
            if path_0 == -1:
                recovered.append(0 if node.search_path == -1 else 1)
            else:
                recovered.append(1 if node.search_path == -1 else 0)
            path = node.search_path
        else:
            path = path_0
        node = node.left if path == -1 else node.right
        if node is None:
            raise RuntimeError("reference decoder entered an invalid branch")

    if node.search_path != 0:
        raise RuntimeError("reference decoder did not reach observed leaf")
    return StepOutcome(int(token_id), tuple(recovered), draws)


def normalized_step(fixture: StepFixture) -> tuple[StepOutcome, StepOutcome]:
    probabilities = np.asarray(fixture.probabilities, dtype=np.float32)
    reference = ReferenceDistribution(probabilities)
    context = StepContext(step_index=0, reference=reference)
    environment = MethodEnvironment(
        output_vocab_size=len(probabilities),
        allowed_token_ids=tuple(range(len(probabilities))),
    )

    secret = FixedSecretSource(fixture.secret_bits)
    encoder = DiscopMethod().create_encoder(
        config={},
        environment=environment,
        secret_source=secret,
        random_source=MethodRandomSource(fixture.seed),
    )
    decision = encoder.step(context)
    encoder_trace = dict(decision.method_trace)
    encoded = StepOutcome(
        token_id=int(decision.token_id),
        bits=tuple(fixture.secret_bits[: int(decision.bits_consumed or 0)]),
        rng_draws=int(encoder_trace["rng_draws"]),
    )

    decoder = DiscopMethod().create_decoder(
        config={},
        environment=environment,
        random_source=MethodRandomSource(fixture.seed),
        expected_payload_bits=len(encoded.bits),
    )
    progress = decoder.observe(context, encoded.token_id)
    decoder_trace = dict(progress.method_trace)
    decoded = StepOutcome(
        token_id=encoded.token_id,
        bits=tuple(progress.recovered_bits),
        rng_draws=int(decoder_trace["rng_draws"]),
    )
    final = decoder.finalize()
    if tuple(final.recovered_bits) != decoded.bits or not final.complete:
        raise RuntimeError("normalized decoder finalization disagrees with step result")
    return encoded, decoded


def fixture_suite(random_cases: int = 128) -> tuple[StepFixture, ...]:
    """Return deterministic fixed + randomized no-boundary conformance cases."""

    fixtures: list[StepFixture] = [
        StepFixture("paper_toy", (0.4, 0.3, 0.2, 0.1), (1, 0, 1, 1, 0, 0), 1234),
        StepFixture("dominant", (0.9, 0.1), (1, 0, 1, 0), 7),
        StepFixture("balanced", (0.5, 0.5), (1, 0, 1, 0), 99),
        StepFixture("leaf_ties", (0.4, 0.2, 0.2, 0.2), (0, 1, 1, 0, 1, 0), 2023),
        StepFixture("eight_way", (0.20, 0.17, 0.15, 0.13, 0.12, 0.10, 0.08, 0.05), tuple([1, 0] * 16), 42),
    ]

    rng = random.Random(20261001)
    for index in range(random_cases):
        size = rng.randint(2, 16)
        # Integer weights make normalization deterministic and avoid values that
        # are intentionally engineered to sit on pointer/partition boundaries.
        weights = [rng.randint(1, 1000) for _ in range(size)]
        total = float(sum(weights))
        probs = tuple(weight / total for weight in weights)
        bits = tuple(rng.randrange(2) for _ in range(64))
        fixtures.append(
            StepFixture(
                fixture_id=f"random_{index:03d}",
                probabilities=probs,
                secret_bits=bits,
                seed=rng.randrange(1, 2**31),
            )
        )
    return tuple(fixtures)
