#!/usr/bin/env python3
"""Stage 5 independent decoder-conformance audit for the six streaming methods.

This diagnostic deliberately does not use private production decoder helpers.
It compares the public normalized decoder sessions against independent
paper/ADR/pinned-source oracles built outside ``src/vkr_benchmark/methods``.

The check is synthetic and CPU-only: no language-model weights, tokenizer, GPU,
or Pilot-B ``results/runs`` artifacts are required.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import heapq
import json
import math
from pathlib import Path
import random
import sys
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from stage4_adg_conformance import Group as ADGGroup
from stage4_adg_conformance import fixture_suite as adg_fixture_suite
from stage4_adg_conformance import partition as adg_paper_partition
from stage4_dairstega_conformance import DAIRFixture, paper_allocate
from stage4_dairstega_conformance import prefix_length_paper
from stage4_discop_conformance import build_reference_tree as discop_reference_tree
from stage4_discop_conformance import _reference_branches as discop_reference_branches
from vkr_benchmark.distributions import ReferenceDistribution, StepContext
from vkr_benchmark.methods import (
    ADGMethod,
    ArithmeticMethod,
    BinsMethod,
    DAIRMethod,
    DiscopMethod,
    HuffmanMethod,
    MethodEnvironment,
)
from vkr_benchmark.randomness import MethodRandomSource

SCHEMA_VERSION = "stage5.decoder_conformance.v1"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _reference(values: Iterable[float]) -> ReferenceDistribution:
    probs = np.asarray(tuple(float(x) for x in values), dtype=np.float64)
    if probs.ndim != 1 or probs.size < 2:
        raise ValueError("decoder conformance fixture needs at least two tokens")
    if np.any(probs <= 0.0) or not np.all(np.isfinite(probs)):
        raise ValueError("fixture probabilities must be positive and finite")
    probs /= float(np.sum(probs, dtype=np.float64))
    return ReferenceDistribution(probs.astype(np.float32))


def _environment(size: int, allowed: Iterable[int] | None = None) -> MethodEnvironment:
    token_ids = tuple(range(size)) if allowed is None else tuple(int(x) for x in allowed)
    return MethodEnvironment(output_vocab_size=size, allowed_token_ids=token_ids)


def _msb_bits(value: int, width: int) -> tuple[int, ...]:
    return tuple((int(value) >> shift) & 1 for shift in range(width - 1, -1, -1))


def _little_bits(value: int, width: int) -> tuple[int, ...]:
    return tuple((int(value) >> shift) & 1 for shift in range(width))


def _failure(method: str, fixture: str, detail: str, **extra: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "method": method,
        "fixture": fixture,
        "detail": detail,
    }
    payload.update(extra)
    return payload


# ---------------------------------------------------------------------------
# Bins: independent Fisher-Yates partition + bin-index -> MSB bits
# ---------------------------------------------------------------------------


def _bins_oracle_partition(
    allowed_token_ids: tuple[int, ...], *, block_size: int, seed: int
) -> dict[int, int]:
    values = list(sorted(int(x) for x in allowed_token_ids))
    rng = random.Random(seed)
    for index in range(len(values) - 1, 0, -1):
        selected = rng.randrange(index + 1)
        values[index], values[selected] = values[selected], values[index]

    num_bins = 1 << block_size
    inverse: dict[int, int] = {}
    for bin_id in range(num_bins):
        start = (bin_id * len(values)) // num_bins
        stop = ((bin_id + 1) * len(values)) // num_bins
        for token_id in values[start:stop]:
            inverse[int(token_id)] = bin_id
    return inverse


def audit_bins() -> dict[str, object]:
    failures: list[dict[str, object]] = []
    checks = 0
    output_vocab_size = 53
    allowed = tuple(token for token in range(output_vocab_size) if token % 7 != 0)
    environment = _environment(output_vocab_size, allowed)
    dummy = StepContext(0, _reference(np.ones(output_vocab_size)))

    for block_size in (1, 2, 4):
        for seed in (7, 1234, 20261031):
            fixture_id = f"b{block_size}-seed{seed}"
            inverse = _bins_oracle_partition(
                environment.allowed_token_ids,
                block_size=block_size,
                seed=seed,
            )
            decoder = BinsMethod().create_decoder(
                config={"block_size": block_size},
                environment=environment,
                random_source=MethodRandomSource(seed),
                expected_payload_bits=None,
            )
            for token_id in environment.allowed_token_ids:
                expected = _msb_bits(inverse[token_id], block_size)
                progress = decoder.observe(dummy, token_id)
                checks += 1
                if tuple(progress.recovered_bits) != expected:
                    failures.append(
                        _failure(
                            "bins",
                            fixture_id,
                            "token-to-bin bit mapping mismatch",
                            token_id=token_id,
                            expected_bits=expected,
                            observed_bits=tuple(progress.recovered_bits),
                        )
                    )
    return {
        "status": "pass" if not failures else "fail",
        "check_count": checks,
        "failure_count": len(failures),
        "failures": failures,
        "oracle": "independent Python Fisher-Yates + floor bin boundaries + MSB bin bits",
    }


# ---------------------------------------------------------------------------
# Huffman: independent deterministic heap/tree implementation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _HNode:
    weight: float
    minimum_token: int
    token_id: int | None = None
    left: "_HNode | None" = None
    right: "_HNode | None" = None


def _huffman_oracle_codes(
    reference: ReferenceDistribution, *, bits_per_word: int
) -> dict[int, tuple[int, ...]]:
    count = 1 << bits_per_word
    candidates = tuple(int(x) for x in reference.token_order[:count])
    if len(candidates) != count:
        raise ValueError("oracle Huffman fixture has insufficient support")

    heap: list[tuple[float, int, _HNode]] = []
    for token_id in candidates:
        node = _HNode(
            weight=float(reference.probabilities[token_id]),
            minimum_token=token_id,
            token_id=token_id,
        )
        heapq.heappush(heap, (node.weight, node.minimum_token, node))

    while len(heap) > 1:
        _, _, left = heapq.heappop(heap)
        _, _, right = heapq.heappop(heap)
        parent = _HNode(
            weight=left.weight + right.weight,
            minimum_token=min(left.minimum_token, right.minimum_token),
            left=left,
            right=right,
        )
        heapq.heappush(heap, (parent.weight, parent.minimum_token, parent))

    codes: dict[int, tuple[int, ...]] = {}

    def walk(node: _HNode, prefix: tuple[int, ...]) -> None:
        if node.token_id is not None:
            codes[node.token_id] = prefix
            return
        if node.left is None or node.right is None:
            raise RuntimeError("invalid independent Huffman tree")
        walk(node.left, (*prefix, 0))
        walk(node.right, (*prefix, 1))

    walk(heap[0][2], ())
    return codes


def audit_huffman(random_cases: int) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    checks = 0
    fixtures: list[tuple[str, ReferenceDistribution, int]] = [
        ("known-unequal", _reference((0.5, 0.25, 0.125, 0.125)), 2),
        ("exact-ties", _reference((0.25, 0.25, 0.25, 0.25)), 2),
        ("mixed-ties", _reference((0.30, 0.20, 0.20, 0.10, 0.08, 0.06, 0.04, 0.02)), 3),
    ]
    rng = np.random.default_rng(20261101)
    for index in range(random_cases):
        bits_per_word = (1, 2, 3, 4)[index % 4]
        size = max(1 << bits_per_word, 20)
        concentration = (0.2, 0.7, 1.5, 4.0)[(index // 4) % 4]
        probabilities = rng.dirichlet(np.full(size, concentration, dtype=np.float64))
        fixtures.append(
            (
                f"random-{index:03d}-e{bits_per_word}",
                _reference(probabilities),
                bits_per_word,
            )
        )

    for fixture_id, reference, bits_per_word in fixtures:
        environment = _environment(reference.vocab_size)
        expected_codes = _huffman_oracle_codes(reference, bits_per_word=bits_per_word)
        context = StepContext(0, reference)
        for token_id, expected in expected_codes.items():
            decoder = HuffmanMethod().create_decoder(
                config={"bits_per_word": bits_per_word},
                environment=environment,
                expected_payload_bits=None,
            )
            progress = decoder.observe(context, token_id)
            checks += 1
            if tuple(progress.recovered_bits) != expected:
                failures.append(
                    _failure(
                        "huffman",
                        fixture_id,
                        "independent Huffman code mismatch",
                        token_id=token_id,
                        expected_bits=expected,
                        observed_bits=tuple(progress.recovered_bits),
                    )
                )
    return {
        "status": "pass" if not failures else "fail",
        "fixture_count": len(fixtures),
        "check_count": checks,
        "failure_count": len(failures),
        "failures": failures,
        "oracle": "independent deterministic (weight, min-token-id) Huffman tree",
    }


# ---------------------------------------------------------------------------
# Arithmetic: independent ADR-0007 integer partition + state transition
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ArithmeticPartitionOracle:
    token_ids: tuple[int, ...]
    widths: tuple[int, ...]
    upper_bounds: tuple[int, ...]
    lower: int
    upper: int

    def bounds(self, rank: int) -> tuple[int, int]:
        lo = self.lower if rank == 0 else self.upper_bounds[rank - 1]
        return lo, self.upper_bounds[rank]


def _arithmetic_oracle_partition(
    reference: ReferenceDistribution,
    *,
    precision: int,
    top_k: int,
    lower: int,
    upper: int,
) -> _ArithmeticPartitionOracle:
    width = upper - lower
    if width < 2:
        raise ValueError("oracle Arithmetic interval collapsed")

    order = [int(token_id) for token_id in reference.token_order]
    probabilities = [float(reference.probabilities[token_id]) for token_id in order]
    threshold = 1.0 / float(width)
    cutoff = len(order)
    for index, probability in enumerate(probabilities):
        if probability < threshold:
            cutoff = index
            break
    candidate_count = min(max(2, cutoff), top_k, len(order))
    token_ids = order[:candidate_count]
    candidate_probs = probabilities[:candidate_count]
    total = math.fsum(candidate_probs)

    # Python round is ties-to-even, matching the integer rounding required by
    # ADR-0007 without calling the production NumPy implementation.
    raw_widths = [int(round((probability / total) * width)) for probability in candidate_probs]
    cumulative: list[int] = []
    running = 0
    keep = len(raw_widths)
    for index, integer_width in enumerate(raw_widths):
        running += integer_width
        if running > width:
            keep = index
            break
        cumulative.append(running)
    if keep == 0 or not cumulative:
        raise ValueError("oracle Arithmetic overfilled at first candidate")
    token_ids = token_ids[:keep]
    raw_widths = raw_widths[:keep]
    cumulative = cumulative[:keep]

    residual = width - cumulative[-1]
    # Shifting every cumulative bound by the residual is equivalent to giving
    # the residual to the first (most probable) token only.
    cumulative = [value + residual for value in cumulative]
    widths = [cumulative[0]]
    widths.extend(cumulative[i] - cumulative[i - 1] for i in range(1, len(cumulative)))
    absolute = tuple(lower + value for value in cumulative)
    return _ArithmeticPartitionOracle(
        token_ids=tuple(token_ids),
        widths=tuple(int(x) for x in widths),
        upper_bounds=absolute,
        lower=lower,
        upper=upper,
    )


def _arithmetic_oracle_renormalize(
    *, lower: int, upper: int, precision: int
) -> tuple[int, int, tuple[int, ...]]:
    left = f"{lower:0{precision}b}"
    right = f"{upper - 1:0{precision}b}"
    count = None
    for index, (a, b) in enumerate(zip(left, right, strict=True)):
        if a != b:
            count = index
            break
    if count is None:
        # Harvard helper intentionally leaves one bit unresolved in the
        # all-equal case; ADR-0007 freezes that normalized behavior.
        count = max(0, precision - 1)
    fixed = tuple(int(bit) for bit in left[:count])
    new_left = left[count:] + ("0" * count)
    new_right = right[count:] + ("1" * count)
    return int(new_left, 2), int(new_right, 2) + 1, fixed


def _arithmetic_contexts(count: int, *, size: int, seed: int) -> tuple[ReferenceDistribution, ...]:
    rng = np.random.default_rng(seed)
    output: list[ReferenceDistribution] = []
    for index in range(count):
        if index % 3 == 1:
            weights = np.geomspace(1.0, 1e-9, size, dtype=np.float64)
            weights = np.roll(weights, index % size)
        elif index % 3 == 2:
            weights = rng.lognormal(mean=0.0, sigma=2.0, size=size)
        else:
            weights = rng.dirichlet(np.full(size, 0.7, dtype=np.float64))
        output.append(_reference(weights))
    return tuple(output)


def audit_arithmetic() -> dict[str, object]:
    failures: list[dict[str, object]] = []
    checks = 0
    configs = (
        (16, 4),
        (16, 24),
        (32, 8),
        (32, 32),
        (48, 32),
    )
    steps_per_config = 30

    for config_index, (precision, top_k) in enumerate(configs):
        fixture_id = f"p{precision}-k{top_k}"
        references = _arithmetic_contexts(
            steps_per_config,
            size=40,
            seed=20261101 + config_index,
        )
        environment = _environment(40)
        decoder = ArithmeticMethod().create_decoder(
            config={"precision": precision, "top_k": top_k},
            environment=environment,
            expected_payload_bits=None,
        )
        lower, upper = 0, 1 << precision
        expected_all: list[int] = []

        for step_index, reference in enumerate(references):
            oracle = _arithmetic_oracle_partition(
                reference,
                precision=precision,
                top_k=top_k,
                lower=lower,
                upper=upper,
            )
            positive_ranks = [index for index, width in enumerate(oracle.widths) if width > 0]
            if not positive_ranks:
                failures.append(_failure("arithmetic_coding", fixture_id, "oracle produced no positive-width candidate"))
                break
            rank = positive_ranks[(step_index * 7 + config_index) % len(positive_ranks)]
            token_id = oracle.token_ids[rank]
            selected_lower, selected_upper = oracle.bounds(rank)
            expected_lower, expected_upper, expected_bits = _arithmetic_oracle_renormalize(
                lower=selected_lower,
                upper=selected_upper,
                precision=precision,
            )

            try:
                progress = decoder.observe(StepContext(step_index, reference), token_id)
            except Exception as exc:  # diagnostic: preserve unexpected production failure
                failures.append(
                    _failure(
                        "arithmetic_coding",
                        fixture_id,
                        "production decoder raised on oracle-valid token",
                        step=step_index,
                        token_id=token_id,
                        exception=f"{type(exc).__name__}: {exc}",
                    )
                )
                break

            checks += 1
            observed_bits = tuple(progress.recovered_bits)
            observed_after = tuple(progress.method_trace.get("interval_after", ()))
            observed_selected = tuple(progress.method_trace.get("selected_interval", ()))
            if observed_bits != expected_bits or observed_after != (expected_lower, expected_upper) or observed_selected != (selected_lower, selected_upper):
                failures.append(
                    _failure(
                        "arithmetic_coding",
                        fixture_id,
                        "stateful integer-range decoder mismatch",
                        step=step_index,
                        token_id=token_id,
                        expected_bits=expected_bits,
                        observed_bits=observed_bits,
                        expected_selected_interval=(selected_lower, selected_upper),
                        observed_selected_interval=observed_selected,
                        expected_interval_after=(expected_lower, expected_upper),
                        observed_interval_after=observed_after,
                    )
                )
                break

            expected_all.extend(expected_bits)
            lower, upper = expected_lower, expected_upper

        else:
            final = decoder.finalize()
            if tuple(final.recovered_bits) != tuple(expected_all) or tuple(final.metadata.get("final_interval", ())) != (lower, upper):
                failures.append(
                    _failure(
                        "arithmetic_coding",
                        fixture_id,
                        "final accumulated state mismatch",
                        expected_bits=tuple(expected_all),
                        observed_bits=tuple(final.recovered_bits),
                        expected_interval=(lower, upper),
                        observed_interval=tuple(final.metadata.get("final_interval", ())),
                    )
                )

    return {
        "status": "pass" if not failures else "fail",
        "configuration_count": len(configs),
        "steps_per_configuration": steps_per_config,
        "check_count": checks,
        "failure_count": len(failures),
        "failures": failures,
        "oracle": "independent ADR-0007 integer allocation, common-prefix and renormalization",
    }


# ---------------------------------------------------------------------------
# ADG: independent Algorithm-1 paper partition + little-endian group path
# ---------------------------------------------------------------------------


def _adg_oracle_decode(reference: ReferenceDistribution, token_id: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    root = ADGGroup(
        token_ids=tuple(range(reference.vocab_size)),
        probabilities=tuple(float(x) for x in reference.probabilities),
    )
    group = root
    recovered: list[int] = []
    path: list[int] = []
    while True:
        children = adg_paper_partition(group, repository_semantics=False)
        if children is None:
            if token_id not in group.token_ids:
                raise ValueError("token outside independent ADG terminal group")
            break
        selected_index = None
        for index, child in enumerate(children):
            if token_id in child.token_ids:
                selected_index = index
                group = child
                break
        if selected_index is None:
            raise ValueError("token outside independent ADG grouping")
        width = int(math.log2(len(children)))
        recovered.extend(_little_bits(selected_index, width))
        path.append(selected_index)
    return tuple(recovered), tuple(path)


def audit_adg(random_cases: int) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    checks = 0
    fixtures = adg_fixture_suite(random_cases)
    for fixture in fixtures:
        reference = _reference(fixture.probabilities)
        environment = _environment(reference.vocab_size)
        context = StepContext(0, reference)
        for token_id in range(reference.vocab_size):
            expected_bits, expected_path = _adg_oracle_decode(reference, token_id)
            decoder = ADGMethod().create_decoder(
                config={},
                environment=environment,
                expected_payload_bits=None,
            )
            progress = decoder.observe(context, token_id)
            checks += 1
            observed_bits = tuple(progress.recovered_bits)
            observed_path = tuple(progress.method_trace.get("group_path", ()))
            if observed_bits != expected_bits or observed_path != expected_path:
                failures.append(
                    _failure(
                        "adg",
                        fixture.fixture_id,
                        "paper Algorithm-1 inverse mapping mismatch",
                        token_id=token_id,
                        expected_bits=expected_bits,
                        observed_bits=observed_bits,
                        expected_group_path=expected_path,
                        observed_group_path=observed_path,
                    )
                )
                break
    return {
        "status": "pass" if not failures else "fail",
        "fixture_count": len(fixtures),
        "check_count": checks,
        "failure_count": len(failures),
        "failures": failures,
        "oracle": "independent Stage-4 paper Algorithm-1 partition + little-endian group-index inverse",
    }


# ---------------------------------------------------------------------------
# Discop: pinned-source oracle with persistent sender/receiver RNG state
# ---------------------------------------------------------------------------


def _discop_reference_encode_persistent(
    reference: ReferenceDistribution,
    *,
    rng: random.Random,
    secret_bits: tuple[int, ...],
    secret_position: int,
) -> tuple[int, tuple[int, ...], int, int]:
    indices = tuple(int(x) for x in reference.token_order)
    probs = tuple(float(reference.probabilities[index]) for index in indices)
    node = discop_reference_tree(indices, probs)
    consumed: list[int] = []
    draws = 0
    while not node.is_leaf:
        path_0, path_1 = discop_reference_branches(node, rng.random())
        draws += 1
        if path_0 != path_1:
            if secret_position >= len(secret_bits):
                raise RuntimeError("independent Discop secret exhausted")
            bit = int(secret_bits[secret_position])
            secret_position += 1
            consumed.append(bit)
            branch = path_0 if bit == 0 else path_1
        else:
            branch = path_0
        node = node.left if branch == -1 else node.right
        if node is None:
            raise RuntimeError("independent Discop encoder entered invalid branch")
    return int(node.token_id), tuple(consumed), draws, secret_position


def _discop_reference_decode_persistent(
    reference: ReferenceDistribution,
    token_id: int,
    *,
    rng: random.Random,
) -> tuple[tuple[int, ...], int]:
    indices = tuple(int(x) for x in reference.token_order)
    probs = tuple(float(reference.probabilities[index]) for index in indices)
    node = discop_reference_tree(indices, probs, search_for=int(token_id))
    recovered: list[int] = []
    draws = 0
    while not node.is_leaf:
        path_0, path_1 = discop_reference_branches(node, rng.random())
        draws += 1
        if path_0 != path_1:
            if node.search_path == 9:
                raise RuntimeError("independent Discop decoder cannot locate token")
            if path_0 == -1:
                recovered.append(0 if node.search_path == -1 else 1)
            else:
                recovered.append(1 if node.search_path == -1 else 0)
            branch = node.search_path
        else:
            branch = path_0
            if branch != node.search_path and node.search_path != 9:
                # If both copies select the other branch, the observed token is
                # incompatible with this synchronized RNG state.
                raise RuntimeError("independent Discop branch unavailable")
        node = node.left if branch == -1 else node.right
        if node is None:
            raise RuntimeError("independent Discop decoder entered invalid branch")
    if node.search_path != 0:
        raise RuntimeError("independent Discop decoder did not reach observed leaf")
    return tuple(recovered), draws


def audit_discop(random_cases: int) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    checks = 0
    rng_probs = np.random.default_rng(20261102)
    sequence_count = max(4, min(12, random_cases if random_cases > 0 else 4))
    steps = 32

    for sequence_index in range(sequence_count):
        seed = 1000 + sequence_index * 17
        sender_rng = random.Random(seed)
        receiver_rng = random.Random(seed)
        secret_rng = random.Random(9000 + sequence_index)
        secret_bits = tuple(secret_rng.randrange(2) for _ in range(8192))
        secret_position = 0
        environment = _environment(16)
        decoder = DiscopMethod().create_decoder(
            config={},
            environment=environment,
            random_source=MethodRandomSource(seed),
            expected_payload_bits=None,
        )
        expected_all: list[int] = []
        expected_total_draws = 0

        for step_index in range(steps):
            concentration = (0.15, 0.5, 1.0, 3.0)[step_index % 4]
            probs = rng_probs.dirichlet(np.full(16, concentration, dtype=np.float64))
            reference = _reference(probs)
            token_id, encoded_bits, sender_draws, secret_position = _discop_reference_encode_persistent(
                reference,
                rng=sender_rng,
                secret_bits=secret_bits,
                secret_position=secret_position,
            )
            expected_bits, receiver_draws = _discop_reference_decode_persistent(
                reference,
                token_id,
                rng=receiver_rng,
            )
            if expected_bits != encoded_bits or receiver_draws != sender_draws:
                failures.append(
                    _failure(
                        "discop",
                        f"sequence-{sequence_index}",
                        "independent pinned-source encoder/decoder disagreed",
                        step=step_index,
                        encoded_bits=encoded_bits,
                        decoded_bits=expected_bits,
                        sender_rng_draws=sender_draws,
                        receiver_rng_draws=receiver_draws,
                    )
                )
                break

            try:
                progress = decoder.observe(StepContext(step_index, reference), token_id)
            except Exception as exc:
                failures.append(
                    _failure(
                        "discop",
                        f"sequence-{sequence_index}",
                        "production decoder raised on pinned-source valid sequence",
                        step=step_index,
                        token_id=token_id,
                        exception=f"{type(exc).__name__}: {exc}",
                    )
                )
                break

            checks += 1
            observed_bits = tuple(progress.recovered_bits)
            observed_draws = int(progress.method_trace.get("rng_draws", -1))
            if observed_bits != expected_bits or observed_draws != receiver_draws:
                failures.append(
                    _failure(
                        "discop",
                        f"sequence-{sequence_index}",
                        "persistent-RNG decoder mismatch",
                        step=step_index,
                        token_id=token_id,
                        expected_bits=expected_bits,
                        observed_bits=observed_bits,
                        expected_rng_draws=receiver_draws,
                        observed_rng_draws=observed_draws,
                    )
                )
                break
            expected_all.extend(expected_bits)
            expected_total_draws += receiver_draws
        else:
            final = decoder.finalize()
            if tuple(final.recovered_bits) != tuple(expected_all) or int(final.metadata.get("rng_draws", -1)) != expected_total_draws:
                failures.append(
                    _failure(
                        "discop",
                        f"sequence-{sequence_index}",
                        "persistent-RNG final state mismatch",
                        expected_bits=tuple(expected_all),
                        observed_bits=tuple(final.recovered_bits),
                        expected_rng_draws=expected_total_draws,
                        observed_rng_draws=int(final.metadata.get("rng_draws", -1)),
                    )
                )

    return {
        "status": "pass" if not failures else "fail",
        "sequence_count": sequence_count,
        "steps_per_sequence": steps,
        "check_count": checks,
        "failure_count": len(failures),
        "failures": failures,
        "oracle": "independent pinned Cython translation with persistent sender/receiver PRNG state",
    }


# ---------------------------------------------------------------------------
# DAIRstega: independent paper allocation + stateful posterior-window inverse
# ---------------------------------------------------------------------------


def _dair_prefix_bits(start: int, stop: int, alpha: int) -> tuple[int, ...]:
    width = prefix_length_paper(
        type("IntervalView", (), {"start": start, "stop": stop})(),
        alpha,
    )
    return _msb_bits(start, alpha)[:width]


def _dair_advance_support(
    lower: int, upper: int, *, prefix_bits: int, alpha: int
) -> tuple[int, int]:
    suffix_bits = alpha - prefix_bits
    if suffix_bits == 0:
        suffix_lower = 0
        suffix_upper = 0
    else:
        mask = (1 << suffix_bits) - 1
        suffix_lower = lower & mask
        suffix_upper = upper & mask
    return suffix_lower << prefix_bits, ((suffix_upper + 1) << prefix_bits) - 1


def audit_dairstega(random_cases: int) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    checks = 0
    # Include all Stage-5 beta families and multiple alpha values.  Small alpha
    # fixtures deliberately exercise singleton intervals and large prefixes.
    configurations = (
        (3, 1.0, 8),
        (8, 1.0, 16),
        (8, 0.75, 16),
        (8, 0.5, 16),
        (16, 1.0, 24),
        (16, 0.75, 24),
        (16, 0.5, 24),
    )
    sequence_count = max(3, min(8, random_cases if random_cases > 0 else 3))
    steps = 24
    rng = np.random.default_rng(20261103)

    for config_index, (alpha, beta, top_k) in enumerate(configurations):
        for sequence_index in range(sequence_count):
            fixture_id = f"a{alpha}-b{beta:g}-k{top_k}-s{sequence_index}"
            vocab_size = max(top_k, 8)
            environment = _environment(vocab_size)
            decoder = DAIRMethod().create_decoder(
                config={"alpha": alpha, "beta": beta, "top_k": top_k},
                environment=environment,
                expected_payload_bits=None,
            )
            window_lower = 0
            window_upper = (1 << alpha) - 1
            expected_all: list[int] = []

            for step_index in range(steps):
                concentration = (0.25, 0.8, 2.0, 5.0)[(step_index + sequence_index) % 4]
                probs = rng.dirichlet(np.full(vocab_size, concentration, dtype=np.float64))
                reference = _reference(probs)
                fixture = DAIRFixture(
                    fixture_id=fixture_id,
                    probabilities=tuple(float(x) for x in reference.probabilities),
                    alpha=alpha,
                    beta=beta,
                    top_k=top_k,
                )
                intervals = paper_allocate(fixture)
                overlapping = [
                    interval
                    for interval in intervals
                    if max(window_lower, interval.start) <= min(window_upper, interval.stop - 1)
                ]
                if not overlapping:
                    failures.append(_failure("dairstega", fixture_id, "independent posterior support has no overlapping interval", step=step_index))
                    break

                # Cycle through the currently reachable intervals rather than
                # always choosing the first one; this exercises conditioned
                # posterior states and partial interval intersections.
                interval = overlapping[(step_index * 5 + config_index + sequence_index) % len(overlapping)]
                posterior_lower = max(window_lower, interval.start)
                posterior_upper = min(window_upper, interval.stop - 1)
                expected_bits = _dair_prefix_bits(interval.start, interval.stop, alpha)
                expected_next = _dair_advance_support(
                    posterior_lower,
                    posterior_upper,
                    prefix_bits=len(expected_bits),
                    alpha=alpha,
                )

                try:
                    progress = decoder.observe(
                        StepContext(step_index, reference),
                        int(interval.token_id),
                    )
                except Exception as exc:
                    failures.append(
                        _failure(
                            "dairstega",
                            fixture_id,
                            "production decoder raised on paper-oracle valid token",
                            step=step_index,
                            token_id=int(interval.token_id),
                            exception=f"{type(exc).__name__}: {exc}",
                        )
                    )
                    break

                checks += 1
                observed_bits = tuple(progress.recovered_bits)
                observed_next = (
                    int(progress.method_trace.get("next_window_lower", -1)),
                    int(progress.method_trace.get("next_window_upper", -1)),
                )
                if observed_bits != expected_bits or observed_next != expected_next:
                    failures.append(
                        _failure(
                            "dairstega",
                            fixture_id,
                            "stateful paper-inverse mismatch",
                            step=step_index,
                            token_id=int(interval.token_id),
                            interval=(interval.start, interval.stop),
                            posterior=(posterior_lower, posterior_upper),
                            expected_bits=expected_bits,
                            observed_bits=observed_bits,
                            expected_next_window=expected_next,
                            observed_next_window=observed_next,
                        )
                    )
                    break

                expected_all.extend(expected_bits)
                window_lower, window_upper = expected_next
            else:
                final = decoder.finalize()
                if tuple(final.recovered_bits) != tuple(expected_all):
                    failures.append(
                        _failure(
                            "dairstega",
                            fixture_id,
                            "final accumulated recovered bits mismatch",
                            expected_bits=tuple(expected_all),
                            observed_bits=tuple(final.recovered_bits),
                        )
                    )

    return {
        "status": "pass" if not failures else "fail",
        "configuration_count": len(configurations),
        "sequence_count_per_configuration": sequence_count,
        "steps_per_sequence": steps,
        "check_count": checks,
        "failure_count": len(failures),
        "failures": failures,
        "oracle": "independent paper/ADR interval allocation + posterior-window state transition",
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def evaluate(random_cases: int = 32) -> dict[str, object]:
    if random_cases < 0:
        raise ValueError("random_cases must be non-negative")
    methods = {
        "bins": audit_bins(),
        "huffman": audit_huffman(random_cases),
        "arithmetic_coding": audit_arithmetic(),
        "adg": audit_adg(random_cases),
        "discop": audit_discop(random_cases),
        "dairstega": audit_dairstega(random_cases),
    }
    failed = [method for method, result in methods.items() if result["status"] != "pass"]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if not failed else "fail",
        "purpose": "independent decoder correctness/conformance gate after Stage-5 Pilot-B direct-token replay",
        "cpu_only": True,
        "production_method_sources_modified": False,
        "method_count": len(methods),
        "failed_methods": failed,
        "methods": methods,
        "interpretation": (
            "PASS means each normalized streaming decoder agrees with a decoder oracle "
            "implemented outside the production method helpers on deterministic synthetic "
            "fixtures. It complements, but does not replace, the 504-run direct-token replay."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--random-cases",
        type=int,
        default=32,
        help="random synthetic cases used by applicable method audits (default: 32)",
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.random_cases < 0:
        parser.error("--random-cases must be non-negative")

    result = evaluate(args.random_cases)
    print("Stage 5 independent decoder conformance audit")
    for method, payload in result["methods"].items():
        print(
            f"{method:20s} {str(payload['status']).upper():4s} "
            f"checks={payload.get('check_count', 0)} failures={payload['failure_count']}"
        )
    print("overall:", str(result["status"]).upper())
    print("CPU-only: yes; no LM weights/GPU required")

    if args.json_output is not None:
        output = args.json_output
        if not output.is_absolute():
            output = REPO_ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("json:", output)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
