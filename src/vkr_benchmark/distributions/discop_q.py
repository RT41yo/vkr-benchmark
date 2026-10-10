"""Independent exact construction of the distribution induced by Discop.

The calculator is deliberately separate from the Discop encoder/decoder.  It
reconstructs the pinned Huffman partitioning on its own and integrates both
rotated distribution copies over the full random-pointer domain.  Uniform,
independent secret bits are marginalized explicitly whenever the copies choose
different branches.

This is the canonical Q_stego construction used by normalized Discop benchmark
runs.  It does not use the encoder's sampled secret bits, sampled PRNG path, or
a Q=P equality certificate.
"""

from __future__ import annotations

import numpy as np

from vkr_benchmark.distributions.types import ReferenceDistribution


def _copy_branches(left_fraction: float, pointer: float) -> tuple[int, int]:
    """Return left/right choices of Discop copies for a normalized node."""

    copy_0 = pointer
    copy_1 = pointer + 0.5
    if copy_1 > 1.0:
        copy_1 -= 1.0
    return (
        0 if copy_0 < left_fraction else 1,
        0 if copy_1 < left_fraction else 1,
    )


def branch_selection_probabilities(
    *, left_probability: float, node_probability: float
) -> tuple[float, float]:
    """Integrate exact Discop branch probabilities over the pointer domain.

    The interval [0, 1) is split at every point where either rotated copy can
    change branch.  Within each resulting interval both copy choices are
    constant.  If the copies disagree, an independent uniform secret bit gives
    each copy weight 1/2; if they agree, the common branch gets the full
    interval mass.
    """

    if not np.isfinite(left_probability) or not np.isfinite(node_probability):
        raise ValueError("Discop probabilities must be finite")
    if node_probability <= 0.0:
        raise ValueError("Discop node probability must be positive")
    if left_probability < 0.0 or left_probability > node_probability:
        raise ValueError("left probability must lie inside the node mass")

    left_fraction = left_probability / node_probability
    left_fraction = min(1.0, max(0.0, left_fraction))

    breakpoints = {0.0, 0.5, 1.0}
    if 0.0 < left_fraction < 1.0:
        breakpoints.add(float(left_fraction))
        shifted = (left_fraction - 0.5) % 1.0
        if 0.0 < shifted < 1.0:
            breakpoints.add(float(shifted))

    ordered = sorted(breakpoints)
    branch_mass = [0.0, 0.0]
    for lower, upper in zip(ordered[:-1], ordered[1:], strict=True):
        if upper <= lower:
            continue
        width = upper - lower
        midpoint = 0.5 * (lower + upper)
        branch_0, branch_1 = _copy_branches(left_fraction, midpoint)
        if branch_0 == branch_1:
            branch_mass[branch_0] += width
        else:
            branch_mass[branch_0] += 0.5 * width
            branch_mass[branch_1] += 0.5 * width

    total = branch_mass[0] + branch_mass[1]
    if not np.isclose(total, 1.0, rtol=0.0, atol=1e-14):
        raise RuntimeError(
            "Discop branch integration lost probability mass: " f"{total!r}"
        )
    return float(branch_mass[0]), float(branch_mass[1])


def explicit_discop_q(reference: ReferenceDistribution) -> np.ndarray:
    """Construct the full induced Discop distribution Q_stego independently.

    The two-queue tree construction mirrors the pinned author's tie semantics,
    but uses an array-backed representation so the exact calculation is
    practical for full language-model vocabularies.  The root carries unit
    probability.  Each internal node is integrated independently over all
    pointer positions and both equally likely secret-bit choices; resulting
    mass is propagated to the leaves.
    """

    ordered = np.asarray(reference.token_order, dtype=np.int64)
    support_size = int(ordered.size)
    if support_size <= 0:
        raise ValueError("Discop explicit Q requires non-empty support")

    # The pinned source receives descending-probability candidates, pushes them
    # into q1 from the end, and therefore consumes leaves in ascending
    # probability order. Reversing canonical token_order reproduces that queue,
    # including deterministic leaf ties.
    leaf_tokens = ordered[::-1].copy()
    if support_size == 1:
        q = np.zeros(reference.vocab_size, dtype=np.float64)
        q[int(leaf_tokens[0])] = 1.0
        q.setflags(write=False)
        return q

    node_count = 2 * support_size - 1
    node_probabilities = np.empty(node_count, dtype=np.float64)
    node_probabilities[:support_size] = reference.probabilities[leaf_tokens].astype(
        np.float64, copy=False
    )
    left_child = np.full(node_count, -1, dtype=np.int64)
    right_child = np.full(node_count, -1, dtype=np.int64)

    leaf_pos = 0
    merged_pos = support_size
    next_internal = support_size

    def pop_queue() -> int:
        nonlocal leaf_pos, merged_pos
        leaves_available = leaf_pos < support_size
        merged_available = merged_pos < next_internal
        if leaves_available and merged_available:
            # Pinned Cython chooses q2 (merged queue) on equality.
            if node_probabilities[leaf_pos] < node_probabilities[merged_pos]:
                result = leaf_pos
                leaf_pos += 1
                return result
            result = merged_pos
            merged_pos += 1
            return result
        if leaves_available:
            result = leaf_pos
            leaf_pos += 1
            return result
        if merged_available:
            result = merged_pos
            merged_pos += 1
            return result
        raise RuntimeError("Discop explicit-Q tree queues unexpectedly became empty")

    for node_index in range(support_size, node_count):
        first = pop_queue()
        second = pop_queue()
        left_child[node_index] = first
        right_child[node_index] = second
        node_probabilities[node_index] = (
            node_probabilities[first] + node_probabilities[second]
        )
        next_internal = node_index + 1

    root = node_count - 1
    node_mass = np.zeros(node_count, dtype=np.float64)
    node_mass[root] = 1.0

    # Children always have smaller indices than their parent, so reverse node
    # order is a valid top-down traversal without recursion.
    for node_index in range(root, support_size - 1, -1):
        left = int(left_child[node_index])
        right = int(right_child[node_index])
        left_mass, right_mass = branch_selection_probabilities(
            left_probability=float(node_probabilities[left]),
            node_probability=float(node_probabilities[node_index]),
        )
        incoming = float(node_mass[node_index])
        node_mass[left] += incoming * left_mass
        node_mass[right] += incoming * right_mass

    q = np.zeros(reference.vocab_size, dtype=np.float64)
    q[leaf_tokens] = node_mass[:support_size]
    total = float(np.sum(q, dtype=np.float64))
    if not np.isclose(total, 1.0, rtol=0.0, atol=1e-12):
        raise RuntimeError(f"explicit Discop Q_stego does not sum to 1: {total!r}")
    q.setflags(write=False)
    return q
