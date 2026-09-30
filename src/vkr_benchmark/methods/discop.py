"""Normalized Discop steganographic method.

Algorithmic reference:
    comydream/Discop, src/stega_cy.pyx,
    commit 3c3a10099a242eae405b49cc4d09fba1abb148ad.

The adapter preserves the text-generation core of Discop: each P_reference is
recursively decomposed into binary distributions using the reference Huffman
construction; at every internal node two pointers separated by half the current
node mass are evaluated; when they select different branches one secret bit
chooses the distribution copy, otherwise the common branch is followed without
consuming payload. Sender and receiver consume exactly one shared-PRNG draw per
visited internal node.

LM/tokenizer ownership, common generation policy, text transport, key storage,
and metrics remain benchmark infrastructure responsibilities. The adapter
reports the paper/reference equality claim Q_stego = P_reference as an analytic
certificate; Stage-4 validation tests independently check that claim on a
synthetic distribution rather than accepting the certificate alone.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from vkr_benchmark.distributions import DistributionInfo, QSource, StepContext
from vkr_benchmark.errors import (
    ConfigurationError,
    MethodError,
    SessionStateError,
)
from vkr_benchmark.methods.base import (
    DecoderSession,
    EncoderSession,
    KeyMaterial,
    MethodConfig,
    StegoMethod,
)
from vkr_benchmark.methods.types import (
    DecodeProgress,
    DecoderFinalization,
    EncodeDecision,
    EncoderFinalization,
    MethodEnvironment,
)
from vkr_benchmark.randomness import RandomSource, SecretSource

_REFERENCE_COMMIT = "3c3a10099a242eae405b49cc4d09fba1abb148ad"


@dataclass(frozen=True, slots=True)
class DiscopConfig:
    """Normalized Discop has no method-local tuning parameter in Stage 4."""

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "DiscopConfig":
        if values:
            raise ConfigurationError(
                f"unknown Discop configuration fields: {sorted(values)}"
            )
        return cls()


@dataclass(frozen=True, slots=True)
class _Node:
    probability: float
    token_id: int | None = None
    left: "_Node | None" = None
    right: "_Node | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.token_id is not None


def _pop_reference_queue(
    leaves: deque[_Node], merged: deque[_Node]
) -> _Node:
    """Mirror ``create_huffman_tree`` queue selection in pinned Cython code.

    The reference implementation prefers the merged-node queue when the two
    front probabilities are exactly equal. That tie rule is preserved here.
    """

    if leaves and merged:
        if leaves[0].probability < merged[0].probability:
            return leaves.popleft()
        return merged.popleft()
    if leaves:
        return leaves.popleft()
    if merged:
        return merged.popleft()
    raise MethodError("Discop Huffman queue unexpectedly became empty")


def _build_tree(context: StepContext, environment: MethodEnvironment) -> _Node:
    reference = context.reference
    if reference.vocab_size != environment.output_vocab_size:
        raise MethodError(
            "Discop MethodEnvironment output vocabulary does not match P_reference"
        )
    if reference.support_size <= 0:
        raise MethodError("Discop requires non-empty P_reference support")

    ordered = tuple(int(token_id) for token_id in reference.token_order)
    for token_id in ordered:
        if not environment.is_allowed(token_id):
            raise MethodError(
                f"P_reference support token {token_id} lies outside V_allowed"
            )

    # The author code receives candidates in descending probability order, then
    # pushes them into q1 from the end to the beginning. q1 is therefore an
    # ascending-probability FIFO queue. P_reference.token_order is the
    # benchmark's deterministic descending order, so reversing it reproduces
    # the same queue construction while also defining deterministic leaf ties.
    leaves: deque[_Node] = deque(
        _Node(
            probability=float(reference.probabilities[token_id]),
            token_id=token_id,
        )
        for token_id in reversed(ordered)
    )
    if len(leaves) == 1:
        return leaves[0]

    merged: deque[_Node] = deque()
    while len(leaves) + len(merged) > 1:
        first = _pop_reference_queue(leaves, merged)
        second = _pop_reference_queue(leaves, merged)
        merged.append(
            _Node(
                probability=first.probability + second.probability,
                left=first,
                right=second,
            )
        )

    return merged[0] if merged else leaves[0]


def _branches(node: _Node, ptr: float) -> tuple[int, int]:
    """Return author-compatible branches for copies 0 and 1.

    Branch ``0`` means left and ``1`` means right.  The strict ``>`` wrapping
    test intentionally matches the pinned Cython implementation rather than
    replacing it with a mathematically cleaner modulo expression at the
    measure-zero boundary.
    """

    if node.left is None or node.right is None:
        raise MethodError("Discop branch query requires an internal tree node")
    if not 0.0 <= ptr < 1.0:
        raise MethodError("RandomSource.random() must return a value in [0, 1)")

    prob_sum = node.probability
    ptr_0 = ptr * prob_sum
    ptr_1 = (ptr + 0.5) * prob_sum
    if ptr_1 > prob_sum:
        ptr_1 -= prob_sum

    partition = node.left.probability
    branch_0 = 0 if ptr_0 < partition else 1
    branch_1 = 0 if ptr_1 < partition else 1
    return branch_0, branch_1


def _child(node: _Node, branch: int) -> _Node:
    if branch == 0 and node.left is not None:
        return node.left
    if branch == 1 and node.right is not None:
        return node.right
    raise MethodError("invalid Discop tree branch")


def _path_to_token(root: _Node, token_id: int) -> tuple[int, ...] | None:
    if root.is_leaf:
        return () if root.token_id == token_id else None
    assert root.left is not None and root.right is not None
    left = _path_to_token(root.left, token_id)
    if left is not None:
        return (0, *left)
    right = _path_to_token(root.right, token_id)
    if right is not None:
        return (1, *right)
    return None


def _distribution_info() -> DistributionInfo:
    return DistributionInfo.reference_equality(
        source=QSource.ANALYTIC_THEORY,
        metadata={
            "method": "discop",
            "reference_commit": _REFERENCE_COMMIT,
            "claim": "distribution_copies_preserve_reference_distribution",
        },
    )


class DiscopEncoderSession(EncoderSession):
    """Streaming Discop encoder with synchronized per-node PRNG draws."""

    def __init__(
        self,
        *,
        environment: MethodEnvironment,
        secret_source: SecretSource,
        random_source: RandomSource,
    ) -> None:
        self._environment = environment
        self._secret_source = secret_source
        self._random_source = random_source
        self._start_secret_position = secret_source.position
        self._steps = 0
        self._rng_draws = 0
        self._finalized = False

    @property
    def done(self) -> bool:
        return self._finalized

    def step(self, context: StepContext) -> EncodeDecision:
        if self._finalized:
            raise SessionStateError("cannot call Discop encoder.step() after finalize()")

        node = _build_tree(context, self._environment)
        bits_consumed = 0
        draws = 0
        path: list[int] = []

        while not node.is_leaf:
            ptr = self._random_source.random()
            draws += 1
            branch_0, branch_1 = _branches(node, ptr)
            if branch_0 != branch_1:
                bit = self._secret_source.read_bits(1)[0]
                bits_consumed += 1
                branch = branch_0 if bit == 0 else branch_1
            else:
                branch = branch_0
            path.append(branch)
            node = _child(node, branch)

        assert node.token_id is not None
        self._steps += 1
        self._rng_draws += draws
        return EncodeDecision(
            token_id=node.token_id,
            bits_consumed=bits_consumed,
            distribution_info=_distribution_info(),
            method_trace={
                "tree_depth": len(path),
                "path": tuple(path),
                "rng_draws": draws,
                "bits_consumed": bits_consumed,
            },
        )

    def finalize(self) -> EncoderFinalization:
        if self._finalized:
            raise SessionStateError("Discop encoder has already been finalized")
        self._finalized = True
        payload_bits = self._secret_source.position - self._start_secret_position
        return EncoderFinalization(
            payload_bits=payload_bits,
            termination_reason="fixed_carrier_tokens",
            metadata={
                "steps": self._steps,
                "rng_draws": self._rng_draws,
                "reference_commit": _REFERENCE_COMMIT,
            },
        )


class DiscopDecoderSession(DecoderSession):
    """Decoder that replays the sender's tree traversal and PRNG sequence."""

    def __init__(
        self,
        *,
        environment: MethodEnvironment,
        random_source: RandomSource,
        expected_payload_bits: int | None,
    ) -> None:
        if expected_payload_bits is not None and expected_payload_bits < 0:
            raise ConfigurationError("expected_payload_bits must be non-negative")
        self._environment = environment
        self._random_source = random_source
        self._expected_payload_bits = expected_payload_bits
        self._recovered: list[int] = []
        self._steps = 0
        self._rng_draws = 0
        self._finalized = False

    @property
    def done(self) -> bool:
        if self._finalized:
            return True
        if self._expected_payload_bits is None:
            return False
        return len(self._recovered) >= self._expected_payload_bits

    def observe(self, context: StepContext, observed_token_id: int) -> DecodeProgress:
        if self._finalized:
            raise SessionStateError("cannot call Discop decoder.observe() after finalize()")

        root = _build_tree(context, self._environment)
        path = _path_to_token(root, int(observed_token_id))
        if path is None:
            raise MethodError(
                f"observed token {observed_token_id} is outside current P_reference support"
            )

        node = root
        recovered: list[int] = []
        draws = 0
        for actual_branch in path:
            ptr = self._random_source.random()
            draws += 1
            branch_0, branch_1 = _branches(node, ptr)
            if branch_0 != branch_1:
                if actual_branch == branch_0:
                    recovered.append(0)
                elif actual_branch == branch_1:
                    recovered.append(1)
                else:  # pragma: no cover - binary branches make this impossible
                    raise MethodError("observed Discop branch cannot be decoded")
            elif actual_branch != branch_0:
                raise MethodError(
                    "Discop sender/receiver state mismatch: observed token follows "
                    "a branch unavailable to both distribution copies"
                )
            node = _child(node, actual_branch)

        self._recovered.extend(recovered)
        self._steps += 1
        self._rng_draws += draws
        return DecodeProgress(
            recovered_bits=tuple(recovered),
            method_trace={
                "tree_depth": len(path),
                "path": path,
                "rng_draws": draws,
                "bits_recovered": len(recovered),
            },
        )

    def finalize(self) -> DecoderFinalization:
        if self._finalized:
            raise SessionStateError("Discop decoder has already been finalized")
        self._finalized = True

        recovered = tuple(self._recovered)
        complete = True
        if self._expected_payload_bits is not None:
            complete = len(recovered) >= self._expected_payload_bits
            recovered = recovered[: self._expected_payload_bits]

        return DecoderFinalization(
            recovered_bits=recovered,
            complete=complete,
            metadata={
                "steps": self._steps,
                "rng_draws": self._rng_draws,
                "raw_recovered_bits": len(self._recovered),
                "reference_commit": _REFERENCE_COMMIT,
            },
        )


class DiscopMethod(StegoMethod):
    """Factory for normalized Discop encoder/decoder sessions."""

    method_id = "discop"

    @staticmethod
    def _require_rng(random_source: RandomSource | None) -> RandomSource:
        if random_source is None:
            raise ConfigurationError(
                "normalized Discop requires synchronized method randomness; "
                "the experiment runner derives it from method.key"
            )
        return random_source

    def create_encoder(
        self,
        *,
        config: MethodConfig,
        environment: MethodEnvironment,
        secret_source: SecretSource,
        random_source: RandomSource | None = None,
        key: KeyMaterial = None,
        target_payload_bits: int | None = None,
    ) -> EncoderSession:
        del key
        if target_payload_bits is not None:
            raise ConfigurationError(
                "Discop normalized adapter currently uses fixed_carrier_tokens termination"
            )
        DiscopConfig.from_mapping(config)
        return DiscopEncoderSession(
            environment=environment,
            secret_source=secret_source,
            random_source=self._require_rng(random_source),
        )

    def create_decoder(
        self,
        *,
        config: MethodConfig,
        environment: MethodEnvironment,
        random_source: RandomSource | None = None,
        key: KeyMaterial = None,
        expected_payload_bits: int | None = None,
    ) -> DecoderSession:
        del key
        DiscopConfig.from_mapping(config)
        return DiscopDecoderSession(
            environment=environment,
            random_source=self._require_rng(random_source),
            expected_payload_bits=expected_payload_bits,
        )
