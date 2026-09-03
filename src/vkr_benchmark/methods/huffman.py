"""Normalized Huffman steganographic method.

Algorithmic reference:
    harvardnlp/NeuralSteganography, huffman_baseline.py + huffman.py,
    commit 14e982564aeaf9a33f7b4de440deda2184d17f12.

The normalized adapter preserves the method's core semantics: at each carrier
step take the top ``2**bits_per_word`` candidates from the current reference
distribution, build a Huffman tree from their probabilities, traverse the tree
with secret bits until a leaf is reached, and emit that leaf's token.  The
decoder rebuilds the same tree from the same P_reference and maps the observed
token back to its Huffman code.

LM/tokenizer ownership, common generation policy, text transport, metrics, and
BPE-repair heuristics are deliberately outside this adapter.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from vkr_benchmark.distributions import DistributionInfo, QMode, QSource, StepContext
from vkr_benchmark.errors import (
    ConfigurationError,
    MethodError,
    SessionStateError,
    UnsupportedConfigurationError,
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


@dataclass(frozen=True, slots=True)
class HuffmanConfig:
    """Normalized Huffman parameters.

    ``bits_per_word`` is the name used by the Harvard reference code.  It does
    *not* mean that every generated token embeds exactly that many bits.  It
    controls only the candidate-set size ``2**bits_per_word``; actual payload
    per token equals the selected Huffman code length and is therefore
    variable.
    """

    bits_per_word: int

    def __post_init__(self) -> None:
        if self.bits_per_word <= 0:
            raise ConfigurationError(
                "Huffman bits_per_word must be a positive integer"
            )

    @property
    def candidate_count(self) -> int:
        return 1 << self.bits_per_word

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "HuffmanConfig":
        unknown = set(values) - {"bits_per_word"}
        if unknown:
            raise ConfigurationError(
                f"unknown Huffman configuration fields: {sorted(unknown)}"
            )
        if "bits_per_word" not in values:
            raise ConfigurationError(
                "Huffman configuration requires bits_per_word"
            )
        bits_per_word = values["bits_per_word"]
        if isinstance(bits_per_word, bool) or not isinstance(bits_per_word, int):
            raise ConfigurationError("Huffman bits_per_word must be an integer")
        return cls(bits_per_word=bits_per_word)


@dataclass(frozen=True, slots=True)
class _HuffmanNode:
    """One deterministic Huffman tree node."""

    weight: float
    min_token_id: int
    token_id: int | None = None
    left: "_HuffmanNode | None" = None
    right: "_HuffmanNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.token_id is not None


@dataclass(frozen=True, slots=True)
class HuffmanCodebook:
    """Per-step candidate set, deterministic tree, codes, and induced Q."""

    candidate_token_ids: tuple[int, ...]
    root: _HuffmanNode
    codes: Mapping[int, tuple[int, ...]]
    q_stego: np.ndarray

    def __post_init__(self) -> None:
        if len(self.candidate_token_ids) < 2:
            raise ValueError("Huffman codebook requires at least two candidates")

        codes = {
            int(token_id): tuple(int(bit) for bit in bits)
            for token_id, bits in self.codes.items()
        }
        if set(codes) != set(self.candidate_token_ids):
            raise ValueError("Huffman codes must cover exactly the candidate tokens")
        if any(not bits for bits in codes.values()):
            raise ValueError("Huffman leaf codes must be non-empty")
        if any(bit not in (0, 1) for bits in codes.values() for bit in bits):
            raise ValueError("Huffman codes must contain only 0/1")

        q = np.asarray(self.q_stego, dtype=np.float64).copy()
        if q.ndim != 1:
            raise ValueError("q_stego must be one-dimensional")
        q.setflags(write=False)

        object.__setattr__(self, "codes", MappingProxyType(codes))
        object.__setattr__(self, "q_stego", q)

    def code_for_token(self, token_id: int) -> tuple[int, ...]:
        try:
            return self.codes[int(token_id)]
        except KeyError as exc:
            raise MethodError(
                f"observed token {token_id} is outside the current Huffman candidate set"
            ) from exc



def _validate_static_configuration(
    *, config: HuffmanConfig, environment: MethodEnvironment
) -> None:
    if config.candidate_count > len(environment.allowed_token_ids):
        raise UnsupportedConfigurationError(
            "Huffman candidate set exceeds V_allowed: "
            f"2**{config.bits_per_word}={config.candidate_count} > "
            f"|V_allowed|={len(environment.allowed_token_ids)}"
        )



def _build_codebook(
    *,
    context: StepContext,
    config: HuffmanConfig,
    environment: MethodEnvironment,
) -> HuffmanCodebook:
    """Build the deterministic per-step Huffman tree and exact induced Q.

    Candidate selection follows the benchmark's canonical token order.  The
    Harvard tree compares heap nodes only by frequency, so exact equal-weight
    merges are not explicitly specified.  Normalized mode adds a deterministic
    tie key: the minimum token ID in each active subtree.  The first popped
    node becomes the left (0) child, the second the right (1) child.
    """

    reference = context.reference
    if reference.vocab_size != environment.output_vocab_size:
        raise MethodError(
            "Huffman MethodEnvironment output vocabulary does not match P_reference"
        )

    count = config.candidate_count
    if reference.support_size < count:
        raise UnsupportedConfigurationError(
            "current P_reference has fewer positive-probability tokens than the "
            f"Huffman candidate set: support={reference.support_size}, "
            f"required={count}"
        )

    candidate_token_ids = tuple(
        int(token_id) for token_id in reference.token_order[:count]
    )
    for token_id in candidate_token_ids:
        if not environment.is_allowed(token_id):
            raise MethodError(
                f"P_reference candidate token {token_id} lies outside V_allowed"
            )

    # Active heap entries use (weight, minimum-leaf-token-id, node).  Active
    # subtrees are disjoint, so min_token_id is unique among them and provides
    # a total deterministic tie order without relying on object identity.
    heap: list[tuple[float, int, _HuffmanNode]] = []
    for token_id in candidate_token_ids:
        weight = float(reference.probabilities[token_id])
        if not np.isfinite(weight) or weight <= 0.0:
            raise MethodError(
                "Huffman candidate probabilities must be finite and positive"
            )
        node = _HuffmanNode(
            weight=weight,
            min_token_id=token_id,
            token_id=token_id,
        )
        heapq.heappush(heap, (node.weight, node.min_token_id, node))

    while len(heap) > 1:
        _, _, left = heapq.heappop(heap)
        _, _, right = heapq.heappop(heap)
        parent = _HuffmanNode(
            weight=left.weight + right.weight,
            min_token_id=min(left.min_token_id, right.min_token_id),
            left=left,
            right=right,
        )
        heapq.heappush(heap, (parent.weight, parent.min_token_id, parent))

    root = heap[0][2]
    codes: dict[int, tuple[int, ...]] = {}

    def visit(node: _HuffmanNode, prefix: tuple[int, ...]) -> None:
        if node.is_leaf:
            assert node.token_id is not None
            codes[node.token_id] = prefix
            return
        assert node.left is not None and node.right is not None
        visit(node.left, (*prefix, 0))
        visit(node.right, (*prefix, 1))

    visit(root, ())

    q = np.zeros(reference.vocab_size, dtype=np.float64)
    for token_id, code in codes.items():
        q[token_id] = 2.0 ** (-len(code))

    # A full binary Huffman tree satisfies Kraft equality.  Check internally so
    # malformed construction cannot silently reach the metric layer.
    if not np.isclose(float(q.sum(dtype=np.float64)), 1.0, rtol=0.0, atol=1e-12):
        raise MethodError("Huffman induced Q_stego failed Kraft normalization")

    return HuffmanCodebook(
        candidate_token_ids=candidate_token_ids,
        root=root,
        codes=codes,
        q_stego=q,
    )


class HuffmanEncoderSession(EncoderSession):
    """Streaming normalized Huffman encoder with variable payload per token."""

    def __init__(
        self,
        *,
        config: HuffmanConfig,
        environment: MethodEnvironment,
        secret_source: SecretSource,
    ) -> None:
        _validate_static_configuration(config=config, environment=environment)
        self._config = config
        self._environment = environment
        self._secret_source = secret_source
        self._start_secret_position = secret_source.position
        self._steps = 0
        self._finalized = False

    @property
    def done(self) -> bool:
        return self._finalized

    def step(self, context: StepContext) -> EncodeDecision:
        if self._finalized:
            raise SessionStateError(
                "cannot call Huffman encoder.step() after finalize()"
            )

        # Build/validate the complete tree before advancing the secret stream.
        codebook = _build_codebook(
            context=context,
            config=self._config,
            environment=self._environment,
        )

        node = codebook.root
        consumed: list[int] = []
        while not node.is_leaf:
            bit = self._secret_source.read_bits(1)[0]
            consumed.append(bit)
            if bit == 0:
                assert node.left is not None
                node = node.left
            else:
                assert node.right is not None
                node = node.right

        assert node.token_id is not None
        token_id = node.token_id
        code = tuple(consumed)
        # The traversal and code table are two independent views of the same
        # tree; keeping this invariant explicit catches implementation drift.
        if codebook.codes[token_id] != code:
            raise MethodError("Huffman tree traversal disagrees with generated codebook")

        self._steps += 1
        return EncodeDecision(
            token_id=token_id,
            bits_consumed=len(code),
            distribution_info=DistributionInfo.explicit(
                codebook.q_stego,
                mode=QMode.ANALYTIC_EXACT,
                source=QSource.ADAPTER_EXACT,
                metadata={
                    "method": "huffman",
                    "bits_per_word": self._config.bits_per_word,
                    "candidate_count": self._config.candidate_count,
                },
            ),
            method_trace={
                "selected_code": code,
                "code_length": len(code),
                "candidate_count": self._config.candidate_count,
            },
        )

    def finalize(self) -> EncoderFinalization:
        if self._finalized:
            raise SessionStateError("Huffman encoder has already been finalized")
        self._finalized = True
        payload_bits = self._secret_source.position - self._start_secret_position
        return EncoderFinalization(
            payload_bits=payload_bits,
            termination_reason="streaming_session_finalized",
            metadata={
                "steps": self._steps,
                "bits_per_word": self._config.bits_per_word,
                "candidate_count": self._config.candidate_count,
            },
        )


class HuffmanDecoderSession(DecoderSession):
    """Streaming decoder that rebuilds the per-prefix Huffman codebook."""

    def __init__(
        self,
        *,
        config: HuffmanConfig,
        environment: MethodEnvironment,
        expected_payload_bits: int | None,
    ) -> None:
        _validate_static_configuration(config=config, environment=environment)
        if expected_payload_bits is not None and expected_payload_bits < 0:
            raise ConfigurationError("expected_payload_bits must be non-negative")
        self._config = config
        self._environment = environment
        self._expected_payload_bits = expected_payload_bits
        self._recovered: list[int] = []
        self._steps = 0
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
            raise SessionStateError(
                "cannot call Huffman decoder.observe() after finalize()"
            )

        codebook = _build_codebook(
            context=context,
            config=self._config,
            environment=self._environment,
        )
        bits = codebook.code_for_token(observed_token_id)
        self._recovered.extend(bits)
        self._steps += 1

        return DecodeProgress(
            recovered_bits=bits,
            method_trace={
                "observed_code": bits,
                "code_length": len(bits),
                "candidate_count": self._config.candidate_count,
            },
        )

    def finalize(self) -> DecoderFinalization:
        if self._finalized:
            raise SessionStateError("Huffman decoder has already been finalized")
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
                "bits_per_word": self._config.bits_per_word,
                "candidate_count": self._config.candidate_count,
                "raw_recovered_bits": len(self._recovered),
            },
        )


class HuffmanMethod(StegoMethod):
    """Factory for matching normalized Huffman encoder/decoder sessions."""

    method_id = "huffman"

    def create_encoder(
        self,
        *,
        config: MethodConfig,
        environment: MethodEnvironment,
        secret_source: SecretSource,
        random_source: RandomSource | None = None,
        key: KeyMaterial = None,
    ) -> EncoderSession:
        # Huffman is deterministic once P_reference and secret bits are fixed.
        del random_source, key
        parsed = HuffmanConfig.from_mapping(config)
        return HuffmanEncoderSession(
            config=parsed,
            environment=environment,
            secret_source=secret_source,
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
        del random_source, key
        parsed = HuffmanConfig.from_mapping(config)
        return HuffmanDecoderSession(
            config=parsed,
            environment=environment,
            expected_payload_bits=expected_payload_bits,
        )
