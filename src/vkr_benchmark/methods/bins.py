"""Normalized Bins (block/binning) steganographic method.

Algorithmic reference:
    harvardnlp/NeuralSteganography, block_baseline.py,
    commit 14e982564aeaf9a33f7b4de440deda2184d17f12.

The normalized adapter preserves the core method semantics — a fixed partition
of the vocabulary into 2**b bins, b secret bits select one bin, and the most
probable token in that bin is emitted — while moving LM/tokenizer handling,
common masking/policy, RNG isolation, transport, and metrics into benchmark
infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass
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
class BinsConfig:
    """Normalized Bins parameters."""

    block_size: int

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ConfigurationError("Bins block_size must be a positive integer")

    @property
    def num_bins(self) -> int:
        return 1 << self.block_size

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "BinsConfig":
        unknown = set(values) - {"block_size"}
        if unknown:
            raise ConfigurationError(
                f"unknown Bins configuration fields: {sorted(unknown)}"
            )
        if "block_size" not in values:
            raise ConfigurationError("Bins configuration requires block_size")
        block_size = values["block_size"]
        if isinstance(block_size, bool) or not isinstance(block_size, int):
            raise ConfigurationError("Bins block_size must be an integer")
        return cls(block_size=block_size)


@dataclass(frozen=True, slots=True)
class BinsPartition:
    """Fixed run-level partition of V_allowed and inverse token-to-bin map."""

    block_size: int
    bins: tuple[tuple[int, ...], ...]
    token_to_bin: np.ndarray

    def __post_init__(self) -> None:
        num_bins = 1 << self.block_size
        if len(self.bins) != num_bins:
            raise ValueError("partition bin count does not match block_size")

        inverse = np.asarray(self.token_to_bin, dtype=np.int64).copy()
        if inverse.ndim != 1:
            raise ValueError("token_to_bin must be one-dimensional")
        inverse.setflags(write=False)
        object.__setattr__(self, "token_to_bin", inverse)

    @property
    def num_bins(self) -> int:
        return len(self.bins)

    def bin_for_token(self, token_id: int) -> int:
        if token_id < 0 or token_id >= self.token_to_bin.size:
            raise MethodError(f"observed token {token_id} lies outside output vocabulary")
        bin_id = int(self.token_to_bin[token_id])
        if bin_id < 0:
            raise MethodError(f"observed token {token_id} is not in V_allowed")
        return bin_id


def _build_partition(
    *,
    config: BinsConfig,
    environment: MethodEnvironment,
    random_source: RandomSource,
) -> BinsPartition:
    """Create the fixed partition used by both encoder and decoder.

    Harvard's reference starts from ascending vocabulary IDs, shuffles once,
    then slices the shuffled sequence at floor(i * |V| / num_bins) boundaries.
    The normalized adapter applies the same partition-shape rule to V_allowed,
    using the benchmark-injected method RNG instead of global NumPy RNG state.
    """

    num_bins = config.num_bins
    allowed = list(environment.allowed_token_ids)
    if num_bins > len(allowed):
        raise UnsupportedConfigurationError(
            f"Bins requires at least one allowed token per bin: "
            f"2**{config.block_size}={num_bins} > |V_allowed|={len(allowed)}"
        )

    random_source.shuffle(allowed)
    n = len(allowed)
    bins: list[tuple[int, ...]] = []
    inverse = np.full(environment.output_vocab_size, -1, dtype=np.int64)

    for bin_id in range(num_bins):
        start = (bin_id * n) // num_bins
        stop = ((bin_id + 1) * n) // num_bins
        members = tuple(allowed[start:stop])
        if not members:
            raise UnsupportedConfigurationError("Bins partition contains an empty bin")
        bins.append(members)
        inverse[np.asarray(members, dtype=np.int64)] = bin_id

    return BinsPartition(
        block_size=config.block_size,
        bins=tuple(bins),
        token_to_bin=inverse,
    )


def _bits_to_int(bits: tuple[int, ...]) -> int:
    value = 0
    for bit in bits:
        if bit not in (0, 1):
            raise ValueError("bits must contain only 0/1")
        value = (value << 1) | bit
    return value


def _int_to_bits(value: int, width: int) -> tuple[int, ...]:
    if value < 0 or value >= (1 << width):
        raise ValueError("value does not fit requested bit width")
    return tuple((value >> shift) & 1 for shift in range(width - 1, -1, -1))


def _representatives_and_q(
    *,
    context: StepContext,
    partition: BinsPartition,
) -> tuple[tuple[int, ...], np.ndarray]:
    """Return the best positive-probability token in each bin and exact Q.

    Under an i.i.d. Bernoulli(0.5) secret stream every b-bit value, hence every
    bin, is selected with probability 2**(-b). Because the encoder emits one
    deterministic representative per selected bin, Q puts exactly that mass on
    each representative.
    """

    reference = context.reference
    if partition.token_to_bin.size != reference.vocab_size:
        raise MethodError(
            "Bins partition output vocabulary does not match P_reference vocabulary"
        )

    representatives = [-1] * partition.num_bins
    remaining = partition.num_bins

    # token_order already implements the benchmark tie rule
    # (-probability, token_id) over strictly positive support.
    for token_id_raw in reference.token_order:
        token_id = int(token_id_raw)
        bin_id = int(partition.token_to_bin[token_id])
        if bin_id >= 0 and representatives[bin_id] < 0:
            representatives[bin_id] = token_id
            remaining -= 1
            if remaining == 0:
                break

    if remaining:
        missing = [i for i, token_id in enumerate(representatives) if token_id < 0]
        raise UnsupportedConfigurationError(
            "current P_reference has no positive-probability token in Bins "
            f"bin(s) {missing}; common truncation/policy is incompatible with "
            "this Bins configuration at this step"
        )

    q = np.zeros(reference.vocab_size, dtype=np.float64)
    mass = 1.0 / partition.num_bins
    q[np.asarray(representatives, dtype=np.int64)] = mass
    return tuple(representatives), q


class BinsEncoderSession(EncoderSession):
    """Streaming normalized Bins encoder."""

    def __init__(
        self,
        *,
        config: BinsConfig,
        partition: BinsPartition,
        secret_source: SecretSource,
    ) -> None:
        self._config = config
        self._partition = partition
        self._secret_source = secret_source
        self._start_secret_position = secret_source.position
        self._steps = 0
        self._finalized = False

    @property
    def partition(self) -> BinsPartition:
        return self._partition

    @property
    def done(self) -> bool:
        # Bins is a streaming method. The runner normally stops it via an
        # external carrier-token limit; finalization closes the session.
        return self._finalized

    def step(self, context: StepContext) -> EncodeDecision:
        if self._finalized:
            raise SessionStateError("cannot call Bins encoder.step() after finalize()")

        representatives, q = _representatives_and_q(
            context=context,
            partition=self._partition,
        )

        # Validate that every secret value is encodable before advancing the
        # secret stream. This prevents partially consumed payload on failure.
        secret_bits = self._secret_source.read_bits(self._config.block_size)
        selected_bin = _bits_to_int(secret_bits)
        token_id = representatives[selected_bin]
        self._steps += 1

        return EncodeDecision(
            token_id=token_id,
            bits_consumed=self._config.block_size,
            distribution_info=DistributionInfo.explicit(
                q,
                mode=QMode.ANALYTIC_EXACT,
                source=QSource.ADAPTER_EXACT,
                metadata={
                    "method": "bins",
                    "block_size": self._config.block_size,
                    "num_bins": self._partition.num_bins,
                },
            ),
            method_trace={
                "selected_bin": selected_bin,
                "secret_bits": secret_bits,
                "representative_token_id": token_id,
            },
        )

    def finalize(self) -> EncoderFinalization:
        if self._finalized:
            raise SessionStateError("Bins encoder has already been finalized")
        self._finalized = True
        payload_bits = self._secret_source.position - self._start_secret_position
        return EncoderFinalization(
            payload_bits=payload_bits,
            termination_reason="streaming_session_finalized",
            metadata={
                "steps": self._steps,
                "block_size": self._config.block_size,
                "num_bins": self._partition.num_bins,
            },
        )


class BinsDecoderSession(DecoderSession):
    """Streaming normalized Bins decoder."""

    def __init__(
        self,
        *,
        config: BinsConfig,
        partition: BinsPartition,
        expected_payload_bits: int | None,
    ) -> None:
        if expected_payload_bits is not None and expected_payload_bits < 0:
            raise ConfigurationError("expected_payload_bits must be non-negative")
        self._config = config
        self._partition = partition
        self._expected_payload_bits = expected_payload_bits
        self._recovered: list[int] = []
        self._steps = 0
        self._finalized = False

    @property
    def partition(self) -> BinsPartition:
        return self._partition

    @property
    def done(self) -> bool:
        if self._finalized:
            return True
        if self._expected_payload_bits is None:
            return False
        return len(self._recovered) >= self._expected_payload_bits

    def observe(self, context: StepContext, observed_token_id: int) -> DecodeProgress:
        del context  # Bins decoding depends only on the fixed partition.
        if self._finalized:
            raise SessionStateError("cannot call Bins decoder.observe() after finalize()")

        bin_id = self._partition.bin_for_token(observed_token_id)
        bits = _int_to_bits(bin_id, self._config.block_size)
        self._recovered.extend(bits)
        self._steps += 1
        return DecodeProgress(
            recovered_bits=bits,
            method_trace={"observed_bin": bin_id},
        )

    def finalize(self) -> DecoderFinalization:
        if self._finalized:
            raise SessionStateError("Bins decoder has already been finalized")
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
                "block_size": self._config.block_size,
                "raw_recovered_bits": len(self._recovered),
            },
        )


class BinsMethod(StegoMethod):
    """Factory for matching normalized Bins encoder/decoder sessions."""

    method_id = "bins"

    @staticmethod
    def _partition(
        *,
        config: BinsConfig,
        environment: MethodEnvironment,
        random_source: RandomSource | None,
    ) -> BinsPartition:
        if random_source is None:
            raise ConfigurationError(
                "normalized Bins requires an injected MethodRandomSource; "
                "the run-level key/seed-to-RNG policy is owned by common infrastructure"
            )
        return _build_partition(
            config=config,
            environment=environment,
            random_source=random_source,
        )

    def create_encoder(
        self,
        *,
        config: MethodConfig,
        environment: MethodEnvironment,
        secret_source: SecretSource,
        random_source: RandomSource | None = None,
        key: KeyMaterial = None,
    ) -> EncoderSession:
        del key
        parsed = BinsConfig.from_mapping(config)
        partition = self._partition(
            config=parsed,
            environment=environment,
            random_source=random_source,
        )
        return BinsEncoderSession(
            config=parsed,
            partition=partition,
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
        del key
        parsed = BinsConfig.from_mapping(config)
        partition = self._partition(
            config=parsed,
            environment=environment,
            random_source=random_source,
        )
        return BinsDecoderSession(
            config=parsed,
            partition=partition,
            expected_payload_bits=expected_payload_bits,
        )
