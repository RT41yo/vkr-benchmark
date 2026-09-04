"""Normalized finite-precision Arithmetic Coding steganographic method.

Algorithmic reference:
    harvardnlp/NeuralSteganography, arithmetic.py,
    commit 14e982564aeaf9a33f7b4de440deda2184d17f12.

The normalized adapter preserves the reference implementation's integer range
coding core while moving LM/tokenizer ownership and the common generation
policy outside the method.  At each step the current integer interval is
partitioned according to canonical ``P_reference``; low-probability candidates
that cannot receive at least one integer point are cut off; remaining masses
are rescaled and rounded to integer widths; the secret look-ahead point selects
one subinterval; and the common binary prefix of that subinterval is confirmed
as embedded payload before the interval is renormalized.

Unlike the author script, normalized fixed-carrier mode does not perform the
source-specific final-token flush.  Only bits that become fixed through common
interval prefixes count as useful payload.  This makes per-step payload
accounting exact and lets the ordinary-text transport determine reliability.
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
class ArithmeticConfig:
    """Normalized finite-precision Arithmetic Coding parameters.

    ``precision`` controls the integer coding range ``[0, 2**precision)``.
    ``top_k`` is the method-internal candidate cap from the Harvard reference;
    it is applied *after* canonical ``P_reference`` and is distinct from the
    common benchmark generation-policy top-k.

    The implementation uses float64 only to convert canonical FP32
    probabilities into integer widths.  ``precision <= 52`` keeps all integer
    endpoints exactly representable while performing that conversion.
    """

    precision: int
    top_k: int = 50_000

    def __post_init__(self) -> None:
        if isinstance(self.precision, bool) or not isinstance(self.precision, int):
            raise ConfigurationError("Arithmetic precision must be an integer")
        if not 2 <= self.precision <= 52:
            raise ConfigurationError("Arithmetic precision must be in [2, 52]")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int):
            raise ConfigurationError("Arithmetic top_k must be an integer")
        if self.top_k < 2:
            raise ConfigurationError("Arithmetic top_k must be >= 2")

    @property
    def max_value(self) -> int:
        return 1 << self.precision

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ArithmeticConfig":
        unknown = set(values) - {"precision", "top_k"}
        if unknown:
            raise ConfigurationError(
                f"unknown Arithmetic configuration fields: {sorted(unknown)}"
            )
        if "precision" not in values:
            raise ConfigurationError("Arithmetic configuration requires precision")
        return cls(
            precision=values["precision"],
            top_k=values.get("top_k", 50_000),
        )


@dataclass(frozen=True, slots=True)
class ArithmeticPartition:
    """One finite-precision partition of the current coding interval."""

    interval_lower: int
    interval_upper: int
    candidate_token_ids: tuple[int, ...]
    integer_widths: tuple[int, ...]
    cumulative_upper_bounds: tuple[int, ...]
    q_stego: np.ndarray

    def __post_init__(self) -> None:
        if not 0 <= self.interval_lower < self.interval_upper:
            raise ValueError("Arithmetic interval must be non-empty")
        if len(self.candidate_token_ids) < 1:
            raise ValueError("Arithmetic partition must contain candidates")
        if len(self.candidate_token_ids) != len(self.integer_widths):
            raise ValueError("candidate ids and integer widths must have equal length")
        if len(self.candidate_token_ids) != len(self.cumulative_upper_bounds):
            raise ValueError("candidate ids and cumulative bounds must have equal length")
        if any(width < 0 for width in self.integer_widths):
            raise ValueError("integer widths must be non-negative")
        if sum(self.integer_widths) != self.interval_width:
            raise ValueError("integer widths must exactly fill the current interval")
        if self.cumulative_upper_bounds[-1] != self.interval_upper:
            raise ValueError("last cumulative upper bound must equal interval upper")

        q = np.asarray(self.q_stego, dtype=np.float64).copy()
        if q.ndim != 1:
            raise ValueError("q_stego must be one-dimensional")
        q.setflags(write=False)
        object.__setattr__(self, "q_stego", q)

    @property
    def interval_width(self) -> int:
        return self.interval_upper - self.interval_lower

    def rank_for_token(self, token_id: int) -> int:
        value = int(token_id)
        try:
            return self.candidate_token_ids.index(value)
        except ValueError as exc:
            raise MethodError(
                f"observed token {value} is outside the current Arithmetic candidate set"
            ) from exc

    def bounds_for_rank(self, rank: int) -> tuple[int, int]:
        if rank < 0 or rank >= len(self.candidate_token_ids):
            raise IndexError("Arithmetic candidate rank out of range")
        lower = (
            self.interval_lower
            if rank == 0
            else self.cumulative_upper_bounds[rank - 1]
        )
        upper = self.cumulative_upper_bounds[rank]
        if upper <= lower:
            raise MethodError(
                "observed Arithmetic token has zero-width finite-precision interval"
            )
        return lower, upper

    def rank_for_point(self, point: int) -> int:
        if point < self.interval_lower or point >= self.interval_upper:
            raise MethodError(
                f"secret point {point} lies outside current Arithmetic interval "
                f"[{self.interval_lower}, {self.interval_upper})"
            )
        rank = int(np.searchsorted(self.cumulative_upper_bounds, point, side="right"))
        if rank >= len(self.candidate_token_ids):
            raise MethodError("Arithmetic partition does not cover the secret point")
        lower, upper = self.bounds_for_rank(rank)
        if not lower <= point < upper:
            raise MethodError("Arithmetic point-to-interval selection is inconsistent")
        return rank


def _bits_to_int_msb(bits: tuple[int, ...] | list[int]) -> int:
    value = 0
    for bit in bits:
        if bit not in (0, 1):
            raise ValueError("bits must contain only 0/1")
        value = (value << 1) | int(bit)
    return value


def _int_to_bits_msb(value: int, precision: int) -> tuple[int, ...]:
    if value < 0 or value >= (1 << precision):
        raise ValueError("integer does not fit configured Arithmetic precision")
    return tuple((value >> shift) & 1 for shift in range(precision - 1, -1, -1))


def _reference_common_prefix_length(
    lower_bits: tuple[int, ...], upper_inclusive_bits: tuple[int, ...]
) -> int:
    """Match the Harvard ``num_same_from_beg`` helper semantics.

    The source helper returns ``precision - 1`` when all bits are equal rather
    than ``precision``.  Encoder and decoder share that behavior, and retaining
    one unresolved bit prevents an implementation drift before the later
    author/conformity comparison.
    """

    if len(lower_bits) != len(upper_inclusive_bits):
        raise ValueError("bit vectors must have equal length")
    for index, (left, right) in enumerate(zip(lower_bits, upper_inclusive_bits)):
        if left != right:
            return index
    return max(0, len(lower_bits) - 1)


def _renormalize_interval(
    *, lower: int, upper: int, precision: int
) -> tuple[int, int, tuple[int, ...]]:
    """Confirm fixed common-prefix bits and expand the residual interval."""

    if not 0 <= lower < upper <= (1 << precision):
        raise MethodError("invalid Arithmetic subinterval")

    lower_bits = _int_to_bits_msb(lower, precision)
    upper_inclusive_bits = _int_to_bits_msb(upper - 1, precision)
    count = _reference_common_prefix_length(lower_bits, upper_inclusive_bits)
    confirmed = lower_bits[:count]

    new_lower_bits = lower_bits[count:] + (0,) * count
    new_upper_inclusive_bits = upper_inclusive_bits[count:] + (1,) * count
    new_lower = _bits_to_int_msb(new_lower_bits)
    new_upper = _bits_to_int_msb(new_upper_inclusive_bits) + 1

    if not 0 <= new_lower < new_upper <= (1 << precision):
        raise MethodError("Arithmetic interval renormalization produced invalid bounds")
    return new_lower, new_upper, confirmed


def _build_partition(
    *,
    context: StepContext,
    config: ArithmeticConfig,
    environment: MethodEnvironment,
    interval_lower: int,
    interval_upper: int,
) -> ArithmeticPartition:
    """Reproduce the reference integer-mass construction from P_reference."""

    reference = context.reference
    if reference.vocab_size != environment.output_vocab_size:
        raise MethodError(
            "Arithmetic MethodEnvironment output vocabulary does not match P_reference"
        )
    if not 0 <= interval_lower < interval_upper <= config.max_value:
        raise MethodError("invalid Arithmetic session interval")

    interval_width = interval_upper - interval_lower
    if interval_width < 2:
        raise MethodError("Arithmetic interval width must remain at least two")
    if reference.support_size < 2:
        raise UnsupportedConfigurationError(
            "Arithmetic Coding requires at least two positive-probability tokens"
        )

    order = tuple(int(token_id) for token_id in reference.token_order)
    for token_id in order:
        if not environment.is_allowed(token_id):
            raise MethodError(
                f"P_reference token {token_id} lies outside Arithmetic V_allowed"
            )

    sorted_probs = np.asarray(
        [reference.probabilities[token_id] for token_id in order],
        dtype=np.float64,
    )
    threshold = 1.0 / float(interval_width)
    below = np.flatnonzero(sorted_probs < threshold)
    cutoff = int(below[0]) if below.size else len(order)
    candidate_count = min(max(2, cutoff), config.top_k, len(order))
    if candidate_count < 2:
        raise UnsupportedConfigurationError(
            "Arithmetic Coding could not construct two finite-precision candidates"
        )

    candidate_ids = list(order[:candidate_count])
    candidate_probs = sorted_probs[:candidate_count]
    prob_sum = float(candidate_probs.sum(dtype=np.float64))
    if not np.isfinite(prob_sum) or prob_sum <= 0.0:
        raise MethodError("Arithmetic candidate probability mass is invalid")

    scaled = candidate_probs / prob_sum * float(interval_width)
    rounded = np.rint(scaled).astype(np.int64)
    cumulative = np.cumsum(rounded, dtype=np.int64)

    # Reference behavior: if rounding overfills the range, remove the first
    # overfilling candidate and everything below it in the ranking.
    overfill = np.flatnonzero(cumulative > interval_width)
    if overfill.size:
        keep = int(overfill[0])
        if keep <= 0:
            raise MethodError("Arithmetic rounding overfilled at the first candidate")
        candidate_ids = candidate_ids[:keep]
        cumulative = cumulative[:keep]

    if cumulative.size == 0:
        raise MethodError("Arithmetic finite-precision candidate set became empty")

    # Reference behavior: any underfilled mass is assigned to the top-ranked
    # candidate by shifting every cumulative boundary by the same residual.
    residual = int(interval_width - int(cumulative[-1]))
    cumulative = cumulative + residual
    if int(cumulative[-1]) != interval_width:
        raise MethodError("Arithmetic integer masses do not fill the current range")
    if np.any(np.diff(cumulative, prepend=0) < 0):
        raise MethodError("Arithmetic integer cumulative masses are not monotonic")

    widths_array = np.diff(cumulative, prepend=0).astype(np.int64)
    absolute_upper = cumulative + int(interval_lower)

    q = np.zeros(reference.vocab_size, dtype=np.float64)
    for token_id, width in zip(candidate_ids, widths_array, strict=True):
        if width > 0:
            q[token_id] = float(width) / float(interval_width)

    if not np.isclose(float(q.sum(dtype=np.float64)), 1.0, rtol=0.0, atol=1e-12):
        raise MethodError("Arithmetic induced Q_stego failed integer normalization")

    return ArithmeticPartition(
        interval_lower=interval_lower,
        interval_upper=interval_upper,
        candidate_token_ids=tuple(candidate_ids),
        integer_widths=tuple(int(width) for width in widths_array),
        cumulative_upper_bounds=tuple(int(bound) for bound in absolute_upper),
        q_stego=q,
    )


class _SecretLookahead:
    """Maintain the overlapping precision-bit window used by Arithmetic Coding."""

    def __init__(self, source: SecretSource, precision: int) -> None:
        self._source = source
        self._precision = precision
        self._bits = list(source.read_bits(precision))
        if len(self._bits) != precision:
            raise MethodError("SecretSource failed to provide Arithmetic look-ahead")

    @property
    def point(self) -> int:
        return _bits_to_int_msb(self._bits)

    @property
    def bits(self) -> tuple[int, ...]:
        return tuple(self._bits)

    def confirm(self, count: int) -> tuple[int, ...]:
        if count < 0 or count > self._precision:
            raise MethodError("invalid Arithmetic confirmed-bit count")
        confirmed = tuple(self._bits[:count])
        if count:
            del self._bits[:count]
            self._bits.extend(self._source.read_bits(count))
        if len(self._bits) != self._precision:
            raise MethodError("Arithmetic look-ahead window lost configured precision")
        return confirmed


class ArithmeticEncoderSession(EncoderSession):
    """Stateful normalized finite-precision Arithmetic Coding encoder."""

    def __init__(
        self,
        *,
        config: ArithmeticConfig,
        environment: MethodEnvironment,
        secret_source: SecretSource,
    ) -> None:
        self._config = config
        self._environment = environment
        self._secret_source = secret_source
        self._source_start_position = secret_source.position
        self._lookahead = _SecretLookahead(secret_source, config.precision)
        self._lower = 0
        self._upper = config.max_value
        self._payload_bits = 0
        self._steps = 0
        self._finalized = False

    @property
    def done(self) -> bool:
        return self._finalized

    @property
    def interval(self) -> tuple[int, int]:
        return self._lower, self._upper

    def step(self, context: StepContext) -> EncodeDecision:
        if self._finalized:
            raise SessionStateError(
                "cannot call Arithmetic encoder.step() after finalize()"
            )

        before = (self._lower, self._upper)
        partition = _build_partition(
            context=context,
            config=self._config,
            environment=self._environment,
            interval_lower=self._lower,
            interval_upper=self._upper,
        )

        point = self._lookahead.point
        rank = partition.rank_for_point(point)
        token_id = partition.candidate_token_ids[rank]
        selected_lower, selected_upper = partition.bounds_for_rank(rank)
        new_lower, new_upper, fixed_prefix = _renormalize_interval(
            lower=selected_lower,
            upper=selected_upper,
            precision=self._config.precision,
        )

        confirmed = self._lookahead.confirm(len(fixed_prefix))
        if confirmed != fixed_prefix:
            raise MethodError(
                "Arithmetic interval prefix disagrees with secret look-ahead bits"
            )

        self._lower, self._upper = new_lower, new_upper
        self._payload_bits += len(confirmed)
        self._steps += 1

        return EncodeDecision(
            token_id=token_id,
            bits_consumed=len(confirmed),
            distribution_info=DistributionInfo.explicit(
                partition.q_stego,
                mode=QMode.ANALYTIC_EXACT,
                source=QSource.ADAPTER_EXACT,
                metadata={
                    "method": "arithmetic_coding",
                    "precision": self._config.precision,
                    "top_k": self._config.top_k,
                    "candidate_count": len(partition.candidate_token_ids),
                    "interval_width": partition.interval_width,
                },
            ),
            method_trace={
                "interval_before": before,
                "selected_rank": rank,
                "selected_interval": (selected_lower, selected_upper),
                "confirmed_prefix": confirmed,
                "candidate_count": len(partition.candidate_token_ids),
                "interval_after": (new_lower, new_upper),
            },
        )

    def finalize(self) -> EncoderFinalization:
        if self._finalized:
            raise SessionStateError("Arithmetic encoder has already been finalized")
        self._finalized = True
        secret_bits_read = self._secret_source.position - self._source_start_position
        return EncoderFinalization(
            payload_bits=self._payload_bits,
            termination_reason="streaming_session_finalized",
            metadata={
                "steps": self._steps,
                "precision": self._config.precision,
                "top_k": self._config.top_k,
                "final_interval": (self._lower, self._upper),
                "secret_bits_read": secret_bits_read,
                "lookahead_bits": self._config.precision,
            },
        )


class ArithmeticDecoderSession(DecoderSession):
    """Stateful decoder mirroring normalized integer interval updates."""

    def __init__(
        self,
        *,
        config: ArithmeticConfig,
        environment: MethodEnvironment,
        expected_payload_bits: int | None,
    ) -> None:
        if expected_payload_bits is not None and expected_payload_bits < 0:
            raise ConfigurationError("expected_payload_bits must be non-negative")
        self._config = config
        self._environment = environment
        self._expected_payload_bits = expected_payload_bits
        self._lower = 0
        self._upper = config.max_value
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

    @property
    def interval(self) -> tuple[int, int]:
        return self._lower, self._upper

    def observe(self, context: StepContext, observed_token_id: int) -> DecodeProgress:
        if self._finalized:
            raise SessionStateError(
                "cannot call Arithmetic decoder.observe() after finalize()"
            )

        before = (self._lower, self._upper)
        partition = _build_partition(
            context=context,
            config=self._config,
            environment=self._environment,
            interval_lower=self._lower,
            interval_upper=self._upper,
        )
        rank = partition.rank_for_token(observed_token_id)
        selected_lower, selected_upper = partition.bounds_for_rank(rank)
        new_lower, new_upper, fixed_prefix = _renormalize_interval(
            lower=selected_lower,
            upper=selected_upper,
            precision=self._config.precision,
        )

        self._recovered.extend(fixed_prefix)
        self._lower, self._upper = new_lower, new_upper
        self._steps += 1

        return DecodeProgress(
            recovered_bits=fixed_prefix,
            method_trace={
                "interval_before": before,
                "observed_rank": rank,
                "selected_interval": (selected_lower, selected_upper),
                "recovered_prefix": fixed_prefix,
                "candidate_count": len(partition.candidate_token_ids),
                "interval_after": (new_lower, new_upper),
            },
        )

    def finalize(self) -> DecoderFinalization:
        if self._finalized:
            raise SessionStateError("Arithmetic decoder has already been finalized")
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
                "precision": self._config.precision,
                "top_k": self._config.top_k,
                "final_interval": (self._lower, self._upper),
                "raw_recovered_bits": len(self._recovered),
            },
        )


class ArithmeticMethod(StegoMethod):
    """Factory for matching normalized Arithmetic encoder/decoder sessions."""

    method_id = "arithmetic_coding"

    def create_encoder(
        self,
        *,
        config: MethodConfig,
        environment: MethodEnvironment,
        secret_source: SecretSource,
        random_source: RandomSource | None = None,
        key: KeyMaterial = None,
    ) -> EncoderSession:
        # Basic Arithmetic Coding is deterministic once P_reference and the
        # secret stream are fixed; method randomness/key are not used here.
        del random_source, key
        parsed = ArithmeticConfig.from_mapping(config)
        return ArithmeticEncoderSession(
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
        parsed = ArithmeticConfig.from_mapping(config)
        return ArithmeticDecoderSession(
            config=parsed,
            environment=environment,
            expected_payload_bits=expected_payload_bits,
        )
