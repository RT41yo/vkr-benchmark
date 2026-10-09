"""Normalized Rotation Range-Coding (RRC) steganography.

Algorithmic reference:
    ryehr/RRC_steganography, ``rrc_core.py``,
    commit dae326259e4fca8bc4fcf460dafdbc0e88a0a71a.

The normalized adapter preserves the current author core while keeping benchmark
boundaries explicit:

* the common LM path constructs the final ``P_reference`` before RRC runs;
* common temperature/top-k/top-p therefore belong to ``GenerationPolicy`` and
  are not repeated inside the method;
* the whole fixed-length secret is read before the first carrier token;
* a fresh synchronized PRNG offset rotates the secret point at every step;
* Decimal precision scales with payload length;
* verified termination replays the receiver's reverse rotation before stopping;
* decoding is finalize-only, matching Algorithm 4;
* explicit ``Q_stego`` is reconstructed independently from interval widths for
  benchmark KL/TVD metrics and is timed as metric instrumentation, not as RRC.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_DOWN, localcontext
from math import ceil, log10
from typing import Any, Mapping

import numpy as np

from vkr_benchmark.distributions import (
    DistributionInfo,
    QMode,
    QSource,
    StepContext,
    explicit_rrc_q,
)
from vkr_benchmark.errors import ConfigurationError, MethodError, SessionStateError
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

_REFERENCE_COMMIT = "dae326259e4fca8bc4fcf460dafdbc0e88a0a71a"
_PRIOR_ANALYSIS_COMMIT = "8b6a86a66e516e3763c907a1ac640c17bcd3641d"


class RRCPrecisionError(MethodError):
    """Raised when Decimal precision cannot represent the requested trajectory."""


def required_precision(bit_length: int, margin: int = 40) -> int:
    """Return Decimal digits required by the pinned RRC precision rule."""

    if isinstance(bit_length, bool) or not isinstance(bit_length, int) or bit_length <= 0:
        raise ValueError("bit_length must be a positive integer")
    if isinstance(margin, bool) or not isinstance(margin, int) or margin < 0:
        raise ValueError("margin must be a non-negative integer")
    return int(ceil(bit_length * log10(2.0))) + margin


def _decimal_mod(value: Decimal, modulus: Decimal) -> Decimal:
    if modulus <= 0:
        raise RRCPrecisionError("RRC interval width must remain positive")
    return value - modulus * (value / modulus).to_integral_value(rounding=ROUND_FLOOR)


def _round_half_down(value: Decimal) -> Decimal:
    return value.to_integral_value(rounding=ROUND_HALF_DOWN)


def _rotate(
    secret_point: Decimal,
    left: Decimal,
    width: Decimal,
    offset: Decimal,
) -> Decimal:
    return left + _decimal_mod((secret_point - left) + offset * width, width)


def _unrotate(
    midpoint: Decimal,
    left: Decimal,
    width: Decimal,
    offset: Decimal,
) -> Decimal:
    return left + _decimal_mod(midpoint - left - offset * width, width)


def _terminated(left: Decimal, right: Decimal, secret_point: Decimal) -> bool:
    diff = (left + right) / 2 - secret_point
    return Decimal("-0.5") < diff <= Decimal("0.5")


@dataclass(frozen=True, slots=True)
class RRCConfig:
    """Method-local RRC controls retained by the normalized adapter."""

    precision_margin: int = 40
    verify_termination: bool = True

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "RRCConfig":
        allowed = {"precision_margin", "verify_termination"}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ConfigurationError(
                "unknown RRC configuration fields: " + ", ".join(unknown)
            )

        margin = values.get("precision_margin", 40)
        verify = values.get("verify_termination", True)
        if isinstance(margin, bool) or not isinstance(margin, int) or margin < 0:
            raise ConfigurationError("RRC precision_margin must be a non-negative integer")
        if not isinstance(verify, bool):
            raise ConfigurationError("RRC verify_termination must be a boolean")
        return cls(precision_margin=margin, verify_termination=verify)


@dataclass(frozen=True, slots=True)
class _StepDistribution:
    """FP64 interval partition derived only from the current P_reference."""

    token_ids: np.ndarray
    boundaries: np.ndarray

    @classmethod
    def from_context(
        cls,
        context: StepContext,
        environment: MethodEnvironment,
    ) -> "_StepDistribution":
        reference = context.reference
        if reference.vocab_size != environment.output_vocab_size:
            raise MethodError(
                "RRC MethodEnvironment output vocabulary does not match P_reference"
            )
        order = np.asarray(reference.token_order, dtype=np.int64)
        if order.size == 0:
            raise MethodError("RRC requires non-empty P_reference support")
        for token_id in order:
            if not environment.is_allowed(int(token_id)):
                raise MethodError(
                    f"P_reference support token {int(token_id)} lies outside V_allowed"
                )

        support = reference.probabilities[order].astype(np.float64, copy=True)
        boundaries = np.empty(order.size + 1, dtype=np.float64)
        boundaries[0] = 0.0
        np.cumsum(support, dtype=np.float64, out=boundaries[1:])
        total = float(boundaries[-1])
        if not np.isfinite(total) or total <= 0.0:
            raise MethodError(f"RRC candidate probability total is invalid: {total!r}")
        boundaries[1:] /= total
        return cls(token_ids=order, boundaries=boundaries)

    def index_of(self, token_id: int) -> int:
        hits = np.flatnonzero(self.token_ids == int(token_id))
        return int(hits[0]) if hits.size else -1


def _select_index(
    dist: _StepDistribution,
    left: Decimal,
    right: Decimal,
    secret_point: Decimal,
) -> tuple[int, Decimal, Decimal]:
    """Pinned Algorithm-3 candidate search using Decimal(str(boundary))."""

    width = right - left
    if width <= 0:
        raise RRCPrecisionError("RRC interval collapsed before candidate selection")
    boundaries = dist.boundaries
    lo, hi = 0, len(boundaries) - 2
    while lo < hi:
        mid = (lo + hi) // 2
        upper = left + Decimal(str(boundaries[mid + 1])) * width
        if upper > secret_point:
            hi = mid
        else:
            lo = mid + 1

    lower = left + Decimal(str(boundaries[lo])) * width
    upper = left + Decimal(str(boundaries[lo + 1])) * width
    if not upper > secret_point:
        raise RRCPrecisionError(
            "no RRC sub-interval contains the rotated secret point"
        )
    return lo, lower, upper


def _narrow(
    dist: _StepDistribution,
    left: Decimal,
    right: Decimal,
    index: int,
) -> tuple[Decimal, Decimal]:
    if index < 0 or index >= len(dist.token_ids):
        raise MethodError("RRC candidate index lies outside the current support")
    width = right - left
    return (
        left + Decimal(str(dist.boundaries[index])) * width,
        left + Decimal(str(dist.boundaries[index + 1])) * width,
    )


def _replay(
    left: Decimal,
    right: Decimal,
    history: list[tuple[Decimal, Decimal, Decimal]],
) -> int:
    midpoint = (left + right) / 2
    for previous_left, previous_width, offset in reversed(history):
        midpoint = _unrotate(midpoint, previous_left, previous_width, offset)
    return int(_round_half_down(midpoint))


def _distribution_info(context: StepContext) -> DistributionInfo:
    q = explicit_rrc_q(context.reference)
    return DistributionInfo.explicit(
        q,
        mode=QMode.ANALYTIC_EXACT,
        source=QSource.INDEPENDENT_ANALYTIC,
        metadata={
            "method": "rrc",
            "reference_commit": _REFERENCE_COMMIT,
            "construction": "independent_interval_measure",
            "randomness_model": "uniform_rotation_offset",
        },
    )


class RRCEncoderSession(EncoderSession):
    """Fixed-message RRC encoder following the pinned Algorithm 3 core."""

    def __init__(
        self,
        *,
        config: RRCConfig,
        environment: MethodEnvironment,
        secret_source: SecretSource,
        random_source: RandomSource,
        target_payload_bits: int,
    ) -> None:
        self._config = config
        self._environment = environment
        self._random_source = random_source
        self._payload_bits = int(target_payload_bits)
        self._precision = required_precision(
            self._payload_bits, config.precision_margin
        )
        self._secret_bits = secret_source.read_bits(self._payload_bits)
        self._secret_integer = int("".join(str(bit) for bit in self._secret_bits), 2)

        with localcontext() as decimal_context:
            decimal_context.prec = self._precision
            self._secret_point = Decimal(self._secret_integer)
            self._left = Decimal(0)
            self._right = Decimal(2) ** self._payload_bits

        self._history: list[tuple[Decimal, Decimal, Decimal]] = []
        self._steps = 0
        self._termination_retries = 0
        self._done = False
        self._finalized = False

    @property
    def done(self) -> bool:
        return self._done

    def step(self, context: StepContext) -> EncodeDecision:
        if self._finalized:
            raise SessionStateError("cannot call RRC encoder.step() after finalize()")
        if self._done:
            raise SessionStateError("cannot call RRC encoder.step() after termination")

        dist = _StepDistribution.from_context(context, self._environment)
        with localcontext() as decimal_context:
            decimal_context.prec = self._precision
            width = self._right - self._left
            if width <= 0:
                raise RRCPrecisionError("RRC interval width became non-positive")
            offset = Decimal(self._random_source.random())
            self._history.append((self._left, width, offset))
            self._secret_point = _rotate(
                self._secret_point,
                self._left,
                width,
                offset,
            )
            if not self._left <= self._secret_point < self._right:
                raise RRCPrecisionError(
                    "RRC rotation left the current interval; precision is exhausted"
                )

            index, self._left, self._right = _select_index(
                dist,
                self._left,
                self._right,
                self._secret_point,
            )
            terminated = _terminated(
                self._left,
                self._right,
                self._secret_point,
            )
            verified = False
            if terminated:
                if not self._config.verify_termination:
                    verified = True
                else:
                    verified = (
                        _replay(self._left, self._right, self._history)
                        == self._secret_integer
                    )
                if verified:
                    self._done = True
                else:
                    self._termination_retries += 1

        self._steps += 1
        return EncodeDecision(
            token_id=int(dist.token_ids[index]),
            bits_consumed=None,
            method_trace={
                "rrc_step": self._steps,
                "candidate_index": int(index),
                "termination_test": bool(terminated),
                "termination_verified": bool(verified),
            },
        )

    def distribution_info(
        self, context: StepContext, decision: EncodeDecision
    ) -> DistributionInfo:
        del decision
        return _distribution_info(context)

    def finalize(self) -> EncoderFinalization:
        if self._finalized:
            raise SessionStateError("RRC encoder has already been finalized")
        if not self._done:
            raise SessionStateError("RRC encoder cannot finalize before termination")
        self._finalized = True
        return EncoderFinalization(
            payload_bits=self._payload_bits,
            termination_reason="fixed_payload_bits",
            metadata={
                "steps": self._steps,
                "precision_digits": self._precision,
                "precision_margin": self._config.precision_margin,
                "verify_termination": self._config.verify_termination,
                "termination_retries": self._termination_retries,
                "reference_commit": _REFERENCE_COMMIT,
            },
        )


class RRCDecoderSession(DecoderSession):
    """Finalize-only RRC decoder following pinned Algorithm 4."""

    def __init__(
        self,
        *,
        config: RRCConfig,
        environment: MethodEnvironment,
        random_source: RandomSource,
        expected_payload_bits: int,
    ) -> None:
        self._config = config
        self._environment = environment
        self._random_source = random_source
        self._payload_bits = int(expected_payload_bits)
        self._precision = required_precision(
            self._payload_bits, config.precision_margin
        )
        with localcontext() as decimal_context:
            decimal_context.prec = self._precision
            self._left = Decimal(0)
            self._right = Decimal(2) ** self._payload_bits
        self._history: list[tuple[Decimal, Decimal, Decimal]] = []
        self._steps = 0
        self._failure_step: int | None = None
        self._failure_reason: str | None = None
        self._finalized = False

    @property
    def done(self) -> bool:
        return self._finalized

    def observe(self, context: StepContext, observed_token_id: int) -> DecodeProgress:
        if self._finalized:
            raise SessionStateError("cannot call RRC decoder.observe() after finalize()")
        if self._failure_reason is not None:
            self._steps += 1
            return DecodeProgress(
                method_trace={"rrc_decode_failed": True, "failure_step": self._failure_step}
            )

        dist = _StepDistribution.from_context(context, self._environment)
        index = dist.index_of(int(observed_token_id))
        if index < 0:
            self._failure_step = context.step_index
            self._failure_reason = (
                f"observed token {int(observed_token_id)} is outside current P_reference support"
            )
            self._steps += 1
            return DecodeProgress(
                method_trace={
                    "rrc_decode_failed": True,
                    "failure_step": self._failure_step,
                }
            )

        with localcontext() as decimal_context:
            decimal_context.prec = self._precision
            width = self._right - self._left
            offset = Decimal(self._random_source.random())
            self._history.append((self._left, width, offset))
            self._left, self._right = _narrow(
                dist,
                self._left,
                self._right,
                index,
            )
        self._steps += 1
        return DecodeProgress(
            # RRC recovers the whole message only after the complete carrier is
            # available; no incremental payload bits are authoritative.
            recovered_bits=(),
            method_trace={"rrc_step": self._steps, "candidate_index": int(index)},
        )

    def finalize(self) -> DecoderFinalization:
        if self._finalized:
            raise SessionStateError("RRC decoder has already been finalized")
        self._finalized = True
        if self._failure_reason is not None or not self._history:
            return DecoderFinalization(
                recovered_bits=(),
                complete=False,
                metadata={
                    "steps": self._steps,
                    "failure_step": self._failure_step,
                    "failure_reason": self._failure_reason or "no carrier tokens",
                    "reference_commit": _REFERENCE_COMMIT,
                },
            )

        with localcontext() as decimal_context:
            decimal_context.prec = self._precision
            recovered_integer = _replay(self._left, self._right, self._history)

        upper = 2 ** self._payload_bits
        if not 0 <= recovered_integer < upper:
            return DecoderFinalization(
                recovered_bits=(),
                complete=False,
                metadata={
                    "steps": self._steps,
                    "failure_reason": (
                        f"recovered integer {recovered_integer} lies outside [0, 2**payload_bits)"
                    ),
                    "reference_commit": _REFERENCE_COMMIT,
                },
            )

        text = bin(recovered_integer)[2:].zfill(self._payload_bits)
        return DecoderFinalization(
            recovered_bits=tuple(int(bit) for bit in text),
            complete=True,
            metadata={
                "steps": self._steps,
                "precision_digits": self._precision,
                "reference_commit": _REFERENCE_COMMIT,
            },
        )


class RRCMethod(StegoMethod):
    """Factory for matching normalized RRC encoder/decoder sessions."""

    method_id = "rrc"

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
        parsed = RRCConfig.from_mapping(config)
        if random_source is None:
            raise ConfigurationError("RRC requires synchronized method randomness")
        if target_payload_bits is None or target_payload_bits <= 0:
            raise ConfigurationError(
                "RRC requires positive target_payload_bits from fixed_payload_bits termination"
            )
        return RRCEncoderSession(
            config=parsed,
            environment=environment,
            secret_source=secret_source,
            random_source=random_source,
            target_payload_bits=target_payload_bits,
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
        parsed = RRCConfig.from_mapping(config)
        if random_source is None:
            raise ConfigurationError("RRC requires synchronized method randomness")
        if expected_payload_bits is None or expected_payload_bits <= 0:
            raise ConfigurationError("RRC decoder requires expected_payload_bits")
        return RRCDecoderSession(
            config=parsed,
            environment=environment,
            random_source=random_source,
            expected_payload_bits=expected_payload_bits,
        )


__all__ = [
    "RRCConfig",
    "RRCDecoderSession",
    "RRCEncoderSession",
    "RRCMethod",
    "RRCPrecisionError",
    "required_precision",
]
