"""Normalized DAIRstega adapter.

Publication:
    Wang et al., "Dynamically Allocated Interval-Based Generative Linguistic
    Steganography with Roulette Wheel", Applied Soft Computing 176 (2025),
    arXiv:2401.15656v5.

Pinned public implementation:
    WangYH-BUPT/DAIRstega,
    commit 8d85edf98d48c3efa827a125b6d4e90f88141ea2,
    ``DAIRstega_t.py``.

Normalized Stage-4 semantics:
* ``alpha`` is the secret look-ahead / integer roulette precision in bits.
* ``beta`` controls the power allocation ``n_j ∝ p_j**beta``.
* ``top_k`` is method-local and is applied *after* canonical P_reference.
* the pinned implementation's alpha-dependent probability-floor restriction,
  integer rounding, overfill truncation, and residual-to-first-token rule are
  retained because they determine the executable roulette partition.
* the paper's extraction rule is implemented as the maximal common binary
  prefix of the selected interval endpoints.  The public helper under-consumes
  by one bit only for singleton intervals; that repository detail is left for
  direct conformance characterization rather than silently becoming the
  normalized scientific definition.

DAIRstega reads alpha bits but advances only the confirmed common prefix.  The
unconsumed suffix is reused on the next token.  This adapter therefore keeps an
explicit overlapping alpha-bit look-ahead buffer while reporting only confirmed
prefix bits as useful payload.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from vkr_benchmark.distributions import (
    DistributionInfo,
    QMode,
    QSource,
    StepContext,
    explicit_dair_q,
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

_REFERENCE_COMMIT = "8d85edf98d48c3efa827a125b6d4e90f88141ea2"


@dataclass(frozen=True, slots=True)
class DAIRConfig:
    alpha: int
    beta: float
    top_k: int

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "DAIRConfig":
        allowed = {"alpha", "beta", "top_k"}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ConfigurationError(f"unknown DAIRstega configuration fields: {unknown}")
        missing = sorted(allowed - set(values))
        if missing:
            raise ConfigurationError(f"missing DAIRstega configuration fields: {missing}")

        alpha = values["alpha"]
        beta = values["beta"]
        top_k = values["top_k"]
        if not isinstance(alpha, int) or isinstance(alpha, bool) or not 1 <= alpha <= 52:
            raise ConfigurationError("DAIRstega alpha must be an integer in [1, 52]")
        if not isinstance(beta, (int, float)) or isinstance(beta, bool):
            raise ConfigurationError("DAIRstega beta must be numeric")
        beta = float(beta)
        if not math.isfinite(beta) or not 0.0 < beta <= 1.0:
            raise ConfigurationError("DAIRstega beta must satisfy 0 < beta <= 1")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 2:
            raise ConfigurationError("DAIRstega top_k must be an integer >= 2")
        return cls(alpha=alpha, beta=beta, top_k=top_k)


@dataclass(frozen=True, slots=True)
class _Interval:
    token_id: int
    start: int
    stop: int  # exclusive

    @property
    def width(self) -> int:
        return self.stop - self.start


def _validate_reference(context: StepContext, environment: MethodEnvironment) -> None:
    if context.reference.vocab_size != environment.output_vocab_size:
        raise MethodError(
            "DAIRstega MethodEnvironment output vocabulary does not match P_reference"
        )
    for token_id in context.reference.token_order:
        if not environment.is_allowed(int(token_id)):
            raise MethodError(
                f"P_reference support token {int(token_id)} lies outside V_allowed"
            )


def _allocate(context: StepContext, config: DAIRConfig) -> tuple[_Interval, ...]:
    reference = context.reference
    token_ids = np.asarray(reference.token_order, dtype=np.int64)
    if token_ids.size < 2:
        raise MethodError("DAIRstega requires at least two positive-probability tokens")
    probabilities = reference.probabilities[token_ids].astype(np.float64, copy=True)

    total_codes = 1 << config.alpha
    threshold = 1.0 / float(total_codes)
    below = np.flatnonzero(probabilities < threshold)
    precision_count = token_ids.size
    if below.size:
        precision_count = max(2, int(below[0]))
    count = min(int(token_ids.size), precision_count, config.top_k)
    token_ids = token_ids[:count]
    probabilities = probabilities[:count]

    weights = np.power(probabilities, config.beta, dtype=np.float64)
    weight_sum = float(np.sum(weights, dtype=np.float64))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        raise MethodError("DAIRstega transformed candidate weights are invalid")
    weights /= weight_sum

    counts = [int(value) for value in np.rint(weights * float(total_codes))]
    retained: list[tuple[int, int]] = []
    cumulative = 0
    for token_id, interval_count in zip(token_ids.tolist(), counts, strict=True):
        next_cumulative = cumulative + interval_count
        if next_cumulative > total_codes:
            break
        retained.append((int(token_id), int(interval_count)))
        cumulative = next_cumulative
    if not retained:
        raise MethodError("DAIRstega integer allocation retained no candidates")

    residual = total_codes - cumulative
    first_token, first_count = retained[0]
    retained[0] = (first_token, first_count + residual)

    intervals: list[_Interval] = []
    cursor = 0
    for token_id, interval_count in retained:
        if interval_count < 0:
            raise MethodError("DAIRstega produced a negative interval width")
        if interval_count == 0:
            continue
        stop = cursor + interval_count
        intervals.append(_Interval(token_id=token_id, start=cursor, stop=stop))
        cursor = stop
    if not intervals or cursor != total_codes:
        raise MethodError("DAIRstega intervals do not exactly cover the code space")
    return tuple(intervals)


def _bits_to_uint_msb(bits: tuple[int, ...]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _uint_prefix_bits(value: int, *, alpha: int, count: int) -> tuple[int, ...]:
    return tuple((value >> (alpha - 1 - i)) & 1 for i in range(count))


def _common_prefix(interval: _Interval, *, alpha: int) -> tuple[int, ...]:
    lower = interval.start
    upper = interval.stop - 1
    xor = lower ^ upper
    count = alpha if xor == 0 else alpha - xor.bit_length()
    return _uint_prefix_bits(lower, alpha=alpha, count=count)


def _find_interval_by_point(intervals: tuple[_Interval, ...], point: int) -> _Interval:
    for interval in intervals:
        if interval.start <= point < interval.stop:
            return interval
    raise MethodError(f"DAIRstega secret point {point} lies outside roulette intervals")


def _find_interval_by_token(
    intervals: tuple[_Interval, ...], token_id: int
) -> _Interval:
    for interval in intervals:
        if interval.token_id == int(token_id):
            return interval
    raise MethodError(
        f"observed token {int(token_id)} has no positive-width DAIRstega interval"
    )


def _intersect_support(
    lower: int, upper: int, interval: _Interval
) -> tuple[int, int]:
    lo = max(lower, interval.start)
    hi = min(upper, interval.stop - 1)
    if lo > hi:
        raise MethodError(
            "observed DAIRstega token is impossible under the current secret-window state"
        )
    return lo, hi


def _advance_support(
    lower: int, upper: int, *, prefix_bits: int, alpha: int
) -> tuple[int, int]:
    if prefix_bits < 0 or prefix_bits > alpha:
        raise MethodError("DAIRstega prefix length lies outside [0, alpha]")
    suffix_bits = alpha - prefix_bits
    if suffix_bits == 0:
        suffix_lower = suffix_upper = 0
    else:
        mask = (1 << suffix_bits) - 1
        suffix_lower = lower & mask
        suffix_upper = upper & mask
    next_lower = suffix_lower << prefix_bits
    next_upper = ((suffix_upper + 1) << prefix_bits) - 1
    return next_lower, next_upper


class DAIREncoderSession(EncoderSession):
    def __init__(
        self,
        *,
        config: DAIRConfig,
        environment: MethodEnvironment,
        secret_source: SecretSource,
    ) -> None:
        self._config = config
        self._environment = environment
        self._secret_source = secret_source
        self._buffer: tuple[int, ...] | None = None
        self._payload_bits = 0
        self._steps = 0
        self._finalized = False
        self._window_lower = 0
        self._window_upper = (1 << config.alpha) - 1
        self._last_q_state: tuple[int, int, int] | None = None

    @property
    def done(self) -> bool:
        return self._finalized

    def _ensure_buffer(self) -> tuple[int, ...]:
        if self._buffer is None:
            self._buffer = self._secret_source.read_bits(self._config.alpha)
        return self._buffer

    def step(self, context: StepContext) -> EncodeDecision:
        if self._finalized:
            raise SessionStateError("cannot call DAIRstega encoder.step() after finalize()")
        _validate_reference(context, self._environment)
        intervals = _allocate(context, self._config)
        bits = self._ensure_buffer()
        point = _bits_to_uint_msb(bits)
        if not self._window_lower <= point <= self._window_upper:
            raise MethodError(
                "DAIRstega secret look-ahead is inconsistent with conditioned window state"
            )

        q_lower, q_upper = self._window_lower, self._window_upper
        interval = _find_interval_by_point(intervals, point)
        posterior_lower, posterior_upper = _intersect_support(
            q_lower, q_upper, interval
        )
        prefix = _common_prefix(interval, alpha=self._config.alpha)
        consumed = len(prefix)

        if consumed:
            suffix = bits[consumed:]
            replenished = self._secret_source.read_bits(consumed)
            self._buffer = suffix + replenished
        else:
            self._buffer = bits

        self._window_lower, self._window_upper = _advance_support(
            posterior_lower,
            posterior_upper,
            prefix_bits=consumed,
            alpha=self._config.alpha,
        )
        self._payload_bits += consumed
        self._steps += 1
        self._last_q_state = (context.step_index, q_lower, q_upper)

        return EncodeDecision(
            token_id=interval.token_id,
            bits_consumed=consumed,
            method_trace={
                "alpha": self._config.alpha,
                "beta": self._config.beta,
                "top_k": self._config.top_k,
                "secret_point": point,
                "interval_start": interval.start,
                "interval_stop": interval.stop,
                "interval_width": interval.width,
                "common_prefix_bits": prefix,
                "candidate_intervals": len(intervals),
                "q_window_lower": q_lower,
                "q_window_upper": q_upper,
                "next_window_lower": self._window_lower,
                "next_window_upper": self._window_upper,
            },
        )

    def distribution_info(
        self, context: StepContext, decision: EncodeDecision
    ) -> DistributionInfo:
        del decision
        if self._last_q_state is None or self._last_q_state[0] != context.step_index:
            raise SessionStateError("DAIRstega distribution_info called out of step order")
        _, lower, upper = self._last_q_state
        q = explicit_dair_q(
            context.reference,
            alpha=self._config.alpha,
            beta=self._config.beta,
            top_k=self._config.top_k,
            window_lower=lower,
            window_upper=upper,
        )
        return DistributionInfo.explicit(
            q,
            mode=QMode.ANALYTIC_EXACT,
            source=QSource.INDEPENDENT_ANALYTIC,
            metadata={
                "method": "dairstega",
                "reference_commit": _REFERENCE_COMMIT,
                "construction": "independent_integer_interval_intersection",
                "alpha": self._config.alpha,
                "beta": self._config.beta,
                "top_k": self._config.top_k,
                "window_lower": lower,
                "window_upper": upper,
            },
        )

    def finalize(self) -> EncoderFinalization:
        if self._finalized:
            raise SessionStateError("DAIRstega encoder has already been finalized")
        self._finalized = True
        return EncoderFinalization(
            payload_bits=self._payload_bits,
            termination_reason="streaming_session_finalized",
            metadata={
                "steps": self._steps,
                "reference_commit": _REFERENCE_COMMIT,
                "alpha": self._config.alpha,
                "beta": self._config.beta,
                "top_k": self._config.top_k,
                "lookahead_bits": self._config.alpha,
            },
        )


class DAIRDecoderSession(DecoderSession):
    def __init__(
        self,
        *,
        config: DAIRConfig,
        environment: MethodEnvironment,
        expected_payload_bits: int | None,
    ) -> None:
        self._config = config
        self._environment = environment
        self._expected_payload_bits = expected_payload_bits
        self._recovered: list[int] = []
        self._steps = 0
        self._finalized = False
        self._window_lower = 0
        self._window_upper = (1 << config.alpha) - 1

    @property
    def done(self) -> bool:
        if self._finalized:
            return True
        if self._expected_payload_bits is None:
            return False
        return len(self._recovered) >= self._expected_payload_bits

    def observe(self, context: StepContext, observed_token_id: int) -> DecodeProgress:
        if self._finalized:
            raise SessionStateError("cannot call DAIRstega decoder.observe() after finalize()")
        _validate_reference(context, self._environment)
        intervals = _allocate(context, self._config)
        interval = _find_interval_by_token(intervals, int(observed_token_id))
        posterior_lower, posterior_upper = _intersect_support(
            self._window_lower, self._window_upper, interval
        )
        prefix = _common_prefix(interval, alpha=self._config.alpha)
        self._recovered.extend(prefix)
        self._window_lower, self._window_upper = _advance_support(
            posterior_lower,
            posterior_upper,
            prefix_bits=len(prefix),
            alpha=self._config.alpha,
        )
        self._steps += 1
        return DecodeProgress(
            recovered_bits=prefix,
            method_trace={
                "interval_start": interval.start,
                "interval_stop": interval.stop,
                "interval_width": interval.width,
                "common_prefix_bits": prefix,
                "next_window_lower": self._window_lower,
                "next_window_upper": self._window_upper,
            },
        )

    def finalize(self) -> DecoderFinalization:
        if self._finalized:
            raise SessionStateError("DAIRstega decoder has already been finalized")
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
                "raw_recovered_bits": len(self._recovered),
                "reference_commit": _REFERENCE_COMMIT,
            },
        )


class DAIRMethod(StegoMethod):
    method_id = "dairstega"

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
        if random_source is not None:
            raise ConfigurationError("normalized DAIRstega does not use method.random_seed")
        if key is not None:
            raise ConfigurationError("normalized DAIRstega does not use method.key")
        if target_payload_bits is not None:
            raise ConfigurationError(
                "normalized DAIRstega currently uses fixed_carrier_tokens, not fixed_payload_bits"
            )
        return DAIREncoderSession(
            config=DAIRConfig.from_mapping(config),
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
        if random_source is not None:
            raise ConfigurationError("normalized DAIRstega does not use method.random_seed")
        if key is not None:
            raise ConfigurationError("normalized DAIRstega does not use method.key")
        return DAIRDecoderSession(
            config=DAIRConfig.from_mapping(config),
            environment=environment,
            expected_payload_bits=expected_payload_bits,
        )
