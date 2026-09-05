"""Raw-LM quality, reliability, and computational-efficiency metrics."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, fsum, isfinite
from typing import Any, Iterable, Sequence

import numpy as np

from vkr_benchmark.errors import MetricError


@dataclass(frozen=True, slots=True)
class RawLMQualityMetrics:
    """Text-quality proxy under the unmodified generating LM."""

    nll_raw_lm_nats_per_token: float
    ppl_raw_lm: float


@dataclass(frozen=True, slots=True)
class ReliabilityMetrics:
    """Run-level recovery and tokenizer round-trip diagnostics."""

    roundtrip_exact: bool
    ber: float | None
    bit_errors: int
    expected_length_bits: int
    recovered_length_bits: int
    length_delta_bits: int
    recovered_extra_bits: int
    first_mismatch_bit: int | None
    first_decode_failure_token: int | None
    token_sequence_roundtrip_exact: bool
    first_token_roundtrip_mismatch: int | None


@dataclass(frozen=True, slots=True)
class TimingBreakdown:
    """Measured runtime components with metric instrumentation excluded."""

    lm_forward_ms: float
    distribution_processing_ms: float
    stego_algorithm_ms: float

    def __post_init__(self) -> None:
        for name, value in (
            ("lm_forward_ms", self.lm_forward_ms),
            ("distribution_processing_ms", self.distribution_processing_ms),
            ("stego_algorithm_ms", self.stego_algorithm_ms),
        ):
            if not isfinite(value) or value < 0.0:
                raise MetricError(f"{name} must be finite and non-negative")

    @property
    def total_ms(self) -> float:
        return float(
            self.lm_forward_ms
            + self.distribution_processing_ms
            + self.stego_algorithm_ms
        )


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """v0.1 computational-efficiency fields plus side-specific breakdowns."""

    encode_total_ms: float
    decode_total_ms: float
    encode_ms_per_token: float
    decode_ms_per_token: float
    payload_bits_per_second_encode: float
    payload_bits_per_second_decode: float
    lm_forward_total_ms: float
    distribution_processing_total_ms: float
    stego_algorithm_total_ms: float
    encode_lm_forward_ms: float
    encode_distribution_processing_ms: float
    encode_stego_algorithm_ms: float
    decode_lm_forward_ms: float
    decode_distribution_processing_ms: float
    decode_stego_algorithm_ms: float


def raw_lm_token_nll_nats(raw_logits: Any, token_id: int) -> float:
    """Return ``-ln P_LM-raw(token_id)`` from untouched next-token logits.

    No benchmark token mask, temperature transform, top-k/top-p filter, or
    steganographic distribution is applied. Torch logits are evaluated by
    FP32 ``logsumexp`` on their current device; NumPy/synthetic inputs use an
    equivalent stable reduction.
    """

    token = int(token_id)

    try:
        import torch
    except ImportError:  # pragma: no cover - core-only environments
        torch = None  # type: ignore[assignment]

    if torch is not None and isinstance(raw_logits, torch.Tensor):
        if raw_logits.ndim != 1:
            raise MetricError("raw LM logits must be one-dimensional")
        if token < 0 or token >= int(raw_logits.numel()):
            raise MetricError(f"selected token id {token} is outside raw LM logits")
        if not bool(torch.isfinite(raw_logits).all().item()):
            raise MetricError("raw LM logits contain NaN or Inf")

        logits = raw_logits.detach().to(dtype=torch.float32)
        nll = float((torch.logsumexp(logits, dim=0) - logits[token]).item())
    else:
        logits = np.asarray(raw_logits, dtype=np.float32)
        if logits.ndim != 1:
            raise MetricError("raw LM logits must be one-dimensional")
        if token < 0 or token >= int(logits.size):
            raise MetricError(f"selected token id {token} is outside raw LM logits")
        if not np.all(np.isfinite(logits)):
            raise MetricError("raw LM logits contain NaN or Inf")

        max_logit = np.max(logits)
        shifted = np.asarray(logits - max_logit, dtype=np.float32)
        exp_sum = np.sum(np.exp(shifted, dtype=np.float32), dtype=np.float32)
        if not np.isfinite(exp_sum) or exp_sum <= 0.0:
            raise MetricError("raw LM logsumexp normalization is invalid")
        log_z = float(max_logit + np.log(exp_sum))
        nll = float(log_z - float(logits[token]))

    if not isfinite(nll):
        raise MetricError(f"raw-LM token NLL is non-finite: {nll!r}")
    # The exact value is non-negative. Tiny negative values can arise only from
    # finite-precision logsumexp reduction when the selected probability is ~1.
    if nll < 0.0:
        if nll >= -1e-6:
            return 0.0
        raise MetricError(f"raw-LM token NLL became negative: {nll!r}")
    return nll


def compute_raw_lm_quality_metrics(
    step_nll_nats: Iterable[float],
) -> RawLMQualityMetrics:
    """Aggregate selected-token raw-LM NLL and perplexity."""

    values = tuple(float(value) for value in step_nll_nats)
    if not values:
        raise MetricError("at least one raw-LM token NLL is required")
    if any(not isfinite(value) or value < 0.0 for value in values):
        raise MetricError("raw-LM token NLL values must be finite and non-negative")

    mean_nll = float(fsum(values) / len(values))
    try:
        ppl = float(exp(mean_nll))
    except OverflowError:
        ppl = float("inf")

    return RawLMQualityMetrics(
        nll_raw_lm_nats_per_token=mean_nll,
        ppl_raw_lm=ppl,
    )


def compute_reliability_metrics(
    *,
    expected_bits: Sequence[int],
    recovered_bits: Sequence[int],
    roundtrip_exact: bool,
    first_mismatch_bit: int | None,
    recovered_extra_bits: int,
    token_sequence_roundtrip_exact: bool,
    first_token_roundtrip_mismatch: int | None,
    first_decode_failure_token: int | None = None,
) -> ReliabilityMetrics:
    """Compute BER with missing expected positions counted as bit errors.

    Extra recovered bits do not enter the BER numerator/denominator; they are
    retained as a separate diagnostic and necessarily make ``roundtrip_exact``
    false at the runner level.
    """

    expected = tuple(int(bit) for bit in expected_bits)
    recovered = tuple(int(bit) for bit in recovered_bits)
    if any(bit not in (0, 1) for bit in expected + recovered):
        raise MetricError("reliability inputs must contain only binary bits")

    expected_length = len(expected)
    recovered_length = len(recovered)
    compared = min(expected_length, recovered_length)
    mismatches = sum(expected[i] != recovered[i] for i in range(compared))
    missing = max(0, expected_length - recovered_length)
    bit_errors = int(mismatches + missing)

    if expected_length == 0:
        ber: float | None = None
    else:
        ber = float(bit_errors / expected_length)

    return ReliabilityMetrics(
        roundtrip_exact=bool(roundtrip_exact),
        ber=ber,
        bit_errors=bit_errors,
        expected_length_bits=expected_length,
        recovered_length_bits=recovered_length,
        length_delta_bits=recovered_length - expected_length,
        recovered_extra_bits=int(recovered_extra_bits),
        first_mismatch_bit=first_mismatch_bit,
        first_decode_failure_token=first_decode_failure_token,
        token_sequence_roundtrip_exact=bool(token_sequence_roundtrip_exact),
        first_token_roundtrip_mismatch=first_token_roundtrip_mismatch,
    )


def compute_performance_metrics(
    *,
    payload_bits: int,
    encode_tokens: int,
    decode_tokens: int,
    encode_timing: TimingBreakdown,
    decode_timing: TimingBreakdown,
) -> PerformanceMetrics:
    """Derive normalized timing/throughput fields from measured components."""

    payload = int(payload_bits)
    encode_count = int(encode_tokens)
    decode_count = int(decode_tokens)
    if payload < 0:
        raise MetricError("payload_bits must be non-negative")
    if encode_count <= 0 or decode_count <= 0:
        raise MetricError("encode/decode token counts must be positive")

    encode_total = encode_timing.total_ms
    decode_total = decode_timing.total_ms
    if encode_total <= 0.0 or decode_total <= 0.0:
        raise MetricError("encode/decode measured time must be positive")

    return PerformanceMetrics(
        encode_total_ms=encode_total,
        decode_total_ms=decode_total,
        encode_ms_per_token=float(encode_total / encode_count),
        decode_ms_per_token=float(decode_total / decode_count),
        payload_bits_per_second_encode=float(payload * 1000.0 / encode_total),
        payload_bits_per_second_decode=float(payload * 1000.0 / decode_total),
        lm_forward_total_ms=float(
            encode_timing.lm_forward_ms + decode_timing.lm_forward_ms
        ),
        distribution_processing_total_ms=float(
            encode_timing.distribution_processing_ms
            + decode_timing.distribution_processing_ms
        ),
        stego_algorithm_total_ms=float(
            encode_timing.stego_algorithm_ms + decode_timing.stego_algorithm_ms
        ),
        encode_lm_forward_ms=float(encode_timing.lm_forward_ms),
        encode_distribution_processing_ms=float(
            encode_timing.distribution_processing_ms
        ),
        encode_stego_algorithm_ms=float(encode_timing.stego_algorithm_ms),
        decode_lm_forward_ms=float(decode_timing.lm_forward_ms),
        decode_distribution_processing_ms=float(
            decode_timing.distribution_processing_ms
        ),
        decode_stego_algorithm_ms=float(decode_timing.stego_algorithm_ms),
    )
