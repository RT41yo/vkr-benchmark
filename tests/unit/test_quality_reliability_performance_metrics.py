from __future__ import annotations

from math import log

import numpy as np
import pytest

from vkr_benchmark.errors import MetricError
from vkr_benchmark.metrics import (
    TimingBreakdown,
    compute_performance_metrics,
    compute_raw_lm_quality_metrics,
    compute_reliability_metrics,
    raw_lm_token_nll_nats,
)


def test_raw_lm_nll_uniform_distribution() -> None:
    logits = np.zeros(4, dtype=np.float32)
    assert raw_lm_token_nll_nats(logits, 2) == pytest.approx(log(4.0), abs=1e-6)


def test_raw_lm_nll_uses_unmodified_logits() -> None:
    logits = np.array([4.0, 1.0, -2.0], dtype=np.float32)
    expected = float(np.log(np.exp(4.0) + np.exp(1.0) + np.exp(-2.0)) - 1.0)
    assert raw_lm_token_nll_nats(logits, 1) == pytest.approx(expected, abs=1e-6)


def test_raw_lm_quality_aggregates_nll_and_ppl() -> None:
    metrics = compute_raw_lm_quality_metrics([log(2.0), log(8.0)])
    assert metrics.nll_raw_lm_nats_per_token == pytest.approx(log(4.0))
    assert metrics.ppl_raw_lm == pytest.approx(4.0)


def test_raw_lm_quality_rejects_empty_sequence() -> None:
    with pytest.raises(MetricError, match="at least one"):
        compute_raw_lm_quality_metrics([])


def test_raw_lm_nll_rejects_invalid_token() -> None:
    with pytest.raises(MetricError, match="outside"):
        raw_lm_token_nll_nats(np.zeros(3, dtype=np.float32), 3)


def test_raw_lm_nll_rejects_nonfinite_logits() -> None:
    with pytest.raises(MetricError, match="NaN or Inf"):
        raw_lm_token_nll_nats(np.array([0.0, np.nan], dtype=np.float32), 0)


def test_reliability_exact_roundtrip_has_zero_ber() -> None:
    metrics = compute_reliability_metrics(
        expected_bits=(0, 1, 0, 1),
        recovered_bits=(0, 1, 0, 1),
        roundtrip_exact=True,
        first_mismatch_bit=None,
        recovered_extra_bits=0,
        token_sequence_roundtrip_exact=True,
        first_token_roundtrip_mismatch=None,
    )
    assert metrics.ber == 0.0
    assert metrics.bit_errors == 0
    assert metrics.length_delta_bits == 0
    assert metrics.roundtrip_exact is True


def test_reliability_counts_mismatched_and_missing_bits_as_errors() -> None:
    metrics = compute_reliability_metrics(
        expected_bits=(0, 1, 0, 1),
        recovered_bits=(0, 0),
        roundtrip_exact=False,
        first_mismatch_bit=1,
        recovered_extra_bits=0,
        token_sequence_roundtrip_exact=True,
        first_token_roundtrip_mismatch=None,
    )
    assert metrics.bit_errors == 3
    assert metrics.ber == pytest.approx(0.75)
    assert metrics.length_delta_bits == -2


def test_reliability_extra_bits_do_not_inflate_ber() -> None:
    metrics = compute_reliability_metrics(
        expected_bits=(0, 1),
        recovered_bits=(0, 1, 1),
        roundtrip_exact=False,
        first_mismatch_bit=2,
        recovered_extra_bits=1,
        token_sequence_roundtrip_exact=True,
        first_token_roundtrip_mismatch=None,
    )
    assert metrics.bit_errors == 0
    assert metrics.ber == 0.0
    assert metrics.recovered_extra_bits == 1
    assert metrics.length_delta_bits == 1


def test_reliability_empty_payload_has_undefined_ber() -> None:
    metrics = compute_reliability_metrics(
        expected_bits=(),
        recovered_bits=(),
        roundtrip_exact=True,
        first_mismatch_bit=None,
        recovered_extra_bits=0,
        token_sequence_roundtrip_exact=True,
        first_token_roundtrip_mismatch=None,
    )
    assert metrics.ber is None


def test_reliability_rejects_nonbinary_values() -> None:
    with pytest.raises(MetricError, match="binary"):
        compute_reliability_metrics(
            expected_bits=(0, 2),
            recovered_bits=(0, 1),
            roundtrip_exact=False,
            first_mismatch_bit=1,
            recovered_extra_bits=0,
            token_sequence_roundtrip_exact=True,
            first_token_roundtrip_mismatch=None,
        )


def test_timing_breakdown_total_is_component_sum() -> None:
    timing = TimingBreakdown(
        lm_forward_ms=10.0,
        distribution_processing_ms=5.0,
        stego_algorithm_ms=2.0,
    )
    assert timing.total_ms == pytest.approx(17.0)


def test_timing_breakdown_rejects_negative_values() -> None:
    with pytest.raises(MetricError, match="non-negative"):
        TimingBreakdown(-1.0, 0.0, 0.0)


def test_performance_metrics_normalize_time_and_throughput() -> None:
    encode = TimingBreakdown(10.0, 5.0, 5.0)  # 20 ms
    decode = TimingBreakdown(8.0, 4.0, 4.0)   # 16 ms
    metrics = compute_performance_metrics(
        payload_bits=40,
        encode_tokens=10,
        decode_tokens=8,
        encode_timing=encode,
        decode_timing=decode,
    )
    assert metrics.encode_total_ms == pytest.approx(20.0)
    assert metrics.decode_total_ms == pytest.approx(16.0)
    assert metrics.encode_ms_per_token == pytest.approx(2.0)
    assert metrics.decode_ms_per_token == pytest.approx(2.0)
    assert metrics.payload_bits_per_second_encode == pytest.approx(2000.0)
    assert metrics.payload_bits_per_second_decode == pytest.approx(2500.0)
    assert metrics.lm_forward_total_ms == pytest.approx(18.0)
    assert metrics.distribution_processing_total_ms == pytest.approx(9.0)
    assert metrics.stego_algorithm_total_ms == pytest.approx(9.0)


def test_performance_metrics_allow_zero_payload_with_zero_throughput() -> None:
    timing = TimingBreakdown(1.0, 1.0, 1.0)
    metrics = compute_performance_metrics(
        payload_bits=0,
        encode_tokens=1,
        decode_tokens=1,
        encode_timing=timing,
        decode_timing=timing,
    )
    assert metrics.payload_bits_per_second_encode == 0.0
    assert metrics.payload_bits_per_second_decode == 0.0


def test_performance_metrics_reject_zero_token_count() -> None:
    timing = TimingBreakdown(1.0, 1.0, 1.0)
    with pytest.raises(MetricError, match="token counts"):
        compute_performance_metrics(
            payload_bits=1,
            encode_tokens=0,
            decode_tokens=1,
            encode_timing=timing,
            decode_timing=timing,
        )


def test_raw_lm_nll_supports_torch_tensor_when_available() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    assert raw_lm_token_nll_nats(logits, 1) == pytest.approx(log(4.0), abs=1e-6)
