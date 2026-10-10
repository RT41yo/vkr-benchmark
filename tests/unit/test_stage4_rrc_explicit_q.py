from __future__ import annotations

import numpy as np

from vkr_benchmark.distributions import ReferenceDistribution
from vkr_benchmark.validation import evaluate_explicit_rrc_q, normalized_rrc_reference


def test_rrc_explicit_q_uses_interval_widths_and_common_metrics() -> None:
    reference = ReferenceDistribution(
        np.asarray([0.4, 0.3, 0.2, 0.1], dtype=np.float32)
    )
    result = evaluate_explicit_rrc_q(reference)
    p = normalized_rrc_reference(reference)
    assert result.distortion.q_mode.value == "analytic_exact"
    assert result.distortion.q_source.value == "independent_analytic"
    assert np.max(np.abs(result.q_probabilities - p)) < 2e-15
    assert result.distortion.kl_ref_to_stego_bits is not None
    assert result.distortion.kl_stego_to_ref_bits is not None
    assert result.distortion.tvd is not None
    assert result.distortion.kl_ref_to_stego_bits < 1e-12
    assert result.distortion.kl_stego_to_ref_bits < 1e-12
    assert result.distortion.tvd < 1e-13


def test_rrc_explicit_q_preserves_sparse_support() -> None:
    reference = ReferenceDistribution(
        np.asarray([0.0, 0.55, 0.0, 0.25, 0.15, 0.05], dtype=np.float32)
    )
    result = evaluate_explicit_rrc_q(reference)
    assert result.q_probabilities[0] == 0.0
    assert result.q_probabilities[2] == 0.0
    assert np.array_equal(
        result.q_probabilities > 0.0,
        result.normalized_reference > 0.0,
    )


def test_rrc_large_vocab_roundoff_is_tiny_but_not_small_support_ulp_scale() -> None:
    vocab_size = 128_256
    probs = np.full(vocab_size, 0.1 / (vocab_size - 1), dtype=np.float64)
    probs[0] = 0.9
    probs = np.asarray(probs, dtype=np.float32)
    probs = np.asarray(probs / np.sum(probs, dtype=np.float32), dtype=np.float32)
    result = evaluate_explicit_rrc_q(ReferenceDistribution(probs))

    assert result.distortion.kl_ref_to_stego_bits is not None
    assert result.distortion.kl_stego_to_ref_bits is not None
    assert result.distortion.tvd is not None
    assert np.isfinite(result.distortion.kl_ref_to_stego_bits)
    assert np.isfinite(result.distortion.kl_stego_to_ref_bits)
    # Sequential FP64 cumulative boundaries across a full vocabulary leave a
    # visible aggregate residue even though the distortion is negligible.
    assert result.distortion.tvd > 1e-13
    assert result.distortion.tvd < 1e-9
