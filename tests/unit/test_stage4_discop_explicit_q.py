from __future__ import annotations

import numpy as np

from vkr_benchmark.distributions import ReferenceDistribution
from vkr_benchmark.validation import (
    branch_selection_probabilities,
    evaluate_explicit_discop_q,
    explicit_discop_q,
    normalized_reference_probabilities,
)


def test_branch_integration_matches_node_mass_ratio_without_using_certificate() -> None:
    for left_fraction in (0.01, 0.1, 0.25, 0.4, 0.5, 0.73, 0.9, 0.99):
        left, right = branch_selection_probabilities(
            left_probability=left_fraction * 7.0,
            node_probability=7.0,
        )
        assert np.isclose(left + right, 1.0, rtol=0.0, atol=1e-14)
        assert np.isclose(left, left_fraction, rtol=0.0, atol=1e-14)
        assert np.isclose(right, 1.0 - left_fraction, rtol=0.0, atol=1e-14)


def test_explicit_q_matches_normalized_reference_on_nontrivial_tree() -> None:
    reference = ReferenceDistribution(
        np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32)
    )
    q = explicit_discop_q(reference)
    p = normalized_reference_probabilities(reference)
    assert np.isclose(np.sum(q, dtype=np.float64), 1.0, rtol=0.0, atol=1e-14)
    assert np.max(np.abs(q - p)) < 1e-14


def test_explicit_q_preserves_zero_support_and_ties() -> None:
    reference = ReferenceDistribution(
        np.array([0.0, 0.4, 0.2, 0.2, 0.2, 0.0], dtype=np.float32)
    )
    q = explicit_discop_q(reference)
    p = normalized_reference_probabilities(reference)
    assert q[0] == 0.0
    assert q[5] == 0.0
    assert np.max(np.abs(q - p)) < 1e-14


def test_explicit_q_uses_common_kl_tvd_formulas() -> None:
    reference = ReferenceDistribution(
        np.array([0.55, 0.25, 0.15, 0.05], dtype=np.float32)
    )
    result = evaluate_explicit_discop_q(reference)
    assert result.distortion.q_mode.value == "exact_enumeration"
    assert result.distortion.q_source.value == "independent_enumeration"
    assert result.max_abs_difference < 1e-14
    assert result.l1_difference < 1e-13
    assert result.distortion.kl_ref_to_stego_bits is not None
    assert result.distortion.kl_stego_to_ref_bits is not None
    assert result.distortion.tvd is not None
    assert result.distortion.kl_ref_to_stego_bits < 1e-12
    assert result.distortion.kl_stego_to_ref_bits < 1e-12
    assert result.distortion.tvd < 1e-13
