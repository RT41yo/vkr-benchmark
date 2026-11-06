from __future__ import annotations

from math import isinf

import numpy as np

from vkr_benchmark.diagnostics import analyze_rrc_boundary_precision
from vkr_benchmark.distributions import ReferenceDistribution


def test_rrc_boundary_probe_detects_tiny_positive_interval_lost_in_fp64_cumsum() -> None:
    reference = ReferenceDistribution(
        np.asarray([0.5, 0.5, 1e-20], dtype=np.float32)
    )

    result = analyze_rrc_boundary_precision(reference)

    assert result.support_loss is True
    assert result.zero_width_count == 1
    assert result.first_lost_rank == 2
    assert result.lost_reference_mass > 0.0
    assert result.max_lost_reference_probability > 0.0
    assert isinf(result.current_distortion.kl_ref_to_stego_bits)
    assert result.current_distortion.kl_stego_to_ref_bits is not None
    assert result.current_distortion.tvd is not None
    assert result.current_distortion.tvd > 0.0
    assert result.ideal_distortion.kl_ref_to_stego_bits is not None
    assert result.ideal_distortion.kl_ref_to_stego_bits < 1e-12
    assert result.ideal_distortion.kl_stego_to_ref_bits is not None
    assert result.ideal_distortion.kl_stego_to_ref_bits < 1e-12
    assert result.ideal_distortion.tvd == 0.0
    assert result.lost_intervals[0].collapse_stage == "fp64_cumsum"


def test_rrc_boundary_probe_reports_no_support_loss_for_well_separated_probabilities() -> None:
    reference = ReferenceDistribution(
        np.asarray([0.4, 0.3, 0.2, 0.1], dtype=np.float32)
    )

    result = analyze_rrc_boundary_precision(reference)

    assert result.support_loss is False
    assert result.zero_width_count == 0
    assert result.lost_reference_mass == 0.0
    assert result.first_lost_rank is None
    assert result.lost_intervals == ()
    assert result.current_distortion.kl_ref_to_stego_bits is not None
    assert result.current_distortion.kl_ref_to_stego_bits < 1e-12
    assert result.current_distortion.tvd is not None
    assert result.current_distortion.tvd < 1e-12
