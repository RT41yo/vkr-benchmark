from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "probe_stage4_dairstega_characteristic.py"
    spec = importlib.util.spec_from_file_location("probe_stage4_dairstega_characteristic", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tradeoff_summary_accepts_capacity_gain_with_quality_cost() -> None:
    module = _module()
    points = {
        "quality_a8_b1": {
            "bits_per_token": 1.2,
            "raw_lm_ppl": 8.0,
            "kl_stego_to_ref_mean_bits_per_token": 0.2,
            "tvd_mean": 0.1,
        },
        "beta_effect_a8_b05": {
            "bits_per_token": 1.8,
            "raw_lm_ppl": 12.0,
            "kl_stego_to_ref_mean_bits_per_token": 0.4,
            "tvd_mean": 0.2,
        },
        "alpha_effect_a32_b1": {
            "bits_per_token": 1.25,
            "raw_lm_ppl": 8.5,
            "kl_stego_to_ref_mean_bits_per_token": 0.22,
            "tvd_mean": 0.11,
        },
        "capacity_a32_b05": {
            "bits_per_token": 2.4,
            "raw_lm_ppl": 18.0,
            "kl_stego_to_ref_mean_bits_per_token": 0.5,
            "tvd_mean": 0.3,
        },
    }
    summary = module._tradeoff_summary(points)
    assert summary["capacity_oriented_point_has_higher_bpt"] is True
    assert summary["concealment_or_quality_cost_observed"] is True
    assert summary["publication_characteristic_observed"] is True
    assert summary["beta_0_5_increases_bpt_at_alpha_8"] is True
    assert summary["beta_0_5_increases_bpt_at_alpha_32"] is True


def test_tradeoff_summary_rejects_capacity_point_without_capacity_gain() -> None:
    module = _module()
    points = {
        "quality_a8_b1": {
            "bits_per_token": 2.0,
            "raw_lm_ppl": 8.0,
            "kl_stego_to_ref_mean_bits_per_token": 0.2,
            "tvd_mean": 0.1,
        },
        "beta_effect_a8_b05": {
            "bits_per_token": 1.9,
            "raw_lm_ppl": 12.0,
            "kl_stego_to_ref_mean_bits_per_token": 0.4,
            "tvd_mean": 0.2,
        },
        "alpha_effect_a32_b1": {
            "bits_per_token": 2.1,
            "raw_lm_ppl": 8.5,
            "kl_stego_to_ref_mean_bits_per_token": 0.22,
            "tvd_mean": 0.11,
        },
        "capacity_a32_b05": {
            "bits_per_token": 1.8,
            "raw_lm_ppl": 18.0,
            "kl_stego_to_ref_mean_bits_per_token": 0.5,
            "tvd_mean": 0.3,
        },
    }
    summary = module._tradeoff_summary(points)
    assert summary["capacity_oriented_point_has_higher_bpt"] is False
    assert summary["publication_characteristic_observed"] is False
