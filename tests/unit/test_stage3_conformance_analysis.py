from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "stage3_conformance_analysis.py"


def _module():
    spec = importlib.util.spec_from_file_location("stage3_conformance_analysis", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config() -> dict:
    return json.loads(
        (REPO_ROOT / "configs" / "reproducibility" / "conformance_analysis.json").read_text(
            encoding="utf-8"
        )
    )


def _summary() -> dict:
    return json.loads(
        (
            REPO_ROOT
            / "results"
            / "stage3"
            / "matched_author_normalized"
            / "summary.json"
        ).read_text(encoding="utf-8")
    )


def _rows():
    module = _module()
    return module.load_csv(
        REPO_ROOT
        / "results"
        / "stage3"
        / "matched_author_normalized"
        / "paired_comparison.csv"
    )


def test_frozen_input_grid_validates() -> None:
    module = _module()
    module.validate_inputs(_rows(), _summary(), _config())


def test_rejects_missing_pair() -> None:
    module = _module()
    with pytest.raises(ValueError, match="expected 32"):
        module.validate_inputs(_rows()[:-1], _summary(), _config())


def test_analysis_classifies_all_four_points_without_numeric_gate() -> None:
    module = _module()
    result = module.analyze(_rows(), _summary(), _config())
    assert result["input_pair_count"] == 32
    assert result["exact_normalized_decode_count"] == 32
    assert result["ready_for_step_3_14_finalization"] is True
    assert result["cross_method_findings"]["core_principle_preserved_for_all_representative_points"] is True
    assert len(result["point_results"]) == 4
    assert _config()["classification_policy"]["numeric_thresholds_are_gates"] is False


def test_bins_capacity_is_exactly_preserved_despite_sequence_divergence() -> None:
    module = _module()
    result = module.analyze(_rows(), _summary(), _config())
    bins = next(x for x in result["point_results"] if x["point_id"] == "bins_b3")
    assert bins["statistics"]["capacity"]["mean_delta_normalized_minus_author"] == 0.0
    assert bins["statistics"]["capacity"]["max_absolute_pair_delta"] == 0.0
    assert bins["statistics"]["token_sequence"]["exact_pair_count"] == 0
    assert bins["classification"]["core_principle_preserved"] is True


def test_huffman_exact_subgroup_is_six_pairs_with_identical_capacity() -> None:
    module = _module()
    result = module.analyze(_rows(), _summary(), _config())
    huffman = next(x for x in result["point_results"] if x["point_id"] == "huffman_e3")
    exact = huffman["statistics"]["exact_sequence_subgroup"]
    assert exact["pair_count"] == 6
    assert exact["max_absolute_capacity_delta"] == 0.0
    assert exact["max_absolute_reverse_kl_delta_bits"] < 0.002


def test_arithmetic_tau09_records_reference_semantics_limitation() -> None:
    module = _module()
    result = module.analyze(_rows(), _summary(), _config())
    item = next(x for x in result["point_results"] if x["point_id"] == "arithmetic_t0.9_k300")
    text = " ".join(item["classification"]["attribution"])
    assert "untempered" in text
    assert item["statistics"]["reverse_kl"]["reference_semantics_exactly_matched"] is False


def test_benchmark_native_kl_is_support_diagnostic_not_finite_ranking_scalar() -> None:
    module = _module()
    result = module.analyze(_rows(), _summary(), _config())
    cross = result["cross_method_findings"]
    assert cross["benchmark_native_kl_infinite_pairs"] == 32
    assert cross["benchmark_native_kl_pair_count"] == 32
    assert "+inf" in cross["benchmark_native_kl_interpretation"]
    assert "without smoothing" in cross["recommended_v1_metric_reading"]
