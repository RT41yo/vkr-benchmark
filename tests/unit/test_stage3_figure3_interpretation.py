from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "stage3_figure3_interpretation.py"


def _module():
    spec = importlib.util.spec_from_file_location("stage3_figure3_interpretation", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _point(method: str, x: float, y: float, **parameters):
    return {
        "method": method,
        "mean_bits_per_word": x,
        "mean_kl_bits": y,
        "se_bits_per_word_runs": 0.01,
        "se_kl_bits_runs": 0.01,
        "parameters": parameters,
    }


def test_linear_interpolation_and_curve_dominance() -> None:
    module = _module()
    arithmetic = [_point("arithmetic", 1.0, 0.5), _point("arithmetic", 2.0, 0.2)]
    huffman = [_point("huffman", 1.0, 1.5), _point("huffman", 2.0, 0.8)]
    result = module.compare_mean_curves(arithmetic, huffman, grid_points=11)
    assert result["arithmetic_lower_at_every_grid_point"] is True
    assert result["common_bits_per_word_interval"] == [1.0, 2.0]
    assert math.isclose(result["minimum_baseline_minus_arithmetic_kl_bits"], 0.6)


def test_huffman_nonincreasing_helper() -> None:
    module = _module()
    assert module.nonincreasing([2.0, 1.5, 1.5, 0.5]) is True
    assert module.nonincreasing([2.0, 2.1, 1.0]) is False


def test_committed_figure3_inputs_build_expected_interpretation() -> None:
    module = _module()
    config = json.loads(
        (REPO_ROOT / "configs" / "reproducibility" / "figure3_interpretation.json").read_text(encoding="utf-8")
    )
    points = module.load_points_csv(REPO_ROOT / config["inputs"]["figure3_points_csv"])
    summary = json.loads((REPO_ROOT / config["inputs"]["figure3_summary_json"]).read_text(encoding="utf-8"))
    precision = json.loads(
        (REPO_ROOT / config["inputs"]["precision_interpretation_json"]).read_text(encoding="utf-8")
    )
    interpretation = module.build_interpretation(points, summary, precision, config)

    assert interpretation["execution_input"]["scheduled_runs"] == 5520
    assert interpretation["execution_input"]["termination_failures"] == 7
    assert interpretation["curves"]["arithmetic_k300"]["minimum"]["temperature"] == 1.0
    assert math.isclose(
        interpretation["curves"]["arithmetic_k300"]["minimum"]["bits_per_word"],
        3.7520964594587127,
    )
    assert interpretation["curve_dominance_diagnostic"]["arithmetic_vs_huffman"]["arithmetic_lower_at_every_grid_point"] is True
    assert interpretation["curve_dominance_diagnostic"]["arithmetic_vs_bins"]["arithmetic_lower_at_every_grid_point"] is True
    assert interpretation["curves"]["arithmetic_k50256"]["ratio_to_paper_anchor"] > 10000
    assert interpretation["precision_discrepancy"]["precision40_to_paper_anchor_ratio"] < 1.0
    assert interpretation["overall_assessment"]["status"] == "partial_reproduction"
    assert interpretation["overall_assessment"]["ready_for_step_3_12_matched_author_vs_normalized"] is True


def test_transport_method_rates_sum_to_global_applicable_count() -> None:
    module = _module()
    config = json.loads(
        (REPO_ROOT / "configs" / "reproducibility" / "figure3_interpretation.json").read_text(encoding="utf-8")
    )
    summary = json.loads((REPO_ROOT / config["inputs"]["figure3_summary_json"]).read_text(encoding="utf-8"))
    by_method = module.summarize_transport_by_method(summary)
    assert sum(item["applicable"] for item in by_method.values()) == 5490
    assert sum(item["exact"] for item in by_method.values()) == 5343
    assert sum(item["failures"] for item in by_method.values()) == 147
    assert by_method["arithmetic"]["exact_prefix_rate"] > by_method["huffman"]["exact_prefix_rate"]
    assert by_method["huffman"]["exact_prefix_rate"] > by_method["bins"]["exact_prefix_rate"]
