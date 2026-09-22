from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = REPO_ROOT / "configs" / "reproducibility" / "paper_reproduction_matrix.json"
CHECKER_PATH = REPO_ROOT / "scripts" / "check_stage3_paper_matrix.py"
EXPECTED_COMMIT = "14e982564aeaf9a33f7b4de440deda2184d17f12"


def _matrix() -> dict[str, object]:
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


def _checker_module():
    spec = importlib.util.spec_from_file_location("stage3_matrix_checker", CHECKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paper_matrix_is_frozen_before_runs() -> None:
    matrix = _matrix()
    assert matrix["schema_version"] == "stage3.paper_reproduction_matrix.v1"
    assert matrix["status"] == "frozen_before_paper_runs"
    assert matrix["reference"]["commit"] == EXPECTED_COMMIT
    assert matrix["common_paper_setup"]["model_id"] == "gpt2-medium"
    assert matrix["go_no_go"]["matrix_frozen"] is True


def test_paper_matrix_matches_figure3_method_ranges() -> None:
    matrix = _matrix()
    sweeps = {item["method"]: item for item in matrix["information_theoretic_sweeps"]}
    assert sweeps["bins"]["values"] == [1, 2, 3, 4, 5]
    assert sweeps["huffman"]["values"] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert sweeps["arithmetic"]["values"] == [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    assert sweeps["arithmetic"]["fixed"] == {"topk": 300, "precision": 26}


def test_paper_matrix_preserves_unmodulated_arithmetic_anchor_and_unit_ambiguity() -> None:
    point = _matrix()["special_points"][0]
    assert point["id"] == "arithmetic_unmodulated"
    assert point["temperature"] == 1.0
    assert point["topk"] == 50256
    assert point["precision"] == 26
    assert point["paper_reported_kl_value"] == 4e-8
    assert point["paper_reported_kl_unit"] == "nats"
    assert point["figure_axis_unit"] == "bits"
    assert point["reference_helper_unit"] == "bits"


def test_paper_matrix_has_exactly_23_information_theoretic_points() -> None:
    matrix = _matrix()
    primary = sum(len(item["values"]) for item in matrix["information_theoretic_sweeps"])
    total = primary + len(matrix["special_points"])
    assert primary == 22
    assert total == 23


def test_paper_matrix_freezes_pilot_and_full_aggregation_plan() -> None:
    plan = _matrix()["execution_plan"]
    assert plan["phase_1_gpt2_medium_pilot"]["contexts"] == 8
    assert plan["phase_1_gpt2_medium_pilot"]["replicates"] == 1
    assert plan["phase_2_information_theoretic_curve"]["contexts_per_point"] == 80
    assert plan["phase_2_information_theoretic_curve"]["replicates"] == 3


def test_paper_matrix_does_not_pretend_dataset_revision_is_known() -> None:
    matrix = _matrix()
    open_items = {item["id"]: item for item in matrix["open_source_details"]}
    assert open_items["cnndm_artifact_revision"]["status"] == "must_pin_before_phase_1"
    assert matrix["go_no_go"]["ready_for_gpt2_medium_pilot"] is False


def test_matrix_checker_accepts_frozen_matrix() -> None:
    checker = _checker_module()
    summary = checker.validate_matrix(_matrix())
    assert summary["model"] == "gpt2-medium"
    assert summary["total_points"] == 23
    assert summary["ready_for_gpt2_medium_pilot"] is False
