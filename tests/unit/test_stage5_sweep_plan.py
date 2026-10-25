from __future__ import annotations

import json
from pathlib import Path

import pytest

from vkr_benchmark.errors import ConfigurationError
from vkr_benchmark.sweeps import SweepManifest


REPO_ROOT = Path(__file__).resolve().parents[2]
TOP_P_MANIFEST = REPO_ROOT / "configs/pilots/stage5/top_p_calibration_llama.json"
PILOT_A_MANIFEST = REPO_ROOT / "configs/pilots/stage5/pilot_a_streaming_llama.json"
PILOT_B_MANIFEST = REPO_ROOT / "configs/pilots/stage5/pilot_b_streaming_llama.json"
PILOT_A_SELECTION = REPO_ROOT / "results/stage5/pilot_a_operating_point_selection.json"
RRC_MARGIN_MANIFEST = (
    REPO_ROOT / "configs/pilots/stage5/rrc_precision_margin_calibration_llama.json"
)


def test_top_p_calibration_manifest_expands_to_42_paired_runs() -> None:
    sweep = SweepManifest.from_json(TOP_P_MANIFEST, project_root=REPO_ROOT)
    runs = sweep.expand()

    assert sweep.planned_run_count() == 42
    assert len(runs) == 42
    assert len({run.run_id for run in runs}) == 42

    combos = {
        (run.generation_point_id, run.config.method.method_id)
        for run in runs
    }
    assert len(combos) == 3 * 7
    assert {run.generation_point_id for run in runs} == {"tp095", "tp098", "tp100"}
    assert {run.config.method.method_id for run in runs} == {
        "bins",
        "huffman",
        "arithmetic_coding",
        "adg",
        "discop",
        "rrc",
        "dairstega",
    }


def test_generation_condition_does_not_change_method_random_material() -> None:
    sweep = SweepManifest.from_json(TOP_P_MANIFEST, project_root=REPO_ROOT)
    runs = sweep.expand()

    bins = [
        run
        for run in runs
        if run.config.method.method_id == "bins"
        and run.config.prompt_id == "s5p0001"
    ]
    assert len(bins) == 3
    assert len({run.config.method.random_seed for run in bins}) == 1
    assert bins[0].config.method.random_seed is not None
    assert all(run.config.method.key is None for run in bins)

    discop = [
        run
        for run in runs
        if run.config.method.method_id == "discop"
        and run.config.prompt_id == "s5p0001"
    ]
    assert len(discop) == 3
    assert len({run.config.method.key for run in discop}) == 1
    assert discop[0].config.method.key is not None
    assert all(run.config.method.random_seed is None for run in discop)


def test_deterministic_methods_get_no_synthetic_rng_or_key() -> None:
    sweep = SweepManifest.from_json(TOP_P_MANIFEST, project_root=REPO_ROOT)
    runs = sweep.expand()

    deterministic = {"huffman", "arithmetic_coding", "dairstega"}
    for run in runs:
        if run.config.method.method_id in deterministic:
            assert run.config.method.random_seed is None
            assert run.config.method.key is None


def test_manifest_rejects_artificial_adg_parameter(tmp_path: Path) -> None:
    raw = json.loads(TOP_P_MANIFEST.read_text(encoding="utf-8"))
    adg = next(item for item in raw["methods"] if item["id"] == "adg")
    adg["points"][0]["params"] = {"fake_tradeoff": 1}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="unknown ADG configuration fields"):
        SweepManifest.from_json(path, project_root=REPO_ROOT)


def test_pilot_a_streaming_manifest_expands_to_360_runs() -> None:
    sweep = SweepManifest.from_json(PILOT_A_MANIFEST, project_root=REPO_ROOT)
    runs = sweep.expand()

    assert sweep.planned_run_count() == 360
    assert len(runs) == 360
    assert len({run.run_id for run in runs}) == 360
    assert {run.generation_point_id for run in runs} == {"main"}
    assert {run.config.prompt_id for run in runs} == {
        "s5p0001",
        "s5p0002",
        "s5p0003",
        "s5p0004",
    }
    assert {run.replicate_id for run in runs} == {"r01", "r02"}
    assert {run.config.method.method_id for run in runs} == {
        "bins",
        "huffman",
        "arithmetic_coding",
        "adg",
        "discop",
        "dairstega",
    }
    assert all(run.config.generation.temperature == 1.0 for run in runs)
    assert all(run.config.generation.top_k is None for run in runs)
    assert all(run.config.generation.top_p == 1.0 for run in runs)
    assert all(run.config.termination.mode == "fixed_carrier_tokens" for run in runs)
    assert all(run.config.termination.target_carrier_tokens == 128 for run in runs)


def test_pilot_a_method_point_grid_is_frozen() -> None:
    sweep = SweepManifest.from_json(PILOT_A_MANIFEST, project_root=REPO_ROOT)
    by_method = {method.method_id: method for method in sweep.methods}

    assert [point.point_id for point in by_method["bins"].points] == [
        f"b{i}" for i in range(1, 9)
    ]
    assert [point.point_id for point in by_method["huffman"].points] == [
        f"e{i}" for i in range(1, 9)
    ]
    assert len(by_method["arithmetic_coding"].points) == 15
    assert len(by_method["adg"].points) == 1
    assert len(by_method["discop"].points) == 1
    assert len(by_method["dairstega"].points) == 12


def test_pilot_a_pairs_secret_and_method_randomness_across_points() -> None:
    sweep = SweepManifest.from_json(PILOT_A_MANIFEST, project_root=REPO_ROOT)
    runs = sweep.expand()

    r01 = [run for run in runs if run.replicate_id == "r01"]
    assert {run.config.secret_id for run in r01} == {"stage5-pilot-a-0001"}

    bins_r01 = [
        run
        for run in r01
        if run.config.method.method_id == "bins"
        and run.config.prompt_id == "s5p0001"
    ]
    assert len(bins_r01) == 8
    assert len({run.config.method.random_seed for run in bins_r01}) == 1

    discop_r01 = [
        run
        for run in r01
        if run.config.method.method_id == "discop"
        and run.config.prompt_id == "s5p0001"
    ]
    assert len(discop_r01) == 1
    assert discop_r01[0].config.method.key is not None


def test_rrc_margin_calibration_expands_to_24_fixed_payload_runs() -> None:
    sweep = SweepManifest.from_json(RRC_MARGIN_MANIFEST, project_root=REPO_ROOT)
    runs = sweep.expand()

    assert sweep.planned_run_count() == 24
    assert len(runs) == 24
    assert len({run.run_id for run in runs}) == 24
    assert {run.config.method.method_id for run in runs} == {"rrc"}
    assert {run.method_point_id for run in runs} == {
        "margin20",
        "margin40",
        "margin60",
    }
    assert all(run.config.generation.top_p == 1.0 for run in runs)
    assert all(run.config.termination.mode == "fixed_payload_bits" for run in runs)
    assert all(run.config.termination.target_payload_bits == 1024 for run in runs)
    assert all(run.config.termination.max_carrier_tokens == 2048 for run in runs)

    paired = [
        run
        for run in runs
        if run.config.prompt_id == "s5p0001" and run.replicate_id == "r01"
    ]
    assert len(paired) == 3
    assert len({run.config.method.key for run in paired}) == 1
    assert paired[0].config.method.key is not None


def test_pilot_b_streaming_manifest_expands_to_504_runs() -> None:
    sweep = SweepManifest.from_json(PILOT_B_MANIFEST, project_root=REPO_ROOT)
    runs = sweep.expand()

    assert sweep.planned_run_count() == 504
    assert len(runs) == 504
    assert len({run.run_id for run in runs}) == 504
    assert {run.generation_point_id for run in runs} == {"main"}
    assert {run.config.prompt_id for run in runs} == {
        f"s5p{i:04d}" for i in range(1, 9)
    }
    assert {run.replicate_id for run in runs} == {"r01", "r02", "r03"}
    assert {run.config.secret_id for run in runs} == {
        "stage5-pilot-b-0001",
        "stage5-pilot-b-0002",
        "stage5-pilot-b-0003",
    }
    assert all(run.config.generation.temperature == 1.0 for run in runs)
    assert all(run.config.generation.top_k is None for run in runs)
    assert all(run.config.generation.top_p == 1.0 for run in runs)
    assert all(run.config.termination.mode == "fixed_carrier_tokens" for run in runs)
    assert all(run.config.termination.target_carrier_tokens == 256 for run in runs)


def test_pilot_b_retained_method_point_grid_is_frozen() -> None:
    sweep = SweepManifest.from_json(PILOT_B_MANIFEST, project_root=REPO_ROOT)
    by_method = {method.method_id: method for method in sweep.methods}

    assert [point.point_id for point in by_method["bins"].points] == [
        "b1", "b2", "b4", "b6"
    ]
    assert [point.point_id for point in by_method["huffman"].points] == [
        "e2", "e4", "e6", "e8"
    ]
    assert [point.point_id for point in by_method["arithmetic_coding"].points] == [
        "p16-k50000",
        "p32-k128",
        "p32-k2048",
        "p32-k8192",
        "p32-k50000",
        "p48-k50000",
    ]
    assert [point.point_id for point in by_method["adg"].points] == ["default"]
    assert [point.point_id for point in by_method["discop"].points] == ["default"]
    assert [point.point_id for point in by_method["dairstega"].points] == [
        "a8-b1-k100",
        "a16-b1-k100",
        "a16-b075-k100",
        "a32-b075-k100",
        "a32-b05-k100",
    ]


def test_pilot_a_selection_matches_pilot_b_manifest() -> None:
    selection = json.loads(PILOT_A_SELECTION.read_text(encoding="utf-8"))
    sweep = SweepManifest.from_json(PILOT_B_MANIFEST, project_root=REPO_ROOT)

    retained = {
        (item["method_id"], item["point_id"])
        for item in selection["points"]
        if item["decision"] == "retained"
    }
    rejected = {
        (item["method_id"], item["point_id"])
        for item in selection["points"]
        if item["decision"] == "rejected"
    }
    manifest_points = {
        (method.method_id, point.point_id)
        for method in sweep.methods
        for point in method.points
    }

    assert selection["pilot_a_method_point_count"] == 45
    assert selection["retained_point_count"] == 21
    assert selection["rejected_point_count"] == 24
    assert len(retained) == 21
    assert len(rejected) == 24
    assert retained.isdisjoint(rejected)
    assert retained == manifest_points

