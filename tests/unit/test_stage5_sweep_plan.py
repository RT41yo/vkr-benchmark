from __future__ import annotations

import json
from pathlib import Path

import pytest

from vkr_benchmark.errors import ConfigurationError
from vkr_benchmark.sweeps import SweepManifest


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "configs/pilots/stage5/top_p_calibration_llama.json"


def test_top_p_calibration_manifest_expands_to_42_paired_runs() -> None:
    sweep = SweepManifest.from_json(MANIFEST, project_root=REPO_ROOT)
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
    sweep = SweepManifest.from_json(MANIFEST, project_root=REPO_ROOT)
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
    sweep = SweepManifest.from_json(MANIFEST, project_root=REPO_ROOT)
    runs = sweep.expand()

    deterministic = {"huffman", "arithmetic_coding", "dairstega"}
    for run in runs:
        if run.config.method.method_id in deterministic:
            assert run.config.method.random_seed is None
            assert run.config.method.key is None


def test_manifest_rejects_artificial_adg_parameter(tmp_path: Path) -> None:
    raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    adg = next(item for item in raw["methods"] if item["id"] == "adg")
    adg["points"][0]["params"] = {"fake_tradeoff": 1}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="unknown ADG configuration fields"):
        SweepManifest.from_json(path, project_root=REPO_ROOT)
