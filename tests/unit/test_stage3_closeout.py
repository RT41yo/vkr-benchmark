from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "stage3_closeout.py"


def _module():
    spec = importlib.util.spec_from_file_location("stage3_closeout", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _json(path: str) -> dict:
    return json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))


def test_comparison_rows_cover_four_representative_points() -> None:
    module = _module()
    conformance = _json("results/stage3/matched_author_normalized/conformance_analysis.json")
    rows = module.build_comparison_rows(conformance)
    assert len(rows) == 4
    assert {row["point_id"] for row in rows} == {
        "bins_b3",
        "huffman_e3",
        "arithmetic_t0.9_k300",
        "arithmetic_t1.0_k50256",
    }
    assert all(row["core_principle_preserved"] is True for row in rows)


def test_closeout_summary_freezes_stage3_conclusions() -> None:
    module = _module()
    closeout = module.build_closeout_summary(
        _json("results/stage3/paper_reproduction/figure3_full/summary.json"),
        _json("results/stage3/paper_reproduction/figure3_full/interpretation.json"),
        _json("results/stage3/matched_author_normalized/summary.json"),
        _json("results/stage3/matched_author_normalized/conformance_analysis.json"),
    )
    assert closeout.figure3_scheduled_runs == 5520
    assert closeout.figure3_terminated_runs == 5513
    assert closeout.figure3_termination_failures == 7
    assert closeout.figure3_overall_assessment == "partial_reproduction"
    assert closeout.exact_unmodulated_anchor_reproduced is False
    assert closeout.matched_pair_count == 32
    assert closeout.core_principle_preserved_count == 4
    assert closeout.benchmark_native_kl_infinite_pair_count == 32
    assert closeout.ready_for_stage4 is True


def test_write_comparison_csv_roundtrip(tmp_path: Path) -> None:
    module = _module()
    rows = module.build_comparison_rows(
        _json("results/stage3/matched_author_normalized/conformance_analysis.json")
    )
    output = module.write_comparison_csv(tmp_path / "comparison.csv", rows)
    text = output.read_text(encoding="utf-8")
    assert "point_id,classification" in text
    assert "huffman_e3" in text
