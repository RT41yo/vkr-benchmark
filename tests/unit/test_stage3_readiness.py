from __future__ import annotations

from pathlib import Path

from vkr_benchmark.validation.stage3 import (
    check_comparison_csv,
    check_conformance,
    check_figure3_summary,
    check_interpretation,
    check_matched_summary,
    check_required_paths,
    evaluate_stage3_readiness,
    format_report,
)
import json

REPO_ROOT = Path(__file__).resolve().parents[2]


def _json(path: str) -> dict:
    return json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))


def test_frozen_figure3_summary_passes_closeout_checks() -> None:
    checks = check_figure3_summary(_json("results/stage3/paper_reproduction/figure3_full/summary.json"))
    assert all(check.ok for check in checks)


def test_frozen_interpretation_has_expected_claim_statuses() -> None:
    checks = check_interpretation(_json("results/stage3/paper_reproduction/figure3_full/interpretation.json"))
    assert all(check.ok for check in checks)


def test_matched_summary_preserves_pairing_and_provenance() -> None:
    checks = check_matched_summary(_json("results/stage3/matched_author_normalized/summary.json"))
    assert all(check.ok for check in checks)


def test_conformance_summary_preserves_all_four_core_mechanisms() -> None:
    checks = check_conformance(_json("results/stage3/matched_author_normalized/conformance_analysis.json"))
    assert all(check.ok for check in checks)


def test_required_paths_pass_without_parquet_in_development_mode() -> None:
    check = check_required_paths(REPO_ROOT, require_parquet=False)
    assert check.ok


def test_comparison_csv_has_four_rows() -> None:
    check = check_comparison_csv(REPO_ROOT / "results/stage3/comparison.csv")
    assert check.ok


def test_full_readiness_can_be_evaluated_without_parquet_for_unit_environment() -> None:
    report = evaluate_stage3_readiness(REPO_ROOT, include_tests=False, require_parquet=False)
    assert report.ready
    text = format_report(report)
    assert "Stage 3 reproducibility readiness: READY" in text
