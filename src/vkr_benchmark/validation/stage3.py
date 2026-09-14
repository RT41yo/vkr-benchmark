from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
import subprocess
from typing import Any, Mapping, Sequence


EXPECTED_REFERENCE_COMMIT = "14e982564aeaf9a33f7b4de440deda2184d17f12"
EXPECTED_MODEL_REVISION = "6dcaa7a952f72f9298047fd5137cd6e4f05f41da"
EXPECTED_CONTEXT_MANIFEST_SHA256 = "cc25b492b26e102496137bc2f8eb78320c7f7768d83ff7e10af9b2e04a14aebc"

REQUIRED_REPOSITORY_PATHS = (
    "docs/stages/stage3/reproducibility_protocol.md",
    "docs/stages/stage3/figure3_full_run.md",
    "docs/stages/stage3/figure3_interpretation.md",
    "docs/stages/stage3/matched_author_normalized.md",
    "docs/stages/stage3/conformance_analysis.md",
    "docs/stages/stage3/reproducibility_report.md",
    "docs/stages/stage3/stage_3.typ",
    "docs/decisions/0012-dual-kl-for-reproducibility.md",
    "docs/decisions/0016-stage3-reproducibility-closeout.md",
    "configs/reproducibility/figure3_full_run.json",
    "configs/reproducibility/figure3_interpretation.json",
    "configs/reproducibility/matched_author_normalized.json",
    "configs/reproducibility/conformance_analysis.json",
    "results/stage3/paper_reproduction/figure3_full/summary.json",
    "results/stage3/paper_reproduction/figure3_full/interpretation.json",
    "results/stage3/paper_reproduction/figure3_full/figure3_reproduction.svg",
    "results/stage3/matched_author_normalized/summary.json",
    "results/stage3/matched_author_normalized/conformance_analysis.json",
    "results/stage3/comparison.csv",
    "results/stage3/stage3_closeout_summary.json",
    "scripts/finalize_stage3.py",
    "scripts/check_stage3_readiness.py",
)


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class Stage3ReadinessReport:
    checks: tuple[CheckResult, ...]

    @property
    def ready(self) -> bool:
        return all(check.ok for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": 3,
            "ready": self.ready,
            "checks": [asdict(check) for check in self.checks],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_required_paths(repo_root: Path, *, require_parquet: bool = True) -> CheckResult:
    required = list(REQUIRED_REPOSITORY_PATHS)
    if require_parquet:
        required.append("results/stage3/comparison.parquet")
    missing = [path for path in required if not (repo_root / path).is_file()]
    return CheckResult(
        "repository_structure",
        not missing,
        f"all {len(required)} required Stage-3 paths exist"
        if not missing
        else "missing: " + ", ".join(missing),
    )


def check_figure3_summary(summary: Mapping[str, Any]) -> tuple[CheckResult, ...]:
    run_ok = (
        summary.get("expected_run_count") == 5520
        and summary.get("run_count") == 5520
        and summary.get("scheduled_run_count") == 5520
        and summary.get("point_count") == 23
        and summary.get("expected_point_count") == 23
    )
    execution_ok = (
        summary.get("terminated_sentence_run_count") == 5513
        and summary.get("termination_failure_count") == 7
        and summary.get("pilot_continuity", {}).get("passed") is True
        and summary.get("reference_worktree_unchanged") is True
        and summary.get("execution_gate_ready_for_step_3_11_interpretation") is True
    )
    return (
        CheckResult(
            "figure3_execution_matrix",
            run_ok,
            "23 points and 5520/5520 scheduled outcomes are present" if run_ok else "Figure-3 matrix/count mismatch",
        ),
        CheckResult(
            "figure3_execution_integrity",
            execution_ok,
            "5513 sentence completions, 7 declared termination failures, pilot continuity and reference integrity preserved"
            if execution_ok
            else "Figure-3 execution integrity mismatch",
        ),
    )


def check_interpretation(interpretation: Mapping[str, Any]) -> tuple[CheckResult, ...]:
    claims = {str(item.get("claim_id")): item for item in interpretation.get("claims", [])}
    expected_statuses = {
        "bins_curve": "trend_reproduction",
        "huffman_curve": "trend_reproduction",
        "arithmetic_temperature_curve": "trend_reproduction",
        "arithmetic_dominates_baselines": "trend_reproduction",
        "unmodulated_near_zero_behavior": "partial_reproduction",
        "unmodulated_4e_minus_8_nats": "not_reproducible",
        "paper_orchestration": "partial_reproduction",
        "historical_mc_estimator": "partial_reproduction",
    }
    claims_ok = len(claims) == 8 and all(
        claims.get(key, {}).get("status") == status for key, status in expected_statuses.items()
    )
    overall_ok = interpretation.get("overall_assessment", {}).get("status") == "partial_reproduction"
    curves = interpretation.get("curves", {})
    curve_ok = (
        curves.get("bins", {}).get("high_kl_tradeoff_reproduced") is True
        and curves.get("huffman", {}).get("kl_nonincreasing") is True
        and curves.get("arithmetic_k300", {}).get("decreases_to_tau_1_then_increases") is True
        and curves.get("arithmetic_k300", {}).get("minimum", {}).get("temperature") == 1.0
    )
    return (
        CheckResult("paper_claim_classification", claims_ok, "all 8 Figure-3 claims have the frozen Step-3.11 classifications" if claims_ok else "paper claim classification mismatch"),
        CheckResult("paper_overall_assessment", overall_ok, "overall paper reproduction status is partial_reproduction" if overall_ok else "unexpected overall paper assessment"),
        CheckResult("paper_curve_behavior", curve_ok, "Bins/Huffman/Arithmetic characteristic curve behavior reproduced" if curve_ok else "curve-level reproduction invariant failed"),
    )


def check_matched_summary(summary: Mapping[str, Any]) -> tuple[CheckResult, ...]:
    pairing_ok = (
        summary.get("pair_count") == 32
        and summary.get("expected_pair_count") == 32
        and summary.get("all_carrier_lengths_matched") is True
        and summary.get("same_secret_stream_verified") is True
        and summary.get("normalized_exact_decode_count") == 32
        and summary.get("all_normalized_token_id_decodes_exact") is True
    )
    provenance_ok = (
        summary.get("reference_commit") == EXPECTED_REFERENCE_COMMIT
        and summary.get("model_revision") == EXPECTED_MODEL_REVISION
        and summary.get("context_manifest_sha256") == EXPECTED_CONTEXT_MANIFEST_SHA256
    )
    dual = summary.get("dual_kl", {})
    kl_ok = dual.get("all_forward_kl_accounted") is True and dual.get("all_reverse_kl_present") is True
    return (
        CheckResult("matched_pair_integrity", pairing_ok, "32/32 pairs have matched carriers, secret streams and exact normalized token-ID decode" if pairing_ok else "matched-pair integrity mismatch"),
        CheckResult("matched_provenance", provenance_ok, "Harvard commit, GPT-2 Medium revision and context manifest are pinned" if provenance_ok else "matched provenance mismatch"),
        CheckResult("dual_kl_contract", kl_ok, "both KL directions are explicitly accounted for in all matched normalized runs" if kl_ok else "dual-KL accounting mismatch"),
    )


def check_conformance(conformance: Mapping[str, Any]) -> tuple[CheckResult, ...]:
    points = list(conformance.get("point_results", []))
    preserved = sum(bool(item.get("classification", {}).get("core_principle_preserved")) for item in points)
    cross = conformance.get("cross_method_findings", {})
    core_ok = len(points) == 4 and preserved == 4 and cross.get("core_principle_preserved_for_all_representative_points") is True
    support_ok = (
        cross.get("benchmark_native_kl_infinite_pairs") == 32
        and cross.get("benchmark_native_kl_pair_count") == 32
        and "without smoothing" in str(cross.get("recommended_v1_metric_reading", ""))
    )
    overall_ok = (
        conformance.get("overall_classification")
        == "normalized_adaptations_preserve_core_method_behavior_with_documented_normalization_divergences"
        and conformance.get("ready_for_step_3_14_finalization") is True
    )
    return (
        CheckResult("core_method_conformance", core_ok, "core embedding/decoding principle preserved at 4/4 representative matched points" if core_ok else "core conformance mismatch"),
        CheckResult("support_mismatch_finding", support_ok, "benchmark-native forward KL is +inf in 32/32 pairs and remains an unsmoothed support diagnostic" if support_ok else "support-diagnostic conclusion mismatch"),
        CheckResult("conformance_overall", overall_ok, "overall conformance classification is frozen and ready for closeout" if overall_ok else "overall conformance mismatch"),
    )


def check_comparison_csv(path: Path) -> CheckResult:
    if not path.is_file():
        return CheckResult("comparison_csv", False, f"missing {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    point_ids = {row.get("point_id") for row in rows}
    expected = {"bins_b3", "huffman_e3", "arithmetic_t0.9_k300", "arithmetic_t1.0_k50256"}
    ok = len(rows) == 4 and point_ids == expected and all(row.get("core_principle_preserved") == "True" for row in rows)
    return CheckResult("comparison_csv", ok, "4 representative matched comparison rows are materialized" if ok else "comparison.csv content mismatch")


def check_comparison_parquet(path: Path) -> CheckResult:
    if not path.is_file():
        return CheckResult("comparison_parquet", False, f"missing {path}")
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return CheckResult("comparison_parquet", False, "pyarrow unavailable; cannot validate comparison.parquet")
    try:
        table = pq.read_table(path)
        rows = table.to_pylist()
    except Exception as exc:  # pragma: no cover - depends on pyarrow error types
        return CheckResult("comparison_parquet", False, f"cannot read comparison.parquet: {exc}")
    point_ids = {row.get("point_id") for row in rows}
    expected = {"bins_b3", "huffman_e3", "arithmetic_t0.9_k300", "arithmetic_t1.0_k50256"}
    ok = len(rows) == 4 and point_ids == expected
    return CheckResult("comparison_parquet", ok, "comparison.parquet is readable and contains the 4 representative points" if ok else "comparison.parquet content mismatch")


def check_closeout_summary(summary: Mapping[str, Any]) -> CheckResult:
    ok = (
        summary.get("stage") == 3
        and summary.get("figure3_scheduled_runs") == 5520
        and summary.get("figure3_terminated_runs") == 5513
        and summary.get("matched_pair_count") == 32
        and summary.get("core_principle_preserved_count") == 4
        and summary.get("benchmark_native_kl_infinite_pair_count") == 32
        and summary.get("ready_for_stage4") is True
    )
    return CheckResult("closeout_summary", ok, "machine-readable Stage-3 closeout summary is internally consistent" if ok else "Stage-3 closeout summary mismatch")


def _run_pytest(repo_root: Path) -> CheckResult:
    proc = subprocess.run(
        ["pytest", "-q"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "no pytest output"
    return CheckResult("pytest_suite", proc.returncode == 0, tail)


def evaluate_stage3_readiness(
    repo_root: Path,
    *,
    include_tests: bool = False,
    require_parquet: bool = True,
) -> Stage3ReadinessReport:
    checks: list[CheckResult] = [check_required_paths(repo_root, require_parquet=require_parquet)]
    figure3 = _load_json(repo_root / "results/stage3/paper_reproduction/figure3_full/summary.json")
    interpretation = _load_json(repo_root / "results/stage3/paper_reproduction/figure3_full/interpretation.json")
    matched = _load_json(repo_root / "results/stage3/matched_author_normalized/summary.json")
    conformance = _load_json(repo_root / "results/stage3/matched_author_normalized/conformance_analysis.json")
    closeout = _load_json(repo_root / "results/stage3/stage3_closeout_summary.json")

    checks.extend(check_figure3_summary(figure3))
    checks.extend(check_interpretation(interpretation))
    checks.extend(check_matched_summary(matched))
    checks.extend(check_conformance(conformance))
    checks.append(check_comparison_csv(repo_root / "results/stage3/comparison.csv"))
    if require_parquet:
        checks.append(check_comparison_parquet(repo_root / "results/stage3/comparison.parquet"))
    checks.append(check_closeout_summary(closeout))
    if include_tests:
        checks.append(_run_pytest(repo_root))
    return Stage3ReadinessReport(tuple(checks))


def format_report(report: Stage3ReadinessReport) -> str:
    lines: list[str] = []
    for check in report.checks:
        status = "PASS" if check.ok else "FAIL"
        lines.append(f"[{status}] {check.name}: {check.detail}")
    lines.append("")
    lines.append("Stage 3 reproducibility readiness: " + ("READY" if report.ready else "NOT READY"))
    return "\n".join(lines) + "\n"
