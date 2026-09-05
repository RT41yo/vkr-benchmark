from __future__ import annotations

from pathlib import Path

from vkr_benchmark.validation.stage2 import (
    REQUIRED_REPOSITORY_PATHS,
    Stage2ReadinessReport,
    CheckResult,
    check_required_paths,
    check_summary_records,
    evaluate_stage2_readiness,
    format_report,
)


def _row(method: str, run_id: str) -> dict[str, object]:
    return {
        "run_id": run_id,
        "method_id": method,
        "method_params_hash": f"hash-{method}",
        "model_id": "meta-llama/Llama-3.2-3B",
        "prompt_id": "p000001",
        "secret_id": "000001",
        "bits_per_token": 2.0,
        "entropy_utilization": 0.6,
        "q_mode": "analytic_exact",
        "kl_mean_bits": float("inf"),
        "tvd_mean": 0.2,
        "nll_raw_lm_nats_per_token": 2.0,
        "ppl_raw_lm": 7.4,
        "ber": 0.0,
        "roundtrip_exact": True,
        "token_sequence_roundtrip_exact": True,
        "encode_ms_per_token": 20.0,
        "decode_ms_per_token": 21.0,
        "status": "ok",
    }


def _valid_rows() -> list[dict[str, object]]:
    return [
        _row("bins", "run-bins"),
        _row("huffman", "run-huffman"),
        _row("arithmetic_coding", "run-ac"),
    ]


def test_summary_records_accept_valid_stage2_smoke() -> None:
    checks = check_summary_records(_valid_rows())
    assert checks
    assert all(check.ok for check in checks)


def test_summary_requires_all_three_baselines() -> None:
    checks = check_summary_records(_valid_rows()[:2])
    check = next(item for item in checks if item.name == "baseline_methods")
    assert not check.ok
    assert "arithmetic_coding" in check.detail


def test_summary_reliability_gate_rejects_nonzero_ber() -> None:
    rows = _valid_rows()
    rows[0]["ber"] = 0.01
    checks = check_summary_records(rows)
    check = next(item for item in checks if item.name == "reliability_smoke")
    assert not check.ok
    assert "bins" in check.detail


def test_summary_metric_sanity_allows_infinite_benchmark_kl() -> None:
    checks = check_summary_records(_valid_rows())
    check = next(item for item in checks if item.name == "metric_sanity")
    assert check.ok


def test_summary_metric_sanity_rejects_invalid_tvd() -> None:
    rows = _valid_rows()
    rows[1]["tvd_mean"] = 1.2
    checks = check_summary_records(rows)
    check = next(item for item in checks if item.name == "metric_sanity")
    assert not check.ok


def test_required_paths_gate(tmp_path: Path) -> None:
    failed = check_required_paths(tmp_path)
    assert not failed.ok
    for relative in REQUIRED_REPOSITORY_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder\n", encoding="utf-8")
    passed = check_required_paths(tmp_path)
    assert passed.ok


def test_evaluate_can_use_injected_records_without_parquet(tmp_path: Path) -> None:
    for relative in REQUIRED_REPOSITORY_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder\n", encoding="utf-8")
    report = evaluate_stage2_readiness(
        tmp_path,
        summary_records=_valid_rows(),
        include_git_check=False,
        include_tests=False,
    )
    assert report.ready


def test_report_format_and_json() -> None:
    report = Stage2ReadinessReport(
        (
            CheckResult("a", True, "ok"),
            CheckResult("b", False, "bad"),
        )
    )
    text = format_report(report)
    assert "[PASS] a: ok" in text
    assert "[FAIL] b: bad" in text
    assert "NOT READY" in text
    payload = report.to_dict()
    assert payload["stage"] == 2
    assert payload["ready"] is False
