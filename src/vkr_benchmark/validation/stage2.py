from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import subprocess
from typing import Iterable, Mapping, Sequence


REQUIRED_METHODS = ("bins", "huffman", "arithmetic_coding")

REQUIRED_REPOSITORY_PATHS = (
    "src/vkr_benchmark/methods/bins.py",
    "src/vkr_benchmark/methods/huffman.py",
    "src/vkr_benchmark/methods/arithmetic.py",
    "src/vkr_benchmark/runner/streaming.py",
    "src/vkr_benchmark/runner/experiment.py",
    "src/vkr_benchmark/metrics/capacity_entropy.py",
    "src/vkr_benchmark/metrics/distribution_distortion.py",
    "src/vkr_benchmark/metrics/quality_reliability_performance.py",
    "src/vkr_benchmark/storage/run_store.py",
    "src/vkr_benchmark/reporting/stage2_summary.py",
    "configs/experiments/stage2_bins.example.json",
    "configs/experiments/stage2_huffman.example.json",
    "configs/experiments/stage2_arithmetic.example.json",
    "data/prompts.jsonl",
    "docs/metrics.md",
    "docs/stage2_validation.md",
    "docs/decisions/0009-unified-experiment-inputs-and-runner.md",
    "docs/decisions/0010-capacity-and-entropy-metrics.md",
    "docs/decisions/0011-kl-tvd-distribution-distortion.md",
    "docs/decisions/0012-dual-kl-for-reproducibility.md",
    "docs/decisions/0013-quality-reliability-and-performance-metrics.md",
    "docs/decisions/0014-canonical-run-storage-and-summary.md",
    "docs/decisions/0015-stage2-readiness-and-v0.2-milestone.md",
    "docs/stage2_closeout.md",
    "docs/stage2_presentation_outline.md",
    "docs/releases/v0.2.md",
)

REQUIRED_SUMMARY_COLUMNS = (
    "run_id",
    "method_id",
    "method_params_hash",
    "model_id",
    "prompt_id",
    "secret_id",
    "bits_per_token",
    "entropy_utilization",
    "q_mode",
    "kl_mean_bits",
    "tvd_mean",
    "nll_raw_lm_nats_per_token",
    "ppl_raw_lm",
    "ber",
    "roundtrip_exact",
    "token_sequence_roundtrip_exact",
    "encode_ms_per_token",
    "decode_ms_per_token",
    "status",
)


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class Stage2ReadinessReport:
    checks: tuple[CheckResult, ...]

    @property
    def ready(self) -> bool:
        return all(check.ok for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": 2,
            "ready": self.ready,
            "checks": [asdict(check) for check in self.checks],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def check_required_paths(repo_root: Path) -> CheckResult:
    missing = [path for path in REQUIRED_REPOSITORY_PATHS if not (repo_root / path).is_file()]
    if missing:
        return CheckResult(
            name="repository_structure",
            ok=False,
            detail="missing: " + ", ".join(missing),
        )
    return CheckResult(
        name="repository_structure",
        ok=True,
        detail=f"all {len(REQUIRED_REPOSITORY_PATHS)} required implementation/documentation paths exist",
    )


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def check_summary_records(records: Sequence[Mapping[str, object]]) -> tuple[CheckResult, ...]:
    checks: list[CheckResult] = []
    if not records:
        return (
            CheckResult("summary_present", False, "summary contains no rows"),
        )

    checks.append(CheckResult("summary_present", True, f"summary contains {len(records)} row(s)"))

    columns = set().union(*(record.keys() for record in records))
    missing_columns = [column for column in REQUIRED_SUMMARY_COLUMNS if column not in columns]
    checks.append(
        CheckResult(
            "summary_schema",
            not missing_columns,
            "required columns present" if not missing_columns else "missing columns: " + ", ".join(missing_columns),
        )
    )

    ok_rows = [record for record in records if record.get("status") == "ok"]
    methods = {str(record.get("method_id")) for record in ok_rows}
    missing_methods = [method for method in REQUIRED_METHODS if method not in methods]
    checks.append(
        CheckResult(
            "baseline_methods",
            not missing_methods,
            "all normalized baselines have an ok row"
            if not missing_methods
            else "missing ok row(s): " + ", ".join(missing_methods),
        )
    )

    baseline_rows = [record for record in ok_rows if str(record.get("method_id")) in REQUIRED_METHODS]
    run_ids = [str(record.get("run_id")) for record in baseline_rows]
    unique_ids = len(run_ids) == len(set(run_ids)) and all(run_id and run_id != "None" for run_id in run_ids)
    checks.append(
        CheckResult(
            "run_id_uniqueness",
            unique_ids,
            "baseline run_id values are present and unique" if unique_ids else "missing or duplicate baseline run_id",
        )
    )

    reliability_ok = bool(baseline_rows)
    reliability_failures: list[str] = []
    for record in baseline_rows:
        method = str(record.get("method_id"))
        ber = _as_float(record.get("ber"))
        if ber is None or ber != 0.0:
            reliability_ok = False
            reliability_failures.append(f"{method}: BER={record.get('ber')!r}")
        if record.get("roundtrip_exact") is not True:
            reliability_ok = False
            reliability_failures.append(f"{method}: roundtrip_exact={record.get('roundtrip_exact')!r}")
        if record.get("token_sequence_roundtrip_exact") is not True:
            reliability_ok = False
            reliability_failures.append(
                f"{method}: token_sequence_roundtrip_exact={record.get('token_sequence_roundtrip_exact')!r}"
            )
    checks.append(
        CheckResult(
            "reliability_smoke",
            reliability_ok,
            "all baseline smoke rows have BER=0 and exact text/token roundtrip"
            if reliability_ok
            else "; ".join(reliability_failures) or "no baseline rows",
        )
    )

    metric_ok = bool(baseline_rows)
    metric_failures: list[str] = []
    for record in baseline_rows:
        method = str(record.get("method_id"))
        bpt = _as_float(record.get("bits_per_token"))
        util = _as_float(record.get("entropy_utilization"))
        tvd = _as_float(record.get("tvd_mean"))
        nll = _as_float(record.get("nll_raw_lm_nats_per_token"))
        ppl = _as_float(record.get("ppl_raw_lm"))
        enc = _as_float(record.get("encode_ms_per_token"))
        dec = _as_float(record.get("decode_ms_per_token"))
        if bpt is None or not math.isfinite(bpt) or bpt <= 0:
            metric_ok = False
            metric_failures.append(f"{method}: invalid BPT")
        if util is None or not math.isfinite(util) or util < 0:
            metric_ok = False
            metric_failures.append(f"{method}: invalid entropy utilization")
        if tvd is None or not math.isfinite(tvd) or not 0.0 <= tvd <= 1.0:
            metric_ok = False
            metric_failures.append(f"{method}: invalid TVD")
        if nll is None or not math.isfinite(nll):
            metric_ok = False
            metric_failures.append(f"{method}: invalid NLL")
        if ppl is None or not math.isfinite(ppl) or ppl <= 0:
            metric_ok = False
            metric_failures.append(f"{method}: invalid PPL")
        if enc is None or not math.isfinite(enc) or enc <= 0:
            metric_ok = False
            metric_failures.append(f"{method}: invalid encode timing")
        if dec is None or not math.isfinite(dec) or dec <= 0:
            metric_ok = False
            metric_failures.append(f"{method}: invalid decode timing")
    checks.append(
        CheckResult(
            "metric_sanity",
            metric_ok,
            "baseline metric vectors satisfy Stage-2 sanity constraints"
            if metric_ok
            else "; ".join(metric_failures) or "no baseline rows",
        )
    )

    q_ok = bool(baseline_rows) and all(record.get("q_mode") == "analytic_exact" for record in baseline_rows)
    checks.append(
        CheckResult(
            "exact_q_available",
            q_ok,
            "all three Stage-2 baselines expose analytic_exact Q_stego"
            if q_ok
            else "one or more baseline rows do not expose analytic_exact Q_stego",
        )
    )

    return tuple(checks)


def read_parquet_records(path: Path) -> list[dict[str, object]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError('Parquet readiness check requires: pip install -e ".[storage]"') from exc

    table = pq.read_table(path)
    return [dict(record) for record in table.to_pylist()]


def check_tracked_hf_token_file(repo_root: Path) -> CheckResult:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "environment/stage1_reference/hf.txt"],
            check=False,
            text=True,
            capture_output=True,
        )
    except OSError as exc:
        return CheckResult("secret_hygiene", False, f"git unavailable: {exc}")

    if proc.returncode != 0:
        return CheckResult("secret_hygiene", False, proc.stderr.strip() or "git ls-files failed")
    tracked = proc.stdout.strip()
    if tracked:
        return CheckResult("secret_hygiene", False, "environment/stage1_reference/hf.txt is still tracked")
    return CheckResult("secret_hygiene", True, "Hugging Face token file is not tracked")


def run_pytest(repo_root: Path) -> CheckResult:
    proc = subprocess.run(
        ["pytest", "-q"],
        cwd=repo_root,
        check=False,
        text=True,
        capture_output=True,
    )
    tail = (proc.stdout + "\n" + proc.stderr).strip().splitlines()
    detail = tail[-1] if tail else f"pytest exited with code {proc.returncode}"
    return CheckResult("test_suite", proc.returncode == 0, detail)


def evaluate_stage2_readiness(
    repo_root: Path,
    *,
    summary_records: Sequence[Mapping[str, object]] | None = None,
    summary_path: Path | None = None,
    include_git_check: bool = True,
    include_tests: bool = False,
) -> Stage2ReadinessReport:
    repo_root = repo_root.resolve()
    checks: list[CheckResult] = [check_required_paths(repo_root)]

    if summary_records is None:
        path = summary_path or (repo_root / "results" / "summary.parquet")
        if not path.is_file():
            checks.append(CheckResult("summary_file", False, f"missing: {path}"))
        else:
            checks.append(CheckResult("summary_file", True, str(path)))
            try:
                summary_records = read_parquet_records(path)
            except RuntimeError as exc:
                checks.append(CheckResult("summary_readable", False, str(exc)))
            else:
                checks.append(CheckResult("summary_readable", True, "Parquet summary decoded successfully"))

    if summary_records is not None:
        checks.extend(check_summary_records(summary_records))

    if include_git_check:
        checks.append(check_tracked_hf_token_file(repo_root))

    if include_tests:
        checks.append(run_pytest(repo_root))

    return Stage2ReadinessReport(tuple(checks))


def format_report(report: Stage2ReadinessReport) -> str:
    lines = []
    for check in report.checks:
        status = "PASS" if check.ok else "FAIL"
        lines.append(f"[{status}] {check.name}: {check.detail}")
    lines.append("")
    lines.append("Stage 2 technical readiness: " + ("READY" if report.ready else "NOT READY"))
    return "\n".join(lines) + "\n"
