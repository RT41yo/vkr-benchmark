#!/usr/bin/env python3
"""Execute a Stage-5 sweep while loading the selected LM only once.

The script consumes the same deterministic plan produced by
``plan_stage5_sweep.py``.  Canonical per-run artifacts remain the ordinary
benchmark ``results/runs/<run_id>/...`` records; this runner only orchestrates
multiple normalized runs efficiently and writes a lightweight sweep log.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vkr_benchmark.config import LocalModelConfig
from vkr_benchmark.inputs import PromptRegistry
from vkr_benchmark.lm import HFCausalLMAdapter
from vkr_benchmark.runner import run_experiment
from vkr_benchmark.storage import (
    ensure_summary_backend_available,
    persist_experiment_execution,
    persist_failed_run,
)
from vkr_benchmark.sweeps import PlannedRun, SweepManifest


def _csv_filter(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    values = {item.strip() for item in raw.split(",") if item.strip()}
    return values or None


def _selected(
    runs: Iterable[PlannedRun],
    *,
    methods: set[str] | None,
    method_points: set[str] | None,
    generation_points: set[str] | None,
    prompts: set[str] | None,
    replicates: set[str] | None,
) -> list[PlannedRun]:
    selected: list[PlannedRun] = []
    for run in runs:
        if methods is not None and run.config.method.method_id not in methods:
            continue
        if method_points is not None and run.method_point_id not in method_points:
            continue
        if generation_points is not None and run.generation_point_id not in generation_points:
            continue
        if prompts is not None and run.config.prompt_id not in prompts:
            continue
        if replicates is not None and run.replicate_id not in replicates:
            continue
        selected.append(run)
    return selected


def _successful_run_exists(results_root: Path, run_id: str) -> bool:
    result_path = results_root / "runs" / run_id / "result.json"
    if not result_path.exists():
        return False
    try:
        raw = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(raw, dict) and raw.get("status") == "ok"


def _append_log(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--method", help="Comma-separated method ids")
    parser.add_argument("--method-point", help="Comma-separated method-point ids")
    parser.add_argument("--generation-point", help="Comma-separated generation-point ids")
    parser.add_argument("--prompt-id", help="Comma-separated prompt ids")
    parser.add_argument("--replicate", help="Comma-separated replicate ids")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "results",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip canonical run ids that already have status=ok.",
    )
    parser.add_argument(
        "--skip-summary-parquet",
        action="store_true",
        help="Persist run artifacts but do not update results/summary.parquet.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort the sweep after the first failed run instead of logging and continuing.",
    )
    args = parser.parse_args()

    manifest = SweepManifest.from_json(args.manifest, project_root=REPO_ROOT)
    prompt_registry = PromptRegistry.from_jsonl(manifest.prompts_path)
    planned = _selected(
        manifest.expand(),
        methods=_csv_filter(args.method),
        method_points=_csv_filter(args.method_point),
        generation_points=_csv_filter(args.generation_point),
        prompts=_csv_filter(args.prompt_id),
        replicates=_csv_filter(args.replicate),
    )
    if not planned:
        raise SystemExit("selection produced zero runs")

    for run in planned:
        prompt_registry.get(run.config.prompt_id)

    results_root = args.results_root.expanduser().resolve()
    update_summary = not args.skip_summary_parquet
    if update_summary:
        ensure_summary_backend_available()

    model_config = LocalModelConfig.from_json(
        manifest.model_config_path,
        project_root=REPO_ROOT,
    )
    log_path = results_root / "stage5" / "sweeps" / manifest.sweep_id / "execution.jsonl"

    print("Stage-5 sweep runner")
    print("sweep_id:", manifest.sweep_id)
    print("selected runs:", len(planned))
    print("model:", model_config.model_id)
    print("revision:", model_config.revision)
    print("sweep log:", log_path)
    print("Loading local model once for the selected sweep...")
    lm = HFCausalLMAdapter.from_local_config(model_config)

    ok = 0
    failed = 0
    skipped = 0
    sweep_started = time.perf_counter()

    for index, planned_run in enumerate(planned, start=1):
        if args.resume and _successful_run_exists(results_root, planned_run.run_id):
            skipped += 1
            print(f"[{index}/{len(planned)}] SKIP {planned_run.run_id}")
            _append_log(
                log_path,
                {
                    **planned_run.metadata_record(),
                    "status": "skipped_existing_ok",
                },
            )
            continue

        print(
            f"[{index}/{len(planned)}] RUN {planned_run.run_id} "
            f"{planned_run.config.method.method_id}/{planned_run.method_point_id} "
            f"{planned_run.generation_point_id} {planned_run.config.prompt_id} "
            f"{planned_run.replicate_id}"
        )
        started = time.perf_counter()
        try:
            execution = run_experiment(
                config=planned_run.config,
                prompt_registry=prompt_registry,
                lm_adapter=lm,
            )
            artifacts = persist_experiment_execution(
                execution,
                model_config=model_config,
                results_root=results_root,
                update_summary=update_summary,
            )
            elapsed = time.perf_counter() - started
            ok += 1
            record = {
                **planned_run.metadata_record(),
                "status": "ok",
                "elapsed_wall_seconds": elapsed,
                "run_directory": str(artifacts.run_directory),
                "bits_per_token": execution.capacity_entropy_metrics.bits_per_token,
                "entropy_utilization": execution.capacity_entropy_metrics.entropy_utilization,
                "kl_ref_to_stego_mean_bits": (
                    execution.distribution_distortion_metrics.kl_ref_to_stego_mean_bits
                ),
                "kl_stego_to_ref_mean_bits": (
                    execution.distribution_distortion_metrics.kl_stego_to_ref_mean_bits
                ),
                "tvd_mean": execution.distribution_distortion_metrics.tvd_mean,
                "ppl_raw_lm": execution.quality_metrics.ppl_raw_lm,
                "roundtrip_exact": execution.reliability_metrics.roundtrip_exact,
                "ber": execution.reliability_metrics.ber,
                "encode_ms_per_token": execution.performance_metrics.encode_ms_per_token,
                "decode_ms_per_token": execution.performance_metrics.decode_ms_per_token,
                "metric_instrumentation_total_ms": (
                    execution.performance_metrics.metric_instrumentation_total_ms
                ),
                "stego_algorithm_total_ms": (
                    execution.performance_metrics.stego_algorithm_total_ms
                ),
            }
            _append_log(log_path, record)
            print(f"  OK in {elapsed:.2f}s")
        except Exception as exc:
            elapsed = time.perf_counter() - started
            failed += 1
            artifacts = persist_failed_run(
                config=planned_run.config,
                model_config=model_config,
                error=exc,
                results_root=results_root,
                update_summary=update_summary,
            )
            _append_log(
                log_path,
                {
                    **planned_run.metadata_record(),
                    "status": "error",
                    "elapsed_wall_seconds": elapsed,
                    "run_directory": str(artifacts.run_directory),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            print(f"  ERROR after {elapsed:.2f}s: {type(exc).__name__}: {exc}")
            if args.stop_on_error:
                raise

    total = time.perf_counter() - sweep_started
    print()
    print("completed:", ok)
    print("failed:", failed)
    print("skipped_existing_ok:", skipped)
    print("sweep_wall_seconds:", f"{total:.2f}")
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
