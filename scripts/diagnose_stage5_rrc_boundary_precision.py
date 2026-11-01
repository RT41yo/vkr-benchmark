#!/usr/bin/env python3
"""Diagnose Stage-5 RRC support loss in FP64 cumulative boundaries.

This is a read-only numerical probe.  It does not modify the production RRC
adapter or persist benchmark runs.  For the selected calibration configs it:

1. reproduces the normalized RRC run in memory;
2. replays the generated sender token sequence through the same LM path;
3. inspects each canonical P_reference before RRC;
4. measures positive reference mass whose FP64 cumulative interval width became
   exactly zero;
5. checks that those steps coincide with forward-KL infinities already produced
   by the benchmark metric layer.
"""

from __future__ import annotations

import argparse
import json
from math import isinf
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vkr_benchmark.config import LocalModelConfig
from vkr_benchmark.diagnostics import analyze_rrc_boundary_precision
from vkr_benchmark.distributions import ReferenceDistributionBuilder
from vkr_benchmark.inputs import PromptRegistry
from vkr_benchmark.lm import HFCausalLMAdapter
from vkr_benchmark.runner import run_experiment
from vkr_benchmark.sweeps import PlannedRun, SweepManifest

DEFAULT_MANIFEST = (
    REPO_ROOT / "configs/pilots/stage5/rrc_precision_margin_calibration_llama.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results/stage5/rrc_boundary_precision_probe"


def _selected_runs(
    manifest: SweepManifest,
    *,
    method_point: str,
    prompt_ids: set[str] | None,
    replicates: set[str] | None,
) -> list[PlannedRun]:
    selected: list[PlannedRun] = []
    for run in manifest.expand():
        if run.config.method.method_id != "rrc":
            continue
        if run.method_point_id != method_point:
            continue
        if prompt_ids is not None and run.config.prompt_id not in prompt_ids:
            continue
        if replicates is not None and run.replicate_id not in replicates:
            continue
        selected.append(run)
    return selected


def _csv_set(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    values = {part.strip() for part in raw.split(",") if part.strip()}
    return values or None


def _json_safe_float(value: float | None) -> float | str | None:
    if value is None:
        return None
    if isinf(value):
        return "+inf" if value > 0 else "-inf"
    return float(value)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _record_path(output_dir: Path, run_id: str) -> Path:
    return output_dir / "records" / f"{run_id}.json"


def _completed(output_dir: Path, run_id: str) -> bool:
    path = _record_path(output_dir, run_id)
    if not path.exists():
        return False
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(raw, dict) and raw.get("status") == "ok"


def _replay_boundary_analysis(
    *,
    lm: HFCausalLMAdapter,
    prompt_text: str,
    config: Any,
    carrier_token_ids: tuple[int, ...],
    original_step_distortion: tuple[Any, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    builder = ReferenceDistributionBuilder(
        token_space=lm.token_space,
        policy=config.generation.to_policy(),
    )
    prompt_token_ids = lm.encode_prompt(prompt_text)
    state = lm.prefill(prompt_token_ids)

    step_records: list[dict[str, Any]] = []
    support_loss_steps = 0
    forward_inf_steps = 0
    replay_metric_classification_mismatches = 0
    total_lost_mass = 0.0
    max_step_lost_mass = 0.0
    max_zero_width_count = 0
    first_support_loss_step: int | None = None

    for step_index, token_id in enumerate(carrier_token_ids):
        raw_logits, state = lm.next_logits(state)
        reference = builder.build(raw_logits)
        analysis = analyze_rrc_boundary_precision(reference)
        original = original_step_distortion[step_index]

        original_forward_inf = (
            original.kl_ref_to_stego_bits is not None
            and isinf(original.kl_ref_to_stego_bits)
        )
        replay_forward_inf = (
            analysis.current_distortion.kl_ref_to_stego_bits is not None
            and isinf(analysis.current_distortion.kl_ref_to_stego_bits)
        )
        if original_forward_inf != replay_forward_inf:
            replay_metric_classification_mismatches += 1

        if analysis.support_loss:
            support_loss_steps += 1
            if first_support_loss_step is None:
                first_support_loss_step = step_index
        if original_forward_inf:
            forward_inf_steps += 1

        total_lost_mass += analysis.lost_reference_mass
        max_step_lost_mass = max(max_step_lost_mass, analysis.lost_reference_mass)
        max_zero_width_count = max(max_zero_width_count, analysis.zero_width_count)

        if analysis.support_loss or original_forward_inf:
            step_records.append(
                {
                    "step_index": step_index,
                    "selected_token_id": int(token_id),
                    "support_size": analysis.support_size,
                    "zero_width_count": analysis.zero_width_count,
                    "lost_reference_mass": analysis.lost_reference_mass,
                    "max_lost_reference_probability": analysis.max_lost_reference_probability,
                    "min_lost_reference_probability": analysis.min_lost_reference_probability,
                    "first_lost_rank": analysis.first_lost_rank,
                    "original_forward_kl_bits": _json_safe_float(
                        original.kl_ref_to_stego_bits
                    ),
                    "replayed_forward_kl_bits": _json_safe_float(
                        analysis.current_distortion.kl_ref_to_stego_bits
                    ),
                    "original_reverse_kl_bits": _json_safe_float(
                        original.kl_stego_to_ref_bits
                    ),
                    "replayed_reverse_kl_bits": _json_safe_float(
                        analysis.current_distortion.kl_stego_to_ref_bits
                    ),
                    "original_tvd": original.tvd,
                    "replayed_tvd": analysis.current_distortion.tvd,
                    "ideal_support_preserving_forward_kl_bits": _json_safe_float(
                        analysis.ideal_distortion.kl_ref_to_stego_bits
                    ),
                    "ideal_support_preserving_reverse_kl_bits": _json_safe_float(
                        analysis.ideal_distortion.kl_stego_to_ref_bits
                    ),
                    "ideal_support_preserving_tvd": analysis.ideal_distortion.tvd,
                    "lost_intervals": [
                        {
                            "rank": item.rank,
                            "token_id": item.token_id,
                            "reference_probability": item.reference_probability,
                            "raw_left_boundary": item.raw_left_boundary,
                            "raw_right_boundary": item.raw_right_boundary,
                            "normalized_left_boundary": item.normalized_left_boundary,
                            "normalized_right_boundary": item.normalized_right_boundary,
                            "raw_boundary_spacing": item.raw_boundary_spacing,
                            "probability_to_spacing_ratio": item.probability_to_spacing_ratio,
                            "collapse_stage": item.collapse_stage,
                        }
                        for item in analysis.lost_intervals
                    ],
                }
            )

        if step_index + 1 < len(carrier_token_ids):
            _, state = lm.next_logits(state, int(token_id))

    summary = {
        "carrier_tokens": len(carrier_token_ids),
        "support_loss_steps": support_loss_steps,
        "forward_infinite_steps": forward_inf_steps,
        "all_forward_infinite_steps_have_support_loss": (
            forward_inf_steps == support_loss_steps
        ),
        "replay_metric_classification_mismatches": replay_metric_classification_mismatches,
        "first_support_loss_step": first_support_loss_step,
        "total_lost_reference_mass_across_steps": total_lost_mass,
        "max_step_lost_reference_mass": max_step_lost_mass,
        "max_zero_width_count_on_one_step": max_zero_width_count,
    }
    return step_records, summary


def _build_summary(output_dir: Path, selected: list[PlannedRun]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for run in selected:
        path = _record_path(output_dir, run.run_id)
        if not path.exists():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(raw, dict) and raw.get("status") == "ok":
            records.append(raw)

    support_loss_runs = sum(bool(r["boundary_summary"]["support_loss_steps"]) for r in records)
    source_inf_runs = sum(int(r["source_forward_kl_infinite_steps"]) > 0 for r in records)
    all_run_level_match = all(
        (int(r["source_forward_kl_infinite_steps"]) > 0)
        == (int(r["boundary_summary"]["support_loss_steps"]) > 0)
        for r in records
    )
    all_step_level_match = all(
        r["boundary_summary"]["all_forward_infinite_steps_have_support_loss"]
        and r["boundary_summary"]["replay_metric_classification_mismatches"] == 0
        for r in records
    )

    by_prompt_replicate = {
        f"{r['prompt_id']}:{r['replicate_id']}": {
            "run_id": r["run_id"],
            "carrier_tokens": r["boundary_summary"]["carrier_tokens"],
            "source_forward_kl_infinite_steps": r["source_forward_kl_infinite_steps"],
            "support_loss_steps": r["boundary_summary"]["support_loss_steps"],
            "first_support_loss_step": r["boundary_summary"]["first_support_loss_step"],
            "max_step_lost_reference_mass": r["boundary_summary"]["max_step_lost_reference_mass"],
            "max_zero_width_count_on_one_step": r["boundary_summary"]["max_zero_width_count_on_one_step"],
        }
        for r in records
    }

    return {
        "schema_version": "stage5.rrc_boundary_precision_probe.summary.v1",
        "selected_count": len(selected),
        "completed_count": len(records),
        "remaining_count": len(selected) - len(records),
        "source_forward_kl_infinite_runs": source_inf_runs,
        "support_loss_runs": support_loss_runs,
        "run_level_inf_matches_support_loss": all_run_level_match,
        "step_level_inf_matches_support_loss": all_step_level_match,
        "root_cause_gate": (
            len(records) == len(selected)
            and all_run_level_match
            and all_step_level_match
            and source_inf_runs == support_loss_runs
        ),
        "by_prompt_replicate": by_prompt_replicate,
        "interpretation": (
            "The probe only diagnoses the current FP64 boundary construction. "
            "It does not change the production RRC algorithm or select a new Stage-6 implementation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--method-point",
        default="margin40",
        help="RRC calibration point to replay; default: margin40",
    )
    parser.add_argument("--prompt-id", help="Comma-separated prompt ids")
    parser.add_argument("--replicate", help="Comma-separated replicate ids")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    manifest = SweepManifest.from_json(args.manifest, project_root=REPO_ROOT)
    selected = _selected_runs(
        manifest,
        method_point=args.method_point,
        prompt_ids=_csv_set(args.prompt_id),
        replicates=_csv_set(args.replicate),
    )
    if not selected:
        raise SystemExit("selection produced zero RRC calibration runs")

    print("Stage 5 RRC FP64 boundary-precision probe")
    print("method point:", args.method_point)
    print("selected:", len(selected))
    for run in selected:
        print(
            f"  {run.run_id} {run.config.prompt_id} {run.replicate_id} "
            f"payload={run.config.termination.target_payload_bits}"
        )

    if args.inventory_only:
        return

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_registry = PromptRegistry.from_jsonl(manifest.prompts_path)
    model_config = LocalModelConfig.from_json(
        manifest.model_config_path,
        project_root=REPO_ROOT,
    )

    pending = [
        run for run in selected if not (args.resume and _completed(output_dir, run.run_id))
    ]
    print("pending:", len(pending))
    if pending:
        print("Loading local model once...")
        lm = HFCausalLMAdapter.from_local_config(model_config)
    else:
        lm = None

    for index, run in enumerate(pending, start=1):
        assert lm is not None
        print(
            f"[{index}/{len(pending)}] {run.run_id} "
            f"{run.config.prompt_id} {run.replicate_id}"
        )
        execution = run_experiment(
            config=run.config,
            prompt_registry=prompt_registry,
            lm_adapter=lm,
        )
        step_records, boundary_summary = _replay_boundary_analysis(
            lm=lm,
            prompt_text=execution.prompt.text,
            config=run.config,
            carrier_token_ids=execution.roundtrip.encode.carrier_token_ids,
            original_step_distortion=execution.roundtrip.encode.step_distribution_distortion,
        )
        source_inf_steps = (
            execution.distribution_distortion_metrics.kl_ref_to_stego_infinite_steps
        )
        record = {
            "schema_version": "stage5.rrc_boundary_precision_probe.record.v1",
            "status": "ok",
            "run_id": run.run_id,
            "method_point_id": run.method_point_id,
            "prompt_id": run.config.prompt_id,
            "replicate_id": run.replicate_id,
            "method_params": dict(run.config.method.params),
            "target_payload_bits": run.config.termination.target_payload_bits,
            "roundtrip_exact": execution.reliability_metrics.roundtrip_exact,
            "ber": execution.reliability_metrics.ber,
            "bits_per_token": execution.capacity_entropy_metrics.bits_per_token,
            "entropy_utilization": execution.capacity_entropy_metrics.entropy_utilization,
            "source_forward_kl_mean_bits": _json_safe_float(
                execution.distribution_distortion_metrics.kl_ref_to_stego_mean_bits
            ),
            "source_reverse_kl_mean_bits": _json_safe_float(
                execution.distribution_distortion_metrics.kl_stego_to_ref_mean_bits
            ),
            "source_tvd_mean": execution.distribution_distortion_metrics.tvd_mean,
            "source_forward_kl_infinite_steps": source_inf_steps,
            "boundary_summary": boundary_summary,
            "support_loss_step_records": step_records,
        }
        _atomic_json(_record_path(output_dir, run.run_id), record)
        print(
            "  forward_inf_steps=",
            source_inf_steps,
            " support_loss_steps=",
            boundary_summary["support_loss_steps"],
            " first=",
            boundary_summary["first_support_loss_step"],
            sep="",
        )

    # Rebuild the compact JSONL deterministically from checkpoint records.
    jsonl_path = output_dir / "rrc_boundary_precision_probe.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()
    for run in selected:
        path = _record_path(output_dir, run.run_id)
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and raw.get("status") == "ok":
                compact = dict(raw)
                compact.pop("support_loss_step_records", None)
                _append_jsonl(jsonl_path, compact)

    summary = _build_summary(output_dir, selected)
    summary_path = output_dir / "summary.json"
    _atomic_json(summary_path, summary)

    print()
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print("records:", output_dir / "records")
    print("jsonl:", jsonl_path)
    print("summary:", summary_path)


if __name__ == "__main__":
    main()
