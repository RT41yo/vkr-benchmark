#!/usr/bin/env python3
"""Aggregate completed Stage-3 Figure-3 shards without loading a language model."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from stage3_figure3_core import (
    aggregate_point,
    pilot_continuity_differences,
    validate_full_config,
    validate_paper_sentence_pilot_result,
    validate_shard_records,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "reproducibility" / "figure3_full_run.json"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL {path}:{line_number}: {exc}") from exc
    return records


def shard_path(output_dir: Path, point_id: str, replicate: int) -> Path:
    return output_dir / "shards" / point_id / f"replicate_{replicate}.jsonl"


def collect_records(config: dict[str, Any], output_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    by_point: dict[str, list[dict[str, Any]]] = {str(point["id"]): [] for point in config["points"]}
    errors: list[str] = []
    for point in config["points"]:
        point_id = str(point["id"])
        for replicate in config["replicates"]:
            path = shard_path(output_dir, point_id, int(replicate))
            if not path.is_file():
                errors.append(f"missing shard: {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}")
                continue
            records = _load_jsonl(path)
            shard_errors = validate_shard_records(records, point=point, replicate=int(replicate))
            errors.extend(f"{point_id}/replicate_{replicate}: {msg}" for msg in shard_errors)
            by_point[point_id].extend(records)
    return by_point, errors


def aggregate(config_path: Path, *, output_dir: Path | None = None, write: bool = True) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    config = _load_json(config_path)
    matrix_path = (REPO_ROOT / str(config["matrix_path"])).resolve()
    pilot_config_path = (REPO_ROOT / str(config["paper_sentence_pilot_config_path"])).resolve()
    pilot_result_path = (REPO_ROOT / str(config["paper_sentence_pilot_result_path"])).resolve()
    if _sha256_file(matrix_path) != str(config["matrix_sha256"]):
        raise RuntimeError("frozen paper matrix SHA-256 mismatch")
    if _sha256_file(pilot_config_path) != str(config["paper_sentence_pilot_config_sha256"]):
        raise RuntimeError("paper-sentence pilot config SHA-256 mismatch")
    matrix = _load_json(matrix_path)
    plan = validate_full_config(config, matrix)
    if not pilot_result_path.is_file():
        raise FileNotFoundError(f"Step-3.9 result missing: {pilot_result_path}")
    pilot_result = _load_json(pilot_result_path)
    validate_paper_sentence_pilot_result(pilot_result)

    output_dir = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else (REPO_ROOT / str(config["storage"]["output_directory"])).resolve()
    )
    by_point, errors = collect_records(config, output_dir)
    if errors:
        raise RuntimeError("full Figure-3 shards are incomplete/invalid:\n- " + "\n- ".join(errors[:30]))

    point_summaries: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    points_by_id = {str(point["id"]): point for point in config["points"]}
    for point in config["points"]:
        point_id = str(point["id"])
        records = by_point[point_id]
        point_summaries.append(aggregate_point(point, records))
        all_records.extend(records)

    continuity = config["pilot_continuity"]
    continuity_errors = pilot_continuity_differences(
        all_records,
        pilot_result,
        point_ids=[str(x) for x in continuity["point_ids"]],
        selection_ranks=[int(x) for x in continuity["selection_ranks"]],
        replicate=int(continuity["replicate"]),
        metric_abs_tol=float(continuity["metric_abs_tol"]),
    )

    run_state_path = output_dir / str(config["storage"]["run_state_path"])
    run_state = _load_json(run_state_path) if run_state_path.is_file() else {}
    expected_total = int(config["execution"]["expected_total_run_count"])
    complete = len(all_records) == expected_total
    accepted_statuses = {"ok", "sentence_termination_failure"}
    all_accepted = all(record.get("status") in accepted_statuses for record in all_records)
    completed_records = [record for record in all_records if record.get("status") == "ok"]
    termination_failures = [record for record in all_records if record.get("status") == "sentence_termination_failure"]
    reference_unchanged = bool(run_state.get("reference", {}).get("worktree_unchanged"))
    execution_gate = bool(complete and all_accepted and not continuity_errors and reference_unchanged)
    zero_payload_count = sum(
        int(record["author_metrics"]["payload_bits_confirmed"]) == 0 for record in completed_records
    )
    recovery_applicable = [
        record for record in completed_records if int(record["author_metrics"]["payload_bits_confirmed"]) > 0
    ]
    recovery_exact = sum(
        bool(record["recovery"]["exact_confirmed_payload_prefix_recovery"]) for record in recovery_applicable
    )
    recovery_failures = len(recovery_applicable) - recovery_exact
    recovery_exceptions = sum(
        record["recovery"].get("status") == "decoder_exception" for record in recovery_applicable
    )
    over_pilot_cap = sum(int(record["author_metrics"]["carrier_tokens"]) > 256 for record in completed_records)
    max_carrier_tokens = max((int(record["author_metrics"]["carrier_tokens"]) for record in completed_records), default=0)
    max_observed_partial_tokens = max((int(record["author_metrics"]["carrier_tokens"]) for record in all_records), default=0)

    arithmetic_records = [record for record in all_records if record.get("method") == "arithmetic"]
    cache_compat_records = [record.get("arithmetic_cache_compatibility") or {} for record in arithmetic_records]
    sentence_guard_records = [((record.get("generation") or {}).get("sentence_guard") or {}) for record in arithmetic_records]
    arithmetic_cache_diag = {
        "max_cache_tokens": int(config["arithmetic_long_context_compatibility"]["max_cache_tokens"]),
        "arithmetic_run_count": len(arithmetic_records),
        "runs_with_encode_trim": sum(int(item.get("encode_trim_events", 0)) > 0 for item in cache_compat_records),
        "total_encode_trim_events": sum(int(item.get("encode_trim_events", 0)) for item in cache_compat_records),
        "max_encode_seen_sequence_tokens": max((int(item.get("encode_max_seen_sequence_tokens", 0)) for item in cache_compat_records), default=0),
        "runs_with_decode_trim": sum(int(item.get("decode_trim_events", 0)) > 0 for item in cache_compat_records),
        "total_decode_trim_events": sum(int(item.get("decode_trim_events", 0)) for item in cache_compat_records),
        "max_decode_seen_sequence_tokens": max((int(item.get("decode_max_seen_sequence_tokens", 0)) for item in cache_compat_records), default=0),
        "figure3_execution_gate": False,
        "role": "compatibility diagnostic for intended historical 1022-token sliding cache",
    }

    sentence_guard_diag = {
        "initial_guard_tokens": int(config["sentence_stop"]["max_generated_tokens"]),
        "max_escalated_guard_tokens": int(config["sentence_stop"]["adaptive_arithmetic_guard"]["cap_sequence_tokens"][-1]),
        "arithmetic_run_count": len(arithmetic_records),
        "runs_with_escalation": sum(int(item.get("escalation_count", 0)) > 0 for item in sentence_guard_records),
        "total_escalations": sum(int(item.get("escalation_count", 0)) for item in sentence_guard_records),
        "max_final_guard_tokens": max((int(item.get("final_guard_tokens", 0)) for item in sentence_guard_records), default=0),
        "final_guard_exhaustion_count": sum(bool(item.get("final_guard_exhausted")) for item in sentence_guard_records),
        "figure3_execution_gate": False,
        "role": "engineering runaway guard diagnostic; final 8192-token exhaustion is classified separately as sentence_termination_failure",
    }

    summary = {
        "schema_version": "stage3.figure3_full_summary.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"path": str(config_path), "sha256": _sha256_file(config_path)},
        "matrix": {"path": str(matrix_path), "sha256": _sha256_file(matrix_path)},
        "mode": "author-compatible paper-sentence Figure-3 operationalization",
        "model_id": str(config["model_id"]),
        "aggregation": config["aggregation"],
        "run_count": len(all_records),
        "scheduled_run_count": len(all_records),
        "terminated_sentence_run_count": len(completed_records),
        "termination_failure_count": len(termination_failures),
        "termination_success_rate": (len(completed_records) / len(all_records)) if all_records else None,
        "expected_run_count": expected_total,
        "point_count": len(point_summaries),
        "expected_point_count": int(plan["point_count"]),
        "pilot_continuity": {
            "passed": not continuity_errors,
            "difference_count": len(continuity_errors),
            "differences": continuity_errors,
        },
        "reference_worktree_unchanged": reference_unchanged,
        "sentence_length_diagnostic": {
            "over_step39_pilot_cap_256_count": over_pilot_cap,
            "max_carrier_tokens": max_carrier_tokens,
            "max_observed_partial_tokens_including_termination_failures": max_observed_partial_tokens,
            "full_run_emergency_cap_tokens": int(config["sentence_stop"]["max_generated_tokens"]),
        },
        "termination_diagnostic": {
            "scheduled_run_count": len(all_records),
            "terminated_sentence_run_count": len(completed_records),
            "termination_failure_count": len(termination_failures),
            "termination_success_rate": (len(completed_records) / len(all_records)) if all_records else None,
            "failure_point_counts": {
                point_id: sum(record.get("point_id") == point_id for record in termination_failures)
                for point_id in sorted({str(record.get("point_id")) for record in termination_failures})
            },
            "metric_conditioning": "Figure-3 sentence metrics are conditional on successful termination under pinned utils.is_sent_finish",
            "figure3_execution_gate": False,
        },
        "arithmetic_cache_compatibility_diagnostic": arithmetic_cache_diag,
        "sentence_guard_escalation_diagnostic": sentence_guard_diag,
        "zero_payload_diagnostic": {
            "count": zero_payload_count,
            "rate_among_completed_sentences": (zero_payload_count / len(completed_records)) if completed_records else None,
            "valid_only_for_method": "arithmetic",
            "included_in_bits_per_word_mean_as_zero": True,
        },
        "transport_recovery_diagnostic": {
            "applicable_positive_payload_count": len(recovery_applicable),
            "not_applicable_zero_payload_count": zero_payload_count,
            "not_applicable_termination_failure_count": len(termination_failures),
            "exact_prefix_count": recovery_exact,
            "failure_count": recovery_failures,
            "exact_prefix_rate": (recovery_exact / len(recovery_applicable)) if recovery_applicable else None,
            "decoder_exception_count": recovery_exceptions,
            "figure3_execution_gate": False,
        },
        "execution_gate_ready_for_step_3_11_interpretation": execution_gate,
        "scientific_claims_evaluated": False,
        "points": point_summaries,
    }

    if write:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / str(config["storage"]["summary_path"])
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        csv_path = output_dir / str(config["storage"]["figure3_points_csv_path"])
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = [
                "point_id", "method", "parameters_json", "run_count",
                "terminated_sentence_run_count", "termination_failure_count", "termination_success_rate",
                "mean_bits_per_word", "se_bits_per_word_runs", "se_bits_per_word_context_clustered",
                "mean_kl_bits", "se_kl_bits_runs", "se_kl_bits_context_clustered",
                "mean_nll_nats", "mean_carrier_tokens", "mean_payload_bits",
                "sentence_over_pilot_cap_256_count", "max_carrier_tokens",
                "zero_payload_count", "zero_payload_rate",
                "arithmetic_cache_runs_with_encode_trim", "arithmetic_cache_total_encode_trim_events",
                "sentence_guard_runs_with_escalation", "sentence_guard_max_final_guard_tokens",
                "transport_recovery_exact_prefix_rate", "transport_recovery_failure_count",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for point_summary in point_summaries:
                metrics = point_summary["metrics"]
                writer.writerow({
                    "point_id": point_summary["point_id"],
                    "method": point_summary["method"],
                    "parameters_json": json.dumps(point_summary["point_parameters"], sort_keys=True, separators=(",", ":")),
                    "run_count": point_summary["run_count"],
                    "terminated_sentence_run_count": point_summary["terminated_sentence_run_count"],
                    "termination_failure_count": point_summary["termination_failure_count"],
                    "termination_success_rate": point_summary["termination_success_rate"],
                    "mean_bits_per_word": metrics["bits_per_word_author"]["mean"],
                    "se_bits_per_word_runs": metrics["bits_per_word_author"]["standard_error_runs"],
                    "se_bits_per_word_context_clustered": metrics["bits_per_word_author"]["standard_error_context_clustered"],
                    "mean_kl_bits": metrics["kl_q_stego_to_p_lm_bits_author"]["mean"],
                    "se_kl_bits_runs": metrics["kl_q_stego_to_p_lm_bits_author"]["standard_error_runs"],
                    "se_kl_bits_context_clustered": metrics["kl_q_stego_to_p_lm_bits_author"]["standard_error_context_clustered"],
                    "mean_nll_nats": metrics["avg_nll_nats_author"]["mean"],
                    "mean_carrier_tokens": metrics["carrier_tokens"]["mean"],
                    "mean_payload_bits": metrics["payload_bits_confirmed"]["mean"],
                    "sentence_over_pilot_cap_256_count": point_summary["sentence_length_diagnostic"]["over_step39_pilot_cap_256_count"],
                    "max_carrier_tokens": point_summary["sentence_length_diagnostic"]["max_carrier_tokens"],
                    "zero_payload_count": point_summary["zero_payload_diagnostic"]["count"],
                    "zero_payload_rate": point_summary["zero_payload_diagnostic"]["rate_among_completed_sentences"],
                    "arithmetic_cache_runs_with_encode_trim": point_summary["arithmetic_cache_compatibility"]["runs_with_encode_trim"],
                    "arithmetic_cache_total_encode_trim_events": point_summary["arithmetic_cache_compatibility"]["total_encode_trim_events"],
                    "sentence_guard_runs_with_escalation": point_summary["sentence_guard_diagnostic"]["runs_with_escalation"],
                    "sentence_guard_max_final_guard_tokens": point_summary["sentence_guard_diagnostic"]["max_final_guard_tokens"],
                    "transport_recovery_exact_prefix_rate": point_summary["transport_recovery"]["exact_prefix_rate"],
                    "transport_recovery_failure_count": point_summary["transport_recovery"]["failure_count"],
                })
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    try:
        summary = aggregate(args.config, output_dir=args.output_dir, write=True)
    except Exception as exc:
        print(f"Stage 3 Figure-3 aggregation: NOT READY ({type(exc).__name__}: {exc})")
        return 1

    print("Stage 3 Figure-3 aggregation")
    print(f"runs: {summary['run_count']}/{summary['expected_run_count']}")
    print(f"points: {summary['point_count']}/{summary['expected_point_count']}")
    print(f"pilot continuity: {summary['pilot_continuity']['passed']}")
    print(f"reference worktree unchanged: {summary['reference_worktree_unchanged']}")
    sentence = summary["sentence_length_diagnostic"]
    print(
        f"sentence-length diagnostic: >256={sentence['over_step39_pilot_cap_256_count']}; "
        f"max={sentence['max_carrier_tokens']}; emergency cap={sentence['full_run_emergency_cap_tokens']}"
    )
    cache_compat = summary["arithmetic_cache_compatibility_diagnostic"]
    print(
        f"Arithmetic cache compatibility: encode-trim-runs={cache_compat['runs_with_encode_trim']}; "
        f"encode-trim-events={cache_compat['total_encode_trim_events']}; "
        f"decode-trim-runs={cache_compat['runs_with_decode_trim']}"
    )
    termination = summary["termination_diagnostic"]
    print(
        f"sentence termination: completed={termination['terminated_sentence_run_count']}/{termination['scheduled_run_count']}; "
        f"failures={termination['termination_failure_count']}; success_rate={termination['termination_success_rate']}"
    )
    zero_payload = summary["zero_payload_diagnostic"]
    print(
        f"zero-payload diagnostic: {zero_payload['count']}/{summary['terminated_sentence_run_count']} completed sentences; "
        "included in bits/word mean as 0"
    )
    recovery = summary["transport_recovery_diagnostic"]
    print(
        f"transport recovery diagnostic: exact={recovery['exact_prefix_count']}/{recovery['applicable_positive_payload_count']}; "
        f"failures={recovery['failure_count']}; decoder exceptions={recovery['decoder_exception_count']}; "
        "Figure-3 gate=False"
    )
    for point in summary["points"]:
        metrics = point["metrics"]
        print(
            f"{point['point_id']}: completed={point['terminated_sentence_run_count']}/{point['scheduled_run_count']}; "
            f"bpw={metrics['bits_per_word_author']['mean']} ± {metrics['bits_per_word_author']['standard_error_runs']}; "
            f"KL={metrics['kl_q_stego_to_p_lm_bits_author']['mean']} ± {metrics['kl_q_stego_to_p_lm_bits_author']['standard_error_runs']} bits"
        )
    print(
        "Stage 3 Figure-3 aggregation: "
        + ("READY FOR STEP 3.11 INTERPRETATION" if summary["execution_gate_ready_for_step_3_11_interpretation"] else "NOT READY")
    )
    return 0 if summary["execution_gate_ready_for_step_3_11_interpretation"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
