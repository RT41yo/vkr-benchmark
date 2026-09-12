#!/usr/bin/env python3
"""Preflight or validate the Stage-3 full Figure-3 author-compatible run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from aggregate_stage3_figure3 import aggregate
from stage3_figure3_core import validate_full_config, validate_paper_sentence_pilot_result

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "reproducibility" / "figure3_full_run.json"
DEFAULT_REFERENCE = REPO_ROOT / "external" / "NeuralSteganography"
EXPECTED_COMMIT = "14e982564aeaf9a33f7b4de440deda2184d17f12"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_reference(path: Path) -> None:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=path, check=True, capture_output=True, text=True).stdout
    if head != EXPECTED_COMMIT:
        raise RuntimeError(f"reference commit mismatch: {head}")
    if status:
        raise RuntimeError("reference worktree is not clean")


def preflight(config_path: Path, reference_dir: Path) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    config = _load_json(config_path)
    matrix_path = (REPO_ROOT / str(config["matrix_path"])).resolve()
    pilot_config_path = (REPO_ROOT / str(config["paper_sentence_pilot_config_path"])).resolve()
    pilot_result_path = (REPO_ROOT / str(config["paper_sentence_pilot_result_path"])).resolve()
    core_path = (REPO_ROOT / str(config["paper_sentence_core_path"])).resolve()
    manifest_path = (REPO_ROOT / str(config["context_manifest_path"])).resolve()
    local_context_path = (REPO_ROOT / str(config["local_context_text_path"])).resolve()

    if _sha256_file(matrix_path) != str(config["matrix_sha256"]):
        raise RuntimeError("paper matrix SHA mismatch")
    if _sha256_file(pilot_config_path) != str(config["paper_sentence_pilot_config_sha256"]):
        raise RuntimeError("Step-3.9 pilot config SHA mismatch")
    if _sha256_file(core_path) != str(config["paper_sentence_core_sha256"]):
        raise RuntimeError("validated paper-sentence core SHA mismatch")
    if _sha256_file(manifest_path) != str(config["context_manifest_sha256"]):
        raise RuntimeError("context manifest SHA mismatch")
    if not local_context_path.is_file():
        raise FileNotFoundError(f"local generated contexts missing: {local_context_path}")

    matrix = _load_json(matrix_path)
    plan = validate_full_config(config, matrix)
    if not pilot_result_path.is_file():
        raise FileNotFoundError(f"Step-3.9 result missing: {pilot_result_path}")
    pilot_result = _load_json(pilot_result_path)
    validate_paper_sentence_pilot_result(pilot_result)
    if str(pilot_result.get("config", {}).get("sha256")) != str(config["paper_sentence_pilot_config_sha256"]):
        raise RuntimeError("Step-3.9 result config SHA mismatch")

    lines = [json.loads(line) for line in local_context_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 80 or sorted(int(item["selection_rank"]) for item in lines) != list(range(80)):
        raise RuntimeError("local context file must contain exactly ranks 0..79")
    _check_reference(reference_dir.expanduser().resolve())
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    try:
        plan = preflight(args.config, args.reference_dir)
    except Exception as exc:
        print(f"Stage 3 Figure-3 preflight: NOT READY ({type(exc).__name__}: {exc})")
        return 1

    print("Stage 3 Figure-3 full-run preflight")
    print(f"points: {plan['point_count']}")
    print(f"shards: {plan['shard_count']}")
    print(f"runs per point: {plan['runs_per_point']}")
    print(f"total runs: {plan['total_runs']}")
    config = _load_json(args.config.expanduser().resolve())
    stop = config["sentence_stop"]
    print(f"Step-3.9 pilot safety cap recorded: {stop['pilot_validated_safety_cap_tokens']} tokens")
    print(f"full-run initial emergency cap: {stop['max_generated_tokens']} tokens")
    adaptive = stop["adaptive_arithmetic_guard"]
    print(f"Arithmetic adaptive guard sequence: {adaptive['cap_sequence_tokens']}")
    termination_policy = stop["termination_failure_policy"]
    print(
        "Arithmetic final-guard non-termination policy: "
        f"status={termination_policy['record_status']}; metrics=excluded; sweep=continues"
    )
    print("zero-confirmed-payload policy: valid for Arithmetic only; included as bits/word=0")
    cache_compat = config["arithmetic_long_context_compatibility"]
    print(
        "Arithmetic long-context cache compatibility: "
        f"sequence-axis sliding window={cache_compat['max_cache_tokens']} tokens"
    )
    print("Step-3.9 pilot gate: READY")
    print("paper-sentence core hash: PINNED")
    print("reference checkout: PINNED + CLEAN")
    print("scientific paper claims are execution gates: False")
    if args.preflight:
        print("Stage 3 Figure-3 preflight: READY TO RUN")
        return 0

    try:
        summary = aggregate(args.config, output_dir=args.output_dir, write=False)
    except Exception as exc:
        print(f"Stage 3 Figure-3 full-run gate: NOT READY ({type(exc).__name__}: {exc})")
        return 1

    print("\nStage 3 Figure-3 full-run gate")
    print(f"runs: {summary['run_count']}/{summary['expected_run_count']}")
    print(f"points: {summary['point_count']}/{summary['expected_point_count']}")
    print(f"pilot continuity: {summary['pilot_continuity']['passed']}")
    print(f"reference worktree unchanged: {summary['reference_worktree_unchanged']}")
    print(f"scientific claims evaluated: {summary['scientific_claims_evaluated']}")
    sentence = summary["sentence_length_diagnostic"]
    print(
        f"sentence-length diagnostic: >256={sentence['over_step39_pilot_cap_256_count']}; "
        f"max={sentence['max_carrier_tokens']}; emergency cap={sentence['full_run_emergency_cap_tokens']}"
    )
    termination = summary.get("termination_diagnostic", {})
    print(
        "sentence termination diagnostic: "
        f"completed={termination.get('terminated_sentence_run_count', 0)}/{termination.get('scheduled_run_count', 0)}; "
        f"failures={termination.get('termination_failure_count', 0)}; "
        f"success_rate={termination.get('termination_success_rate')}"
    )
    cache_compat = summary.get("arithmetic_cache_compatibility_diagnostic", {})
    print(
        "Arithmetic cache compatibility diagnostic: "
        f"runs with encode trim={cache_compat.get('runs_with_encode_trim', 0)}; "
        f"encode trim events={cache_compat.get('total_encode_trim_events', 0)}; "
        f"runs with decode trim={cache_compat.get('runs_with_decode_trim', 0)}"
    )
    guard = summary.get("sentence_guard_escalation_diagnostic", {})
    print(
        "Arithmetic sentence-guard diagnostic: "
        f"runs with escalation={guard.get('runs_with_escalation', 0)}; "
        f"total escalations={guard.get('total_escalations', 0)}; "
        f"max final guard={guard.get('max_final_guard_tokens', 0)}"
    )
    zero_payload = summary["zero_payload_diagnostic"]
    print(
        f"zero-payload diagnostic: {zero_payload['count']}/{summary['terminated_sentence_run_count']} completed sentences; "
        "valid only for Arithmetic"
    )
    recovery = summary["transport_recovery_diagnostic"]
    print(
        f"transport recovery diagnostic: exact={recovery['exact_prefix_count']}/{recovery['applicable_positive_payload_count']}; "
        f"failures={recovery['failure_count']}; decoder exceptions={recovery['decoder_exception_count']}; "
        "Figure-3 gate=False"
    )
    ready = bool(summary["execution_gate_ready_for_step_3_11_interpretation"])
    print(
        "Stage 3 Figure-3 full-run gate: "
        + ("READY FOR STEP 3.11 INTERPRETATION" if ready else "NOT READY")
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
