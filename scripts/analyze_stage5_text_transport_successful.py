#!/usr/bin/env python3
"""CPU-only Stage-5 Pilot-B text-transport analysis for process-successful runs.

This diagnostic reads only persisted JSON artifacts. It never loads the LM and
never re-runs encoder/decoder code. For each process-successful Pilot-B run it
compares persisted sender and receiver token IDs and links that evidence with
both the original Pilot-B reliability result and the completed direct-token
audit.

The primary target is the 62 process-successful runs with roundtrip_exact=false.
A run is classified as text_transport_attributed only when all of the following
are true:
  * Pilot B completed the process but roundtrip_exact=false;
  * the direct-token audit reports direct_exact for the same run;
  * persisted sender and receiver token sequences differ.

Exact Pilot-B runs are retained as controls, including cases where transport
changed tokenization but decoding still remained exact.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXECUTION = ROOT / "results/stage5/sweeps/stage5-pilot-b-streaming-llama-v1/execution.jsonl"
DEFAULT_RUN_ROOT = ROOT / "results/runs"
DEFAULT_DIRECT = ROOT / "decoder_audit_local_full/direct_token_replay.jsonl"
DEFAULT_OUTPUT = ROOT / "results/stage5/decoder_audit"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return rows


def first_mismatch(left: tuple[int, ...], right: tuple[int, ...]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right, strict=False)):
        if a != b:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def common_prefix_length(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    mismatch = first_mismatch(left, right)
    return min(len(left), len(right)) if mismatch is None else mismatch


def common_suffix_length(left: tuple[int, ...], right: tuple[int, ...], prefix: int) -> int:
    limit = min(len(left), len(right)) - prefix
    matched = 0
    while matched < limit and left[len(left) - 1 - matched] == right[len(right) - 1 - matched]:
        matched += 1
    return matched


def sequence_diagnostics(sender: tuple[int, ...], receiver: tuple[int, ...]) -> dict[str, Any]:
    prefix = common_prefix_length(sender, receiver)
    suffix = common_suffix_length(sender, receiver, prefix)
    mismatch = first_mismatch(sender, receiver)
    aligned = min(len(sender), len(receiver))
    aligned_mismatch_count = sum(sender[i] != receiver[i] for i in range(aligned))
    sender_middle = max(0, len(sender) - prefix - suffix)
    receiver_middle = max(0, len(receiver) - prefix - suffix)
    return {
        "token_sequence_exact": mismatch is None,
        "first_token_mismatch": mismatch,
        "sender_token_count": len(sender),
        "receiver_token_count": len(receiver),
        "token_count_delta": len(receiver) - len(sender),
        "common_prefix_tokens": prefix,
        "common_suffix_tokens": suffix,
        "sender_changed_middle_tokens": sender_middle,
        "receiver_changed_middle_tokens": receiver_middle,
        "aligned_position_mismatch_count": aligned_mismatch_count,
        "sender_token_at_first_mismatch": None if mismatch is None or mismatch >= len(sender) else sender[mismatch],
        "receiver_token_at_first_mismatch": None if mismatch is None or mismatch >= len(receiver) else receiver[mismatch],
    }


def _median(values: Iterable[int]) -> float | None:
    data = list(values)
    return None if not data else float(statistics.median(data))


def _range(values: Iterable[int]) -> dict[str, Any]:
    data = list(values)
    if not data:
        return {"min": None, "median": None, "max": None}
    return {"min": min(data), "median": float(statistics.median(data)), "max": max(data)}


def classify(*, source_exact: bool, direct_exact: bool | None, token_exact: bool) -> str:
    if source_exact:
        return "exact_control_transport_exact" if token_exact else "exact_control_transport_changed"
    if direct_exact is not True:
        return "silent_inexact_direct_gate_not_exact"
    if token_exact:
        return "silent_inexact_inconsistent_token_evidence"
    return "text_transport_attributed"


def load_direct_rows(path: Path, execution_sha: str) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Direct-token audit JSONL not found: {path}")
    summary_path = path.parent / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Direct-token audit summary not found next to JSONL: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("source_execution_sha256") != execution_sha:
        raise ValueError("Direct-token audit was produced from a different Pilot-B execution log")
    rows = read_jsonl(path)
    mapped = {row["run_id"]: row for row in rows}
    if len(mapped) != len(rows):
        raise ValueError("Duplicate run IDs in direct-token audit")
    return mapped


def read_result(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc


def analyze(
    *,
    execution_rows: list[dict[str, Any]],
    direct_rows: dict[str, dict[str, Any]],
    run_root: Path,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in execution_rows:
        if source.get("status") != "ok":
            continue
        run_id = source["run_id"]
        run_dir = run_root / run_id
        token_path = run_dir / "token_ids.json"
        result_path = run_dir / "result.json"
        if not token_path.is_file():
            raise FileNotFoundError(f"Missing token_ids.json for process-successful run {run_id}")
        if not result_path.is_file():
            raise FileNotFoundError(f"Missing result.json for process-successful run {run_id}")
        token_record = read_result(token_path)
        result_record = read_result(result_path)
        sender = tuple(token_record["sender_token_ids"])
        receiver = tuple(token_record["receiver_token_ids"])
        if not all(type(x) is int for x in sender + receiver):
            raise ValueError(f"Non-integer token ID in {token_path}")
        seq = sequence_diagnostics(sender, receiver)

        persisted_transport_exact = token_record.get("token_sequence_roundtrip_exact")
        persisted_first_mismatch = token_record.get("first_token_mismatch")
        persisted_delta = token_record.get("token_count_delta")
        if persisted_transport_exact is not seq["token_sequence_exact"]:
            raise ValueError(f"Persisted transport-exact flag disagrees with token IDs for {run_id}")
        if persisted_first_mismatch != seq["first_token_mismatch"]:
            raise ValueError(f"Persisted first-token mismatch disagrees with token IDs for {run_id}")
        if persisted_delta != seq["token_count_delta"]:
            raise ValueError(f"Persisted token-count delta disagrees with token IDs for {run_id}")

        source_exact = bool(source.get("roundtrip_exact"))
        if result_record.get("roundtrip_exact") is not source_exact:
            raise ValueError(f"execution.jsonl and result.json disagree on roundtrip_exact for {run_id}")

        direct = direct_rows.get(run_id)
        if direct is None:
            raise ValueError(f"Run {run_id} missing from direct-token audit")
        direct_exact = direct.get("direct_roundtrip_exact")
        if direct.get("audit_status") != "direct_exact" or direct_exact is not True:
            direct_exact = False

        classification = classify(
            source_exact=source_exact,
            direct_exact=direct_exact,
            token_exact=seq["token_sequence_exact"],
        )
        output.append({
            "run_id": run_id,
            "method_id": source["method_id"],
            "method_point_id": source["method_point_id"],
            "prompt_id": source["prompt_id"],
            "replicate_id": source["replicate_id"],
            "source_roundtrip_exact": source_exact,
            "source_ber": source.get("ber"),
            "source_bit_errors": result_record.get("bit_errors"),
            "source_expected_length_bits": result_record.get("expected_length_bits"),
            "source_recovered_length_bits": result_record.get("recovered_length_bits"),
            "source_length_delta_bits": result_record.get("length_delta_bits"),
            "source_recovered_extra_bits": result_record.get("recovered_extra_bits"),
            "source_first_mismatch_bit": result_record.get("first_mismatch_bit"),
            "direct_token_exact": direct_exact,
            **seq,
            "classification": classification,
        })
    return output


def summarize(rows: list[dict[str, Any]], *, execution_sha: str) -> dict[str, Any]:
    counts = Counter(row["classification"] for row in rows)
    successful = len(rows)
    silent = [row for row in rows if not row["source_roundtrip_exact"]]
    exact = [row for row in rows if row["source_roundtrip_exact"]]
    silent_changed = [row for row in silent if not row["token_sequence_exact"]]
    exact_changed = [row for row in exact if not row["token_sequence_exact"]]

    by_method: dict[str, Any] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["method_id"]].append(row)
    for method_id, method_rows in sorted(grouped.items()):
        method_silent = [r for r in method_rows if not r["source_roundtrip_exact"]]
        method_exact = [r for r in method_rows if r["source_roundtrip_exact"]]
        by_method[method_id] = {
            "successful_runs": len(method_rows),
            "source_exact": len(method_exact),
            "source_silent_inexact": len(method_silent),
            "transport_changed_total": sum(not r["token_sequence_exact"] for r in method_rows),
            "silent_inexact_transport_changed": sum(not r["token_sequence_exact"] for r in method_silent),
            "silent_inexact_text_transport_attributed": sum(r["classification"] == "text_transport_attributed" for r in method_silent),
            "exact_controls_transport_changed": sum(not r["token_sequence_exact"] for r in method_exact),
        }

    by_point: dict[str, Any] = {}
    point_grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        point_grouped[(row["method_id"], row["method_point_id"])].append(row)
    for (method_id, point_id), point_rows in sorted(point_grouped.items()):
        silent_rows = [r for r in point_rows if not r["source_roundtrip_exact"]]
        by_point[f"{method_id}:{point_id}"] = {
            "successful_runs": len(point_rows),
            "source_silent_inexact": len(silent_rows),
            "silent_inexact_transport_changed": sum(not r["token_sequence_exact"] for r in silent_rows),
            "silent_inexact_text_transport_attributed": sum(r["classification"] == "text_transport_attributed" for r in silent_rows),
        }

    return {
        "schema_version": "stage5.text_transport_successful_artifact_analysis.v1",
        "source_execution_sha256": execution_sha,
        "process_successful_count": successful,
        "source_exact_count": len(exact),
        "source_silent_inexact_count": len(silent),
        "classification_counts": dict(sorted(counts.items())),
        "silent_inexact_transport_changed_count": len(silent_changed),
        "silent_inexact_transport_exact_count": len(silent) - len(silent_changed),
        "silent_inexact_text_transport_attributed_count": counts.get("text_transport_attributed", 0),
        "exact_controls_transport_changed_count": len(exact_changed),
        "silent_first_token_mismatch": _range(r["first_token_mismatch"] for r in silent_changed if r["first_token_mismatch"] is not None),
        "silent_sender_changed_middle_tokens": _range(r["sender_changed_middle_tokens"] for r in silent_changed),
        "silent_receiver_changed_middle_tokens": _range(r["receiver_changed_middle_tokens"] for r in silent_changed),
        "by_method": by_method,
        "by_operating_point": by_point,
        "interpretation_gate": {
            "all_silent_inexact_have_direct_exact": all(r["direct_token_exact"] for r in silent),
            "all_silent_inexact_have_transport_change": all(not r["token_sequence_exact"] for r in silent),
            "all_silent_inexact_attributed_to_text_transport": all(r["classification"] == "text_transport_attributed" for r in silent),
        },
    }


def protected_output(path: Path, run_root: Path) -> None:
    resolved = path.resolve()
    protected = run_root.resolve()
    if resolved == protected or protected in resolved.parents:
        raise ValueError(f"Refusing to write inside canonical results/runs: {resolved}")
    if resolved.exists():
        raise ValueError(f"Refusing to overwrite existing output: {resolved}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-log", type=Path, default=DEFAULT_EXECUTION)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--direct-audit", type=Path, default=DEFAULT_DIRECT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-successful", type=int, default=466)
    parser.add_argument("--expected-silent-inexact", type=int, default=62)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()

    execution_rows = read_jsonl(args.execution_log)
    execution_sha = sha256(args.execution_log)
    successful_rows = [row for row in execution_rows if row.get("status") == "ok"]
    silent_rows = [row for row in successful_rows if row.get("roundtrip_exact") is False]
    inventory = {
        "execution_count": len(execution_rows),
        "process_successful_count": len(successful_rows),
        "silent_inexact_count": len(silent_rows),
        "token_artifacts_present": sum((args.run_root / row["run_id"] / "token_ids.json").is_file() for row in successful_rows),
        "result_artifacts_present": sum((args.run_root / row["run_id"] / "result.json").is_file() for row in successful_rows),
        "direct_audit_present": args.direct_audit.is_file(),
    }
    print(json.dumps(inventory, indent=2))
    if len(successful_rows) != args.expected_successful:
        parser.error(f"Expected {args.expected_successful} process-successful runs, found {len(successful_rows)}")
    if len(silent_rows) != args.expected_silent_inexact:
        parser.error(f"Expected {args.expected_silent_inexact} silent-inexact runs, found {len(silent_rows)}")
    if args.inventory_only:
        return
    if inventory["token_artifacts_present"] != len(successful_rows) or inventory["result_artifacts_present"] != len(successful_rows):
        parser.error("Restore complete local results/runs artifacts for all 466 process-successful Pilot-B runs")

    direct_rows = load_direct_rows(args.direct_audit, execution_sha)
    rows = analyze(execution_rows=execution_rows, direct_rows=direct_rows, run_root=args.run_root)
    summary = summarize(rows, execution_sha=execution_sha)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = args.output_dir / "pilot_b_text_transport_successful.jsonl"
    out_summary = args.output_dir / "pilot_b_text_transport_successful_summary.json"
    for path in (out_jsonl, out_summary):
        protected_output(path, args.run_root)
    with out_jsonl.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Stage 5 Pilot-B process-successful text-transport artifact analysis")
    print(f"successful: {summary['process_successful_count']}")
    print(f"silent inexact: {summary['source_silent_inexact_count']}")
    print(f"silent with token transport change: {summary['silent_inexact_transport_changed_count']}")
    print(f"silent attributed to text transport: {summary['silent_inexact_text_transport_attributed_count']}")
    print(f"exact controls with token transport change: {summary['exact_controls_transport_changed_count']}")
    print(f"all silent attributed: {summary['interpretation_gate']['all_silent_inexact_attributed_to_text_transport']}")
    print(f"jsonl: {out_jsonl}")
    print(f"summary: {out_summary}")


if __name__ == "__main__":
    main()
