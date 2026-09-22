#!/usr/bin/env python3
"""Validate the locally generated Stage-3 paper-sentence pilot result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = REPO_ROOT / "results" / "stage3" / "paper_reproduction" / "paper_sentence_pilot" / "result.json"
EXPECTED_POINTS = {
    "bins_b3",
    "huffman_e3",
    "arithmetic_t0.9_k300",
    "arithmetic_t1.0_k50256",
}


def check(path: Path) -> int:
    path = path.expanduser().resolve()
    result = json.loads(path.read_text(encoding="utf-8"))
    records = result.get("records") or []
    parity = result.get("parity_checks") or []
    expected_pairs = {(point, rank) for point in EXPECTED_POINTS for rank in range(8)}
    actual_pairs = {(str(r.get("point_id")), int(r.get("selection_rank", -1))) for r in records}
    parity_points = {str(item.get("point_id")) for item in parity}
    gate = result.get("gate") or {}

    checks = {
        "schema": result.get("schema_version") == "stage3.paper_sentence_pilot_result.v1",
        "status": result.get("status") == "ok",
        "model": result.get("model_id") == "gpt2-medium",
        "record_count": len(records) == 32,
        "point_context_cartesian_product": actual_pairs == expected_pairs,
        "parity_point_set": parity_points == EXPECTED_POINTS,
        "parity": len(parity) == 4 and all(bool(item.get("passed")) for item in parity),
        "first_boundary_stop": all(
            r.get("generation", {}).get("terminal_reason") == "sentence_boundary"
            and bool(r.get("generation", {}).get("first_sentence_boundary_is_final_token"))
            for r in records
        ),
        "positive_payload": all(int(r.get("author_metrics", {}).get("payload_bits_confirmed", 0)) > 0 for r in records),
        "payload_recovery": all(bool(r.get("recovery", {}).get("exact_confirmed_payload_prefix_recovery")) for r in records),
        "no_zero_padding": all(not bool(r.get("secret_stream", {}).get("used_implicit_zero_lookahead")) for r in records),
        "no_stream_exhaustion": all(r.get("generation", {}).get("terminal_reason") != "bitstream_exhausted" for r in records),
        "no_safety_cap": all(r.get("generation", {}).get("terminal_reason") != "max_generated_tokens" for r in records),
        "reference_unchanged": bool(result.get("reference", {}).get("worktree_unchanged")),
        "ready_for_full_runner": bool(gate.get("ready_for_full_figure3_runner_implementation")),
        "explicit_operationalization": "operationalization" in str(result.get("metric_semantics", {}).get("historical_claim", "")).lower(),
    }
    failed = [name for name, ok in checks.items() if not ok]

    print("Stage 3 paper-sentence pilot gate")
    print(f"result: {path}")
    print(f"status: {result.get('status')}")
    print(f"model: {result.get('model_id')}")
    print(f"records: {len(records)}/32")
    print(f"mirror parity checks: {sum(bool(item.get('passed')) for item in parity)}/4")
    print(f"all runs stop at first boundary: {checks['first_boundary_stop']}")
    print(f"all confirmed payload prefixes recovered: {checks['payload_recovery']}")
    print(f"all measured steps free of implicit zero look-ahead: {checks['no_zero_padding']}")
    print(f"reference worktree unchanged: {checks['reference_unchanged']}")
    for point_id, summary in sorted((result.get("summary_by_point") or {}).items()):
        print(
            f"{point_id}: ok={summary.get('ok_count')}/8; "
            f"mean tokens={summary.get('mean_carrier_tokens')}; "
            f"mean payload bits={summary.get('mean_payload_bits_confirmed')}; "
            f"mean author bpw={summary.get('mean_bits_per_word_author')}; "
            f"mean author KL bits={summary.get('mean_kl_q_stego_to_p_lm_bits_author')}"
        )

    if failed:
        print("FAILED checks: " + ", ".join(failed))
        print("Stage 3 paper-sentence pilot gate: NOT READY")
        return 1
    print("Stage 3 paper-sentence pilot gate: READY FOR FULL FIGURE-3 RUNNER IMPLEMENTATION")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    args = parser.parse_args()
    return check(args.result)


if __name__ == "__main__":
    raise SystemExit(main())
