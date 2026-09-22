#!/usr/bin/env python3
"""Validate the locally generated Stage-3 GPT-2 Medium pilot result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = REPO_ROOT / "results" / "stage3" / "paper_reproduction" / "gpt2_medium_pilot" / "result.json"
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
    point_ids = {str(record.get("point_id")) for record in records}
    pairs = {(str(record.get("point_id")), int(record.get("selection_rank", -1))) for record in records}
    expected_pairs = {(point, rank) for point in EXPECTED_POINTS for rank in range(8)}

    checks = {
        "schema": result.get("schema_version") == "stage3.gpt2_medium_pilot_result.v1",
        "status": result.get("status") == "ok",
        "model": result.get("model_id") == "gpt2-medium",
        "record_count": len(records) == 32,
        "points": point_ids == EXPECTED_POINTS,
        "point_context_cartesian_product": pairs == expected_pairs,
        "payload_recovery": all(
            bool(record.get("recovery", {}).get("exact_payload_prefix_recovery")) for record in records
        ),
        "sentence_completion": all(
            bool(record.get("generation", {}).get("final_token_finishes_sentence")) for record in records
        ),
        "reference_unchanged": bool(result.get("reference", {}).get("worktree_unchanged")),
        "not_claimed_as_final_figure3": (
            result.get("metric_semantics", {}).get("figure3_status")
            == "not a final Figure-3 reproduction measurement"
        ),
    }
    failed = [name for name, ok in checks.items() if not ok]

    print("Stage 3 GPT-2 Medium pilot gate")
    print(f"result: {path}")
    print(f"status: {result.get('status')}")
    print(f"model: {result.get('model_id')}")
    print(f"records: {len(records)}/32")
    gate = result.get("gate") or {}
    print(f"early sentence-finish runs (diagnostic): {gate.get('early_sentence_finish_run_count')}")
    print(f"paper sentence-shape review required: {gate.get('paper_sentence_shape_review_required')}")
    for point_id, summary in sorted((result.get("summary_by_point") or {}).items()):
        print(
            f"{point_id}: ok={summary.get('ok_count')}/8; "
            f"mean author bpw={summary.get('mean_bits_per_word_author_stats_prefix')}; "
            f"mean author KL bits={summary.get('mean_kl_q_stego_to_p_lm_bits_author')}"
        )

    if failed:
        print("FAILED checks: " + ", ".join(failed))
        print("Stage 3 GPT-2 Medium pilot gate: NOT READY")
        return 1
    print("Stage 3 GPT-2 Medium pilot gate: READY")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    args = parser.parse_args()
    return check(args.result)


if __name__ == "__main__":
    raise SystemExit(main())
