#!/usr/bin/env python3
"""Derive a deterministic sentence-shape audit from the committed GPT-2 Medium pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "results" / "stage3" / "paper_reproduction" / "gpt2_medium_pilot" / "result.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "stage3" / "paper_reproduction" / "gpt2_medium_pilot" / "sentence_shape_analysis.json"
EXPECTED_SOURCE_SHA256 = "ee669fde58f880cca961baebca88e8c5b383bd2716c6775afb13b5a2d475c47d"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file():
        print(f"missing pilot result: {source}", file=sys.stderr)
        return 1
    source_sha = sha256(source)
    if source_sha != EXPECTED_SOURCE_SHA256:
        print(f"pilot result SHA-256 mismatch: expected {EXPECTED_SOURCE_SHA256}, got {source_sha}", file=sys.stderr)
        return 1
    data = json.loads(source.read_text(encoding="utf-8"))
    records = data.get("records") or []
    early: list[dict[str, object]] = []
    one_boundary = 0
    for record in records:
        gen = record.get("generation") or {}
        positions = [int(x) for x in gen.get("sentence_finish_token_positions_zero_based", [])]
        total = int(gen.get("generated_token_count_total", 0))
        if len(positions) == 1 and positions and positions[0] == total - 1:
            one_boundary += 1
        if bool(gen.get("has_early_sentence_finish")):
            early.append({
                "point_id": record.get("point_id"),
                "selection_rank": int(record.get("selection_rank")),
                "generated_token_count_total": total,
                "sentence_finish_token_positions_zero_based": positions,
                "first_sentence_finish_position_zero_based": positions[0] if positions else None,
                "tokens_after_first_sentence_finish": total - positions[0] - 1 if positions else None,
            })
    by_point: dict[str, int] = {}
    for item in early:
        point = str(item["point_id"])
        by_point[point] = by_point.get(point, 0) + 1
    out = {
        "schema_version": "stage3.gpt2_medium_pilot_sentence_shape.v1",
        "source_result_path": str(source),
        "source_result_sha256": source_sha,
        "total_run_count": len(records),
        "single_sentence_shape_count": one_boundary,
        "early_sentence_finish_run_count": len(early),
        "early_sentence_finish_count_by_point": dict(sorted(by_point.items())),
        "early_sentence_finish_runs": early,
        "interpretation": {
            "public_finish_sent_semantics": "The public author loop consumes the fixed payload first and only then greedily extends until a sentence-finish token is observed.",
            "paper_sentence_shape_issue_observed": len(early) > 0,
            "fixed_payload_then_finish_can_cross_first_sentence_boundary": len(early) > 0,
            "future_paper_driver_requirement": "Use a sufficiently long uniform bitstream and define the first sentence boundary as the candidate generation stop; do not reuse this audit as a Figure-3 measurement.",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Stage 3 GPT-2 Medium sentence-shape audit")
    print(f"source SHA-256: {source_sha}")
    print(f"runs: {len(records)}")
    print(f"single-sentence-shape runs: {one_boundary}")
    print(f"early sentence-finish runs: {len(early)}")
    print(f"by point: {dict(sorted(by_point.items()))}")
    print(f"output: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
