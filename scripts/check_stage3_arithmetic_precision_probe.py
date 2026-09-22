#!/usr/bin/env python3
"""Validate the completed Stage-3 Arithmetic precision diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = REPO_ROOT / "results" / "stage3" / "paper_reproduction" / "arithmetic_precision_probe" / "result.json"
EXPECTED_PRECISIONS = {26, 32, 40, 48}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    args = parser.parse_args()
    path = args.result.expanduser().resolve()
    if not path.is_file():
        print(f"Stage 3 Arithmetic precision probe gate: NOT READY (missing {path})")
        return 1
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("records") or []
    gate = data.get("gate") or {}
    summary = data.get("summary_by_precision") or {}
    errors: list[str] = []
    if data.get("schema_version") != "stage3.arithmetic_precision_probe_result.v1":
        errors.append("unexpected result schema")
    if data.get("status") != "ok":
        errors.append(f"status={data.get('status')}")
    if len(records) != 32:
        errors.append(f"expected 32 records, got {len(records)}")
    if not gate.get("mirror_parity_passed"):
        errors.append("instrumented mirror parity did not pass")
    if not gate.get("reference_worktree_unchanged"):
        errors.append("reference worktree changed")
    precisions = {int(r.get("precision", -1)) for r in records}
    if precisions != EXPECTED_PRECISIONS:
        errors.append(f"precision set mismatch: {sorted(precisions)}")
    for p in EXPECTED_PRECISIONS:
        subset = [r for r in records if int(r.get("precision", -1)) == p]
        if len(subset) != 8:
            errors.append(f"precision {p}: expected 8 contexts, got {len(subset)}")
        item = summary.get(str(p)) or {}
        if int(item.get("run_count", -1)) != 8:
            errors.append(f"precision {p}: summary run_count != 8")
        if int(item.get("zero_padding_free_step_count", 0)) <= 0:
            errors.append(f"precision {p}: no zero-padding-free diagnostic steps")

    print("Stage 3 Arithmetic precision probe gate")
    print(f"result: {path}")
    print(f"status: {data.get('status')}")
    print(f"records: {len(records)}/32")
    print(f"mirror parity: {bool(gate.get('mirror_parity_passed'))}")
    print(f"reference worktree unchanged: {bool(gate.get('reference_worktree_unchanged'))}")
    for p in sorted(EXPECTED_PRECISIONS):
        item = summary.get(str(p)) or {}
        print(
            f"precision={p}: full-run mean author KL={item.get('mean_run_author_kl_bits')}; "
            f"zero-padding-free author KL={item.get('mean_zero_padding_free_author_kl_bits')}; "
            f"zero-padding-free trunc-only={item.get('mean_zero_padding_free_truncation_only_kl_bits')}; "
            f"zero-padding-free terminal-fill={item.get('mean_zero_padding_free_terminal_fill_counterfactual_kl_bits')}; "
            f"min clean effective bits={item.get('minimum_effective_precision_bits_before')}"
        )
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        print("Stage 3 Arithmetic precision probe gate: NOT READY")
        return 1
    print("full Figure-3 sweep remains blocked pending interpretation of this diagnostic")
    print("Stage 3 Arithmetic precision probe gate: READY FOR REVIEW")
    return 0


if __name__ == "__main__":
    sys.exit(main())
