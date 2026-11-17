#!/usr/bin/env python3
"""Audit Stage-6 canonical texts read-only and prepare local external test index."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vkr_benchmark.steganalysis.stage6 import build_stage6_test_index  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=ROOT / "results/runs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        frozen = json.loads((ROOT / "results/stage6/dataset_v1.0_manifest.json").read_text())
        prompts = json.loads((ROOT / "results/stage5/stage6_prompt_manifest.json").read_text())
        rows, audit = build_stage6_test_index(run_root=args.run_root, frozen_manifest=frozen,
                                               stage6_prompts=prompts)
        if not args.dry_run:
            target = ROOT / "data/stage7/dataset/stage6_test_index.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
            report = ROOT / "results/stage7/stage6_source_audit.json"
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(audit, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Stage-7 source audit: ERROR ({type(exc).__name__}: {exc})")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
