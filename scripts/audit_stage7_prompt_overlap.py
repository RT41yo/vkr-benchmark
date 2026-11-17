#!/usr/bin/env python3
"""Flag cross-partition near-duplicate CNN/DailyMail prompts; never reshuffle."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from vkr_benchmark.steganalysis.dataset import read_json, read_jsonl  # noqa: E402
from vkr_benchmark.steganalysis.leakage import audit_prompt_overlap  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--split-manifest', type=Path, default=ROOT/'results/stage7/splits/split_manifest.json')
    parser.add_argument('--stage6-prompts', type=Path, default=ROOT/'data/stage6/main_prompts.jsonl')
    parser.add_argument('--stage7-prompts', type=Path, default=ROOT/'data/stage7/prompts.jsonl')
    parser.add_argument('--output', type=Path, default=ROOT/'results/stage7/splits/prompt_overlap_audit.json')
    parser.add_argument('--threshold', type=float, default=0.85)
    args = parser.parse_args()
    try:
        audit = audit_prompt_overlap(
            split_manifest=read_json(args.split_manifest),
            stage6_prompts=read_jsonl(args.stage6_prompts),
            stage7_prompts=read_jsonl(args.stage7_prompts), threshold=args.threshold)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2)+'\n'
        if args.output.exists() and args.output.read_text(encoding='utf-8') != text:
            raise ValueError(f'overlap audit already exists with different result: {args.output}')
        args.output.write_text(text, encoding='utf-8')
        print(f"Stage-7 cross-split near-duplicate audit: {audit['status']} "
              f"(candidates={audit['cross_split_candidate_count']})")
        return 0 if audit['status']=='ok' else 2
    except Exception as exc:
        print(f'Stage-7 prompt overlap audit: ERROR ({type(exc).__name__}: {exc})')
        return 1


if __name__=='__main__':
    raise SystemExit(main())
