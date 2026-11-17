#!/usr/bin/env python3
"""Freeze matched cover plan after ALL Stage-7 train/val stegos exist."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from vkr_benchmark.errors import ConfigurationError  # noqa: E402
from vkr_benchmark.steganalysis.dataset import (  # noqa: E402
    cover_plan_from_stegos, jsonl_text, read_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainval-plan', type=Path, default=ROOT/'data/stage7/dataset/trainval_stego_plan.jsonl')
    parser.add_argument('--stage6-index', type=Path, default=ROOT/'data/stage7/dataset/stage6_test_index.jsonl')
    parser.add_argument('--stego-root', type=Path, default=ROOT/'results/stage7/stego_runs')
    parser.add_argument('--stage6-prompts', type=Path, default=ROOT/'data/stage6/main_prompts.jsonl')
    parser.add_argument('--stage7-prompts', type=Path, default=ROOT/'data/stage7/prompts.jsonl')
    parser.add_argument('--output', type=Path, default=ROOT/'data/stage7/dataset/all_cover_plan.jsonl')
    parser.add_argument('--merged-prompts', type=Path, default=ROOT/'data/stage7/dataset/all_prompts.jsonl')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        plan = cover_plan_from_stegos(
            trainval_plan=read_jsonl(args.trainval_plan), stage6_index=read_jsonl(args.stage6_index),
            stego_root=args.stego_root)
        old_prompts = read_jsonl(args.stage6_prompts)
        new_prompts = read_jsonl(args.stage7_prompts)
        merged = old_prompts + new_prompts
        if len(merged) != 152 or len({p['prompt_id'] for p in merged}) != 152:
            raise ConfigurationError('expected 24 frozen + 128 Stage-7 unique prompts')
        plan_prompts = {r['prompt_id'] for r in plan}
        if plan_prompts != {r['prompt_id'] for r in merged}:
            raise ConfigurationError('cover plan/prompts mismatch')
        payload = jsonl_text(plan)
        prompt_payload = jsonl_text(sorted(merged, key=lambda r: r['prompt_id']))
        if not args.dry_run:
            for target, content in ((args.output, payload), (args.merged_prompts, prompt_payload)):
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and target.read_text(encoding='utf-8') != content:
                    raise ConfigurationError(f'will not overwrite different frozen Stage-7 file: {target}')
            args.output.write_text(payload, encoding='utf-8')
            args.merged_prompts.write_text(prompt_payload, encoding='utf-8')
        print(f'Stage-7 matched cover plan: {len(plan)} pairs, 152 prompt groups, dry_run={args.dry_run}')
        return 0
    except Exception as exc:
        print(f'Stage-7 matched cover plan: ERROR ({type(exc).__name__}: {exc})')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
