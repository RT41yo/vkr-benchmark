#!/usr/bin/env python3
"""Expand frozen 15 operating points × 128 new articles × 2 models.

No LM weights loaded; does not touch canonical Stage-6 artifacts. The plan is
local and includes method RNG/key material; never commit it to Git.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vkr_benchmark.steganalysis.plan import build_trainval_stego_plan  # noqa: E402
from vkr_benchmark.errors import ConfigurationError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--split-manifest', type=Path, default=ROOT/'results/stage7/splits/split_manifest.json')
    parser.add_argument('--frozen-manifest', type=Path, default=ROOT/'results/stage6/dataset_v1.0_manifest.json')
    parser.add_argument('--output', type=Path, default=ROOT/'data/stage7/dataset/trainval_stego_plan.jsonl')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        split = json.loads(args.split_manifest.read_text(encoding='utf-8'))
        frozen = json.loads(args.frozen_manifest.read_text(encoding='utf-8'))
        rows = build_trainval_stego_plan(split_manifest=split, frozen_manifest=frozen, project_root=ROOT)
        payload = ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows)
        h = hashlib.sha256(payload.encode()).hexdigest()
        if not args.dry_run:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if args.output.exists() and args.output.read_text(encoding='utf-8') != payload:
                raise ConfigurationError('Stage-7 trainval plan differs from existing file; refuse to overwrite')
            args.output.write_text(payload, encoding='utf-8')
        print(f"Stage-7 train/val plan: {len(rows)} runs (train=2880 validation=960), sha256={h}, dry_run={args.dry_run}")
        return 0
    except Exception as exc:
        print(f'Stage-7 train/val plan: ERROR ({type(exc).__name__}: {exc})')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
