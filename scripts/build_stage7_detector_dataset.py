#!/usr/bin/env python3
"""Validate all 5640 matched pairs and write local detector JSONL + safe manifest."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from vkr_benchmark.errors import ConfigurationError  # noqa: E402
from vkr_benchmark.steganalysis.dataset import (  # noqa: E402
    jsonl_text, paired_dataset, read_json, read_jsonl,
)


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cover-plan', type=Path, default=ROOT/'data/stage7/dataset/all_cover_plan.jsonl')
    parser.add_argument('--cover-root', type=Path, default=ROOT/'results/stage7/cover_runs')
    parser.add_argument('--stego-root', type=Path, default=ROOT/'results/stage7/stego_runs')
    parser.add_argument('--stage6-run-root', type=Path, default=ROOT/'results/runs')
    parser.add_argument('--split-manifest', type=Path, default=ROOT/'results/stage7/splits/split_manifest.json')
    parser.add_argument('--output', type=Path, default=ROOT/'data/stage7/dataset/detector_records.jsonl')
    parser.add_argument('--manifest-output', type=Path, default=ROOT/'results/stage7/dataset_manifest.json')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        samples, manifest = paired_dataset(
            cover_plan=read_jsonl(args.cover_plan), cover_root=args.cover_root,
            stego_root=args.stego_root, stage6_run_root=args.stage6_run_root,
            split_manifest=read_json(args.split_manifest),
        )
        content = jsonl_text(samples)
        manifest['dataset_sha256'] = hashlib.sha256(content.encode('utf-8')).hexdigest()
        manifest['cover_plan_sha256'] = _sha_file(args.cover_plan)
        manifest['split_manifest_sha256'] = _sha_file(args.split_manifest)
        manifest['stage6_frozen_manifest_sha256'] = _sha_file(ROOT/'results/stage6/dataset_v1.0_manifest.json')
        if not args.dry_run:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if args.output.exists() and args.output.read_text(encoding='utf-8') != content:
                raise ConfigurationError('existing detector dataset differs; refuse to overwrite')
            args.output.write_text(content, encoding='utf-8')
            args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
            metadata = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + '\n'
            if args.manifest_output.exists() and args.manifest_output.read_text(encoding='utf-8') != metadata:
                raise ConfigurationError('existing Stage-7 dataset manifest differs')
            args.manifest_output.write_text(metadata, encoding='utf-8')
        print(json.dumps({**manifest, 'dry_run': args.dry_run}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f'Stage-7 detector dataset: ERROR ({type(exc).__name__}: {exc})')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
