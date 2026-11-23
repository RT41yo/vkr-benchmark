#!/usr/bin/env python3
"""Pin and hash the RoBERTa-base source snapshot before any Stage-7 GPU training.

Requires an explicitly supplied 40-hex Hugging Face revision, never floating main.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vkr_benchmark.errors import ConfigurationError  # noqa: E402

REQUIRED = ('config.json', 'tokenizer.json', 'vocab.json', 'merges.txt')
WEIGHTS = ('model.safetensors', 'pytorch_model.bin')
INCLUDE = REQUIRED + WEIGHTS + ('tokenizer_config.json','special_tokens_map.json')


def manifest_for_snapshot(snapshot: Path, revision: str) -> dict:
    if re.fullmatch(r'[0-9a-f]{40}', revision) is None:
        raise ConfigurationError('RoBERTa base revision must be a full 40-character commit SHA')
    if not snapshot.is_dir():
        raise ConfigurationError('RoBERTa snapshot directory does not exist')
    missing = [name for name in REQUIRED if not (snapshot/name).is_file()]
    if missing or not any((snapshot/name).is_file() for name in WEIGHTS):
        raise ConfigurationError(f'RoBERTa snapshot missing mandatory assets: {missing}; weights required')
    files = {}
    for name in INCLUDE:
        path = snapshot/name
        if path.is_file():
            files[name] = sha256(path.read_bytes()).hexdigest()
    return {'schema_version':'stage7.roberta_base_snapshot.v1',
            'base_model':'FacebookAI/roberta-base', 'revision':revision,
            'asset_sha256':files, 'status':'pinned'}


def verify_snapshot(snapshot: Path, report: dict) -> None:
    actual = manifest_for_snapshot(snapshot, report['revision'])
    if report != actual:
        raise ConfigurationError('RoBERTa model/tokenizer snapshot has changed')


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--revision', required=True, help='Full 40-character immutable Hugging Face commit SHA')
    p.add_argument('--snapshot', type=Path, help='Already downloaded snapshot directory')
    p.add_argument('--report', type=Path, default=ROOT/'results/stage7/checkpoints/roberta_base_snapshot.json')
    p.add_argument('--download', action='store_true', help='Fetch immutable revision into HF cache')
    a = p.parse_args()
    try:
        if re.fullmatch(r'[0-9a-f]{40}', a.revision) is None:
            raise ConfigurationError('RoBERTa revision must be full immutable commit SHA')
        if a.download == bool(a.snapshot):
            raise ConfigurationError('provide exactly one of --download or --snapshot')
        if a.download:
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise ConfigurationError('huggingface-hub is required for snapshot download') from exc
            snapshot = Path(snapshot_download(repo_id='FacebookAI/roberta-base',revision=a.revision,
                allow_patterns=list(INCLUDE)))
        else:
            snapshot = a.snapshot
        report = manifest_for_snapshot(snapshot, a.revision)
        if a.report.exists():
            stored = json.loads(a.report.read_text(encoding='utf-8'))
            if report != stored:
                raise ConfigurationError('existing base snapshot pin differs; refuse overwrite')
        else:
            a.report.parent.mkdir(parents=True, exist_ok=True)
            a.report.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
        print(json.dumps({'status':'pinned','snapshot_path':str(snapshot),
                          'report_path':str(a.report),'revision':a.revision,
                          'assets':sorted(report['asset_sha256'])},indent=2))
        return 0
    except (OSError,ValueError,ConfigurationError) as exc:
        print(f'Stage-7 RoBERTa base preflight: ERROR ({type(exc).__name__}: {exc})',file=sys.stderr)
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
