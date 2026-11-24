#!/usr/bin/env python3
"""Pre-test Stage-7 final evaluation lock: only validation artifacts and hashes.

This command NEVER reads dataset test texts, loads trained weights or scores test.
--write-lock creates a small reproducible JSON suitable for commit before test.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vkr_benchmark.errors import ConfigurationError  # noqa: E402
from vkr_benchmark.steganalysis.final_evaluation import (  # noqa: E402
    make_lock, canonical_bytes, file_sha,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, default=ROOT / 'configs/stage7/final_evaluation.json')
    p.add_argument('--detector-config', type=Path, default=ROOT / 'configs/stage7/detector_training.json')
    p.add_argument('--primary-audit', type=Path, default=ROOT / 'results/stage7/roberta_primary_audit.json')
    p.add_argument('--manifest', type=Path, default=ROOT / 'results/stage7/dataset_manifest.json')
    p.add_argument('--baseline-results', type=Path, default=ROOT / 'results/stage7/checkpoints/baseline')
    p.add_argument('--roberta-results', type=Path, default=ROOT / 'results/stage7/checkpoints/roberta')
    p.add_argument('--models-root', type=Path, default=ROOT / 'models/detectors/stage7')
    p.add_argument('--lock', type=Path, default=ROOT / 'results/stage7/final_evaluation_lock.json')
    p.add_argument('--write-lock', action='store_true')
    args = p.parse_args()
    try:
        lock = make_lock(protocol=args.protocol, detector_config=args.detector_config,
            primary_audit=args.primary_audit, dataset_manifest=args.manifest,
            baseline_results=args.baseline_results, roberta_results=args.roberta_results,
            models_root=args.models_root)
        content = canonical_bytes(lock)
        digest = sha256(content).hexdigest()
        if args.lock.exists() and args.lock.read_bytes() != content:
            raise ConfigurationError('existing pre-test lock differs; refusing overwrite')
        if args.write_lock:
            args.lock.parent.mkdir(parents=True, exist_ok=True)
            args.lock.write_bytes(content)
        print(json.dumps({
            'status': 'sealed_pretest', 'test_access': 'none',
            'lock_sha256': digest, 'lock_written': args.write_lock,
            'dataset_sha256': lock['dataset_sha256'],
            'selected': [{'model_id': r['model_id'], 'roberta_seed': r['roberta_e4']['seed'],
                          'roberta_epoch': r['roberta_e4']['epoch'],
                          'roberta_validation_roc_auc': r['roberta_e4']['validation_roc_auc'],
                          'baseline_validation_roc_auc': r['baseline']['validation_roc_auc']}
                         for r in lock['selected']],
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, ConfigurationError) as exc:
        print(f'Stage-7 final evaluation lock: ERROR ({type(exc).__name__}: {exc})',file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
