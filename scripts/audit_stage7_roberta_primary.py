#!/usr/bin/env python3
"""Read-only audit of all six frozen Stage-7 primary RoBERTa checkpoints.

Validates *validation* scores and model artifacts; never scores test.
The frozen dataset loader may read test bytes for source-integrity checks,
but never returns test text to the audit or to a detector.
Optionally writes a compact, content-addressed audit report (no carrier text).
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vkr_benchmark.errors import ConfigurationError  # noqa: E402
from vkr_benchmark.steganalysis.detectors import (  # noqa: E402
    FROZEN_DATASET_SHA256, LLAMA_ID, QWEN_ID,
    choose_validation_threshold, load_detector_training_data,
)

MODEL_SLUGS = ((LLAMA_ID, 'llama-3.2-3b'), (QWEN_ID, 'qwen3-4b-base'))
REQUIRED_FILES = ('config.json', 'model.safetensors', 'tokenizer.json',
                  'tokenizer_config.json', 'vocab.json', 'merges.txt')


def file_hash(path: Path) -> str:
    value = sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ConfigurationError(f'expected JSON object: {path}')
    return value


def _near(actual: object, expected: float, *, what: str) -> None:
    if not isinstance(actual, (int, float)) or isinstance(actual, bool) or not math.isfinite(actual) or \
       not math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-10):
        raise ConfigurationError(f'{what} differs from validation predictions')


def audit_primary(*, dataset: Path, manifest: Path, config: Path,
                  base_report: Path, results_root: Path, model_root: Path,
                  frozen_sha256: str = FROZEN_DATASET_SHA256,
                  frozen_structure: bool = True) -> dict:
    """Audit without writing or loading model weights. Return safe metadata only."""
    from sklearn.metrics import roc_auc_score

    config_sha = file_hash(config)
    cfg = _read_json(config)
    revision = cfg.get('roberta', {}).get('base_revision')
    base_sha = file_hash(base_report)
    report = _read_json(base_report)
    if (cfg.get('schema_version') != 'stage7.detector_training.v1' or
            cfg.get('dataset_sha256') != frozen_sha256 or
            not isinstance(revision, str) or len(revision) != 40 or
            any(c not in '0123456789abcdef' for c in revision) or
            report.get('revision') != revision or report.get('status') != 'pinned'):
        raise ConfigurationError('primary config/revision/base report mismatch')
    if cfg['roberta'].get('max_epochs') != 4 or cfg['roberta'].get('training_seeds') != [711, 712, 713]:
        raise ConfigurationError('primary training protocol changed')
    audited = []
    for model_id, slug in MODEL_SLUGS:
        data = load_detector_training_data(dataset=dataset, manifest=manifest,
            model_id=model_id, expected_sha256=frozen_sha256,
            require_frozen_structure=frozen_structure)
        validation_by_id = {row['record_id']: row for row in data.validation}
        for seed in (711, 712, 713):
            result_dir = results_root / slug / f'seed-{seed}'
            model_dir = model_root / slug / 'roberta' / f'seed-{seed}'
            summary_path = result_dir / 'summary.json'
            predictions_path = result_dir / 'validation_predictions.jsonl'
            summary = _read_json(summary_path)
            if any((summary.get(key) != value) for key, value in {
                'status': 'complete', 'seed': seed, 'model_id': model_id,
                'dataset_sha256': frozen_sha256,
                'configuration_sha256': config_sha,
                'base_report_sha256': base_sha,
                'base_revision': revision, 'test_access': 'none',
                'max_epochs': 4, 'counts': data.counts,
                'microbatch': 4, 'gradient_accumulation': 8,
                'effective_batch': 32,
            }.items()):
                raise ConfigurationError(f'inconsistent primary training summary: {model_id} seed={seed}')
            history = summary.get('history')
            if not isinstance(history, list) or not (1 <= len(history) <= 4):
                raise ConfigurationError('invalid primary training history')
            best = max(history, key=lambda row: row['validation_roc_auc'])
            if [row.get('epoch') for row in history] != list(range(1, len(history) + 1)) or \
                    best['epoch'] != summary.get('best_epoch'):
                raise ConfigurationError('primary best epoch differs from history')
            _near(summary.get('validation_roc_auc'), best['validation_roc_auc'], what='best-epoch ROC-AUC')
            if not predictions_path.is_file():
                raise ConfigurationError(f'missing validation predictions: {predictions_path}')
            scores_by_id = {}
            for line in predictions_path.read_text(encoding='utf-8').splitlines():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ConfigurationError('non-object validation prediction')
                record_id = row.get('record_id')
                source = validation_by_id.get(record_id)
                score = row.get('score_stego')
                if (source is None or record_id in scores_by_id or
                        row.get('partition') != 'validation' or
                        row.get('label') != source['label'] or
                        row.get('pair_id') != source['pair_id'] or
                        not isinstance(score, (int, float)) or isinstance(score, bool) or
                        not math.isfinite(score) or not (0 <= score <= 1)):
                    raise ConfigurationError(f'invalid validation prediction: {model_id} seed={seed}')
                scores_by_id[record_id] = float(score)
            if len(scores_by_id) != len(validation_by_id):
                raise ConfigurationError(f'incomplete validation predictions: {model_id} seed={seed}')
            labels = [row['label'] for row in data.validation]
            scores = [scores_by_id[row['record_id']] for row in data.validation]
            auc = float(roc_auc_score(labels, scores))
            threshold, bacc = choose_validation_threshold(labels, scores)
            _near(summary.get('validation_roc_auc'), auc, what='validation ROC-AUC')
            _near(summary.get('validation_threshold'), threshold, what='validation threshold')
            _near(summary.get('validation_balanced_accuracy'), bacc, what='balanced accuracy')
            if not model_dir.is_dir():
                raise ConfigurationError(f'missing primary model checkpoint: {model_dir}')
            hashes = {}
            for filename in REQUIRED_FILES:
                path = model_dir / filename
                if not path.is_file() or path.stat().st_size == 0:
                    raise ConfigurationError(f'missing or empty checkpoint file: {path}')
                hashes[filename] = file_hash(path)
            audited.append({
                'model_id': model_id, 'seed': seed,
                'best_epoch': summary['best_epoch'],
                'validation_roc_auc': auc,
                'validation_threshold': threshold,
                'validation_balanced_accuracy': bacc,
                'validation_count': len(data.validation),
                'summary_sha256': file_hash(summary_path),
                'predictions_sha256': file_hash(predictions_path),
                'checkpoint_asset_sha256': hashes,
            })
    return {
        'schema_version': 'stage7.roberta_primary_audit.v1',
        'status': 'verified', 'test_access': 'none',
        'dataset_sha256': frozen_sha256,
        'primary_configuration_sha256': config_sha,
        'base_report_sha256': base_sha, 'base_revision': revision,
        'detector_count': len(audited),
        'runs': audited,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=ROOT/'data/stage7/dataset/detector_records.jsonl')
    p.add_argument('--manifest', type=Path, default=ROOT/'results/stage7/dataset_manifest.json')
    p.add_argument('--config', type=Path, default=ROOT/'configs/stage7/detector_training.json')
    p.add_argument('--base-report', type=Path, default=ROOT/'results/stage7/checkpoints/roberta_base_snapshot.json')
    p.add_argument('--results-root', type=Path, default=ROOT/'results/stage7/checkpoints/roberta')
    p.add_argument('--model-root', type=Path, default=ROOT/'models/detectors/stage7')
    p.add_argument('--report', type=Path, default=ROOT/'results/stage7/roberta_primary_audit.json')
    p.add_argument('--write-report', action='store_true', help='Write content-addressed report after complete audit')
    a = p.parse_args()
    try:
        report = audit_primary(dataset=a.dataset, manifest=a.manifest, config=a.config,
            base_report=a.base_report, results_root=a.results_root, model_root=a.model_root)
        content = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + '\n'
        if a.write_report:
            if a.report.exists() and a.report.read_text(encoding='utf-8') != content:
                raise ConfigurationError('existing audit report differs; refuse overwrite')
            a.report.parent.mkdir(parents=True, exist_ok=True)
            a.report.write_text(content, encoding='utf-8')
        print(json.dumps({
            'status': report['status'], 'detector_count': report['detector_count'],
            'dataset_sha256': report['dataset_sha256'],
            'audit_sha256': sha256(content.encode('utf-8')).hexdigest(),
            'report_written': a.write_report,
            'selected': [
                {'model_id': model_id, 'seed': max((r for r in report['runs'] if r['model_id'] == model_id),
                     key=lambda r: (r['validation_roc_auc'], -r['best_epoch'], -r['seed']))['seed']}
                for model_id, _ in MODEL_SLUGS
            ],
        }, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, ConfigurationError) as exc:
        print(f'Stage-7 RoBERTa primary audit: ERROR ({type(exc).__name__}: {exc})', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
