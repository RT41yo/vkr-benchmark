#!/usr/bin/env python3
"""Compare six exploratory 8-epoch validation runs with six frozen primary runs.

Validation-only diagnostic: never selects/replaces primary RoBERTa or opens test.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vkr_benchmark.errors import ConfigurationError  # noqa: E402
from vkr_benchmark.steganalysis.detectors import FROZEN_DATASET_SHA256, LLAMA_ID, QWEN_ID  # noqa: E402
from train_stage7_roberta_epochs8 import (  # noqa: E402
    E8_STUDY, EXPECTED_ABLATION, PINNED_PRIMARY_CONFIG, PINNED_BASE_REPORT,
)
from audit_stage7_roberta_primary import file_hash  # noqa: E402


def compare(*, primary_audit: Path, primary_results: Path, e8_results: Path,
            ablation_config: Path) -> dict:
    audit = json.loads(primary_audit.read_text(encoding='utf-8'))
    if (audit.get('status') != 'verified' or audit.get('detector_count') != 6 or
            audit.get('test_access') != 'none' or
            audit.get('dataset_sha256') != FROZEN_DATASET_SHA256 or
            audit.get('primary_configuration_sha256') != PINNED_PRIMARY_CONFIG or
            audit.get('base_report_sha256') != PINNED_BASE_REPORT or
            json.loads(ablation_config.read_text(encoding='utf-8')) != EXPECTED_ABLATION):
        raise ConfigurationError('primary audit/ablation configuration differs from frozen inputs')
    output = []
    for model_id, slug in ((LLAMA_ID, 'llama-3.2-3b'), (QWEN_ID, 'qwen3-4b-base')):
        pairs = []
        for seed in (711, 712, 713):
            p_path = primary_results / slug / f'seed-{seed}' / 'summary.json'
            a_path = e8_results / slug / f'seed-{seed}' / 'summary.json'
            p = json.loads(p_path.read_text(encoding='utf-8'))
            a = json.loads(a_path.read_text(encoding='utf-8'))
            audit_row = next((r for r in audit['runs']
                              if r.get('model_id') == model_id and r.get('seed') == seed), None)
            if (audit_row is None or file_hash(p_path) != audit_row['summary_sha256'] or
                    p.get('validation_roc_auc') != audit_row['validation_roc_auc'] or
                    a.get('status') != 'complete' or a.get('study_id') != E8_STUDY or
                    a.get('model_id') != model_id or a.get('seed') != seed or
                    a.get('test_access') != 'none' or a.get('device') != 'cuda' or
                    a.get('max_epochs') != 8 or
                    a.get('dataset_sha256') != FROZEN_DATASET_SHA256 or
                    a.get('primary_configuration_sha256') != PINNED_PRIMARY_CONFIG or
                    a.get('base_report_sha256') != PINNED_BASE_REPORT or
                    a.get('ablation_configuration_sha256') != file_hash(ablation_config) or
                    a.get('primary_audit_sha256') != file_hash(primary_audit) or
                    a.get('counts') != p.get('counts')):
                raise ConfigurationError(f'inconsistent primary/E8 pair: {model_id}, {seed}')
            history = a.get('history')
            if (not isinstance(history, list) or not (1 <= len(history) <= 8) or
                    [h.get('epoch') for h in history] != list(range(1, len(history)+1)) or
                    max(history, key=lambda h: h['validation_roc_auc'])['epoch'] != a.get('best_epoch') or
                    abs(max(h['validation_roc_auc'] for h in history) - a['validation_roc_auc']) > 1e-10):
                raise ConfigurationError(f'bad ablation epoch selection: {model_id}, {seed}')
            pairs.append({
                'seed': seed,
                'primary_best_epoch': p['best_epoch'],
                'primary_validation_roc_auc': p['validation_roc_auc'],
                'ablation_best_epoch': a['best_epoch'],
                'ablation_validation_roc_auc': a['validation_roc_auc'],
                'ablation_epochs_executed': len(history),
                'ablation_minus_primary_roc_auc': a['validation_roc_auc'] - p['validation_roc_auc'],
                'primary_summary_sha256': file_hash(p_path),
                'ablation_summary_sha256': file_hash(a_path),
            })
        primary_mean = statistics.mean(r['primary_validation_roc_auc'] for r in pairs)
        ablation_mean = statistics.mean(r['ablation_validation_roc_auc'] for r in pairs)
        output.append({
            'model_id': model_id,
            'primary_mean_validation_roc_auc': primary_mean,
            'ablation_mean_validation_roc_auc': ablation_mean,
            'ablation_minus_primary_mean_roc_auc': ablation_mean - primary_mean,
            'per_seed': pairs,
        })
    return {
        'schema_version': 'stage7.roberta_epochs8_comparison.v1',
        'status': 'validation_only_exploratory',
        'test_access': 'none',
        'primary_checkpoint_selection_changed': False,
        'scheduler_caveat': '8-epoch linear LR schedule differs from 4-epoch primary LR schedule',
        'dataset_sha256': FROZEN_DATASET_SHA256,
        'primary_audit_sha256': file_hash(primary_audit),
        'ablation_config_sha256': file_hash(ablation_config),
        'models': output,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--primary-audit', type=Path, default=ROOT/'results/stage7/roberta_primary_audit.json')
    p.add_argument('--primary-results', type=Path, default=ROOT/'results/stage7/checkpoints/roberta')
    p.add_argument('--ablation-results', type=Path, default=ROOT/'results/stage7/checkpoints/roberta_epochs8')
    p.add_argument('--ablation-config', type=Path, default=ROOT/'configs/stage7/roberta_epochs8_ablation.json')
    p.add_argument('--output', type=Path, default=ROOT/'results/stage7/roberta_epochs8_comparison.json')
    p.add_argument('--write-report', action='store_true')
    a = p.parse_args()
    try:
        result = compare(primary_audit=a.primary_audit, primary_results=a.primary_results,
                         e8_results=a.ablation_results, ablation_config=a.ablation_config)
        content = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True)+'\n'
        if a.write_report:
            if a.output.exists() and a.output.read_text(encoding='utf-8') != content:
                raise ConfigurationError('existing E8 comparison differs; refuse overwrite')
            a.output.parent.mkdir(parents=True, exist_ok=True)
            a.output.write_text(content, encoding='utf-8')
        print(content, end='')
        return 0
    except (OSError, KeyError, ValueError, ConfigurationError) as exc:
        print(f'Stage-7 RoBERTa E8 comparison: ERROR ({type(exc).__name__}: {exc})', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
