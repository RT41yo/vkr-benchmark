#!/usr/bin/env python3
"""Select a single RoBERTa seed by pooled validation ROC-AUC, never by test."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vkr_benchmark.errors import ConfigurationError  # noqa: E402
from vkr_benchmark.steganalysis.detectors import LLAMA_ID,QWEN_ID,FROZEN_DATASET_SHA256  # noqa: E402


def select_summary(rows: list[dict]) -> dict:
    if {row.get('seed') for row in rows} != {711,712,713} or len(rows) != 3:
        raise ConfigurationError('selection requires exactly three frozen training seeds')
    if any(row.get('status')!='complete' or row.get('dataset_sha256')!=FROZEN_DATASET_SHA256 or
           row.get('test_access')!='none' or row.get('best_epoch') not in (1,2,3,4)
           for row in rows):
        raise ConfigurationError('incomplete or contaminated RoBERTa run')
    if len({row['model_id'] for row in rows})!=1 or len({row['base_revision'] for row in rows})!=1 or \
       len({row['base_report_sha256'] for row in rows})!=1 or len({row['configuration_sha256'] for row in rows})!=1:
        raise ConfigurationError('RoBERTa seeds were trained with inconsistent inputs')
    for row in rows:
        auc = row['validation_roc_auc']
        if not isinstance(auc,(int,float)) or not (0 <= auc <= 1):
            raise ConfigurationError('invalid validation ROC-AUC')
    selected = min(rows, key=lambda row:(-row['validation_roc_auc'],row['best_epoch'],row['seed']))
    return {'schema_version':'stage7.roberta_selection.v1',
            'model_id':selected['model_id'],
            'selected_seed':selected['seed'], 'selected_epoch':selected['best_epoch'],
            'validation_roc_auc':selected['validation_roc_auc'],
            'validation_threshold':selected['validation_threshold'],
            'validation_balanced_accuracy':selected['validation_balanced_accuracy'],
            'dataset_sha256':FROZEN_DATASET_SHA256,
            'base_revision':selected['base_revision'],
            'config_sha256':selected['configuration_sha256'],
            'test_used_for_selection':False,
            'all_seeds':[{'seed':r['seed'],'best_epoch':r['best_epoch'],
                          'validation_roc_auc':r['validation_roc_auc']} for r in sorted(rows,key=lambda x:x['seed'])]}


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-id',required=True,choices=[LLAMA_ID,QWEN_ID])
    p.add_argument('--results-root',type=Path,default=ROOT/'results/stage7/checkpoints/roberta')
    p.add_argument('--selection-root',type=Path,default=ROOT/'results/stage7/checkpoints/roberta_selection')
    a=p.parse_args()
    slug='llama-3.2-3b' if a.model_id==LLAMA_ID else 'qwen3-4b-base'
    try:
        rows=[]
        for seed in (711,712,713):
            path=a.results_root/slug/f'seed-{seed}'/'summary.json'
            rows.append(json.loads(path.read_text(encoding='utf-8')))
        result=select_summary(rows)
        if result['model_id']!=a.model_id:
            raise ConfigurationError('selection model mismatch')
        target=a.selection_root/f'{slug}.json'
        content=json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+'\n'
        if target.exists() and target.read_text(encoding='utf-8')!=content:
            raise ConfigurationError('selected checkpoint already exists with different content')
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(content,encoding='utf-8')
        print(content,end='')
        return 0
    except (OSError,ValueError,ConfigurationError) as exc:
        print(f'Stage-7 RoBERTa selection: ERROR ({type(exc).__name__}: {exc})',file=sys.stderr)
        return 1

if __name__=='__main__':
    raise SystemExit(main())
