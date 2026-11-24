#!/usr/bin/env python3
"""Exploratory Stage-7 8-epoch RoBERTa training, isolated from primary 4 epochs.

This is a new run from the pinned pretrained base, NOT a continuation of the
4-epoch weights. Linear learning rate decay is stretched over eight epochs.
Only train/validation is accessible; external Stage-6 test remains sealed.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vkr_benchmark.errors import ConfigurationError  # noqa: E402
from vkr_benchmark.steganalysis.detectors import (  # noqa: E402
    FROZEN_DATASET_SHA256, LLAMA_ID, QWEN_ID,
    choose_validation_threshold, load_detector_training_data,
)
from prepare_stage7_roberta_base import verify_snapshot  # noqa: E402
from train_stage7_roberta import _validation_scores, _write, _slug  # noqa: E402

E8_SCHEMA = 'stage7.roberta_epochs8_ablation.v1'
E8_STUDY = 'exploratory_epochs8_fresh_linear'
PINNED_REVISION = 'e2da8e2f811d1448a5b465c236feacd80ffbac7b'
PINNED_PRIMARY_CONFIG = '2eb26ae60e768b0e55f041c53b2221a87bbb433f357e7523a5665a96a60b17ac'
PINNED_BASE_REPORT = '9acbccb24045fc2697633978cf80aed0ae52bbc3824834f9ed7b1f32e6e6d6f3'
EXPECTED_ABLATION = {
    'schema_version': E8_SCHEMA,
    'study_id': E8_STUDY,
    'dataset_sha256': FROZEN_DATASET_SHA256,
    'primary_configuration_sha256': PINNED_PRIMARY_CONFIG,
    'base_report_sha256': PINNED_BASE_REPORT,
    'base_revision': PINNED_REVISION,
    'max_epochs': 8,
    'early_stopping_patience': 2,
    'training_seeds': [711, 712, 713],
    'start_from': 'pinned_base_not_primary_checkpoint',
    'scheduler': 'fresh_linear_over_eight_epochs',
    'selection_metric': 'pooled_validation_roc_auc',
    'test_access': 'none',
    'primary_results_unchanged': True,
}
FROZEN_PRIMARY_ROBERTA = {
    'base_model': 'FacebookAI/roberta-base', 'base_revision': PINNED_REVISION,
    'max_length': 512, 'truncation': 'right', 'dynamic_padding': True,
    'learning_rate': 0.00002, 'weight_decay': 0.01, 'max_epochs': 4,
    'warmup_ratio': 0.1, 'microbatch': 4, 'gradient_accumulation': 8,
    'training_seeds': [711, 712, 713], 'early_stopping_patience': 2,
}


def _disjoint(path: Path, forbidden: Path) -> bool:
    a, b = path.resolve(), forbidden.resolve()
    return a != b and a not in b.parents and b not in a.parents


def validate_ablation_preflight(*, cfg_path: Path, ablation_path: Path, report_path: Path,
                                snapshot: Path, audit_path: Path,
                                results_root: Path, model_root: Path,
                                dataset: Path, manifest: Path, model_id: str, seed: int,
                                frozen_structure: bool = True) -> tuple[dict, object]:
    """Full frozen input contract and output isolation, no GPU imports or writes."""
    primary_sha = sha256(cfg_path.read_bytes()).hexdigest()
    base_sha = sha256(report_path.read_bytes()).hexdigest()
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    ablation_raw = ablation_path.read_bytes()
    experiment = json.loads(ablation_raw)
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    base_report = json.loads(report_path.read_text(encoding='utf-8'))
    if (primary_sha != PINNED_PRIMARY_CONFIG or
            cfg.get('roberta') != FROZEN_PRIMARY_ROBERTA or
            cfg.get('dataset_sha256') != FROZEN_DATASET_SHA256):
        raise ConfigurationError('primary Stage-7 detector configuration changed')
    if experiment != EXPECTED_ABLATION:
        raise ConfigurationError('8-epoch ablation configuration changed from frozen protocol')
    if base_sha != PINNED_BASE_REPORT or base_report.get('revision') != PINNED_REVISION:
        raise ConfigurationError('pinned RoBERTa source report differs')
    verify_snapshot(snapshot, base_report)
    if (audit.get('status') != 'verified' or audit.get('detector_count') != 6 or
            audit.get('test_access') != 'none' or
            audit.get('dataset_sha256') != FROZEN_DATASET_SHA256 or
            audit.get('primary_configuration_sha256') != primary_sha or
            audit.get('base_report_sha256') != base_sha or
            {(r.get('model_id'), r.get('seed')) for r in audit.get('runs', [])} !=
            {(model, s) for model in (LLAMA_ID, QWEN_ID) for s in (711, 712, 713)}):
        raise ConfigurationError('all six primary models must pass audit first')
    if not _disjoint(results_root, ROOT / 'results/stage7/checkpoints/roberta') or \
            not _disjoint(model_root, ROOT / 'models/detectors/stage7') or \
            not _disjoint(results_root, ROOT / 'results/stage7/checkpoints/roberta_selection'):
        raise ConfigurationError('8-epoch outputs overlap primary checkpoints')
    if seed not in (711, 712, 713) or model_id not in (LLAMA_ID, QWEN_ID):
        raise ConfigurationError('unsupported exploratory model/seed')
    data = load_detector_training_data(dataset=dataset, manifest=manifest,
        model_id=model_id, expected_sha256=FROZEN_DATASET_SHA256,
        require_frozen_structure=frozen_structure)
    info = {
        'status': 'ablation_preflight_ok', 'study_id': E8_STUDY,
        'model_id': model_id, 'seed': seed, 'max_epochs': 8,
        'base_revision': PINNED_REVISION,
        'dataset_sha256': FROZEN_DATASET_SHA256,
        'primary_configuration_sha256': primary_sha,
        'base_report_sha256': base_sha,
        'ablation_configuration_sha256': sha256(ablation_raw).hexdigest(),
        'primary_audit_sha256': sha256(audit_path.read_bytes()).hexdigest(),
        'counts': data.counts, 'test_access': 'none',
        'scheduler': 'fresh_linear_over_eight_epochs',
    }
    return info, data


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-id', required=True, choices=[LLAMA_ID, QWEN_ID])
    p.add_argument('--seed', required=True, type=int, choices=[711, 712, 713])
    p.add_argument('--snapshot', required=True, type=Path)
    p.add_argument('--base-report', type=Path, default=ROOT/'results/stage7/checkpoints/roberta_base_snapshot.json')
    p.add_argument('--dataset', type=Path, default=ROOT/'data/stage7/dataset/detector_records.jsonl')
    p.add_argument('--manifest', type=Path, default=ROOT/'results/stage7/dataset_manifest.json')
    p.add_argument('--config', type=Path, default=ROOT/'configs/stage7/detector_training.json')
    p.add_argument('--ablation-config', type=Path, default=ROOT/'configs/stage7/roberta_epochs8_ablation.json')
    p.add_argument('--primary-audit', type=Path, default=ROOT/'results/stage7/roberta_primary_audit.json')
    p.add_argument('--output-root', type=Path, default=ROOT/'results/stage7/checkpoints/roberta_epochs8')
    p.add_argument('--model-root', type=Path, default=ROOT/'models/detectors/stage7_epochs8')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    try:
        info, data = validate_ablation_preflight(cfg_path=a.config, ablation_path=a.ablation_config,
            report_path=a.base_report, snapshot=a.snapshot, audit_path=a.primary_audit,
            results_root=a.output_root, model_root=a.model_root,
            dataset=a.dataset, manifest=a.manifest, model_id=a.model_id, seed=a.seed)
        slug = _slug(a.model_id)
        result_dir = a.output_root / slug / f'seed-{a.seed}'
        model_dir = a.model_root / slug / 'roberta' / f'seed-{a.seed}'
        if result_dir.exists() or model_dir.exists():
            raise ConfigurationError('8-epoch output already exists; refuse overwrite')
        if a.dry_run:
            print(json.dumps(info, indent=2, sort_keys=True))
            return 0
        try:
            import numpy as np
            import torch
            from sklearn.metrics import roc_auc_score
            from torch.utils.data import DataLoader
            from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                                      DataCollatorWithPadding, get_linear_schedule_with_warmup)
        except ImportError as exc:
            raise ConfigurationError('8-epoch study requires numpy, torch, sklearn, transformers') from exc
        random.seed(a.seed)
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(a.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
            torch.backends.cuda.matmul.allow_tf32 = False
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type != 'cuda':
            raise ConfigurationError('exploratory training requires CUDA GPU (no silent CPU fallback)')
        tokenizer = AutoTokenizer.from_pretrained(str(a.snapshot), local_files_only=True, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(str(a.snapshot),
            local_files_only=True, num_labels=2)
        model.to(device)
        train, validation = list(data.train), list(data.validation)
        def encode(rows):
            return [dict(tokenizer(row['text'], max_length=512, truncation=True,
                                   add_special_tokens=True), labels=row['label']) for row in rows]
        collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors='pt')
        train_loader = DataLoader(encode(train), batch_size=4, shuffle=True,
            collate_fn=collator, generator=torch.Generator().manual_seed(a.seed))
        val_loader = DataLoader(encode(validation), batch_size=4, shuffle=False, collate_fn=collator)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        steps_per_epoch = math.ceil(len(train_loader) / 8)
        total_steps = steps_per_epoch * 8
        warmup_steps = math.ceil(total_steps * 0.1)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                     num_training_steps=total_steps)
        labels = [row['label'] for row in validation]
        best_auc, best_epoch, best_scores, patience = -1.0, None, None, 0
        history = []
        for epoch in range(1, 9):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_total = 0.0
            batches = len(train_loader)
            for step, batch in enumerate(train_loader):
                batch = {k:v.to(device) for k,v in batch.items()}
                result = model(**batch)
                group_size = min(8, batches - (step // 8) * 8)
                (result.loss / group_size).backward()
                loss_total += float(result.loss.detach().cpu())
                if (step + 1) % 8 == 0 or step + 1 == batches:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            scores = _validation_scores(model, val_loader, device, torch)
            auc = float(roc_auc_score(labels, scores))
            history.append({'epoch': epoch, 'validation_roc_auc': auc,
                            'mean_batch_loss': loss_total / batches})
            print(f'Stage-7 RoBERTa E8 {slug} seed={a.seed} epoch={epoch}: '
                  f'validation ROC-AUC={auc:.6f}', flush=True)
            if auc > best_auc + 1e-12:
                best_auc, best_epoch, best_scores, patience = auc, epoch, scores, 0
                model_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(model_dir, safe_serialization=True)
                tokenizer.save_pretrained(model_dir)
            else:
                patience += 1
                if patience >= 2:
                    break
        if best_scores is None:
            raise ConfigurationError('8-epoch validation never produced scores')
        threshold, bacc = choose_validation_threshold(labels, best_scores)
        summary = {
            **info, 'status': 'complete', 'best_epoch': best_epoch,
            'validation_roc_auc': best_auc, 'validation_threshold': threshold,
            'validation_balanced_accuracy': bacc, 'history': history,
            'optimizer': 'AdamW', 'microbatch': 4, 'gradient_accumulation': 8,
            'effective_batch': 32, 'warmup_steps': warmup_steps,
            'total_scheduled_steps': total_steps, 'device': str(device),
            'torch_version': torch.__version__,
        }
        result_dir.mkdir(parents=True, exist_ok=False)
        _write(result_dir/'summary.json', summary)
        predictions = [
            {'record_id': row['record_id'], 'pair_id': row['pair_id'],
             'label': row['label'], 'partition': 'validation', 'score_stego': score}
            for row, score in zip(validation, best_scores)
        ]
        (result_dir/'validation_predictions.jsonl').write_text(
            ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True)+'\n'
                    for row in predictions), encoding='utf-8')
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, ConfigurationError) as exc:
        print(f'Stage-7 RoBERTa E8: ERROR ({type(exc).__name__}: {exc})', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
