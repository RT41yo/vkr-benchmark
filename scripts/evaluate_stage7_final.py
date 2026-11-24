#!/usr/bin/env python3
"""Locked Stage-7 frozen-test ROC-AUC and article-cluster bootstrap (E4 primary only).

No test rows are read in --dry-run. Full evaluation requires --open-frozen-test,
a SHA-256 of a previously committed lock, and verification of the lock's
content in HEAD. Both detectors are text-only and use validation thresholds.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vkr_benchmark.errors import ConfigurationError  # noqa: E402
from vkr_benchmark.steganalysis.final_evaluation import (  # noqa: E402
    MODEL_SLUGS, canonical_bytes, make_lock, read_json, file_sha, validate_protocol,
    load_frozen_test, bootstrap_article_weights, evaluate_scored_rows,
)


def _committed(path: Path) -> None:
    """Check exact file bytes are in HEAD before touching test labels or text."""
    try:
        relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
        result = subprocess.run(['git', 'show', f'HEAD:{relative}'], cwd=ROOT,
                                check=True, capture_output=True)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        raise ConfigurationError(f'final evaluation file must be committed before test: {path}') from exc
    if result.stdout != path.read_bytes():
        raise ConfigurationError(f'final evaluation file differs from committed HEAD: {path}')


def _roberta_scores(rows: list[dict], model_dir: Path, *, batch_size: int,
                    max_length: int) -> tuple[list[float], list[bool]]:
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as exc:
        raise ConfigurationError('RoBERTa scoring requires torch and transformers') from exc
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True)
    if model.config.num_labels != 2:
        raise ConfigurationError('selected RoBERTa must have two labels')
    model.to(device)
    model.eval()
    scores, trunc = [], []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            subset = rows[start:start + batch_size]
            texts = [r['text'] for r in subset]
            # Text only; token counts are a diagnostic (not detector input).
            original = tok(texts, add_special_tokens=True, truncation=False)['input_ids']
            trunc.extend(len(ids) > max_length for ids in original)
            batch = tok(texts, add_special_tokens=True, truncation=True,
                        max_length=max_length, padding=True, return_tensors='pt')
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            scores.extend(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist())
    return scores, trunc


def _baseline_scores(rows: list[dict], path: Path, lock_meta: dict) -> list[float]:
    import joblib
    saved = joblib.load(path)
    if (saved.get('dataset_sha256') != lock_meta['dataset_sha256'] or
            saved.get('config_sha256') != lock_meta['historic_training_config_sha256']):
        raise ConfigurationError('baseline fitted model metadata differs from validation lock')
    vectorizer, clf = saved['vectorizer'], saved['classifier']
    if list(clf.classes_) != [0, 1]:
        raise ConfigurationError('baseline classifier class ordering changed')
    x = vectorizer.transform([r['text'] for r in rows])
    return clf.predict_proba(x)[:, 1].tolist()


def _length_stats(rows: list[dict], trunc: list[bool]) -> dict:
    """Descriptive controls; never used to select models/thresholds."""
    result = {}
    for label, role in ((0, 'cover'), (1, 'stego')):
        indices = [i for i, row in enumerate(rows) if row['label'] == label]
        subset = [rows[i] for i in indices]
        result[role] = {
            'mean_characters': sum(len(r['text']) for r in subset) / len(subset),
            'mean_words': sum(len(r['text'].split()) for r in subset) / len(subset),
            'ending_punctuation_rate': sum(r['text'].rstrip().endswith(('.', '?', '!'))
                                           for r in subset) / len(subset),
            'roberta_truncation_rate': sum(trunc[i] for i in indices) / len(indices),
        }
    return result


def _evaluate(rows_by_model: dict, *, lock: dict, protocol: dict, model_root: Path) -> tuple[dict, list[dict]]:
    from vkr_benchmark.steganalysis.final_evaluation import MODEL_SLUGS
    all_articles = sorted({r['article_id'] for rows in rows_by_model.values() for r in rows})
    if len(all_articles) != 24:
        raise ConfigurationError('test cluster count changed')
    weights = bootstrap_article_weights(all_articles,
        iterations=protocol['bootstrap']['resamples'], seed=protocol['bootstrap']['seed'])
    selected = {r['model_id']: r for r in lock['selected']}
    results, raw_predictions = [], []
    for model_id, slug in MODEL_SLUGS.items():
        rows = rows_by_model[model_id]
        reference = selected[model_id]
        baseline_path = model_root / slug / 'tfidf_logreg.joblib'
        if file_sha(baseline_path) != reference['baseline']['joblib_sha256']:
            raise ConfigurationError('baseline fitted model differs from committed lock')
        baseline_scores = _baseline_scores(rows, baseline_path, {
            **reference['baseline'], 'dataset_sha256': lock['dataset_sha256']})
        roberta_scores, trunc = _roberta_scores(rows, model_root / slug / 'roberta' / 'seed-713',
            batch_size=protocol['roberta_batch_size'], max_length=protocol['roberta_max_length'])
        if len(baseline_scores) != len(rows) or len(roberta_scores) != len(rows):
            raise ConfigurationError('detector returned wrong score count')
        test_index = {r['record_id']: i for i, r in enumerate(rows)}
        for name, scores, meta in (('tfidf_logreg', baseline_scores, reference['baseline']),
                                   ('roberta_e4', roberta_scores, reference['roberta_e4'])):
            for r, score in zip(rows, scores):
                raw_predictions.append({
                    'record_id': r['record_id'], 'pair_id': r['pair_id'],
                    'article_id': r['article_id'], 'model_id': model_id,
                    'method_id': r['method_id'], 'method_point_id': r['method_point_id'],
                    'partition': 'test', 'detector': name,
                    'label': r['label'], 'score_stego': float(score),
                })
            subsets = [(None, None, rows, scores)]
            for method, point in protocol['operating_points']:
                idx = [i for i, r in enumerate(rows)
                       if (r['method_id'], r['method_point_id']) == (method, point)]
                subsets.append((method, point, [rows[i] for i in idx], [scores[i] for i in idx]))
            for method, point, selected_rows, selected_scores in subsets:
                subset_flags = [trunc[test_index[r['record_id']]] for r in selected_rows]
                record = {
                    'model_id': model_id, 'detector': name,
                    'scope': 'pooled' if method is None else 'operating_point',
                    'method_id': method, 'method_point_id': point,
                    **evaluate_scored_rows(selected_rows, selected_scores,
                         threshold=meta['threshold'], article_ids=all_articles, draw_counts=weights),
                    'length_diagnostics': _length_stats(selected_rows, subset_flags),
                }
                results.append(record)
        print(f'Stage-7 test scoring complete: {model_id}, two frozen detectors', flush=True)
    if len(results) != 64 or len(raw_predictions) != 7200:
        raise ConfigurationError('test evaluation must produce 60 point + 4 pooled results')
    report = {
        'schema_version': 'stage7.final_evaluation.v1', 'status': 'complete',
        'dataset_sha256': lock['dataset_sha256'],
        'lock_sha256': sha256(canonical_bytes(lock)).hexdigest(),
        'protocol_sha256': lock['protocol_sha256'],
        'test_article_groups': len(all_articles), 'test_pair_count': 1800,
        'detector_scoring': 'text_only', 'primary_model': 'E4_seed_713',
        'ablation_e8_used_for_test': False,
        'threshold_selection': 'pooled_validation_only',
        'bootstrap': protocol['bootstrap'],
        'row_count': len(results), 'point_rows': 60, 'pooled_rows': 4,
        'results': results,
    }
    return report, raw_predictions


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, default=ROOT/'configs/stage7/final_evaluation.json')
    p.add_argument('--lock', type=Path, default=ROOT/'results/stage7/final_evaluation_lock.json')
    p.add_argument('--detector-config', type=Path, default=ROOT/'configs/stage7/detector_training.json')
    p.add_argument('--primary-audit', type=Path, default=ROOT/'results/stage7/roberta_primary_audit.json')
    p.add_argument('--manifest', type=Path, default=ROOT/'results/stage7/dataset_manifest.json')
    p.add_argument('--dataset', type=Path, default=ROOT/'data/stage7/dataset/detector_records.jsonl')
    p.add_argument('--baseline-results', type=Path, default=ROOT/'results/stage7/checkpoints/baseline')
    p.add_argument('--roberta-results', type=Path, default=ROOT/'results/stage7/checkpoints/roberta')
    p.add_argument('--models-root', type=Path, default=ROOT/'models/detectors/stage7')
    p.add_argument('--report', type=Path, default=ROOT/'results/stage7/final_evaluation_report.json')
    p.add_argument('--predictions', type=Path, default=ROOT/'results/stage7/predictions/raw/final_test_predictions.jsonl')
    p.add_argument('--dry-run', action='store_true', help='Check committed selections; never read test')
    p.add_argument('--open-frozen-test', action='store_true', help='Explicit permission for first external test scoring')
    p.add_argument('--lock-sha256', help='Exact SHA-256 of committed lock (required to open test)')
    args = p.parse_args()
    try:
        if not args.lock.is_file():
            raise ConfigurationError('pre-test lock is missing; run preflight and commit first')
        _committed(args.lock)
        _committed(args.protocol)
        _committed(Path(__file__))
        _committed(ROOT / 'src/vkr_benchmark/steganalysis/final_evaluation.py')
        _committed(args.detector_config)
        _committed(args.primary_audit)
        _committed(args.manifest)
        expected = make_lock(protocol=args.protocol, detector_config=args.detector_config,
            primary_audit=args.primary_audit, dataset_manifest=args.manifest,
            baseline_results=args.baseline_results, roberta_results=args.roberta_results,
            models_root=args.models_root)
        if args.lock.read_bytes() != canonical_bytes(expected):
            raise ConfigurationError('committed lock is stale relative to current model bytes')
        digest = file_sha(args.lock)
        protocol = read_json(args.protocol)
        validate_protocol(protocol)
        if args.dry_run:
            if args.open_frozen_test:
                raise ConfigurationError('dry-run and opening test are mutually exclusive')
            print(json.dumps({'status': 'committed_lock_verified', 'test_access': 'none',
                              'lock_sha256': digest, 'model_selection': 'E4 seeds 713/713'}, indent=2))
            return 0
        if not args.open_frozen_test or args.lock_sha256 != digest:
            raise ConfigurationError('test remains sealed: require --open-frozen-test and exact --lock-sha256')
        if args.report.exists() or args.predictions.exists():
            raise ConfigurationError('final test output exists; refuse overwrite')
        rows = load_frozen_test(dataset=args.dataset, manifest=args.manifest,
            expected_sha256=expected['dataset_sha256'], protocol=protocol)
        report, predictions = _evaluate(rows, lock=expected, protocol=protocol,
            model_root=args.models_root)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        # No carrier text in either report; raw record-level scores are ignored by Git.
        args.predictions.write_bytes(b''.join(canonical_bytes(r) for r in predictions))
        args.report.write_bytes(canonical_bytes({**report,
            'predictions_sha256': file_sha(args.predictions)}))
        print(json.dumps({
            'status': report['status'], 'test_pair_count': report['test_pair_count'],
            'point_rows': report['point_rows'], 'pooled_rows': report['pooled_rows'],
            'test_access': 'evaluation_only',
            'report_sha256': file_sha(args.report),
            'predictions_sha256': file_sha(args.predictions),
        }, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, ConfigurationError) as exc:
        print(f'Stage-7 final evaluation: ERROR ({type(exc).__name__}: {exc})',file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
