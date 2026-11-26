#!/usr/bin/env python3
"""Separate post-test RRC length diagnostic; never modifies frozen artifacts."""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from audit_stage7_final_evaluation import (  # noqa: E402
    DATASET_SHA, LOCK_SHA, PRED_SHA, REPORT_SHA, PROTOCOL_SHA, MODELS,
    DETECTORS, checked_bytes, require, weighted_rank_auc,
)
from evaluate_stage7_final import _baseline_scores, _roberta_scores, _committed  # noqa: E402
from vkr_benchmark.steganalysis.final_evaluation import (  # noqa: E402
    MODEL_SLUGS, canonical_bytes, jsonl_bytes, parse_jsonl_records, file_sha,
    read_json, bootstrap_article_weights, evaluate_scored_rows,
)
from vkr_benchmark.errors import ConfigurationError  # noqa: E402
from vkr_benchmark.steganalysis.protocol import cover_seed  # noqa: E402
from vkr_benchmark.steganalysis.textio import read_carrier_text  # noqa: E402

POINT = 'payload1024-margin40'
PLAN_SHA = 'bbc4a184f16c96155c2fb6a8513a3c07b94032f541c57b84e574b565bc9aea88'
SPLIT_SHA = 'bfb6b6323177f834d72fe2e1236e53b9e559cd7d907c37890b058ab0fece52d6'
REVISIONS = {'meta-llama/Llama-3.2-3B': '13afe5124825b4f3751f836b40dafda64c1ed062',
             'Qwen/Qwen3-4B-Base': '906bfd4b4dc7f14ee4320094d8b41684abff8539'}
CACHE = ROOT / 'data/stage7/cache/rrc_length'
PLAN = CACHE / 'cover_plan.jsonl'
COVERS = CACHE / 'cover_runs'
RAW = CACHE / 'unmatched_predictions.jsonl'
REPORT = ROOT / 'results/stage7/analysis/rrc_unmatched_length.json'


def build_plan(matched, predictions, prompts, split):
    groups = {r['prompt_id']: r for r in split['rows'] if r['partition'] == 'test'}
    prompt_map = {r['prompt_id']: r for r in prompts}
    require(len(groups) == len(prompt_map) == len(prompts) == 24 and set(groups) == set(prompt_map),
            'frozen test prompts differ')
    for pid, row in groups.items():
        require(sha256(prompt_map[pid]['text'].encode()).hexdigest() == row['context_sha256'],
                f'prompt text changed: {pid}')
    selected = [r for r in matched if r['partition'] == 'test' and r['method_id'] == 'rrc']
    counts = Counter((r['model_id'], r['article_id']) for r in selected)
    require(counts == {(m, r['article_id']): n for m, (_, n) in MODELS.items() for r in groups.values()},
            'RRC test coverage changed')
    require(len(selected) == len({r['sample_id'] for r in selected}) == 120, 'RRC pair IDs changed')
    original = [p for p in predictions if p['method_id'] == 'rrc']
    expected = {(r['sample_id'], d, y) for r in selected for d in DETECTORS for y in (0, 1)}
    index = {(p['pair_id'], p['detector'], p['label']): p for p in original}
    require(len(original) == len(index) == len(expected) and set(index) == expected, 'RRC score support changed')
    result = []
    for r in selected:
        require(r['method_point_id'] == POINT and r['carrier_tokens'] > 0 and
                groups[r['prompt_id']]['article_id'] == r['article_id'], 'invalid matched RRC row')
        require(r['cover_seed'] == cover_seed(model_id=r['model_id'], article_id=r['article_id'],
                method_point_id=f'rrc/{POINT}', replicate_id=r['replicate_id']), 'cover seed changed')
        for d in DETECTORS:
            for y in (0, 1):
                p = index[r['sample_id'], d, y]
                require(all(p[k] == r[k] for k in ('model_id', 'article_id', 'method_id', 'method_point_id', 'partition')),
                        'original RRC score metadata differs')
        result.append(dict(r, matched_carrier_tokens=r['carrier_tokens'], carrier_tokens=256))
    return sorted(result, key=lambda r: (r['model_id'], r['sample_id']))


def read_cover(row, root):
    folder = root / row['sample_id']
    marker = read_json(folder / 'cover_result.json')
    text = read_carrier_text(folder / 'cover.txt')
    ids = read_json(folder / 'token_ids.json')['sender_token_ids']
    require(all(marker[k] == row[k] for k in ('sample_id', 'stego_run_id', 'model_id', 'prompt_id', 'cover_seed', 'carrier_tokens'))
            and marker['model_revision'] == REVISIONS[row['model_id']]
            and len(ids) == row['carrier_tokens'] and all(type(i) is int and i >= 0 for i in ids)
            and bool(text) and sha256(text.encode()).hexdigest() == marker['text_sha256'],
            f'invalid unmatched cover: {row["sample_id"]}')
    return text


def common_prefix_exact(row, matched_root, unmatched_root):
    # Separate generation with the same seed should preserve common sender-token
    # prefixes. Record any deviation instead of assuming numerical determinism.
    read_cover(dict(row, carrier_tokens=row['matched_carrier_tokens']), matched_root)
    old = read_json(matched_root / row['sample_id'] / 'token_ids.json')['sender_token_ids']
    new = read_json(unmatched_root / row['sample_id'] / 'token_ids.json')['sender_token_ids']
    n = min(len(old), len(new))
    return old[:n] == new[:n]


def paired_delta(matched, unmatched, articles, draws):
    def samples(rows):
        return weighted_rank_auc([r['label'] for r in rows], [r['score_stego'] for r in rows],
                                 [articles.index(r['article_id']) for r in rows], draws)
    diff = samples(unmatched) - samples(matched)
    low, high = np.quantile(diff, [.025, .975])
    return {'ci95_lower': float(low), 'ci95_upper': float(high), 'bootstrap_n': len(diff)}


def write_new(path, blob):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(blob)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'evaluate'])
    p.add_argument('--write-plan', action='store_true')
    args = p.parse_args()
    if args.action == 'evaluate' and args.write_plan:
        p.error('--write-plan is only for prepare')
    if args.action == 'evaluate':
        # Same byte-verified weights and thresholds as the primary experiment.
        _committed(Path(__file__))
        subprocess.run([sys.executable, str(ROOT/'scripts/evaluate_stage7_final.py'), '--dry-run'],
                       cwd=ROOT, check=True)
        require(not REPORT.exists() and not RAW.exists(), 'diagnostic output exists; refuse overwrite')
    lock = json.loads(checked_bytes(ROOT/'results/stage7/final_evaluation_lock.json', LOCK_SHA))
    report = json.loads(checked_bytes(ROOT/'results/stage7/final_evaluation_report.json', REPORT_SHA))
    predictions = parse_jsonl_records(checked_bytes(ROOT/'results/stage7/predictions/raw/final_test_predictions.jsonl', PRED_SHA))
    protocol = json.loads(checked_bytes(ROOT/'configs/stage7/final_evaluation.json', PROTOCOL_SHA))
    manifest = read_json(ROOT/'results/stage7/dataset_manifest.json')
    require(manifest['dataset_sha256'] == DATASET_SHA and manifest['cover_plan_sha256'] == PLAN_SHA
            and manifest['split_manifest_sha256'] == SPLIT_SHA, 'frozen manifest changed')
    matched = parse_jsonl_records(checked_bytes(ROOT/'data/stage7/dataset/all_cover_plan.jsonl', PLAN_SHA))
    split = json.loads(checked_bytes(ROOT/'results/stage7/splits/split_manifest.json', SPLIT_SHA))
    prompts = parse_jsonl_records((ROOT/'data/stage6/main_prompts.jsonl').read_bytes())
    plan = build_plan(matched, predictions, prompts, split)
    blob = jsonl_bytes(plan)
    if args.action == 'prepare':
        if args.write_plan:
            if PLAN.exists():
                require(PLAN.read_bytes() == blob, 'existing diagnostic plan differs')
            else:
                write_new(PLAN, blob)
        print(json.dumps({'status': 'plan_verified', 'plan_sha256': sha256(blob).hexdigest(),
                          'plan_written': args.write_plan, 'cover_count': len(plan),
                          'per_model': dict(Counter(r['model_id'] for r in plan)),
                          'seed_policy': 'same_as_matched', 'inference_repeated': False}, indent=2))
        return
    require(PLAN.read_bytes() == blob, 'diagnostic plan changed')
    new_texts = {r['sample_id']: read_cover(r, COVERS) for r in plan}
    prefix_exact = {r['sample_id']: common_prefix_exact(r, ROOT/'results/stage7/cover_runs', COVERS) for r in plan}
    articles = sorted({r['article_id'] for r in plan})
    draws = bootstrap_article_weights(articles, iterations=2000, seed=731)
    selected = {r['model_id']: r for r in lock['selected']}
    orig = {(r['pair_id'], r['detector'], r['label']): r for r in predictions if r['method_id'] == 'rrc'}
    results, raw = [], []
    for model, slug in MODEL_SLUGS.items():
        covers = [dict(record_id=r['sample_id']+'-cover-unmatched256', pair_id=r['sample_id'],
                       article_id=r['article_id'], model_id=model, method_id='rrc', method_point_id=POINT,
                       partition='test', label=0, text=new_texts[r['sample_id']]) for r in plan if r['model_id'] == model]
        ref = selected[model]
        model_root = ROOT/'models/detectors/stage7'/slug
        base = _baseline_scores(covers, model_root/'tfidf_logreg.joblib',
                                {**ref['baseline'], 'dataset_sha256': DATASET_SHA})
        rob, trunc = _roberta_scores(covers, model_root/'roberta/seed-713',
                         batch_size=protocol['roberta_batch_size'], max_length=protocol['roberta_max_length'])
        # _length_stats expects both classes, so calculate the new cover statistics directly.
        cover_stats = dict(mean_characters=sum(len(r['text']) for r in covers)/len(covers),
                           mean_words=sum(len(r['text'].split()) for r in covers)/len(covers),
                           ending_punctuation_rate=sum(r['text'].rstrip().endswith(('.', '?', '!')) for r in covers)/len(covers),
                           roberta_truncation_rate=sum(trunc)/len(covers))
        for detector, scores, meta in (('tfidf_logreg', base, ref['baseline']), ('roberta_e4', rob, ref['roberta_e4'])):
            require(len(scores) == len(covers), 'cover score count changed')
            old, new = [], []
            for r, score in zip(covers, scores):
                sid = r['pair_id']
                stego = orig[sid, detector, 1]
                old.extend([orig[sid, detector, 0], stego])
                new_cover = {k:v for k,v in r.items() if k != 'text'} | {'detector': detector, 'score_stego': float(score)}
                new_cover['text_sha256'] = sha256(r['text'].encode()).hexdigest()
                new.extend([new_cover, stego])
                raw.append(new_cover)
            prior = next(r for r in report['results'] if r['model_id']==model and r['detector']==detector and r['method_id']=='rrc')
            kwargs = dict(threshold=meta['threshold'], article_ids=articles, draw_counts=draws)
            old_metrics = evaluate_scored_rows(old, [r['score_stego'] for r in old], **kwargs)
            require(abs(old_metrics['roc_auc']-prior['roc_auc']) < 1e-10, 'matched AUC differs')
            new_metrics = evaluate_scored_rows(new, [r['score_stego'] for r in new], **kwargs)
            delta = paired_delta(old, new, articles, draws)
            results.append(dict(model_id=model, detector=detector, matched_roc_auc=prior['roc_auc'],
                                unmatched=new_metrics, delta_roc_auc=new_metrics['roc_auc']-prior['roc_auc'],
                                delta_ci95=delta, length_diagnostics={'cover': cover_stats,
                                'stego': prior['length_diagnostics']['stego']},
                                matched_length_diagnostics=prior['length_diagnostics']))
    require(len(raw)==240 and len(results)==4, 'incomplete diagnostic scoring')
    payload = jsonl_bytes(raw)
    result = dict(schema_version='stage7.rrc_unmatched_length.v1', status='complete',
                  role='diagnostic_only', dataset_sha256=DATASET_SHA, lock_sha256=LOCK_SHA,
                  original_report_sha256=REPORT_SHA, original_predictions_sha256=PRED_SHA,
                  plan_sha256=sha256(blob).hexdigest(), new_cover_predictions_sha256=sha256(payload).hexdigest(),
                  new_cover_count=120, stego_scores_reused=True, cover_seed_policy='same_as_matched',
                  cover_tokens=256, bootstrap=protocol['bootstrap'],
                  common_prefix_exact_count=sum(prefix_exact.values()),
                  common_prefix_mismatch_sample_ids=sorted(sid for sid, exact in prefix_exact.items() if not exact),
                  results=results)
    write_new(RAW, payload)
    write_new(REPORT, canonical_bytes(result))
    print(json.dumps({'status': 'complete', 'diagnostic_rows': len(results), 'new_cover_scores': len(raw),
                      'common_prefix_exact_count': sum(prefix_exact.values()),
                      'report_sha256': file_sha(REPORT), 'predictions_sha256': file_sha(RAW)}, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, ConfigurationError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f'RRC length diagnostic: ERROR ({type(exc).__name__}: {exc})')
