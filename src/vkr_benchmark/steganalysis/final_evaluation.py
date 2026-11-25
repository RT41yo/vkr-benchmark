"""Locked, text-only Stage-7 external-test metrics and article-cluster bootstrap.

The lock is prepared exclusively from validation summaries and checkpoint hashes.
The frozen test is opened only by the explicit evaluation runner.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

from vkr_benchmark.errors import ConfigurationError
from vkr_benchmark.steganalysis.detectors import FROZEN_DATASET_SHA256, LLAMA_ID, QWEN_ID

MODEL_SLUGS = {LLAMA_ID: 'llama-3.2-3b', QWEN_ID: 'qwen3-4b-base'}
PINNED_PRIMARY_CONFIG = '2eb26ae60e768b0e55f041c53b2221a87bbb433f357e7523a5665a96a60b17ac'
PINNED_PRIMARY_AUDIT = 'bdcdd4443418e2d6b529d1cdea334ba23c08523c6e5e12614fb2aea6e54d37f5'
PINNED_REVISION = 'e2da8e2f811d1448a5b465c236feacd80ffbac7b'
RO_BERT_ASSETS = ('config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json',
                  'vocab.json', 'merges.txt')
FROZEN_POINTS = (
    ('bins', 'b1'), ('bins', 'b4'), ('bins', 'b6'),
    ('huffman', 'e2'), ('huffman', 'e4'), ('huffman', 'e8'),
    ('arithmetic_coding', 'p32-k128'), ('arithmetic_coding', 'p32-k2048'),
    ('arithmetic_coding', 'p32-k50000'), ('adg', 'default'), ('discop', 'default'),
    ('dairstega', 'a8-b1-k100'), ('dairstega', 'a16-b075-k100'),
    ('dairstega', 'a32-b05-k100'), ('rrc', 'payload1024-margin40'),
)


def file_sha(path: Path) -> str:
    hasher = sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ConfigurationError(f'expected JSON object: {path}')
    return data


def canonical_bytes(item: Any) -> bytes:
    return (json.dumps(item, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode('utf-8')


def parse_jsonl_records(blob: bytes) -> list[dict[str, Any]]:
    """Parse canonical JSONL using literal LF record delimiters only.

    splitlines() is not valid here: it also splits at U+0085/U+2028/U+2029,
    which are permitted inside JSON strings serialized with ensure_ascii=False.
    The dataset's frozen byte SHA-256 is validated by load_frozen_test first.
    """
    text = blob.decode('utf-8')
    if not text.endswith('\n'):
        raise ConfigurationError('frozen detector JSONL must end with LF')
    records = [json.loads(line) for line in text.split('\n') if line]
    if any(not isinstance(row, dict) for row in records):
        raise ConfigurationError('frozen detector JSONL requires object records')
    return records


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    """Serialize one compact JSON object per literal LF-delimited record."""
    return ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n'
                   for row in rows).encode('utf-8')


def close_number(value: Any, expected: float) -> bool:
    return (isinstance(value, (float, int)) and not isinstance(value, bool)
            and math.isfinite(value) and math.isclose(value, expected, rel_tol=0, abs_tol=1e-10))


def validate_protocol(cfg: dict) -> None:
    if (cfg.get('schema_version') != 'stage7.final_evaluation_protocol.v1' or
            cfg.get('study') != 'primary_e4_only' or
            cfg.get('dataset_sha256') != FROZEN_DATASET_SHA256 or
            cfg.get('primary_configuration_sha256') != PINNED_PRIMARY_CONFIG or
            cfg.get('primary_audit_sha256') != PINNED_PRIMARY_AUDIT or
            cfg.get('roberta_revision') != PINNED_REVISION or
            cfg.get('selected_roberta_seeds') != {LLAMA_ID: 713, QWEN_ID: 713} or
            cfg.get('detectors') != ['tfidf_logreg', 'roberta_e4'] or
            cfg.get('roberta_max_length') != 512 or
            cfg.get('roberta_batch_size') != 4 or
            cfg.get('threshold_source') != 'pooled_validation_only' or
            cfg.get('scores') != 'positive_class_stego_probability' or
            cfg.get('pr_auc_definition') != 'trapezoidal_precision_recall_curve' or
            cfg.get('test_used_for_selection') is not False or
            cfg.get('bootstrap') != {'unit': 'article_id', 'resamples': 2000,
                                      'seed': 731, 'confidence': 0.95,
                                      'method': 'percentile', 'shared_draws': True} or
            cfg.get('test_article_groups') != 24 or cfg.get('test_pair_count') != 1800 or
            cfg.get('test_pairs_per_model') != {LLAMA_ID: 1080, QWEN_ID: 720} or
            tuple(tuple(x) for x in cfg.get('operating_points', [])) != FROZEN_POINTS):
        raise ConfigurationError('Stage-7 final evaluation protocol differs from frozen decision')


def validate_validation_predictions(path: Path, summary: dict) -> None:
    """Verify baseline threshold and ROC-AUC using validation predictions only."""
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from vkr_benchmark.steganalysis.detectors import choose_validation_threshold
    labels, scores, ids = [], [], set()
    for line in path.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        score, label, rid = row['score_stego'], row['label'], row['record_id']
        if (row.get('partition') != 'validation' or
                type(label) is not int or label not in (0, 1) or
                not isinstance(score, (float, int)) or isinstance(score, bool) or
                not math.isfinite(score) or not (0 <= score <= 1) or
                not isinstance(rid, str) or not rid or rid in ids):
            raise ConfigurationError('invalid/duplicate baseline validation prediction')
        ids.add(rid)
        labels.append(label)
        scores.append(float(score))
    if len(labels) != summary['counts']['validation'] or Counter(labels)[0] != Counter(labels)[1]:
        raise ConfigurationError('incomplete baseline validation predictions')
    auc = float(roc_auc_score(labels, scores))
    threshold, bacc = choose_validation_threshold(labels, scores)
    if not (close_number(summary.get('validation_roc_auc'), auc) and
            close_number(summary.get('validation_threshold'), threshold) and
            close_number(summary.get('validation_balanced_accuracy'), bacc)):
        raise ConfigurationError('baseline validation predictions differ from summary')


def make_lock(*, protocol: Path, detector_config: Path, primary_audit: Path,
              dataset_manifest: Path, baseline_results: Path, roberta_results: Path,
              models_root: Path) -> dict:
    """Seal validation selections and model bytes. Does not read the test dataset."""
    cfg = read_json(protocol)
    validate_protocol(cfg)
    if file_sha(detector_config) != PINNED_PRIMARY_CONFIG:
        raise ConfigurationError('pinned four-epoch detector configuration SHA-256 mismatch')
    dcfg = read_json(detector_config)
    if (dcfg.get('dataset_sha256') != FROZEN_DATASET_SHA256 or
            dcfg.get('roberta', {}).get('base_revision') != PINNED_REVISION or
            dcfg.get('roberta', {}).get('max_epochs') != 4):
        raise ConfigurationError('four-epoch detector config differs')
    audit_sha = file_sha(primary_audit)
    if audit_sha != PINNED_PRIMARY_AUDIT:
        raise ConfigurationError('primary six-detector audit SHA-256 mismatch')
    audit = read_json(primary_audit)
    if (audit.get('status') != 'verified' or audit.get('test_access') != 'none' or
            audit.get('detector_count') != 6 or
            audit.get('dataset_sha256') != FROZEN_DATASET_SHA256 or
            audit.get('primary_configuration_sha256') != PINNED_PRIMARY_CONFIG or
            audit.get('base_revision') != PINNED_REVISION or
            len(audit.get('runs', [])) != 6 or
            {(r['model_id'], r['seed']) for r in audit['runs']} !=
            {(m, s) for m in MODEL_SLUGS for s in (711, 712, 713)}):
        raise ConfigurationError('invalid or incomplete RoBERTa primary audit')
    manifest = read_json(dataset_manifest)
    if (manifest.get('dataset_sha256') != FROZEN_DATASET_SHA256 or
            manifest.get('status') != 'complete_with_documented_nontermination' or
            manifest.get('sample_count') != 11278 or manifest.get('pair_count') != 5639 or
            manifest.get('stage6_test_sources') != 1800 or
            manifest.get('group_count') != 152 or
            manifest.get('class_counts') != {'cover': 5639, 'stego': 5639} or
            manifest.get('partition_counts') != {'train': 5760, 'validation': 1918, 'test': 3600} or
            manifest.get('excluded_nontermination_run_ids') != ['0b0e22af25613edb']):
        raise ConfigurationError('frozen dataset manifest differs')
    selected = []
    for model_id, slug in MODEL_SLUGS.items():
        rows = [r for r in audit['runs'] if r['model_id'] == model_id]
        chosen = min(rows, key=lambda r: (-r['validation_roc_auc'], r['best_epoch'], r['seed']))
        if chosen['seed'] != 713 or chosen['best_epoch'] != 4:
            raise ConfigurationError('RoBERTa E4 validation-selected checkpoint changed')
        roberta_summary = roberta_results / slug / 'seed-713' / 'summary.json'
        if file_sha(roberta_summary) != chosen['summary_sha256']:
            raise ConfigurationError('selected RoBERTa summary differs from audit')
        s = read_json(roberta_summary)
        if (s.get('status') != 'complete' or s.get('test_access') != 'none' or
                s.get('dataset_sha256') != FROZEN_DATASET_SHA256 or
                s.get('seed') != 713 or s.get('model_id') != model_id or
                not close_number(s.get('validation_threshold'), chosen['validation_threshold'])):
            raise ConfigurationError('selected RoBERTa checkpoint metadata changed')
        weight_dir = models_root / slug / 'roberta' / 'seed-713'
        hashes = {name: file_sha(weight_dir / name) for name in RO_BERT_ASSETS}
        if hashes != chosen['checkpoint_asset_sha256']:
            raise ConfigurationError('selected RoBERTa checkpoint weights/tokenizer changed')
        baseline_summary_path = baseline_results / slug / 'summary.json'
        b = read_json(baseline_summary_path)
        if (b.get('status') != 'complete' or b.get('model_id') != model_id or
                b.get('dataset_sha256') != FROZEN_DATASET_SHA256 or
                b.get('test_access') != 'none' or b.get('baseline') != dcfg['baseline']):
            raise ConfigurationError('baseline summary/configuration differs')
        validate_validation_predictions(baseline_results / slug / 'validation_predictions.jsonl', b)
        joblib_path = models_root / slug / 'tfidf_logreg.joblib'
        selected.append({
            'model_id': model_id, 'slug': slug,
            'baseline': {
                'validation_roc_auc': b['validation_roc_auc'],
                'threshold': b['validation_threshold'],
                'summary_sha256': file_sha(baseline_summary_path),
                'validation_predictions_sha256': file_sha(baseline_results / slug / 'validation_predictions.jsonl'),
                'joblib_sha256': file_sha(joblib_path),
                'historic_training_config_sha256': b['config_sha256'],
            },
            'roberta_e4': {
                'seed': 713, 'epoch': 4,
                'validation_roc_auc': chosen['validation_roc_auc'],
                'threshold': chosen['validation_threshold'],
                'summary_sha256': chosen['summary_sha256'],
                'checkpoint_asset_sha256': hashes,
            },
        })
    return {
        'schema_version': 'stage7.final_evaluation_lock.v1',
        'status': 'sealed', 'test_access': 'none',
        'dataset_sha256': FROZEN_DATASET_SHA256,
        'protocol_sha256': file_sha(protocol),
        'dataset_manifest_sha256': file_sha(dataset_manifest),
        'primary_config_sha256': PINNED_PRIMARY_CONFIG,
        'primary_audit_sha256': PINNED_PRIMARY_AUDIT,
        'selected': selected,
    }


def load_frozen_test(*, dataset: Path, manifest: Path, expected_sha256: str,
                     protocol: dict) -> dict[str, list[dict]]:
    """Only call this *after* verifying the committed evaluation lock."""
    validate_protocol(protocol)
    if expected_sha256 != FROZEN_DATASET_SHA256 or file_sha(dataset) != expected_sha256:
        raise ConfigurationError('frozen test dataset SHA-256 mismatch')
    meta = read_json(manifest)
    if (meta.get('dataset_sha256') != expected_sha256 or
            meta.get('sample_count') != 11278 or meta.get('pair_count') != 5639):
        raise ConfigurationError('frozen test manifest mismatch')
    test_rows = []
    non_test_articles = set()
    for row in parse_jsonl_records(dataset.read_bytes()):
        if row['partition'] == 'test':
            if (not isinstance(row.get('text'), str) or not row['text'] or
                    sha256(row['text'].encode('utf-8')).hexdigest() != row.get('text_sha256')):
                raise ConfigurationError('test carrier text SHA-256 mismatch')
            test_rows.append(row)
        else:
            non_test_articles.add(row['article_id'])
    if len(test_rows) != 3600:
        raise ConfigurationError('external test row count changed')
    articles = {r['article_id'] for r in test_rows}
    if len(articles) != 24 or articles & non_test_articles:
        raise ConfigurationError('test article groups overlap training/validation')
    expected_points = {tuple(p) for p in protocol['operating_points']}
    counts = Counter((r['model_id'], r['method_id'], r['method_point_id'], r['label'])
                     for r in test_rows)
    by_article = Counter((r['model_id'], r['method_id'], r['method_point_id'],
                          r['label'], r['article_id']) for r in test_rows)
    ids = set()
    pairs: dict[str, list[dict]] = defaultdict(list)
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in test_rows:
        if (r['model_id'] not in MODEL_SLUGS or
                (r['method_id'], r['method_point_id']) not in expected_points or
                type(r['label']) is not int or r['label'] not in (0, 1) or
                r['record_id'] in ids or
                not isinstance(r['article_id'], str) or not r['article_id']):
            raise ConfigurationError('invalid test stratum/record metadata')
        ids.add(r['record_id'])
        pairs[r['pair_id']].append(r)
        by_model[r['model_id']].append(r)
    if len(pairs) != 1800:
        raise ConfigurationError('external test pair count changed')
    for pair_id, pair in pairs.items():
        if (len(pair) != 2 or {r['label'] for r in pair} != {0, 1} or
                {r['record_id'] for r in pair} != {f'{pair_id}-cover', f'{pair_id}-stego'} or
                len({(r['model_id'], r['article_id'], r['method_id'], r['method_point_id'])
                     for r in pair}) != 1):
            raise ConfigurationError('broken external test cover/stego pair')
    for model in MODEL_SLUGS:
        if len(by_model[model]) != protocol['test_pairs_per_model'][model] * 2:
            raise ConfigurationError('test LM size changed')
        if {r['article_id'] for r in by_model[model]} != articles:
            raise ConfigurationError('test article groups differ across LMs')
        for method, point in expected_points:
            n = protocol['test_pairs_per_model'][model] // 15
            if counts[model, method, point, 0] != n or counts[model, method, point, 1] != n:
                raise ConfigurationError('test operating-point support changed')
            for article in articles:
                for label in (0, 1):
                    if by_article[model, method, point, label, article] != n // 24:
                        raise ConfigurationError('test article/point/replicate support changed')
    for model in MODEL_SLUGS:
        by_model[model].sort(key=lambda row: row['record_id'])
    return dict(by_model)


def bootstrap_article_weights(article_ids: list[str], *, iterations: int = 2000,
                              seed: int = 731):
    import numpy as np
    if not article_ids or len(article_ids) != len(set(article_ids)):
        raise ConfigurationError('bootstrap needs unique article groups')
    rng = np.random.default_rng(seed)
    positions = rng.integers(0, len(article_ids), size=(iterations, len(article_ids)))
    return (np.arange(len(article_ids))[None, None, :] == positions[:, :, None]).sum(axis=1)


def article_bootstrap_auc(*, rows: list[dict], scores: list[float],
                          article_ids: list[str], draw_counts):
    """Exact article-cluster ROC-AUC via weighted pairwise positive/negative wins.

    All carrier repeats of a sampled article receive the same multiplicity.
    The same draw matrix is reused for all methods and detectors.
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score
    if len(scores) != len(rows) or not rows:
        raise ConfigurationError('bootstrap scores/rows mismatch')
    y = np.array([r['label'] for r in rows], dtype=int)
    s = np.array(scores, dtype=float)
    if not np.isfinite(s).all() or not set(y) == {0, 1}:
        raise ConfigurationError('bootstrap requires both classes and finite scores')
    index = {a: i for i, a in enumerate(article_ids)}
    if any(r['article_id'] not in index for r in rows):
        raise ConfigurationError('unknown test article in bootstrap')
    g = np.array([index[r['article_id']] for r in rows], dtype=int)
    p = np.where(y == 1)[0]
    n = np.where(y == 0)[0]
    wins = (s[p, None] > s[n][None, :]).astype(float)
    wins += 0.5 * (s[p, None] == s[n][None, :])
    # Aggregate all cross-article positive-negative comparisons by pair of articles.
    matrix = np.zeros((len(article_ids), len(article_ids)), dtype=float)
    np.add.at(matrix, (g[p, None], g[n][None, :]), wins)
    pos = np.bincount(g[p], minlength=len(article_ids))
    neg = np.bincount(g[n], minlength=len(article_ids))
    w = np.asarray(draw_counts, dtype=np.float64)
    if w.ndim != 2 or w.shape[1] != len(article_ids):
        raise ConfigurationError('invalid shared bootstrap draws')
    denom = (w @ pos) * (w @ neg)
    if (denom <= 0).any():
        raise ConfigurationError('bootstrap replicate has only one class')
    samples = np.einsum('bi,ij,bj->b', w, matrix, w, optimize=True) / denom
    auc = float(roc_auc_score(y, s))
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {'roc_auc': auc, 'ci95_lower': float(lower), 'ci95_upper': float(upper),
            'bootstrap_n': len(samples)}


def evaluate_scored_rows(rows: list[dict], scores: list[float], *,
                         threshold: float, article_ids: list[str], draw_counts) -> dict:
    import numpy as np
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                                 recall_score, f1_score, confusion_matrix,
                                 precision_recall_curve, auc, average_precision_score)
    y = [r['label'] for r in rows]
    if (len(rows) != len(scores) or not rows or
            not math.isfinite(threshold) or not (0 <= threshold <= 1) or
            any(not isinstance(s, (int, float)) or isinstance(s, bool) or
                not math.isfinite(s) or not (0 <= s <= 1) for s in scores)):
        raise ConfigurationError('invalid locked detector scores/threshold')
    pred = (np.asarray(scores) >= threshold).astype(int)
    tp = confusion_matrix(y, pred, labels=[0, 1])
    prec, recall, _ = precision_recall_curve(y, scores)
    return {
        **article_bootstrap_auc(rows=rows, scores=scores, article_ids=article_ids,
                                draw_counts=draw_counts),
        'record_count': len(rows), 'pair_count': len(rows) // 2,
        'article_group_count': len({r['article_id'] for r in rows}),
        'threshold_from_pooled_validation': float(threshold),
        'accuracy': float(accuracy_score(y, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall': float(recall_score(y, pred, zero_division=0)),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'pr_auc_trapezoid': float(auc(recall, prec)),
        'average_precision': float(average_precision_score(y, scores)),
        'confusion': {'tn': int(tp[0, 0]), 'fp': int(tp[0, 1]),
                      'fn': int(tp[1, 0]), 'tp': int(tp[1, 1])},
    }
