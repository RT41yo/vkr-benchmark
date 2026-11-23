"""Frozen, text-only Stage-7 detector input and validation-only scoring.

No test text is returned to detector training.  All dataset validation is
performed before any tokenizer or vectorizer is fitted.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from vkr_benchmark.errors import ConfigurationError

LLAMA_ID = 'meta-llama/Llama-3.2-3B'
QWEN_ID = 'Qwen/Qwen3-4B-Base'
MODEL_IDS = (LLAMA_ID, QWEN_ID)
FROZEN_DATASET_SHA256 = '43ea3b2ded135a8004a2ada7f3c3897ee3873baf0cd9283408418f99298587d9'


@dataclass(frozen=True)
class DetectorData:
    model_id: str
    train: tuple[dict[str, Any], ...]
    validation: tuple[dict[str, Any], ...]
    dataset_sha256: str
    counts: dict[str, int]


def _parse_jsonl(blob: bytes) -> list[dict[str, Any]]:
    try:
        text = blob.decode('utf-8')
        if not text.endswith('\n'):
            raise ConfigurationError('Stage-7 detector JSONL must end with newline')
        items = [json.loads(line) for line in text.split('\n') if line]
        if any(not isinstance(x, dict) for x in items):
            raise ValueError('non-object JSONL entry')
        return items
    except (ValueError, UnicodeError) as exc:
        raise ConfigurationError(f'invalid detector dataset JSONL: {exc}') from exc


def load_detector_training_data(
    *, dataset: Path, manifest: Path, model_id: str,
    expected_sha256: str = FROZEN_DATASET_SHA256,
    require_frozen_structure: bool = True,
) -> DetectorData:
    """Validate the complete corpus and return *only train/validation* texts.

    The frozen test split is inspected exclusively for metadata integrity and
    article isolation. Its text is never returned to a training caller.
    """
    if model_id not in MODEL_IDS:
        raise ConfigurationError(f'unknown Stage-7 source LM: {model_id}')
    if not expected_sha256 or len(expected_sha256) != 64:
        raise ConfigurationError('expected dataset SHA-256 must be provided')
    raw = dataset.read_bytes()
    actual = sha256(raw).hexdigest()
    metadata = json.loads(manifest.read_text(encoding='utf-8'))
    if actual != expected_sha256 or actual != metadata.get('dataset_sha256'):
        raise ConfigurationError('Stage-7 detector dataset differs from frozen SHA-256')
    if metadata.get('schema_version') != 'stage7.detector_dataset.v1':
        raise ConfigurationError('unsupported detector dataset schema')
    if metadata.get('status') != 'complete_with_documented_nontermination':
        raise ConfigurationError('Stage-7 detector dataset exception status changed')
    if metadata.get('excluded_nontermination_run_ids') != ['0b0e22af25613edb']:
        raise ConfigurationError('Stage-7 documented RRC exclusion changed')
    rows = _parse_jsonl(raw)
    if len(rows) != metadata.get('sample_count'):
        raise ConfigurationError('detector dataset sample count differs from manifest')
    if require_frozen_structure and (len(rows) != 11278 or metadata.get('pair_count') != 5639 or
            metadata.get('stage6_test_sources') != 1800 or metadata.get('group_count') != 152):
        raise ConfigurationError('Stage-7 frozen dataset cardinality changed')

    ids: set[str] = set()
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    articles: dict[str, str] = {}
    prompts: dict[str, tuple[str, str]] = {}
    counts: Counter = Counter()
    safe: dict[str, list[dict[str, Any]]] = {'train': [], 'validation': []}
    mandatory = ('record_id', 'pair_id', 'partition', 'article_id', 'prompt_id',
                 'model_id', 'method_id', 'method_point_id', 'replicate_id',
                 'label', 'text', 'text_sha256')
    for row in rows:
        if any(key not in row for key in mandatory):
            raise ConfigurationError('detector dataset missing required metadata')
        partition = row['partition']
        if partition not in ('train', 'validation', 'test'):
            raise ConfigurationError('unknown detector partition')
        if row['model_id'] not in MODEL_IDS or type(row['label']) is not int or row['label'] not in (0, 1):
            raise ConfigurationError('invalid detector source model or class')
        for field in mandatory:
            if field not in ('label', 'text') and (not isinstance(row[field], str) or not row[field]):
                raise ConfigurationError(f'invalid detector {field}')
        if not isinstance(row['text'], str) or not row['text']:
            raise ConfigurationError('empty or invalid detector carrier text')
        if sha256(row['text'].encode('utf-8')).hexdigest() != row['text_sha256']:
            raise ConfigurationError('detector text hash mismatch')
        if row['record_id'] in ids:
            raise ConfigurationError('duplicate detector record ID')
        ids.add(row['record_id'])
        article = row['article_id']
        prompt = row['prompt_id']
        if article in articles and articles[article] != partition:
            raise ConfigurationError('article crosses detector splits')
        articles[article] = partition
        if prompt in prompts and prompts[prompt] != (article, partition):
            raise ConfigurationError('prompt identity crosses detector splits')
        prompts[prompt] = (article, partition)
        pairs[row['pair_id']].append(row)
        counts[(partition, row['model_id'], row['label'])] += 1
        if row['model_id'] == model_id and partition != 'test':
            safe[partition].append(row)
    for pair_id, pair in pairs.items():
        if len(pair) != 2 or {r['label'] for r in pair} != {0, 1}:
            raise ConfigurationError(f'missing/duplicate cover-stego pairing: {pair_id}')
        for field in ('partition', 'model_id', 'article_id', 'prompt_id', 'method_id',
                      'method_point_id', 'replicate_id'):
            if pair[0][field] != pair[1][field]:
                raise ConfigurationError(f'mismatched matched-pair {field}: {pair_id}')
        if {r['record_id'] for r in pair} != {f'{pair_id}-cover', f'{pair_id}-stego'}:
            raise ConfigurationError(f'wrong record IDs for matched pair: {pair_id}')
    if len(pairs) != metadata.get('pair_count'):
        raise ConfigurationError('Stage-7 pair count does not match manifest')
    if dict(Counter(r['partition'] for r in rows)) != metadata.get('partition_counts'):
        raise ConfigurationError('Stage-7 partition counts differ from manifest')
    if dict(Counter(r['model_id'] for r in rows)) != metadata.get('per_model_counts'):
        raise ConfigurationError('Stage-7 per-model counts differ from manifest')
    if sum(r['label']==0 for r in rows) != metadata.get('class_counts', {}).get('cover') or \
            sum(r['label']==1 for r in rows) != metadata.get('class_counts', {}).get('stego'):
        raise ConfigurationError('Stage-7 class counts differ from manifest')
    if require_frozen_structure:
        expected = {
            ('train', LLAMA_ID): 1440, ('train', QWEN_ID): 1440,
            ('validation', LLAMA_ID): 480, ('validation', QWEN_ID): 479,
            ('test', LLAMA_ID): 1080, ('test', QWEN_ID): 720,
        }
        for (split, model), count in expected.items():
            if any(counts[split, model, label] != count for label in (0, 1)):
                raise ConfigurationError(f'Stage-7 frozen {split}/{model} allocation differs')
        if len(articles) != 152 or len(prompts) != 152:
            raise ConfigurationError('Stage-7 article/prompt group total changed')
    for model in MODEL_IDS:
        for split in ('train', 'validation', 'test'):
            if counts[split, model, 0] != counts[split, model, 1]:
                raise ConfigurationError('unbalanced detector split')
    for partition in safe:
        safe[partition].sort(key=lambda r: r['record_id'])
    return DetectorData(model_id=model_id, train=tuple(safe['train']),
                        validation=tuple(safe['validation']), dataset_sha256=actual,
                        counts={'train': len(safe['train']), 'validation': len(safe['validation'])})


def choose_validation_threshold(labels: list[int], scores: list[float]) -> tuple[float, float]:
    """One pooled-validation threshold; deterministic lower-threshold tiebreak."""
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score
    if not labels or len(labels) != len(scores) or set(labels) != {0, 1}:
        raise ConfigurationError('threshold selection needs both validation classes')
    values = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ConfigurationError('nonfinite detector scores')
    candidates = np.unique(np.r_[values, np.nextafter(values.max(), np.inf)])
    best_threshold, best_score = None, -1.0
    for threshold in candidates:
        value = float(balanced_accuracy_score(labels, (values >= threshold).astype(int)))
        if value > best_score + 1e-12:
            best_score, best_threshold = value, float(threshold)
    assert best_threshold is not None
    return best_threshold, best_score
