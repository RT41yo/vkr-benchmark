"""Stage-7 paired sample integrity, no model invocation or Stage-6 mutation."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from vkr_benchmark.errors import ConfigurationError


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f'cannot read {path}: {exc}') from exc
    if not isinstance(result, dict):
        raise ConfigurationError(f'expected JSON object: {path}')
    return result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
        records = [json.loads(line) for line in lines if line.strip()]
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f'cannot read {path}: {exc}') from exc
    if not all(isinstance(r, dict) for r in records):
        raise ConfigurationError(f'invalid JSONL object row in {path}')
    return records


def jsonl_text(rows: list[dict[str, Any]]) -> str:
    return ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in rows)


def cover_plan_from_stegos(
    *, trainval_plan: list[dict[str, Any]], stage6_index: list[dict[str, Any]],
    stego_root: Path,
) -> list[dict[str, Any]]:
    """Compute RRC matched lengths from completed Stage-7 sender token IDs."""
    if len(trainval_plan) != 3840 or len(stage6_index) != 1800:
        raise ConfigurationError('Stage-7 cover plan requires 3840+1800 stego sources')
    if any(row['partition'] != 'test' for row in stage6_index):
        raise ConfigurationError('external Stage-6 data must be test-only')
    output: list[dict[str, Any]] = []
    for row in trainval_plan:
        run_id = row['stego_run_id']
        directory = stego_root/run_id
        result = read_json(directory/'stego_result.json')
        ids = read_json(directory/'token_ids.json').get('sender_token_ids')
        text = (directory/'stegotext.txt').read_text(encoding='utf-8')
        if (not isinstance(ids, list) or not ids or
                any(type(v) is not int or v < 0 for v in ids)):
            raise ConfigurationError(f'invalid Stage-7 sender tokens: {run_id}')
        if (result.get('run_id') != run_id or result.get('sample_id') != row['sample_id'] or
                result.get('canonical_run_config') != row['canonical_run_config'] or
                result.get('carrier_tokens') != len(ids) or
                result.get('text_sha256') != sha_text(text)):
            raise ConfigurationError(f'corrupt Stage-7 stego source: {run_id}')
        if row['method_id'] != 'rrc' and len(ids) != 256:
            raise ConfigurationError(f'streaming length changed: {run_id}')
        if row['method_id'] == 'rrc' and result.get('payload_bits') != 1024:
            raise ConfigurationError(f'RRC payload changed: {run_id}')
        output.append({key: row[key] for key in (
            'sample_id', 'partition', 'stego_run_id', 'model_id', 'prompt_id',
            'article_id', 'method_id', 'method_point_id', 'replicate_id', 'cover_seed',
        )} | {'carrier_tokens': len(ids), 'stego_text_sha256': sha_text(text),
              'stego_status': 'generated'})
    for row in stage6_index:
        if not isinstance(row.get('carrier_tokens'), int) or row['carrier_tokens'] < 1:
            raise ConfigurationError('invalid frozen test carrier length')
        output.append({key: row[key] for key in (
            'sample_id', 'partition', 'stego_run_id', 'model_id', 'prompt_id',
            'article_id', 'method_id', 'method_point_id', 'replicate_id',
            'cover_seed', 'carrier_tokens', 'stego_text_sha256', 'stego_status',
        )})
    if len({r['sample_id'] for r in output}) != 5640 or len({r['stego_run_id'] for r in output}) != 5640:
        raise ConfigurationError('Stage-7 cover plan contains duplicates')
    if Counter(r['partition'] for r in output) != {'train': 2880, 'validation': 960, 'test': 1800}:
        raise ConfigurationError('Stage-7 cover plan has unexpected split counts')
    by_model = Counter(r['model_id'] for r in output)
    if sorted(by_model.values()) != [2520, 3120]:
        raise ConfigurationError('Stage-7 cover plan has unexpected model counts')
    strata = Counter((r['partition'], r['model_id'], r['method_id'], r['method_point_id'])
                     for r in output)
    model_test_counts = Counter(r['model_id'] for r in stage6_index)
    for (partition, model_id, _, _), count in strata.items():
        required = 96 if partition == 'train' else 32 if partition == 'validation' else model_test_counts[model_id] // 15
        if count != required:
            raise ConfigurationError(f'Stage-7 stratum has {count} samples, expected {required}')
    if len(strata) != 90:
        raise ConfigurationError('Stage-7 cover plan missing model/point/split strata')
    output.sort(key=lambda r: (r['partition'], r['model_id'], r['sample_id']))
    return output


def paired_dataset(
    *, cover_plan: list[dict[str, Any]], cover_root: Path,
    stego_root: Path, stage6_run_root: Path,
    split_manifest: dict[str, Any],
    expected_pairs: int = 5640, expected_groups: int = 152,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return local text-only detector records and commit-safe metadata."""
    if len(cover_plan) != expected_pairs:
        raise ConfigurationError(f'paired dataset requires all {expected_pairs} cover/stego pairs')
    split_rows = {r['prompt_id']: r for r in split_manifest['rows']}
    if len(split_rows) != expected_groups:
        raise ConfigurationError('invalid split manifest prompt groups')
    texts: list[dict[str, Any]] = []
    counts = Counter()
    seen: set[str] = set()
    for row in cover_plan:
        sid = row['sample_id']
        if sid in seen:
            raise ConfigurationError(f'duplicate sample ID: {sid}')
        seen.add(sid)
        group = split_rows.get(row['prompt_id'])
        if group is None or group['article_id'] != row['article_id'] or group['partition'] != row['partition']:
            raise ConfigurationError(f'leaked or mismatched prompt group: {sid}')
        if row['partition'] == 'test':
            run = stage6_run_root/row['stego_run_id']
            stego = (run/'stegotext.txt').read_text(encoding='utf-8') if row['stego_status'] == 'ok' else \
                    read_json(run/'failure_diagnostics.json')['transport_text']
        else:
            run = stego_root/row['stego_run_id']
            result = read_json(run/'stego_result.json')
            if (result.get('run_id') != row['stego_run_id'] or
                    result.get('carrier_tokens') != row['carrier_tokens']):
                raise ConfigurationError(f'stego result mismatch: {sid}')
            stego = (run/'stegotext.txt').read_text(encoding='utf-8')
        if not isinstance(stego, str) or not stego or sha_text(stego) != row['stego_text_sha256']:
            raise ConfigurationError(f'stego source text corrupt: {sid}')
        cover_dir = cover_root/sid
        cover_result = read_json(cover_dir/'cover_result.json')
        cover_ids = read_json(cover_dir/'token_ids.json').get('sender_token_ids')
        if (not isinstance(cover_ids, list) or len(cover_ids) != row['carrier_tokens'] or
                any(type(token_id) is not int or token_id < 0 for token_id in cover_ids)):
            raise ConfigurationError(f'cover sender carrier length/IDs corrupt: {sid}')
        cover = (cover_dir/'cover.txt').read_text(encoding='utf-8')
        if (not cover or cover_result.get('sample_id') != sid or
                cover_result.get('stego_run_id') != row['stego_run_id'] or
                cover_result.get('carrier_tokens') != row['carrier_tokens'] or
                cover_result.get('model_id') != row['model_id'] or
                cover_result.get('cover_seed') != row['cover_seed'] or
                sha_text(cover) != cover_result.get('text_sha256')):
            raise ConfigurationError(f'cover source text/metadata corrupt: {sid}')
        for label, role, text in ((0, 'cover', cover), (1, 'stego', stego)):
            texts.append({
                'record_id': f'{sid}-{role}', 'pair_id': sid,
                'partition': row['partition'], 'article_id': row['article_id'],
                'prompt_id': row['prompt_id'], 'model_id': row['model_id'],
                'method_id': row['method_id'], 'method_point_id': row['method_point_id'],
                'replicate_id': row['replicate_id'], 'label': label, 'text': text,
                'text_sha256': sha_text(text),
            })
            counts[(row['partition'], row['model_id'], row['method_id'], row['method_point_id'], label)] += 1
    if len(texts) != 2 * expected_pairs or len({r['record_id'] for r in texts}) != 2 * expected_pairs:
        raise ConfigurationError('paired dataset must contain exactly two unique records per pair')
    for (partition, model, method, point, label), count in counts.items():
        if counts[(partition, model, method, point, 1-label)] != count:
            raise ConfigurationError('class imbalance within a method/model/split stratum')
    texts.sort(key=lambda r: (r['partition'], r['model_id'], r['pair_id'], r['label']))
    manifest = {
        'schema_version': 'stage7.detector_dataset.v1', 'status': 'complete',
        'sample_count': len(texts), 'pair_count': len(seen),
        'class_counts': {'cover': len(seen), 'stego': len(seen)},
        'partition_counts': dict(Counter(r['partition'] for r in texts)),
        'group_count': expected_groups,
        'stage6_test_sources': sum(r['partition']=='test' for r in cover_plan),
        'per_model_counts': dict(Counter(r['model_id'] for r in texts)),
    }
    return texts, manifest
