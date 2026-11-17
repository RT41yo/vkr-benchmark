from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vkr_benchmark.errors import ConfigurationError
from vkr_benchmark.steganalysis.dataset import paired_dataset


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding='utf-8')


def _sha(text): return hashlib.sha256(text.encode()).hexdigest()


def _setup_pair(root: Path):
    stego_root, cover_root, stage6_root = (root/x for x in ('stegos','covers','stage6'))
    plan = []
    for name, partition, status, carrier in (
        ('new1','train','generated',256), ('old1','test','error',256),
    ):
        text = 'hello stego ' + name
        base = {
            'sample_id': 'sample-'+name, 'stego_run_id': name,
            'partition': partition, 'article_id': 'article-'+name,
            'prompt_id': 'prompt-'+name, 'model_id': 'test/model',
            'method_id': 'bins', 'method_point_id': 'b1', 'replicate_id': 'r01',
            'cover_seed': 123, 'carrier_tokens': carrier,
            'stego_status': status, 'stego_text_sha256': _sha(text),
        }
        plan.append(base)
        if partition == 'train':
            rd = stego_root/name
            _write_json(rd/'stego_result.json', {
                'run_id': name, 'carrier_tokens': carrier,
                'text_sha256': _sha(text),
            })
            (rd/'stegotext.txt').write_text(text, encoding='utf-8')
        else:
            _write_json(stage6_root/name/'failure_diagnostics.json', {'transport_text': text})
        cover = 'ordinary cover ' + name
        _write_json(cover_root/('sample-'+name)/'cover_result.json', {
            'sample_id': 'sample-'+name, 'stego_run_id': name,
            'model_id': 'test/model', 'carrier_tokens': carrier,
            'cover_seed': 123, 'text_sha256': _sha(cover),
        })
        (cover_root/('sample-'+name)/'cover.txt').write_text(cover, encoding='utf-8')
        _write_json(cover_root/('sample-'+name)/'token_ids.json',
                    {'sender_token_ids':[1]*carrier})
    split = {'rows': [
        {'prompt_id': p['prompt_id'], 'article_id': p['article_id'],
         'partition': p['partition']} for p in plan
    ]}
    return plan, cover_root, stego_root, stage6_root, split


def test_paired_dataset_includes_stage6_hard_failure_without_leaking_metadata(tmp_path):
    plan, cover, stego, stage6, split = _setup_pair(tmp_path)
    texts, manifest = paired_dataset(
        cover_plan=plan, cover_root=cover, stego_root=stego,
        stage6_run_root=stage6, split_manifest=split,
        expected_pairs=2, expected_groups=2)
    assert len(texts) == 4
    assert manifest['class_counts'] == {'cover': 2, 'stego': 2}
    assert manifest['stage6_test_sources'] == 1
    assert sorted(r['label'] for r in texts) == [0,0,1,1]
    assert next(r for r in texts if r['partition']=='test' and r['label']==1)['text'] == 'hello stego old1'
    assert all('cover_seed' not in r and 'stego_run_id' not in r for r in texts)
    assert len({r['record_id'] for r in texts}) == 4


def test_paired_dataset_fails_if_cover_corrupted(tmp_path):
    plan, cover, stego, stage6, split = _setup_pair(tmp_path)
    (cover/'sample-new1'/'cover.txt').write_text('tampered', encoding='utf-8')
    with pytest.raises(ConfigurationError, match='cover source'):
        paired_dataset(cover_plan=plan, cover_root=cover, stego_root=stego,
                       stage6_run_root=stage6, split_manifest=split,
                       expected_pairs=2, expected_groups=2)


def test_paired_dataset_rejects_split_leakage(tmp_path):
    plan, cover, stego, stage6, split = _setup_pair(tmp_path)
    split['rows'][1]['partition'] = 'train'
    with pytest.raises(ConfigurationError, match='leaked or mismatched'):
        paired_dataset(cover_plan=plan, cover_root=cover, stego_root=stego,
                       stage6_run_root=stage6, split_manifest=split,
                       expected_pairs=2, expected_groups=2)


def test_paired_dataset_does_not_silently_drop_missing_stage6_text(tmp_path):
    plan, cover, stego, stage6, split = _setup_pair(tmp_path)
    (stage6/'old1'/'failure_diagnostics.json').unlink()
    with pytest.raises(ConfigurationError, match='cannot read'):
        paired_dataset(cover_plan=plan, cover_root=cover, stego_root=stego,
                       stage6_run_root=stage6, split_manifest=split,
                       expected_pairs=2, expected_groups=2)


def test_paired_dataset_rejects_wrong_cover_carrier_count(tmp_path):
    plan, cover, stego, stage6, split = _setup_pair(tmp_path)
    _write_json(cover/'sample-new1'/'token_ids.json', {'sender_token_ids':[1]})
    with pytest.raises(ConfigurationError, match='cover sender carrier'):
        paired_dataset(cover_plan=plan, cover_root=cover, stego_root=stego,
                       stage6_run_root=stage6, split_manifest=split,
                       expected_pairs=2, expected_groups=2)
