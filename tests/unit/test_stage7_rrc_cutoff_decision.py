"""Stage-7 single-cutoff allowlist, replay gate and exact paired accounting."""
from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from vkr_benchmark.errors import ConfigurationError
from vkr_benchmark.steganalysis import dataset
from vkr_benchmark.steganalysis.rrc_diagnostic import _interval_snapshot
from vkr_benchmark.steganalysis.rrc_failure import (
    EXCEPTION_SCHEMA, FAILURE_SCHEMA, KNOWN_CUTOFF, REPLAY_SCHEMA,
    canonical_config_sha256, validate_registered_cutoffs,
)


ROOT = Path(__file__).resolve().parents[2]


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding='utf-8')


def _evidence(tmp_path: Path):
    """Exactly the observed failed run, minus secret-bearing config values."""
    info = KNOWN_CUTOFF
    raw = {
        'method': {'id': 'rrc', 'params': {'precision_margin': 40,
                                          'verify_termination': True}, 'key': 'secret-method-key'},
        'model': {'id': info['model_id'], 'revision': 'pinned-revision'},
        'termination': {'mode': 'fixed_payload_bits', 'target_payload_bits': 1024,
                        'max_carrier_tokens': 2048},
    }
    row = {
        'stego_run_id': info['run_id'], 'canonical_run_config': raw,
        'context_sha256': hashlib.sha256(b'toy prompt').hexdigest(),
        **{key: info[key] for key in ('sample_id','partition','model_id','prompt_id',
                                      'article_id','method_id','method_point_id')},
    }
    exc = {'schema_version': EXCEPTION_SCHEMA,
           'decision': 'conditional_on_matching_replay',
           'expected_total_failed_runs': 1, 'exceptions': [dict(info)]}
    record = dict(info) | {
        'schema_version': FAILURE_SCHEMA,
        'canonical_run_config': raw,
        'error_type': 'Stage7FixedPayloadNonTermination',
        'is_usable_stegotext': False,
    }
    report = {
        'schema_version': REPLAY_SCHEMA, 'status': 'reproduced',
        'run_id': info['run_id'], 'sample_id': info['sample_id'],
        'model_id': info['model_id'], 'prompt_id': info['prompt_id'],
        'prompt_sha256': hashlib.sha256(b'toy prompt').hexdigest(),
        'canonical_config_sha256': canonical_config_sha256(raw),
        'model_revision': 'pinned-revision',
        'replays': [{
            'terminated': False, 'generated_carrier_tokens': 2048,
            'termination_candidates': 0, 'verification_rejections': 0,
            'token_ids_sha256': 'f'*64,
            'snapshots': [{'step': 2048, 'interval_log2_width': 16.2,
                           'interval_width_at_most_one': False}],
        }],
    }
    excpath = tmp_path/'exception.json'
    failure_root = tmp_path/'failures'
    replaypath = tmp_path/'replay.json'
    _write(excpath, exc)
    _write(failure_root/f'{info["run_id"]}.json', record)
    _write(replaypath, report)
    args = dict(trainval_plan=[row], stego_root=tmp_path/'stegos',
                failure_root=failure_root, exception_config_path=excpath,
                replay_report_path=replaypath)
    return args, excpath, failure_root, replaypath


def test_exact_observed_rrc_failure_admitted_only_with_matching_replay(tmp_path):
    args, _, _, _ = _evidence(tmp_path)
    accepted = validate_registered_cutoffs(**args)
    assert set(accepted) == {KNOWN_CUTOFF['run_id']}
    assert accepted[KNOWN_CUTOFF['run_id']]['partition'] == 'validation'


@pytest.mark.parametrize('mutation,match', [
    ('missing_replay', 'cannot read'),
    ('missing_failure', 'registry differs'),
    ('extra_failure', 'registry differs'),
    ('replay_success', 'replay does not confirm'),
    ('wrong_token_digest', 'different sender token streams'),
    ('different_run_id', 'exception fields changed'),
    ('has_success_directory', 'also has a stego success'),
    ('corrupt_record', 'does not match allowlisted'),
    ('width_too_small', 'does not confirm'),
    ('prompt_changed', 'missing or inconsistent'),
])
def test_rrc_cutoff_decision_fails_closed(tmp_path, mutation, match):
    args, excpath, failures, replaypath = _evidence(tmp_path)
    if mutation == 'missing_replay':
        replaypath.unlink()
    elif mutation == 'missing_failure':
        (failures/f'{KNOWN_CUTOFF["run_id"]}.json').unlink()
    elif mutation == 'extra_failure':
        _write(failures/'arbitrary-other-failure.json', {})
    elif mutation == 'replay_success':
        obj = json.loads(replaypath.read_text())
        obj['replays'][0]['terminated'] = True
        _write(replaypath, obj)
    elif mutation == 'wrong_token_digest':
        obj = json.loads(replaypath.read_text())
        obj['replays'].append(obj['replays'][0] | {'token_ids_sha256': 'a'*64})
        _write(replaypath, obj)
    elif mutation == 'different_run_id':
        obj = json.loads(excpath.read_text())
        obj['exceptions'][0]['run_id'] = 'arbitrary-failure'
        _write(excpath, obj)
    elif mutation == 'has_success_directory':
        (args['stego_root']/KNOWN_CUTOFF['run_id']).mkdir(parents=True)
    elif mutation == 'width_too_small':
        obj = json.loads(replaypath.read_text())
        obj['replays'][0]['snapshots'][-1]['interval_width_at_most_one'] = True
        _write(replaypath, obj)
    elif mutation == 'prompt_changed':
        obj = json.loads(replaypath.read_text())
        obj['prompt_sha256'] = 'a'*64
        _write(replaypath, obj)
    elif mutation == 'corrupt_record':
        file = failures/f'{KNOWN_CUTOFF["run_id"]}.json'
        obj = json.loads(file.read_text())
        obj['termination_candidates'] = 1
        _write(file, obj)
    with pytest.raises(ConfigurationError, match=match):
        validate_registered_cutoffs(**args)


def test_rrc_report_does_not_contain_secret_key_from_config(tmp_path):
    args, *_ = _evidence(tmp_path)
    report = args['replay_report_path'].read_text()
    assert 'secret-method-key' not in report


def test_rrc_interval_snapshot_is_readonly_and_reports_log2_width():
    encoder = SimpleNamespace(_left=Decimal(0), _right=Decimal(8),
                              _secret_point=Decimal(2))
    report = _interval_snapshot(encoder, 512, 0, 0)
    assert report['interval_log2_width'] == 3
    assert report['interval_width_below_one'] is False
    assert encoder._right == 8
    encoder._right = Decimal('0.25')
    assert _interval_snapshot(encoder, 1024, 1, 0)['interval_width_below_one'] is True


POINTS = [
    ('bins','b1'), ('bins','b4'), ('bins','b6'),
    ('huffman','e2'), ('huffman','e4'), ('huffman','e8'),
    ('arithmetic_coding','p32-k128'),
    ('arithmetic_coding','p32-k2048'),
    ('arithmetic_coding','p32-k50000'),
    ('adg','default'), ('discop','default'),
    ('dairstega','a8-b1-k100'), ('dairstega','a16-b075-k100'),
    ('dairstega','a32-b05-k100'), ('rrc','payload1024-margin40'),
]


def _plans():
    trainval, test = [], []
    for model, test_n in [('meta-llama/Llama-3.2-3B',72),
                          ('Qwen/Qwen3-4B-Base',48)]:
        for method, point in POINTS:
            for part, n in [('train',96),('validation',32),('test',test_n)]:
                for idx in range(n):
                    run = f'run-{hashlib.sha256(model.encode()).hexdigest()[:8]}-{method}-{point}-{part}-{idx}'
                    row = {
                        'sample_id': f'sample-{run}', 'stego_run_id': run,
                        'partition': part, 'model_id': model,
                        'method_id': method, 'method_point_id': point,
                        'prompt_id': f'{part}-{idx}', 'article_id': f'article-{part}-{idx}',
                        'replicate_id': 'r01', 'cover_seed': idx,
                        'carrier_tokens': 300 if method=='rrc' else 256,
                    }
                    if part == 'test':
                        test.append(row | {
                            'stego_text_sha256': hashlib.sha256(b'stego').hexdigest(),
                            'stego_status': 'ok',
                        })
                    else:
                        trainval.append(row | {
                            'canonical_run_config': {'run':run},
                        })
    return trainval, test


def test_cover_plan_counts_are_exact_both_with_and_without_one_failure(monkeypatch, tmp_path):
    trainval, stage6 = _plans()
    index = {r['stego_run_id']: r for r in trainval}
    def read_json(path):
        row = index[path.parent.name]
        if path.name == 'token_ids.json':
            return {'sender_token_ids':[1]*row['carrier_tokens']}
        return {'run_id': row['stego_run_id'], 'sample_id':row['sample_id'],
                'canonical_run_config':row['canonical_run_config'],
                'carrier_tokens':row['carrier_tokens'],
                'payload_bits':1024 if row['method_id']=='rrc' else 256,
                'text_sha256':hashlib.sha256(b'stego').hexdigest()}
    monkeypatch.setattr(dataset, 'read_json', read_json)
    monkeypatch.setattr(dataset, 'read_carrier_text', lambda path: 'stego')
    all_rows = dataset.cover_plan_from_stegos(
        trainval_plan=trainval, stage6_index=stage6, stego_root=tmp_path)
    assert len(all_rows) == 5640
    assert sorted(Counter(x['model_id'] for x in all_rows).values()) == [2640, 3000]
    assert Counter(r['partition'] for r in all_rows) == {'train':2880,'validation':960,'test':1800}
    failure_row = next(row for row in trainval if row['partition']=='validation' and
                       row['model_id']=='Qwen/Qwen3-4B-Base' and row['method_id']=='rrc')
    exception = {failure_row['stego_run_id']:failure_row}
    paired_rows = dataset.cover_plan_from_stegos(
        trainval_plan=trainval, stage6_index=stage6, stego_root=tmp_path,
        accepted_failures=exception)
    assert len(paired_rows) == 5639
    assert Counter(r['partition'] for r in paired_rows) == {'train':2880,'validation':959,'test':1800}
    assert Counter(r['model_id'] for r in paired_rows) == {
        'meta-llama/Llama-3.2-3B':3000, 'Qwen/Qwen3-4B-Base':2639}
    assert failure_row['sample_id'] not in {r['sample_id'] for r in paired_rows}
    stratum = Counter((r['partition'],r['model_id'],r['method_id'],r['method_point_id'])
                      for r in paired_rows)
    assert stratum[('validation','Qwen/Qwen3-4B-Base','rrc','payload1024-margin40')] == 31
    assert len(stratum) == 90


def test_dataset_rejects_excluded_record_if_it_appears_in_cover_plan(tmp_path):
    row = {'stego_run_id':KNOWN_CUTOFF['run_id'], 'sample_id':KNOWN_CUTOFF['sample_id']}
    with pytest.raises(ConfigurationError, match='leaked into paired dataset'):
        dataset.paired_dataset(cover_plan=[row], cover_root=tmp_path,
            stego_root=tmp_path, stage6_run_root=tmp_path,
            split_manifest={'rows':[]}, expected_pairs=5639,
            documented_failures={KNOWN_CUTOFF['run_id']:KNOWN_CUTOFF})
