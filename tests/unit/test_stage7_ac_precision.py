from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import stage7_ac_precision as ac
from vkr_benchmark.errors import ConfigurationError
from vkr_benchmark.steganalysis.plan import build_trainval_stego_plan
from vkr_benchmark.methods.arithmetic import ArithmeticConfig


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    rows, prompts = [], []
    for i, partition in enumerate(['test']*24+['train']*96+['validation']*32):
        pid=f's6p{i+1:04d}' if partition=='test' else f's7p{i-23:04d}'
        text=f'unique context {i}'
        rows.append(dict(prompt_id=pid, article_id=f'article-{i}',partition=partition,
                         context_sha256=ac.sha(text.encode())))
        if partition!='test':
            prompts.append(dict(prompt_id=pid,text=text,source='test',source_revision='rev',language='en'))
    prompt_file=tmp_path/'prompts.jsonl'
    prompt_file.write_text(''.join(json.dumps(r)+'\n' for r in prompts),encoding='utf-8')
    split=dict(schema_version='stage7.prompt_split.v1', rows=rows,
               local_prompt_text_sha256=ac.sha(prompt_file.read_bytes()))
    split_file=tmp_path/'split.json'; split_file.write_bytes(ac.json_bytes(split))
    monkeypatch.setattr(ac,'SPLIT',split_file)
    monkeypatch.setattr(ac,'PROMPTS',prompt_file)
    return split, prompt_file


def test_selection_is_validation_only_and_order_independent(inputs):
    split,_=inputs
    groups=ac.select_groups(split)
    assert len(groups)==16 and all(r['partition']=='validation' for r in groups)
    assert groups==ac.select_groups(dict(split,rows=split['rows'][::-1]))


def test_plan_matches_existing_p32_config_and_controls(inputs):
    split,_=inputs
    plan,prepared,_=ac.build_plan()
    frozen=ac.read_json(ac.FROZEN)
    existing=build_trainval_stego_plan(split_manifest=split,frozen_manifest=frozen,project_root=ac.ROOT)
    baseline={(r['model_id'],r['prompt_id']):r for r in existing if r['method_id']=='arithmetic_coding' and r['method_point_id']=='p32-k2048'}
    assert len(prepared)==len(plan['rows'])==96
    for model in ac.MODELS:
        rows=[r for r in plan['rows'] if r['model_id']==model]
        assert len(rows)==48
        for group in plan['groups']:
            subset=[r for r in rows if r['article_id']==group['article_id']]
            assert [r['precision'] for r in subset]==[16,32,48]
            expected=baseline[(model,group['prompt_id'])]
            for row in subset:
                conf=row['config']
                assert ArithmeticConfig.from_mapping(conf['method']['params']).precision==row['precision']
                assert conf['method']['params']['top_k']==2048
                reference=json.loads(json.dumps(expected['canonical_run_config']))
                reference['method']['params']['precision']=row['precision']
                assert reference==conf
                if row['precision']==32:
                    assert row['run_id']==expected['stego_run_id']


def test_plan_rejects_prompt_hash_mismatch(inputs):
    _,path=inputs
    path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ConfigurationError,match='prompt file hash'):
        ac.build_plan()


def test_write_once_refuses_change(tmp_path):
    path=tmp_path/'plan.json'
    ac.write_once(path,b'first')
    ac.write_once(path,b'first')
    with pytest.raises(ConfigurationError,match='refuse to overwrite'):
        ac.write_once(path,b'second')
    assert path.read_bytes()==b'first'


def test_hard_failure_even_with_exact_token_roundtrip(tmp_path,inputs):
    plan,_,_=ac.build_plan(); row=plan['rows'][0]
    assert ac.existing_status(tmp_path,row) is None
    directory=tmp_path/'runs'/row['run_id']; directory.mkdir(parents=True)
    (directory/'config.json').write_bytes(ac.json_bytes(row['config']))
    (directory/'result.json').write_bytes(ac.json_bytes({'status':'error','run_id':row['run_id']}))
    (directory/'failure_diagnostics.json').write_bytes(ac.json_bytes({'failure_phase':'decode','sender_token_count':256,'payload_bits':100,'token_sequence_roundtrip_exact':True}))
    with pytest.raises(FileNotFoundError):
        ac.existing_status(tmp_path,row)
    (directory/'sender_replay_metrics.json').write_bytes(ac.json_bytes({'run_id':row['run_id'],'carrier_tokens':256,'payload_bits':100,'sender_replay_exact':True}))
    assert ac.existing_status(tmp_path,row)=='hard_failure'
    (directory/'failure_diagnostics.json').unlink()
    with pytest.raises(FileNotFoundError):
        ac.existing_status(tmp_path,row)


@pytest.mark.parametrize('precision', [16, 32, 48])
def test_sender_replay_recovers_dual_kl_and_rejects_changed_payload(inputs, precision):
    from dataclasses import replace
    from test_stage7_stego_plan import TinyLM
    from vkr_benchmark.distributions import ReferenceDistributionBuilder
    from vkr_benchmark.errors import RoundtripFailureDiagnostics
    from vkr_benchmark.inputs import create_secret_source
    from vkr_benchmark.runner import create_method_runtime, run_experiment
    from vkr_benchmark.runner.streaming import encode_fixed_carrier_tokens, method_environment_from_builder
    plan,prepared,prompts=ac.build_plan()
    row=next(r for r in plan['rows'] if r['precision']==precision)
    cfg,_=prepared[row['run_id']]
    lm=TinyLM()
    execution=run_experiment(config=cfg,prompt_registry=prompts,lm_adapter=lm)
    encoded=execution.roundtrip.encode
    diag=RoundtripFailureDiagnostics(failure_phase='decode',sender_token_ids=encoded.carrier_token_ids,
        payload_bits=encoded.payload_bits,secret_bits_read=encoded.secret_bits_read)
    metrics=ac.replay_sender_metrics(cfg,lm,prompts,diag)
    assert metrics['sender_replay_exact']
    assert metrics['carrier_tokens']==256
    assert metrics['tvd_mean']==execution.distribution_distortion_metrics.tvd_mean
    for direction in ('ref_to_stego','stego_to_ref'):
        name=f'kl_{direction}_mean_bits'
        expected=getattr(execution.distribution_distortion_metrics,name)
        assert metrics[name]==('inf' if expected==float('inf') else expected)
    with pytest.raises(ConfigurationError,match='replay differs'):
        ac.replay_sender_metrics(cfg,lm,prompts,replace(diag,payload_bits=diag.payload_bits+1))
