#!/usr/bin/env python3
"""Isolated Stage-7 validation-only Arithmetic Coding precision study."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vkr_benchmark.config import (ExperimentConfig, GenerationRunConfig, LocalModelConfig,
                                  MethodRunConfig, TerminationConfig)
from vkr_benchmark.errors import ConfigurationError, MethodError, get_roundtrip_failure_diagnostics
from vkr_benchmark.inputs import PromptRegistry
from vkr_benchmark.steganalysis.protocol import DOMAIN, secret_id_for_prompt
from vkr_benchmark.storage import persist_experiment_execution, persist_failed_run, successful_run_is_complete
from vkr_benchmark.storage.run_store import materialize_run_config, compute_run_id

CACHE = ROOT / 'data/stage7/cache/ac_precision'
PLAN = ROOT / 'results/stage7/analysis/ac_precision_plan.json'
SPLIT = ROOT / 'results/stage7/splits/split_manifest.json'
FROZEN = ROOT / 'results/stage6/dataset_v1.0_manifest.json'
PROMPTS = ROOT / 'data/stage7/prompts.jsonl'
PRECISIONS = (16, 32, 48)
MODELS = ('meta-llama/Llama-3.2-3B', 'Qwen/Qwen3-4B-Base')


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_bytes())


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')


def write_once(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise ConfigurationError(f'refuse to overwrite different artifact: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(data)


def select_groups(split: dict) -> list[dict]:
    rows = split['rows']
    val = [r for r in rows if r['partition'] == 'validation']
    if (split['schema_version'] != 'stage7.prompt_split.v1' or len(rows) != 152 or
            Counter(r['partition'] for r in rows) != {'test':24,'train':96,'validation':32} or
            len({r['article_id'] for r in rows}) != 152 or
            len({r['prompt_id'] for r in rows}) != 152 or
            len({r['context_sha256'] for r in rows}) != 152):
        raise ConfigurationError('invalid Stage-7 split')
    return sorted(val, key=lambda r: (sha(f"{DOMAIN}|implementation-factors|seed=741|article_id={r['article_id']}".encode()), r['article_id']))[:16]


def experiment(group: dict, info: dict, precision: int, point: dict) -> ExperimentConfig:
    if precision not in PRECISIONS:
        raise ConfigurationError('unexpected precision')
    return ExperimentConfig(
        benchmark_version='1.0', run_kind='normalized', model_config_path=ROOT / info['config_path'],
        prompt_id=group['prompt_id'], secret_id=secret_id_for_prompt(group['prompt_id'], 'r01'),
        method=MethodRunConfig(method_id='arithmetic_coding', params={'precision':precision, 'top_k':2048},
                               implementation_revision=point['implementation_revision']),
        generation=GenerationRunConfig(),
        termination=TerminationConfig(mode='fixed_carrier_tokens', target_carrier_tokens=256))


def build_plan():
    split, frozen = read_json(SPLIT), read_json(FROZEN)
    if sha(PROMPTS.read_bytes()) != split['local_prompt_text_sha256']:
        raise ConfigurationError('Stage-7 prompt file hash mismatch')
    prompts = PromptRegistry.from_jsonl(PROMPTS)
    groups = select_groups(split)
    points = [p for p in frozen['operating_points'] if p['method_id']=='arithmetic_coding' and p['method_point_id']=='p32-k2048']
    if len(points)!=1 or points[0]['params']!={'precision':32,'top_k':2048} or points[0]['termination']!={'mode':'fixed_carrier_tokens','target_carrier_tokens':256}:
        raise ConfigurationError('frozen AC point changed')
    point = points[0]
    infos = {i['model_id']: i for i in frozen['inputs']['models'].values()}
    if set(infos)!=set(MODELS):
        raise ConfigurationError('frozen model inventory changed')
    rows, prepared = [], {}
    for model_id in MODELS:
        info=infos[model_id]
        model_path=ROOT / info['config_path']
        if sha(model_path.read_bytes()) != info['config_sha256']:
            raise ConfigurationError('frozen model configuration hash mismatch')
        model = LocalModelConfig.from_json(model_path, project_root=ROOT)
        if (model.model_id, model.revision)!=(model_id, info['revision']):
            raise ConfigurationError('model identity mismatch')
        for group in groups:
            if sha(prompts.get(group['prompt_id']).text.encode()) != group['context_sha256']:
                raise ConfigurationError('prompt context hash mismatch')
            for precision in PRECISIONS:
                cfg=experiment(group,info,precision,point)
                raw=materialize_run_config(cfg,model)
                run_id=compute_run_id(raw)
                rows.append({'run_id':run_id,'model_id':model_id,'article_id':group['article_id'],
                             'prompt_id':group['prompt_id'],'precision':precision,'config':raw})
                prepared[run_id]=(cfg,model)
    if len(rows)!=96 or len(prepared)!=96:
        raise ConfigurationError('expected 96 unique runs')
    plan={'schema_version':'stage7.ac_precision_plan.v1','role':'validation_only_diagnostic',
          'selection_seed':741,'selection_rule':f'{DOMAIN}|implementation-factors|seed=741|article_id=<id>',
          'split_sha256':sha(SPLIT.read_bytes()),'frozen_manifest_sha256':sha(FROZEN.read_bytes()),
          'prompts_sha256':sha(PROMPTS.read_bytes()),'groups':groups,'rows':rows}
    return plan,prepared,prompts


def replay_sender_metrics(cfg, lm, prompts, diag):
    """Recover sender metrics lost by the canonical hard-failure result.

    Accept only a deterministic replay matching the original captured sender.
    This does not recover receiver BER, which remains unavailable.
    """
    from vkr_benchmark.distributions import ReferenceDistributionBuilder
    from vkr_benchmark.inputs import create_secret_source
    from vkr_benchmark.metrics import (compute_capacity_entropy_metrics,
        compute_distribution_distortion_metrics, compute_raw_lm_quality_metrics)
    from vkr_benchmark.runner import create_method_runtime
    from vkr_benchmark.runner.streaming import encode_fixed_carrier_tokens, method_environment_from_builder
    builder=ReferenceDistributionBuilder(token_space=lm.token_space,policy=cfg.generation.to_policy())
    runtime=create_method_runtime(cfg)
    encoded=encode_fixed_carrier_tokens(lm_adapter=lm,reference_builder=builder,method=runtime.method,
        method_config=cfg.method.params,environment=method_environment_from_builder(builder),
        prompt_text=prompts.get(cfg.prompt_id).text,carrier_tokens=256,
        secret_source=create_secret_source(cfg.secret_id),method_random_source=runtime.encoder_random_source,key=cfg.method.key)
    if (encoded.carrier_token_ids!=diag.sender_token_ids or encoded.payload_bits!=diag.payload_bits or
            encoded.secret_bits_read!=diag.secret_bits_read):
        raise ConfigurationError('hard failure sender replay differs from captured sender')
    capacity=compute_capacity_entropy_metrics(payload_bits=encoded.payload_bits,
        reference_entropies_bits=encoded.step_reference_entropy_bits)
    distortion=compute_distribution_distortion_metrics(encoded.step_distribution_distortion)
    quality=compute_raw_lm_quality_metrics(encoded.step_raw_lm_nll_nats)
    result={'role':'hard_failure_sender_replay','sender_replay_exact':True,
            'payload_bits':capacity.payload_bits,'carrier_tokens':capacity.carrier_tokens,
            'bits_per_token':capacity.bits_per_token,'entropy_utilization':capacity.entropy_utilization,
            'ppl_raw_lm':quality.ppl_raw_lm,'tvd_mean':distortion.tvd_mean}
    for direction in ('ref_to_stego','stego_to_ref'):
        for suffix in ('mean_bits','median_bits','p95_bits','max_bits','infinite_steps','finite_steps'):
            name=f'kl_{direction}_{suffix}'
            result[name]=getattr(distortion,name)
    return {k:('inf' if isinstance(v,float) and v==math.inf else v) for k,v in result.items()}


def existing_status(root: Path, row: dict) -> str | None:
    directory=root/'runs'/row['run_id']
    if not directory.exists():
        return None
    if read_json(directory/'config.json') != row['config']:
        raise ConfigurationError(f"existing config mismatch: {row['run_id']}")
    result=read_json(directory/'result.json')
    if result['run_id']!=row['run_id']:
        raise ConfigurationError('existing result identity mismatch')
    if result['status']=='ok' and successful_run_is_complete(root,row['run_id']):
        return 'ok'
    if result['status']=='error':
        diag=read_json(directory/'failure_diagnostics.json')
        if diag['failure_phase'] in ('decode', 'text_transport') and diag['sender_token_count']==256:
            replay=read_json(directory/'sender_replay_metrics.json')
            if (replay['sender_replay_exact'] is True and replay['run_id']==row['run_id'] and
                    replay['carrier_tokens']==256 and replay['payload_bits']==diag['payload_bits']):
                return 'hard_failure'
    raise ConfigurationError(f"incomplete or unexpected result: {row['run_id']}")


def execute(plan,prepared,prompts,args):
    from vkr_benchmark.lm import HFCausalLMAdapter
    from vkr_benchmark.runner import run_experiment
    if PLAN.read_bytes()!=json_bytes(plan):
        raise ConfigurationError('saved plan differs from deterministic reconstruction')
    rows=[r for r in plan['rows'] if r['model_id']==args.model_id]
    if args.limit is not None:
        rows=rows[:args.limit]
    # Fixed isolated output root, never canonical Stage 6 or detector dataset.
    root=CACHE/'execution'
    pending=[]
    counts={'ok':0,'hard_failure':0,'skipped_ok':0,'skipped_hard_failure':0}
    for row in rows:
        status=existing_status(root,row)
        if status is None:
            pending.append(row)
        elif not args.resume:
            raise ConfigurationError('existing run: use --resume; never overwrite')
        else:
            counts['skipped_'+status]+=1
    if pending:
        lm=HFCausalLMAdapter.from_local_config(prepared[pending[0]['run_id']][1])
        for row in pending:
            cfg,model=prepared[row['run_id']]
            try:
                output=run_experiment(config=cfg,prompt_registry=prompts,lm_adapter=lm)
            except MethodError as exc:
                diag=get_roundtrip_failure_diagnostics(exc)
                # Preserve attributed failures even if token IDs match: a numerical
                # decoder failure must not be silently discarded. Other failures stop execution.
                if (diag is None or diag.failure_phase not in ('decode', 'text_transport')
                        or len(diag.sender_token_ids)!=256):
                    raise
                # Preserve the original failure before any diagnostic replay.
                artifacts=persist_failed_run(config=cfg,model_config=model,error=exc,results_root=root,update_summary=False)
                replay=replay_sender_metrics(cfg,lm,prompts,diag)
                write_once(artifacts.run_directory/'sender_replay_metrics.json',json_bytes(dict(replay,run_id=row['run_id'])))
                status='hard_failure'
            else:
                persist_experiment_execution(output,model_config=model,results_root=root,update_summary=False)
                status='ok'
            if existing_status(root,row)!=status:
                raise ConfigurationError('persisted run verification failed')
            counts[status]+=1
            print(f"{status} {row['run_id']} {row['prompt_id']} p={row['precision']}",flush=True)
    print(json.dumps({'status':'complete','model_id':args.model_id,'selected':len(rows),**counts},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    prepare=sub.add_parser('prepare')
    prepare.add_argument('--write-plan',action='store_true')
    run=sub.add_parser('run')
    run.add_argument('--model-id',choices=MODELS,required=True)
    mode=run.add_mutually_exclusive_group(required=True)
    mode.add_argument('--limit',type=int,choices=range(1,49),metavar='N')
    mode.add_argument('--all',action='store_true')
    run.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    plan,prepared,prompts=build_plan()
    if args.command=='prepare':
        payload=json_bytes(plan)
        if PLAN.exists() and PLAN.read_bytes()!=payload:
            raise ConfigurationError('existing plan differs')
        if args.write_plan:
            write_once(PLAN,payload)
        print(json.dumps({'status':'plan_verified','plan_sha256':sha(payload),'plan_written':args.write_plan,
                          'article_groups':16,'run_count':96,'per_model':48,'precision':list(PRECISIONS),
                          'partition':'validation','test_access':'none'},indent=2))
    else:
        execute(plan,prepared,prompts,args)


if __name__=='__main__':
    main()
