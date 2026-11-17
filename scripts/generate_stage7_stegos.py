#!/usr/bin/env python3
"""Create Stage-7 train/validation stegotexts without decode or metric overhead.

All files are confined to results/stage7/stego_runs/, never results/runs/ or
results/summary.parquet. Fail closed on any invalid/incomplete pre-existing run.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from vkr_benchmark.config import LocalModelConfig  # noqa: E402
from vkr_benchmark.distributions import ReferenceDistributionBuilder  # noqa: E402
from vkr_benchmark.errors import ConfigurationError  # noqa: E402
from vkr_benchmark.inputs import PromptRegistry, create_secret_source  # noqa: E402
from vkr_benchmark.runner.experiment import create_method_runtime  # noqa: E402
from vkr_benchmark.runner.streaming import (  # noqa: E402
    encode_fixed_carrier_tokens, encode_fixed_payload_bits, method_environment_from_builder,
)
from vkr_benchmark.lm import HFCausalLMAdapter  # noqa: E402
from vkr_benchmark.steganalysis.plan import validate_stego_plan_row  # noqa: E402
from vkr_benchmark.steganalysis.stego import generate_stage7_stego  # noqa: E402


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')


def _verify_existing(directory: Path, row: dict) -> bool:
    marker = directory/'stego_result.json'
    if not marker.is_file():
        raise ConfigurationError(f'incomplete pre-existing stego run: {directory}')
    old = _json(marker)
    if old.get('run_id') != row['stego_run_id'] or old.get('sample_id') != row['sample_id']:
        raise ConfigurationError(f'Stage-7 stego run identity mismatch: {directory}')
    if old.get('canonical_run_config') != row['canonical_run_config']:
        raise ConfigurationError(f'Stage-7 stego run config mismatch: {directory}')
    if not (directory/'stegotext.txt').is_file() or not (directory/'token_ids.json').is_file():
        raise ConfigurationError(f'incomplete Stage-7 stego run: {directory}')
    raw_text = (directory/'stegotext.txt').read_text(encoding='utf-8')
    ids = _json(directory/'token_ids.json')
    if hashlib.sha256(raw_text.encode()).hexdigest() != old.get('text_sha256'):
        raise ConfigurationError(f'Stage-7 stego text hash mismatch: {directory}')
    if (not isinstance(ids.get('sender_token_ids'), list) or
            len(ids['sender_token_ids']) != old.get('carrier_tokens')):
        raise ConfigurationError(f'Stage-7 sender token count mismatch: {directory}')
    if old.get('carrier_tokens') != (256 if row['method_id'] != 'rrc' else old.get('carrier_tokens')):
        raise ConfigurationError(f'Stage-7 streaming length mismatch: {directory}')
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config', type=Path, required=True)
    parser.add_argument('--prompts-file', type=Path, default=ROOT/'data/stage7/prompts.jsonl')
    parser.add_argument('--plan', type=Path, default=ROOT/'data/stage7/dataset/trainval_stego_plan.jsonl')
    parser.add_argument('--output-root', type=Path, default=ROOT/'results/stage7/stego_runs')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--method', help='Optional method id for staged smoke/execution')
    parser.add_argument('--method-point', help='Optional method-point id')
    parser.add_argument('--verify-encoder', action='store_true',
                        help='Re-run one smoke with the instrumented canonical encoder and assert identical tokens')
    args = parser.parse_args()
    if (args.limit is None) == (not args.all) or (args.limit is not None and args.limit < 1):
        parser.error('choose exactly one of --limit N (N>0) or --all')
    if args.verify_encoder and (args.all or args.limit != 1):
        parser.error('--verify-encoder is restricted to --limit 1 smoke tests')
    model_cfg = LocalModelConfig.from_json(args.model_config, project_root=ROOT)
    frozen = _json(ROOT/'results/stage6/dataset_v1.0_manifest.json')
    frozen_points = {(p['method_id'], p['method_point_id']): p
                     for p in frozen['operating_points']}
    if len(frozen_points) != 15:
        parser.error('frozen Stage-6 operating points are incomplete')
    prompts = PromptRegistry.from_jsonl(args.prompts_file)
    plan = [json.loads(line) for line in args.plan.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(plan) != 3840 or len({r['stego_run_id'] for r in plan}) != 3840:
        parser.error('Stage-7 plan must contain 3840 unique train/validation stego runs')
    model_rows = [r for r in plan if r['model_id'] == model_cfg.model_id]
    if len(model_rows) != 1920:
        parser.error('Stage-7 plan must contain exactly 1920 runs per model')
    selected = [r for r in model_rows
                if (args.method is None or r['method_id'] == args.method)
                and (args.method_point is None or r['method_point_id'] == args.method_point)]
    if not selected:
        parser.error('no Stage-7 runs matched model/method filters')
    if args.limit is not None:
        selected = selected[:args.limit]
    # Validate selected plans *before* loading the LM (multiple GB).
    prepared = []
    for row in selected:
        cfg = validate_stego_plan_row(row, model_cfg)
        frozen_point = frozen_points.get((row['method_id'], row['method_point_id']))
        if (frozen_point is None or dict(cfg.method.params) != frozen_point['params'] or
                cfg.method.implementation_revision != frozen_point['implementation_revision'] or
                row['canonical_run_config']['termination'] != frozen_point['termination']):
            raise ConfigurationError('Stage-7 planned operating point differs from frozen Stage 6')
        prompts.get(row['prompt_id'])
        prepared.append((row, cfg))
    lm = HFCausalLMAdapter.from_local_config(model_cfg)
    ok = skipped = 0
    for row, cfg in prepared:
        directory = args.output_root / row['stego_run_id']
        if directory.exists():
            if not args.resume:
                raise ConfigurationError(f'Stage-7 run exists (use --resume): {directory}')
            _verify_existing(directory, row)
            skipped += 1
            continue
        builder = ReferenceDistributionBuilder(token_space=lm.token_space,
                                               policy=cfg.generation.to_policy())
        prompt_text = prompts.get(row['prompt_id']).text
        output = generate_stage7_stego(
            config=cfg, lm_adapter=lm, reference_builder=builder, prompt_text=prompt_text)
        if args.verify_encoder:
            runtime = create_method_runtime(cfg)
            common = dict(lm_adapter=lm, reference_builder=builder, method=runtime.method,
                          method_config=cfg.method.params,
                          environment=method_environment_from_builder(builder),
                          prompt_text=prompt_text,
                          secret_source=create_secret_source(cfg.secret_id),
                          method_random_source=runtime.encoder_random_source, key=cfg.method.key)
            if cfg.termination.mode == 'fixed_carrier_tokens':
                expected = encode_fixed_carrier_tokens(
                    **common, carrier_tokens=cfg.termination.target_carrier_tokens)
            else:
                expected = encode_fixed_payload_bits(
                    **common, target_payload_bits=cfg.termination.target_payload_bits,
                    max_carrier_tokens=cfg.termination.max_carrier_tokens)
            if (output.sender_token_ids != expected.carrier_token_ids or
                    output.payload_bits != expected.payload_bits or
                    output.secret_bits_read != expected.secret_bits_read):
                raise ConfigurationError(
                    f'Stage-7 generation-only deviates from canonical sender: {row["stego_run_id"]}')
            print(f'Stage-7 canonical sender equivalence: PASS ({row["stego_run_id"]})', flush=True)
        # Write only after generation has finished; a partial directory is not
        # silently accepted by --resume.
        directory.mkdir(parents=True, exist_ok=False)
        (directory/'stegotext.txt').write_text(output.text, encoding='utf-8')
        _write(directory/'token_ids.json', {
            'prompt_token_ids': output.prompt_token_ids,
            'sender_token_ids': output.sender_token_ids,
            'receiver_token_ids': output.receiver_token_ids,
            'token_sequence_roundtrip_exact': output.token_sequence_roundtrip_exact,
            'first_token_mismatch': output.first_token_mismatch,
        })
        _write(directory/'stego_result.json', {
            'schema_version': 'stage7.stego_run.v1', 'run_id': row['stego_run_id'],
            'sample_id': row['sample_id'], 'partition': row['partition'],
            'method_id': row['method_id'], 'method_point_id': row['method_point_id'],
            'model_id': row['model_id'], 'model_revision': row['model_revision'],
            'prompt_id': row['prompt_id'], 'article_id': row['article_id'],
            'canonical_run_config': row['canonical_run_config'],
            'carrier_tokens': len(output.sender_token_ids),
            'payload_bits': output.payload_bits,
            'secret_bits_read': output.secret_bits_read,
            'termination_reason': output.termination_reason,
            'token_sequence_roundtrip_exact': output.token_sequence_roundtrip_exact,
            'text_sha256': hashlib.sha256(output.text.encode()).hexdigest(),
        })
        ok += 1
        print(f"STEGO OK {row['stego_run_id']} model={row['model_id']} "
              f"method={row['method_id']}/{row['method_point_id']} "
              f"tokens={len(output.sender_token_ids)}", flush=True)
    print(f'Stage-7 stego generation complete: {ok} new, {skipped} skipped')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
