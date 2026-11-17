"""Deterministic Stage-7 generation plans derived from frozen Stage-6 points."""
from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from vkr_benchmark.config import (
    ExperimentConfig, GenerationRunConfig, LocalModelConfig, MethodRunConfig,
    TerminationConfig,
)
from vkr_benchmark.errors import ConfigurationError
from vkr_benchmark.steganalysis.protocol import (
    cover_seed, method_random_material, secret_id_for_prompt,
)
from vkr_benchmark.storage.run_store import compute_run_id, materialize_run_config


def _require_points(points: list[dict[str, Any]]) -> None:
    if len(points) != 15 or len({(p['method_id'], p['method_point_id']) for p in points}) != 15:
        raise ConfigurationError("frozen Stage-6 must define 15 distinct method-qualified points")
    if sum(p['method_id'] == 'rrc' for p in points) != 1:
        raise ConfigurationError("frozen Stage-6 must define exactly one RRC point")
    for point in points:
        term = point['termination']
        if point['method_id'] == 'rrc':
            if term != {'mode': 'fixed_payload_bits', 'target_payload_bits': 1024,
                        'max_carrier_tokens': 2048}:
                raise ConfigurationError("RRC fixed payload semantics changed")
        elif term != {'mode': 'fixed_carrier_tokens', 'target_carrier_tokens': 256}:
            raise ConfigurationError("Stage-6 streaming carrier semantics changed")


def build_trainval_stego_plan(
    *, split_manifest: dict[str, Any], frozen_manifest: dict[str, Any],
    project_root: Path,
) -> list[dict[str, Any]]:
    """Return 3840 immutable per-run plan rows, no generated text or LM weights."""
    if split_manifest.get('schema_version') != 'stage7.prompt_split.v1':
        raise ConfigurationError("Stage-7 split manifest version mismatch")
    split_rows = split_manifest['rows']
    counts = Counter(r['partition'] for r in split_rows)
    if counts != {'train': 96, 'validation': 32, 'test': 24}:
        raise ConfigurationError(f"unexpected split counts: {dict(counts)}")
    if len({r['article_id'] for r in split_rows}) != 152 or len({r['prompt_id'] for r in split_rows}) != 152:
        raise ConfigurationError("split articles/prompts are duplicated")
    if len({r['context_sha256'] for r in split_rows}) != 152:
        raise ConfigurationError("split contexts are duplicated")
    points = frozen_manifest['operating_points']
    _require_points(points)
    lm_infos = frozen_manifest['inputs']['models']
    if len(lm_infos) != 2:
        raise ConfigurationError("Stage-7 protocol requires both frozen LMs")
    models: list[tuple[dict[str, Any], LocalModelConfig]] = []
    for info in lm_infos.values():
        model_cfg = LocalModelConfig.from_json(
            project_root / info['config_path'], project_root=project_root)
        if model_cfg.model_id != info['model_id'] or model_cfg.revision != info['revision']:
            raise ConfigurationError("Stage-6 model revision/config mismatch")
        models.append((info, model_cfg))
    records = []
    for group in sorted(split_rows, key=lambda x: x['prompt_id']):
        if group['partition'] == 'test':
            continue
        pid = group['prompt_id']
        if not pid.startswith('s7p'):
            raise ConfigurationError("new train/validation prompt must have s7p id")
        rep = 'r01'
        for info, model in models:
            for point in points:
                method_id = point['method_id']
                seed, key = method_random_material(
                    article_id=group['article_id'], method_id=method_id, replicate_id=rep)
                term = point['termination']
                exp = ExperimentConfig(
                    benchmark_version='1.0', run_kind='normalized',
                    model_config_path=project_root / info['config_path'],
                    prompt_id=pid,
                    method=MethodRunConfig(
                        method_id=method_id, params=point['params'],
                        implementation_revision=point['implementation_revision'],
                        random_seed=seed, key=key),
                    generation=GenerationRunConfig(
                        temperature=1.0, top_k=None, top_p=1.0,
                        exclude_special_tokens=True, kv_cache=True),
                    secret_id=secret_id_for_prompt(pid, rep),
                    termination=TerminationConfig(**term),
                )
                cfg = materialize_run_config(exp, model)
                run_id = compute_run_id(cfg)
                sample_id = hashlib.sha256(f'stage7-cover|{run_id}'.encode()).hexdigest()[:24]
                records.append({
                    'sample_id': sample_id, 'partition': group['partition'],
                    'stego_run_id': run_id, 'stego_status': 'pending',
                    'model_id': model.model_id, 'model_revision': model.revision,
                    'model_config': info['config_path'], 'prompt_id': pid,
                    'article_id': group['article_id'], 'context_sha256': group['context_sha256'],
                    'method_id': method_id, 'method_point_id': point['method_point_id'],
                    'replicate_id': rep, 'carrier_tokens': (
                        256 if method_id != 'rrc' else None),
                    'cover_seed': cover_seed(
                        model_id=model.model_id, article_id=group['article_id'],
                        method_point_id=f"{method_id}/{point['method_point_id']}", replicate_id=rep),
                    'canonical_run_config': cfg,
                })
    if len(records) != 3840 or len({r['sample_id'] for r in records}) != len(records) or len({r['stego_run_id'] for r in records}) != len(records):
        raise ConfigurationError('Stage-7 train/validation plan not exactly 3840 unique runs')
    if Counter(r['partition'] for r in records) != {'train': 2880, 'validation': 960}:
        raise ConfigurationError('Stage-7 train/validation plan partition imbalance')
    records.sort(key=lambda r: (r['model_id'], r['prompt_id'], r['method_id'], r['method_point_id']))
    return records


def validate_stego_plan_row(row: dict[str, Any], model_cfg: LocalModelConfig) -> ExperimentConfig:
    """Rebuild experiment config from untrusted local plan and check identity."""
    raw = row['canonical_run_config']
    method = raw['method']
    term = raw['termination']
    if raw.get('model', {}).get('id') != model_cfg.model_id or raw['model']['revision'] != model_cfg.revision:
        raise ConfigurationError('plan model identity/revision mismatch')
    if row['model_id'] != model_cfg.model_id or row['model_revision'] != model_cfg.revision:
        raise ConfigurationError('plan row does not match selected LM')
    if row['prompt_id'] != raw['prompt_id'] or row['method_id'] != method['id']:
        raise ConfigurationError('plan run identity mismatch')
    if raw.get('secret_id') != secret_id_for_prompt(row['prompt_id'], row['replicate_id']):
        raise ConfigurationError('plan secret id mismatch')
    seed, key = method_random_material(article_id=row['article_id'], method_id=row['method_id'],
                                       replicate_id=row['replicate_id'])
    if method.get('random_seed') != seed or method.get('key') != key:
        raise ConfigurationError('plan method random material mismatch')
    if row['cover_seed'] != cover_seed(model_id=model_cfg.model_id, article_id=row['article_id'],
                                     method_point_id=f"{row['method_id']}/{row['method_point_id']}",
                                     replicate_id=row['replicate_id']):
        raise ConfigurationError('plan cover seed mismatch')
    exp = ExperimentConfig(
        benchmark_version=raw['benchmark_version'], run_kind=raw['run_kind'],
        model_config_path=Path(row['model_config']), prompt_id=raw['prompt_id'],
        method=MethodRunConfig(method_id=method['id'], params=method['params'],
                               implementation_revision=method.get('implementation_revision'),
                               random_seed=method.get('random_seed'), key=method.get('key')),
        generation=GenerationRunConfig(**raw['generation']),
        secret_id=raw['secret_id'], termination=TerminationConfig(**term),
    )
    if materialize_run_config(exp, model_cfg) != raw:
        raise ConfigurationError('plan canonical config mismatch')
    if compute_run_id(raw) != row['stego_run_id']:
        raise ConfigurationError('plan run id mismatch')
    if row['sample_id'] != hashlib.sha256(f"stage7-cover|{row['stego_run_id']}".encode()).hexdigest()[:24]:
        raise ConfigurationError('plan sample id mismatch')
    if exp.generation != GenerationRunConfig():
        raise ConfigurationError('Stage-7 generation differs from frozen P_reference policy')
    if (row['carrier_tokens'] is None) != (row['method_id'] == 'rrc'):
        raise ConfigurationError('plan RRC/streaming carrier semantics mismatch')
    if row['method_id'] != 'rrc' and row['carrier_tokens'] != 256:
        raise ConfigurationError('plan streaming carrier tokens not 256')
    term = exp.termination
    if row['method_id'] == 'rrc':
        if (term.mode != 'fixed_payload_bits' or term.target_payload_bits != 1024
                or term.max_carrier_tokens != 2048):
            raise ConfigurationError('Stage-7 RRC fixed payload semantics changed')
    elif term.mode != 'fixed_carrier_tokens' or term.target_carrier_tokens != 256:
        raise ConfigurationError('Stage-7 streaming termination semantics changed')
    return exp
