#!/usr/bin/env python3
"""Replay one recorded Stage-7 Qwen RRC cutoff without modifying stego data.

Only aggregated interval-width checkpoints and token-ID hashes are persisted.
The original RRC encoder, reference distribution, 1024-bit payload and
2048-token ceiling are *unchanged*; no secret or partial text is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from vkr_benchmark.config import LocalModelConfig  # noqa: E402
from vkr_benchmark.distributions import ReferenceDistributionBuilder  # noqa: E402
from vkr_benchmark.errors import ConfigurationError  # noqa: E402
from vkr_benchmark.inputs import PromptRegistry  # noqa: E402
from vkr_benchmark.lm import HFCausalLMAdapter  # noqa: E402
from vkr_benchmark.steganalysis.dataset import read_jsonl  # noqa: E402
from vkr_benchmark.steganalysis.plan import validate_stego_plan_row  # noqa: E402
from vkr_benchmark.steganalysis.rrc_diagnostic import replay_rrc  # noqa: E402
from vkr_benchmark.steganalysis.rrc_failure import (  # noqa: E402
    KNOWN_CUTOFF, REPLAY_SCHEMA, canonical_config_sha256,
    validate_registered_cutoffs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-config', type=Path, required=True)
    parser.add_argument('--prompts-file', type=Path, default=ROOT/'data/stage7/prompts.jsonl')
    parser.add_argument('--trainval-plan', type=Path, default=ROOT/'data/stage7/dataset/trainval_stego_plan.jsonl')
    parser.add_argument('--stego-root', type=Path, default=ROOT/'results/stage7/stego_runs')
    parser.add_argument('--failure-root', type=Path, default=ROOT/'results/stage7/stego_failures')
    parser.add_argument('--exception-config', type=Path, default=ROOT/'configs/stage7/rrc_nontermination_exception.json')
    parser.add_argument('--output', type=Path, default=ROOT/'results/stage7/rrc_nontermination_replay.json')
    parser.add_argument('--replays', type=int, default=1, choices=(1, 2))
    args = parser.parse_args()

    plan = read_jsonl(args.trainval_plan)
    if len(plan) != 3840:
        raise ConfigurationError('expected exactly 3840 frozen Stage-7 train/val plans')
    allowed = validate_registered_cutoffs(
        trainval_plan=plan, stego_root=args.stego_root,
        failure_root=args.failure_root,
        exception_config_path=args.exception_config,
        require_replay=False,
    )
    assert set(allowed) == {KNOWN_CUTOFF['run_id']}
    row = next(r for r in plan if r['stego_run_id'] == KNOWN_CUTOFF['run_id'])
    model_cfg = LocalModelConfig.from_json(args.model_config, project_root=ROOT)
    cfg = validate_stego_plan_row(row, model_cfg)
    prompts = PromptRegistry.from_jsonl(args.prompts_file)
    prompt = prompts.get(row['prompt_id']).text
    prompt_hash = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
    if args.output.exists():
        raise ConfigurationError(f'RRC replay report already exists; do not overwrite: {args.output}')

    lm = HFCausalLMAdapter.from_local_config(model_cfg)
    reference = ReferenceDistributionBuilder(
        token_space=lm.token_space, policy=cfg.generation.to_policy())
    outcomes = []
    for i in range(args.replays):
        outcome = replay_rrc(
            config=cfg, lm_adapter=lm, prompt_text=prompt,
            reference_builder=reference,
        )
        outcomes.append(outcome)
        print(f'RRC replay {i+1}/{args.replays}: '
              f'terminated={outcome["terminated"]}, '
              f'carriers={outcome["generated_carrier_tokens"]}, '
              f'candidates={outcome["termination_candidates"]}, '
              f'rejections={outcome["verification_rejections"]}', flush=True)
    matching = all(
        outcome['terminated'] is False and
        all(outcome[key] == KNOWN_CUTOFF[key] for key in (
            'generated_carrier_tokens', 'termination_candidates',
            'verification_rejections'))
        for outcome in outcomes
    ) and len({outcome['token_ids_sha256'] for outcome in outcomes}) == 1 and all(
        out['snapshots'] and out['snapshots'][-1]['step'] == 2048 and
        out['snapshots'][-1]['interval_width_at_most_one'] is False
        for out in outcomes)
    report = {
        'schema_version': REPLAY_SCHEMA,
        'status': 'reproduced' if matching else 'mismatch',
        'run_id': row['stego_run_id'], 'sample_id': row['sample_id'],
        'model_id': model_cfg.model_id, 'model_revision': model_cfg.revision,
        'prompt_id': row['prompt_id'], 'prompt_sha256': prompt_hash,
        'canonical_config_sha256': canonical_config_sha256(row['canonical_run_config']),
        'replays': outcomes,
        'notes': ('No secret bits, method key, plaintext, or incomplete stegotext are retained. '
                  'Width is read-only from the unchanged RRC Decimal encoder. '
                  'A matching outcome establishes replay of a cutoff, not the root cause.'),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True,
                                      ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'RRC cutoff replay: {report["status"]}; diagnostics={args.output}')
    return 0 if matching else 2


if __name__ == '__main__':
    raise SystemExit(main())
