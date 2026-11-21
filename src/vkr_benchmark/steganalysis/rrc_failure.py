"""Stage-7-only, strictly allowlisted RRC cutoff decision and replay gate.

No secret bits, method keys, raw stegotext or incomplete sender token IDs are
emitted. This module never modifies the frozen Stage-6 data or any run folder.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from vkr_benchmark.errors import ConfigurationError

EXCEPTION_SCHEMA = 'stage7.rrc_nontermination_exception.v1'
FAILURE_SCHEMA = 'stage7.stego_nontermination.v1'
REPLAY_SCHEMA = 'stage7.rrc_nontermination_replay.v1'
KNOWN_CUTOFF = {
    'run_id': '0b0e22af25613edb',
    'sample_id': '79775985a169f917bfe94a05',
    'model_id': 'Qwen/Qwen3-4B-Base',
    'prompt_id': 's7p0082',
    'article_id': '19496d97273879cee7e2a41de48fa2d6f445459e',
    'partition': 'validation',
    'method_id': 'rrc',
    'method_point_id': 'payload1024-margin40',
    'target_payload_bits': 1024,
    'max_carrier_tokens': 2048,
    'generated_carrier_tokens': 2048,
    'termination_candidates': 0,
    'verification_rejections': 0,
}
EXCEPTION_FIELDS = (
    'run_id', 'sample_id', 'model_id', 'prompt_id', 'article_id',
    'partition', 'method_id', 'method_point_id', 'target_payload_bits',
    'max_carrier_tokens', 'generated_carrier_tokens',
    'termination_candidates', 'verification_rejections',
)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f'cannot read Stage-7 RRC evidence {path}: {exc}') from exc
    if not isinstance(value, dict):
        raise ConfigurationError(f'expected JSON object in {path}')
    return value


def canonical_config_sha256(config: dict[str, Any]) -> str:
    """Fingerprint canonical run config; never disclose its method key."""
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=False,
                           separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(canonical).hexdigest()


def validate_registered_cutoffs(
    *, trainval_plan: list[dict[str, Any]], stego_root: Path,
    failure_root: Path, exception_config_path: Path,
    replay_report_path: Path | None = None,
    require_replay: bool = True,
) -> dict[str, dict[str, Any]]:
    """Fail closed unless one exact, *reproduced* failure is documented.

    All generated successes remain mandatory; only an explicitly enumerated
    Stage-7 RRC cutoff may lack a matched control. No arbitrary missing sample
    can be filtered by this function.
    """
    config = _json(exception_config_path)
    if (config.get('schema_version') != EXCEPTION_SCHEMA or
            config.get('decision') != 'conditional_on_matching_replay' or
            config.get('expected_total_failed_runs') != 1):
        raise ConfigurationError('unexpected Stage-7 RRC exception decision')
    entries = config.get('exceptions')
    if not isinstance(entries, list) or len(entries) != 1:
        raise ConfigurationError('Stage-7 exception must enumerate exactly one known cutoff')
    expected = entries[0]
    if (not isinstance(expected, dict) or
            expected != KNOWN_CUTOFF or
            set(expected) != set(EXCEPTION_FIELDS) or
            expected.get('method_id') != 'rrc' or
            expected.get('method_point_id') != 'payload1024-margin40' or
            expected.get('partition') != 'validation' or
            expected.get('target_payload_bits') != 1024 or
            expected.get('max_carrier_tokens') != 2048 or
            expected.get('generated_carrier_tokens') != 2048 or
            expected.get('termination_candidates') != 0 or
            expected.get('verification_rejections') != 0):
        raise ConfigurationError('Stage-7 RRC exception fields changed')
    plan_by_run = {row['stego_run_id']: row for row in trainval_plan}
    if len(plan_by_run) != len(trainval_plan):
        raise ConfigurationError('Stage-7 trainval plan has duplicate run IDs')
    rid = expected['run_id']
    row = plan_by_run.get(rid)
    if row is None or any(row.get(field) != expected[field] for field in (
            'sample_id', 'model_id', 'prompt_id', 'article_id',
            'partition', 'method_id', 'method_point_id')):
        raise ConfigurationError('allowlisted RRC cutoff disagrees with frozen trainval plan')
    term = row['canonical_run_config']['termination']
    method = row['canonical_run_config']['method']
    if (term != {'mode': 'fixed_payload_bits', 'target_payload_bits': 1024,
                 'max_carrier_tokens': 2048} or
            method['id'] != 'rrc' or
            method['params'] != {'precision_margin': 40, 'verify_termination': True}):
        raise ConfigurationError('RRC cutoff config no longer equals frozen Stage-6 point')

    actual_files = {p.stem for p in failure_root.glob('*.json')} if failure_root.exists() else set()
    if actual_files != {rid}:
        raise ConfigurationError('Stage-7 RRC failure registry differs from allowlist: '
                                 f'actual={sorted(actual_files)} expected={[rid]}')
    if (stego_root / rid).exists():
        raise ConfigurationError('allowlisted RRC failure also has a stego success directory')
    record = _json(failure_root / f'{rid}.json')
    if (record.get('schema_version') != FAILURE_SCHEMA or
            record.get('error_type') != 'Stage7FixedPayloadNonTermination' or
            record.get('is_usable_stegotext') is not False or
            record.get('canonical_run_config') != row['canonical_run_config'] or
            any(record.get(field) != expected[field] for field in EXCEPTION_FIELDS)):
        raise ConfigurationError('recorded nontermination does not match allowlisted RRC evidence')

    if not require_replay:
        return {rid: expected}
    if replay_report_path is None:
        raise ConfigurationError('explicit RRC replay report path is required')
    replay = _json(replay_report_path)
    if (replay.get('schema_version') != REPLAY_SCHEMA or
            replay.get('status') != 'reproduced' or
            replay.get('run_id') != rid or
            replay.get('sample_id') != expected['sample_id'] or
            replay.get('model_id') != expected['model_id'] or
            replay.get('prompt_id') != expected['prompt_id'] or
            replay.get('prompt_sha256') != row['context_sha256'] or
            replay.get('canonical_config_sha256') !=
            canonical_config_sha256(row['canonical_run_config']) or
            replay.get('model_revision') != row['canonical_run_config']['model']['revision'] or
            not isinstance(replay.get('replays'), list) or not replay['replays']):
        raise ConfigurationError('RRC replay report missing or inconsistent with frozen run')
    outcomes = replay['replays']
    if any(not isinstance(out, dict) or
           out.get('terminated') is not False or
           any(out.get(field) != expected[field] for field in (
               'generated_carrier_tokens', 'termination_candidates',
               'verification_rejections')) or
           not isinstance(out.get('token_ids_sha256'), str) or
           len(out['token_ids_sha256']) != 64 or
           not isinstance(out.get('snapshots'), list) or not out['snapshots'] or
           out['snapshots'][-1].get('step') != 2048 or
           out['snapshots'][-1].get('interval_width_at_most_one') is not False or
           not isinstance(out['snapshots'][-1].get('interval_log2_width'), (int, float)) or
           not 0 < out['snapshots'][-1]['interval_log2_width'] <= 1024.0001
           for out in outcomes):
        raise ConfigurationError('RRC replay does not confirm recorded nontermination')
    if len({out['token_ids_sha256'] for out in outcomes}) != 1:
        raise ConfigurationError('RRC repeated replay generated different sender token streams')
    return {rid: expected}
