"""Strict Stage-7-only reconciliation of Stage-6 newline-normalized index hashes.

The early Stage-7 Stage-6 audit used Path.read_text() and thus calculated a
SHA-256 of a newline-normalized text rather than the on-disk carrier bytes.
This module allows an explicitly audited, metadata-only correction, but *only*
if each mismatched digest is exactly reproduced by that legacy behaviour.
The frozen Stage-6 run artifacts are never written to.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from vkr_benchmark.errors import ConfigurationError


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _by_id(rows: list[dict[str, Any]], field: str, description: str) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row.get(field)
        if not isinstance(key, str) or not key or key in values:
            raise ConfigurationError(f'duplicate/invalid {field} in {description}: {key!r}')
        values[key] = row
    return values


def reconcile_stage6_hashes(
    *, old_index: list[dict[str, Any]], fresh_index: list[dict[str, Any]],
    cover_plan: list[dict[str, Any]], stage6_run_root: Path,
    expected_test_rows: int = 1800, expected_pairs: int = 5639,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return corrected *copies* after proving every change is legacy newline folding.

    No filesystem writes. All metadata besides Stage-6 stego_text_sha256 must
    match the fresh audit exactly; every Stage-6 plan row must match the old
    index. Any unexplained mismatch aborts the complete operation.
    """
    if len(old_index) != expected_test_rows or len(fresh_index) != expected_test_rows:
        raise ConfigurationError('Stage-6 hash reconciliation requires the full frozen test inventory')
    if len(cover_plan) != expected_pairs:
        raise ConfigurationError('Stage-6 hash reconciliation requires the full existing cover plan')
    old_by_run = _by_id(old_index, 'stego_run_id', 'old Stage-6 index')
    fresh_by_run = _by_id(fresh_index, 'stego_run_id', 'fresh Stage-6 audit')
    _by_id(old_index, 'sample_id', 'old Stage-6 index')
    _by_id(cover_plan, 'sample_id', 'existing Stage-7 cover plan')
    if set(old_by_run) != set(fresh_by_run):
        raise ConfigurationError('Stage-6 run membership changed between frozen index and fresh audit')
    if Counter(row.get('partition') for row in cover_plan) != Counter({
        'train': 2880, 'validation': 959, 'test': 1800,
    }) and expected_pairs == 5639:
        raise ConfigurationError('existing Stage-7 cover plan has unexpected split counts')

    changed: list[dict[str, str]] = []
    fresh_hashes: dict[str, str] = {}
    for run_id, old in old_by_run.items():
        fresh = fresh_by_run[run_id]
        old_without = {k: v for k, v in old.items() if k != 'stego_text_sha256'}
        fresh_without = {k: v for k, v in fresh.items() if k != 'stego_text_sha256'}
        if old_without != fresh_without:
            raise ConfigurationError(f'Stage-6 source metadata changed, not just hashes: {run_id}')
        before = old.get('stego_text_sha256')
        after = fresh.get('stego_text_sha256')
        if not all(isinstance(d, str) and len(d) == 64 for d in (before, after)):
            raise ConfigurationError(f'invalid Stage-6 text digests for {run_id}')
        fresh_hashes[run_id] = after
        if before != after:
            # The old audit was performed with Path.read_text(). This only
            # affects successful runs with raw CR or CRLF in stegotext.txt.
            if old.get('stego_status') != 'ok' or old.get('stego_text_source') != 'stegotext.txt':
                raise ConfigurationError(f'non-newline Stage-6 hash mismatch: {run_id}')
            path = stage6_run_root / run_id / 'stegotext.txt'
            try:
                raw = path.read_bytes()
                legacy = path.read_text(encoding='utf-8')
                exact = raw.decode('utf-8')
            except (OSError, UnicodeError) as exc:
                raise ConfigurationError(f'cannot verify Stage-6 original bytes: {run_id}: {exc}') from exc
            if (b'\r' not in raw or _sha(legacy) != before or
                    hashlib.sha256(raw).hexdigest() != after or legacy == exact):
                raise ConfigurationError(f'unexplained Stage-6 hash mismatch: {run_id}; refuse repair')
            changed.append({
                'sample_id': old['sample_id'], 'run_id': run_id,
                'legacy_normalized_sha256': before, 'raw_utf8_sha256': after,
            })

    new_plan: list[dict[str, Any]] = []
    seen_test: set[str] = set()
    for row in cover_plan:
        run_id = row.get('stego_run_id')
        if row.get('partition') != 'test':
            if run_id in old_by_run:
                raise ConfigurationError(f'frozen Stage-6 run leaked into non-test partition: {run_id}')
            new_plan.append(dict(row))
            continue
        old = old_by_run.get(run_id)
        if old is None or run_id in seen_test:
            raise ConfigurationError(f'extra/duplicate Stage-6 test run in cover plan: {run_id}')
        seen_test.add(run_id)
        for key, value in old.items():
            if key == 'stego_text_source':
                continue  # source provenance is intentionally omitted from cover plan
            if key not in row or row[key] != value:
                raise ConfigurationError(f'Stage-6 cover plan differs from original source index: {run_id} / {key}')
        updated = dict(row)
        updated['stego_text_sha256'] = fresh_hashes[run_id]
        new_plan.append(updated)
    if seen_test != set(old_by_run):
        raise ConfigurationError('Stage-6 test sources missing from cover plan')

    new_index = [dict(fresh_by_run[row['stego_run_id']]) for row in old_index]
    changed.sort(key=lambda r: r['run_id'])
    return new_index, new_plan, {
        'schema_version': 'stage7.stage6_text_hash_reconciliation.v1',
        'status': 'verified_legacy_newline_normalization',
        'test_source_count': len(old_index),
        'cover_plan_pair_count': len(cover_plan),
        'changed_hash_count': len(changed),
        'unchanged_hash_count': len(old_index) - len(changed),
        'changed_stage6_sources': changed,
        'scope': 'Stage-7 derived metadata only; Stage-6 source artifacts untouched',
    }
