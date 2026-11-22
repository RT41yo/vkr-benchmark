#!/usr/bin/env python3
"""Verify/reconcile stale Stage-6 test SHA-256 fields in Stage-7 derived files.

Default mode is read-only. Explicit --apply backs up the two local derived
JSONL files and rewrites them *only* if every mismatch is proven to be legacy
Path.read_text() newline normalization. Never mutates Stage-6 inputs, runs,
or generated stego/cover texts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from vkr_benchmark.errors import ConfigurationError  # noqa: E402
from vkr_benchmark.steganalysis.dataset import jsonl_text, read_jsonl  # noqa: E402
from vkr_benchmark.steganalysis.stage6 import build_stage6_test_index  # noqa: E402
from vkr_benchmark.steganalysis.stage6_hash_reconciliation import reconcile_stage6_hashes  # noqa: E402


def _atomic_replace(path: Path, data: bytes) -> None:
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f'.{path.name}.',
                                         suffix='.tmp', delete=False) as stream:
            tmp_name = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    finally:
        if tmp_name is not None and tmp_name.exists():
            tmp_name.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage6-root', type=Path, default=ROOT/'results/runs')
    parser.add_argument('--stage6-index', type=Path, default=ROOT/'data/stage7/dataset/stage6_test_index.jsonl')
    parser.add_argument('--cover-plan', type=Path, default=ROOT/'data/stage7/dataset/all_cover_plan.jsonl')
    parser.add_argument('--backup-dir', type=Path, default=ROOT/'data/stage7/cache/stage6_hash_reconciliation')
    parser.add_argument('--report', type=Path, default=ROOT/'results/stage7/stage6_hash_reconciliation.json')
    parser.add_argument('--apply', action='store_true', help='Backup and rewrite ONLY Stage-7 derived hash metadata')
    args = parser.parse_args()
    try:
        frozen = json.loads((ROOT/'results/stage6/dataset_v1.0_manifest.json').read_text(encoding='utf-8'))
        prompts = json.loads((ROOT/'results/stage5/stage6_prompt_manifest.json').read_text(encoding='utf-8'))
        fresh, audit = build_stage6_test_index(run_root=args.stage6_root, frozen_manifest=frozen,
                                                stage6_prompts=prompts)
        old_index = read_jsonl(args.stage6_index)
        old_plan = read_jsonl(args.cover_plan)
        new_index, new_plan, report = reconcile_stage6_hashes(
            old_index=old_index, fresh_index=fresh, cover_plan=old_plan,
            stage6_run_root=args.stage6_root)
        report['fresh_audit_status'] = audit['status']
        print(f"Stage-7 Stage-6 hash reconciliation: {report['changed_hash_count']} legacy "
              f"newline-normalized hashes of {report['test_source_count']} test sources; "
              f"plan={report['cover_plan_pair_count']} pairs; apply={args.apply}")
        for change in report['changed_stage6_sources']:
            print(f"  LEGACY CR/CRLF: sample={change['sample_id']} run={change['run_id']}")
        if not args.apply:
            print('READ-ONLY: no data changed. Use --apply only after reviewing these results.')
            return 0
        if not report['changed_hash_count']:
            print('No derived hash changes needed; no files written.')
            return 0
        if (ROOT/'data/stage7/dataset/detector_records.jsonl').exists():
            raise ConfigurationError('detector_records.jsonl already exists; cannot change assembled dataset metadata')
        staged = (
            (args.stage6_index, jsonl_text(new_index).encode('utf-8')),
            (args.cover_plan, jsonl_text(new_plan).encode('utf-8')),
        )
        args.backup_dir.mkdir(parents=True, exist_ok=True)
        # Deterministic backups, with no overwriting even of identical names.
        backups = []
        for path, _ in staged:
            old_bytes = path.read_bytes()
            sha = hashlib.sha256(old_bytes).hexdigest()
            backup = args.backup_dir / (path.name + '.' + sha[:16] + '.bak')
            if backup.exists() and backup.read_bytes() != old_bytes:
                raise ConfigurationError(f'backup path collision: {backup}')
            if not backup.exists():
                backup.write_bytes(old_bytes)
            backups.append((path, backup, old_bytes))
        report['previous_index_sha256'] = hashlib.sha256(backups[0][2]).hexdigest()
        report['previous_cover_plan_sha256'] = hashlib.sha256(backups[1][2]).hexdigest()
        report['updated_index_sha256'] = hashlib.sha256(staged[0][1]).hexdigest()
        report['updated_cover_plan_sha256'] = hashlib.sha256(staged[1][1]).hexdigest()
        report_bytes = (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode('utf-8')
        if args.report.exists() and args.report.read_bytes() != report_bytes:
            raise ConfigurationError(f'existing reconciliation report differs: {args.report}')
        replaced = []
        try:
            for path, new_bytes in staged:
                _atomic_replace(path, new_bytes)
                replaced.append(path)
            args.report.parent.mkdir(parents=True, exist_ok=True)
            _atomic_replace(args.report, report_bytes)
        except Exception:
            for path, backup, old_bytes in backups:
                if path in replaced:
                    _atomic_replace(path, old_bytes)
            raise
        print(f'APPLIED: two Stage-7 derived metadata files updated; backups: {args.backup_dir}')
        print(f'Diagnostic report: {args.report}')
        return 0
    except Exception as exc:
        print(f'Stage-7 Stage-6 hash reconciliation: ERROR ({type(exc).__name__}: {exc})')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
