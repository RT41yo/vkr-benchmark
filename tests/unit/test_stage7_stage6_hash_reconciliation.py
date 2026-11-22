from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from vkr_benchmark.errors import ConfigurationError
from vkr_benchmark.steganalysis.stage6_hash_reconciliation import reconcile_stage6_hashes


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _fixture(root: Path, *, text: str = 'visible\rnew line\r\nlast'):
    run = root/'run-a'
    run.mkdir(parents=True)
    (run/'stegotext.txt').write_bytes(text.encode('utf-8'))
    old = {'sample_id': 'sample-a', 'partition': 'test', 'stego_run_id': 'run-a',
           'stego_status': 'ok', 'stego_text_source': 'stegotext.txt',
           'stego_text_sha256': _sha(text.replace('\r\n','\n').replace('\r','\n')),
           'carrier_tokens': 256, 'method_id': 'bins'}
    fresh = {**old, 'stego_text_sha256': _sha(text)}
    plan = [{key: val for key, val in old.items() if key != 'stego_text_source'}]
    return old, fresh, plan


def _reconcile(root, old, fresh, plan):
    return reconcile_stage6_hashes(
        old_index=[old], fresh_index=[fresh], cover_plan=plan,
        stage6_run_root=root, expected_test_rows=1, expected_pairs=1)


def test_reconciles_only_legacy_cr_or_crlf_hash_and_preserves_other_fields(tmp_path):
    old, fresh, plan = _fixture(tmp_path)
    updated_index, updated_plan, report = _reconcile(tmp_path, old, fresh, plan)
    assert report['changed_hash_count'] == 1
    assert report['changed_stage6_sources'][0]['run_id'] == 'run-a'
    assert updated_index[0]['stego_text_sha256'] == fresh['stego_text_sha256']
    assert updated_plan[0]['stego_text_sha256'] == fresh['stego_text_sha256']
    assert {k: v for k, v in updated_plan[0].items() if k != 'stego_text_sha256'} == {
        k: v for k, v in plan[0].items() if k != 'stego_text_sha256'}
    assert plan[0]['stego_text_sha256'] == old['stego_text_sha256']  # no in-place edits
    assert (tmp_path/'run-a'/'stegotext.txt').read_bytes() == b'visible\rnew line\r\nlast'


def test_rejects_actual_unknown_hash_mismatch(tmp_path):
    old, fresh, plan = _fixture(tmp_path)
    old['stego_text_sha256'] = 'f'*64
    plan[0]['stego_text_sha256'] = old['stego_text_sha256']
    with pytest.raises(ConfigurationError, match='unexplained Stage-6 hash mismatch'):
        _reconcile(tmp_path, old, fresh, plan)


def test_rejects_metadata_tampering(tmp_path):
    old, fresh, plan = _fixture(tmp_path)
    fresh['carrier_tokens'] = 257
    with pytest.raises(ConfigurationError, match='metadata changed'):
        _reconcile(tmp_path, old, fresh, plan)


def test_rejects_preexisting_plan_hash_inconsistency(tmp_path):
    old, fresh, plan = _fixture(tmp_path)
    plan[0]['stego_text_sha256'] = 'a'*64
    with pytest.raises(ConfigurationError, match='cover plan differs'):
        _reconcile(tmp_path, old, fresh, plan)


def test_noop_if_hashes_already_exact(tmp_path):
    old, fresh, plan = _fixture(tmp_path, text='ordinary\ntext')
    a, b, report = _reconcile(tmp_path, old, fresh, plan)
    assert report['changed_hash_count'] == 0
    assert a == [old]
    assert b == plan


def test_rejects_failure_diagnostic_hash_change(tmp_path):
    old, fresh, plan = _fixture(tmp_path)
    old['stego_status'] = 'error'
    fresh['stego_status'] = 'error'
    old['stego_text_source'] = 'failure_diagnostics.json:transport_text'
    fresh['stego_text_source'] = 'failure_diagnostics.json:transport_text'
    with pytest.raises(ConfigurationError, match='non-newline Stage-6 hash mismatch'):
        _reconcile(tmp_path, old, fresh, plan)


def test_rejects_missing_test_run(tmp_path):
    old, fresh, plan = _fixture(tmp_path)
    plan.clear()
    with pytest.raises(ConfigurationError, match='requires the full existing cover plan'):
        _reconcile(tmp_path, old, fresh, plan)
