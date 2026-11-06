"""Guardrails for the read-only Pilot-B direct-token audit."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/audit_stage5_direct_tokens.py"
spec = importlib.util.spec_from_file_location("audit_stage5_direct_tokens", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def result(final, incremental, complete=True):
    return SimpleNamespace(recovered_bits=tuple(final),
                           incremental_recovered_bits=tuple(incremental),
                           finalization=SimpleNamespace(complete=complete))


def test_strict_exact():
    row = audit.exact_result((1, 0), result((1, 0), (1, 0)))
    assert row["direct_roundtrip_exact"]
    assert row["recovered_extra_bits"] == 0


def test_finalization_truncation_cannot_hide_extra_incremental_bits():
    row = audit.exact_result((1, 0), result((1, 0), (1, 0, 1)))
    assert not row["direct_roundtrip_exact"]
    assert row["recovered_extra_bits"] == 1


def test_incomplete_or_length_mismatch_is_not_exact():
    assert not audit.exact_result((1, 0), result((1, 0), (1, 0), False))["direct_roundtrip_exact"]
    assert not audit.exact_result((1, 0), result((1,), (1,)))["direct_roundtrip_exact"]


def test_refuses_overwrite_and_protected_results(tmp_path):
    results = tmp_path / "results"
    with pytest.raises(ValueError, match="protected"):
        audit.protected_output(results / "runs" / "summary.json", [results])
    outside = tmp_path / "audit" / "summary.json"
    outside.parent.mkdir()
    outside.write_text("existing")
    with pytest.raises(ValueError, match="overwrite"):
        audit.protected_output(outside, [results])
