"""Unit tests for CPU-only Pilot-B text-transport artifact analysis."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/analyze_stage5_text_transport_successful.py"
spec = importlib.util.spec_from_file_location("analyze_stage5_text_transport_successful", SCRIPT)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_sequence_diagnostics_exact():
    row = analysis.sequence_diagnostics((1, 2, 3), (1, 2, 3))
    assert row["token_sequence_exact"] is True
    assert row["first_token_mismatch"] is None
    assert row["common_prefix_tokens"] == 3
    assert row["common_suffix_tokens"] == 0
    assert row["sender_changed_middle_tokens"] == 0
    assert row["receiver_changed_middle_tokens"] == 0


def test_sequence_diagnostics_local_retokenization_change():
    row = analysis.sequence_diagnostics((10, 20, 30, 40, 50), (10, 21, 22, 40, 50))
    assert row["token_sequence_exact"] is False
    assert row["first_token_mismatch"] == 1
    assert row["common_prefix_tokens"] == 1
    assert row["common_suffix_tokens"] == 2
    assert row["sender_changed_middle_tokens"] == 2
    assert row["receiver_changed_middle_tokens"] == 2
    assert row["token_count_delta"] == 0


def test_sequence_diagnostics_length_change():
    row = analysis.sequence_diagnostics((1, 2, 3, 4), (1, 2, 9, 3, 4))
    assert row["first_token_mismatch"] == 2
    assert row["common_prefix_tokens"] == 2
    assert row["common_suffix_tokens"] == 2
    assert row["sender_changed_middle_tokens"] == 0
    assert row["receiver_changed_middle_tokens"] == 1
    assert row["token_count_delta"] == 1


def test_classification_requires_direct_exact_and_transport_change():
    assert analysis.classify(source_exact=False, direct_exact=True, token_exact=False) == "text_transport_attributed"
    assert analysis.classify(source_exact=False, direct_exact=True, token_exact=True) == "silent_inexact_inconsistent_token_evidence"
    assert analysis.classify(source_exact=False, direct_exact=False, token_exact=False) == "silent_inexact_direct_gate_not_exact"
    assert analysis.classify(source_exact=True, direct_exact=True, token_exact=False) == "exact_control_transport_changed"


def test_summary_gate_for_attributed_silent_runs():
    rows = [
        {
            "classification": "text_transport_attributed",
            "source_roundtrip_exact": False,
            "direct_token_exact": True,
            "token_sequence_exact": False,
            "first_token_mismatch": 4,
            "sender_changed_middle_tokens": 1,
            "receiver_changed_middle_tokens": 2,
            "method_id": "bins",
            "method_point_id": "b1",
        },
        {
            "classification": "exact_control_transport_changed",
            "source_roundtrip_exact": True,
            "direct_token_exact": True,
            "token_sequence_exact": False,
            "first_token_mismatch": 10,
            "sender_changed_middle_tokens": 1,
            "receiver_changed_middle_tokens": 1,
            "method_id": "bins",
            "method_point_id": "b1",
        },
    ]
    summary = analysis.summarize(rows, execution_sha="abc")
    assert summary["source_silent_inexact_count"] == 1
    assert summary["silent_inexact_text_transport_attributed_count"] == 1
    assert summary["interpretation_gate"]["all_silent_inexact_attributed_to_text_transport"] is True
    assert summary["exact_controls_transport_changed_count"] == 1


def test_refuses_to_write_inside_results_runs_or_overwrite(tmp_path):
    run_root = tmp_path / "results" / "runs"
    run_root.mkdir(parents=True)
    with pytest.raises(ValueError, match="canonical results/runs"):
        analysis.protected_output(run_root / "x.json", run_root)
    safe = tmp_path / "audit" / "x.json"
    safe.parent.mkdir()
    safe.write_text("existing", encoding="utf-8")
    with pytest.raises(ValueError, match="overwrite"):
        analysis.protected_output(safe, run_root)
