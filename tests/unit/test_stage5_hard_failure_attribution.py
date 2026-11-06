"""Unit tests for final Stage-5 hard-failure attribution infrastructure."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/audit_stage5_hard_failure_attribution.py"
spec = importlib.util.spec_from_file_location("audit_stage5_hard_failure_attribution", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_classification_requires_exact_source_error_reproduction():
    base = {
        "direct_audit_exact": True,
        "first_transport_mismatch": 10,
        "failure_step": 12,
        "replayed_error_matches_source": True,
    }
    assert audit.classify_replay(base) == "text_transport_reproduced_on_reconstructed_sender"
    assert audit.classify_replay({**base, "direct_audit_exact": False}) == "direct_gate_not_exact"
    assert audit.classify_replay({**base, "first_transport_mismatch": None}) == "no_transport_token_change"
    assert audit.classify_replay({**base, "failure_step": None}) == "source_hard_failure_not_reproduced"
    assert audit.classify_replay({**base, "failure_step": 9}) == "failure_precedes_transport_mismatch"
    assert (
        audit.classify_replay({**base, "replayed_error_matches_source": False})
        == "transport_failure_reproduced_but_source_error_differs"
    )


def test_summary_preserves_reconstruction_caveat_and_completion_gate():
    rows = [
        {
            "method_id": "huffman",
            "classification": "text_transport_reproduced_on_reconstructed_sender",
            "direct_audit_exact": True,
            "first_transport_mismatch": 5,
            "replayed_error_matches_source": True,
        },
        {
            "method_id": "discop",
            "classification": "source_hard_failure_not_reproduced",
            "direct_audit_exact": True,
            "first_transport_mismatch": 7,
            "replayed_error_matches_source": False,
        },
    ]
    summary = audit.summarize(
        rows,
        execution_sha="exec",
        direct_sha="direct",
        total_source_hard_failures=38,
    )
    assert summary["checkpointed_count"] == 2
    assert summary["remaining_count"] == 36
    assert summary["attributed_count"] == 1
    assert summary["all_checkpointed_attributed"] is False
    assert summary["all_38_attributed"] is False
    assert "not persisted" in summary["interpretation_caveat"]


def test_checkpoint_loader_rejects_wrong_provenance(tmp_path):
    records = tmp_path / "records"
    records.mkdir()
    (records / "a.json").write_text(
        json.dumps({"run_id": "a", "source_execution_sha256": "wrong", "direct_audit_sha256": "d"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="different execution log"):
        audit.load_checkpoint_records(records, execution_sha="e", direct_sha="d")


def test_checkpoint_outputs_are_resumable(tmp_path):
    records_dir = tmp_path / "records"
    records_dir.mkdir()
    row = {
        "run_id": "r1",
        "method_id": "dairstega",
        "classification": "text_transport_reproduced_on_reconstructed_sender",
        "direct_audit_exact": True,
        "first_transport_mismatch": 1,
        "replayed_error_matches_source": True,
        "source_execution_sha256": "exec",
        "direct_audit_sha256": "direct",
    }
    audit._atomic_json(records_dir / "r1.json", row)
    loaded = audit.load_checkpoint_records(records_dir, execution_sha="exec", direct_sha="direct")
    assert loaded == {"r1": row}
    summary = audit.write_checkpoint_outputs(
        tmp_path,
        loaded,
        execution_sha="exec",
        direct_sha="direct",
        total_source_hard_failures=38,
    )
    assert summary["checkpointed_count"] == 1
    assert (tmp_path / "hard_failure_attribution.jsonl").is_file()
    assert (tmp_path / "summary.json").is_file()


def test_pilot_b_inventory_is_38_across_four_methods():
    execution = (
        Path(__file__).resolve().parents[2]
        / "results/stage5/sweeps/stage5-pilot-b-streaming-llama-v1/execution.jsonl"
    )
    failures = audit.load_hard_failure_records(execution, methods=audit.HARD_METHODS)
    counts = {}
    for item in failures:
        counts[item.method_id] = counts.get(item.method_id, 0) + 1
    assert len(failures) == 38
    assert counts == audit.EXPECTED_HARD_COUNTS
