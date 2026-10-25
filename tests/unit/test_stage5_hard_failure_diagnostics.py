from __future__ import annotations

from pathlib import Path

from vkr_benchmark.diagnostics import (
    classify_hard_failure,
    load_hard_failure_records,
    summarize_hard_failure_diagnostics,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PILOT_A_EXECUTION = (
    REPO_ROOT
    / "results/stage5/sweeps/stage5-pilot-a-streaming-llama-v1/execution.jsonl"
)


def test_pilot_a_log_contains_expected_13_hard_failures() -> None:
    failures = load_hard_failure_records(PILOT_A_EXECUTION)

    assert len(failures) == 13
    assert sum(item.method_id == "huffman" for item in failures) == 8
    assert sum(item.method_id == "dairstega" for item in failures) == 5
    assert len({item.run_id for item in failures}) == 13


def test_hard_failure_classifier_requires_direct_exact_and_transport_mismatch() -> None:
    base = {
        "direct_token_decode_exact": True,
        "first_transport_mismatch": 17,
        "failure_step": 19,
    }
    assert classify_hard_failure(base) == "text_transport_induced"
    assert (
        classify_hard_failure({**base, "direct_token_decode_exact": False})
        == "requires_investigation"
    )
    assert (
        classify_hard_failure({**base, "first_transport_mismatch": None})
        == "requires_investigation"
    )
    assert (
        classify_hard_failure({**base, "failure_step": 16})
        == "requires_investigation"
    )


def test_hard_failure_summary_keeps_unresolved_cases_visible() -> None:
    rows = [
        {
            "method_id": "huffman",
            "direct_token_decode_exact": True,
            "first_transport_mismatch": 2,
            "failure_step": 2,
        },
        {
            "method_id": "dairstega",
            "direct_token_decode_exact": False,
            "first_transport_mismatch": 4,
            "failure_step": 5,
        },
    ]
    summary = summarize_hard_failure_diagnostics(rows)

    assert summary["diagnosed_runs"] == 2
    assert summary["by_method"] == {"dairstega": 1, "huffman": 1}
    assert summary["by_classification"] == {
        "requires_investigation": 1,
        "text_transport_induced": 1,
    }
    assert summary["direct_token_decode_exact_count"] == 1
    assert summary["transport_mismatch_count"] == 2
