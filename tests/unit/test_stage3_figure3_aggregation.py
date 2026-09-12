from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_PATH = REPO_ROOT / "scripts" / "stage3_figure3_core.py"


def _core():
    spec = importlib.util.spec_from_file_location("stage3_figure3_core_agg_test", CORE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(CORE_PATH.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(CORE_PATH.parent))
    return module


def _record(rank: int, replicate: int, *, point_id: str = "bins_b1", value: float = 1.0):
    return {
        "point_id": point_id,
        "method": "bins",
        "replicate": replicate,
        "selection_rank": rank,
        "status": "ok",
        "generation": {
            "terminal_reason": "sentence_boundary",
            "first_sentence_boundary_is_final_token": True,
            "text_token_roundtrip_exact": True,
            "sender_token_ids": [13],
        },
        "recovery": {
            "exact_confirmed_payload_prefix_recovery": True,
            "status": "exact_prefix",
            "figure3_execution_gate": False,
            "decoder_stdout": None,
            "decoder_exception": None,
        },
        "secret_stream": {"used_implicit_zero_lookahead": False},
        "author_metrics": {
            "payload_bits_confirmed": 1,
            "carrier_tokens": 1,
            "bits_per_word_author": value,
            "kl_q_stego_to_p_lm_bits_author": value / 10,
            "avg_nll_nats_author": 2.0,
        },
    }


def test_shard_validator_keeps_transport_recovery_as_diagnostic() -> None:
    core = _core()
    point = {"id": "bins_b1", "method": "bins", "block_size_bits": 1}
    records = [_record(rank, 0) for rank in range(80)]
    assert core.validate_shard_records(records, point=point, replicate=0) == []
    records[5]["recovery"]["exact_confirmed_payload_prefix_recovery"] = False
    records[5]["recovery"]["status"] = "prefix_mismatch"
    errors = core.validate_shard_records(records, point=point, replicate=0)
    assert errors == []
    records[5]["recovery"]["figure3_execution_gate"] = True
    errors = core.validate_shard_records(records, point=point, replicate=0)
    assert any("incorrectly promoted" in error for error in errors)


def test_point_aggregation_uses_240_runs_and_reports_two_standard_errors() -> None:
    core = _core()
    point = {"id": "bins_b1", "method": "bins", "block_size_bits": 1}
    records = []
    for replicate in range(3):
        for rank in range(80):
            records.append(_record(rank, replicate, value=1.0 + rank / 1000 + replicate / 100))
    summary = core.aggregate_point(point, records)
    assert summary["run_count"] == 240
    assert summary["context_count"] == 80
    assert summary["replicate_count"] == 3
    metric = summary["metrics"]["bits_per_word_author"]
    assert metric["standard_error_runs"] > 0
    assert metric["standard_error_context_clustered"] > 0
    assert summary["transport_recovery"]["exact_prefix_rate"] == 1.0
    assert summary["transport_recovery"]["failure_count"] == 0


def test_pilot_continuity_detects_token_or_metric_drift() -> None:
    core = _core()
    points = ["bins_b3"]
    ranks = [0]
    pilot = {"records": [_record(0, 0, point_id="bins_b3", value=3.0)]}
    full = [_record(0, 0, point_id="bins_b3", value=3.0)]
    assert core.pilot_continuity_differences(
        full, pilot, point_ids=points, selection_ranks=ranks, replicate=0, metric_abs_tol=1e-10
    ) == []
    full[0]["generation"]["sender_token_ids"] = [999]
    errors = core.pilot_continuity_differences(
        full, pilot, point_ids=points, selection_ranks=ranks, replicate=0, metric_abs_tol=1e-10
    )
    assert any("token IDs differ" in error for error in errors)


def test_zero_payload_is_valid_only_for_arithmetic_and_is_kept_in_aggregation() -> None:
    core = _core()
    arithmetic_point = {"id": "arithmetic_t0.4_k300", "method": "arithmetic", "temperature": 0.4, "topk": 300, "precision": 26}
    records = []
    for rank in range(80):
        record = _record(rank, 0, point_id="arithmetic_t0.4_k300", value=1.0)
        record["method"] = "arithmetic"
        if rank in {7, 10}:
            record["author_metrics"]["payload_bits_confirmed"] = 0
            record["author_metrics"]["bits_per_word_author"] = 0.0
        records.append(record)
    assert core.validate_shard_records(records, point=arithmetic_point, replicate=0) == []

    bins_point = {"id": "bins_b1", "method": "bins", "block_size_bits": 1}
    bins_records = [_record(rank, 0) for rank in range(80)]
    bins_records[7]["author_metrics"]["payload_bits_confirmed"] = 0
    bins_records[7]["author_metrics"]["bits_per_word_author"] = 0.0
    errors = core.validate_shard_records(bins_records, point=bins_point, replicate=0)
    assert any("zero payload is only valid for Arithmetic" in error for error in errors)

    # Aggregate requires 3 replicates; zero-payload samples stay in the mean as 0.
    full_records = []
    for replicate in range(3):
        for rank in range(80):
            record = _record(rank, replicate, point_id="arithmetic_t0.4_k300", value=1.0)
            record["method"] = "arithmetic"
            if replicate == 0 and rank in {7, 10}:
                record["author_metrics"]["payload_bits_confirmed"] = 0
                record["author_metrics"]["bits_per_word_author"] = 0.0
            full_records.append(record)
    summary = core.aggregate_point(arithmetic_point, full_records)
    assert summary["zero_payload_diagnostic"]["count"] == 2
    assert summary["metrics"]["bits_per_word_author"]["mean"] == 238 / 240
    assert summary["transport_recovery"]["not_applicable_zero_payload_count"] == 2
    assert summary["transport_recovery"]["applicable_positive_payload_count"] == 238


def test_arithmetic_cache_compatibility_diagnostic_aggregates_trim_events() -> None:
    core = _core()
    point = {"id": "arithmetic_t0.4_k300", "method": "arithmetic", "temperature": 0.4, "topk": 300, "precision": 26}
    records = []
    for replicate in range(3):
        for rank in range(80):
            trim = 2 if (replicate == 0 and rank == 7) else 0
            records.append({
                "selection_rank": rank,
                "replicate": replicate,
                "status": "ok",
                "generation": {"first_sentence_boundary_is_final_token": True, "text_token_roundtrip_exact": True},
                "recovery": {"exact_confirmed_payload_prefix_recovery": True, "status": "exact_prefix"},
                "secret_stream": {"used_implicit_zero_lookahead": False},
                "arithmetic_cache_compatibility": {
                    "max_cache_tokens": 1022,
                    "encode_trim_events": trim,
                    "encode_max_seen_sequence_tokens": 1023 if trim else 100,
                    "decode_trim_events": 1 if trim else 0,
                    "decode_max_seen_sequence_tokens": 1023 if trim else 100,
                },
                "author_metrics": {
                    "bits_per_word_author": 1.0,
                    "kl_q_stego_to_p_lm_bits_author": 0.1,
                    "avg_nll_nats_author": 2.0,
                    "carrier_tokens": 10,
                    "payload_bits_confirmed": 10,
                },
            })
    summary = core.aggregate_point(point, records)
    cache = summary["arithmetic_cache_compatibility"]
    assert cache["applicable"] is True
    assert cache["runs_with_encode_trim"] == 1
    assert cache["total_encode_trim_events"] == 2
    assert cache["runs_with_decode_trim"] == 1
    assert cache["total_decode_trim_events"] == 1
    assert cache["max_cache_tokens"] == 1022


def _termination_failure_record(rank: int, replicate: int, *, point_id: str = "arithmetic_t0.4_k300"):
    return {
        "point_id": point_id,
        "method": "arithmetic",
        "replicate": replicate,
        "selection_rank": rank,
        "status": "sentence_termination_failure",
        "generation": {
            "terminal_reason": "max_generated_tokens",
            "first_sentence_boundary_is_final_token": False,
            "sentence_finish_token_positions_zero_based": [],
            "text_token_roundtrip_exact": None,
            "sentence_guard": {
                "final_guard_tokens": 8192,
                "reached_real_boundary": False,
                "final_guard_exhausted": True,
                "escalation_count": 3,
            },
        },
        "recovery": {
            "exact_confirmed_payload_prefix_recovery": None,
            "status": "not_applicable_termination_failure",
            "figure3_execution_gate": False,
            "decoder_stdout": None,
            "decoder_exception": None,
        },
        "secret_stream": {"used_implicit_zero_lookahead": False},
        "arithmetic_cache_compatibility": {
            "max_cache_tokens": 1022,
            "encode_trim_events": 100,
            "encode_max_seen_sequence_tokens": 1023,
            "decode_trim_events": 0,
            "decode_max_seen_sequence_tokens": 0,
        },
        "author_metrics": {
            "payload_bits_confirmed": 100,
            "carrier_tokens": 8192,
            "bits_per_word_author": 0.012,
            "kl_q_stego_to_p_lm_bits_author": 9.9,
            "avg_nll_nats_author": 9.9,
        },
    }


def test_sentence_termination_failure_is_valid_only_for_arithmetic_final_guard() -> None:
    core = _core()
    arithmetic_point = {
        "id": "arithmetic_t0.4_k300", "method": "arithmetic",
        "temperature": 0.4, "topk": 300, "precision": 26,
    }
    records = []
    for rank in range(80):
        if rank == 49:
            records.append(_termination_failure_record(rank, 0))
        else:
            record = _record(rank, 0, point_id="arithmetic_t0.4_k300", value=1.0)
            record["method"] = "arithmetic"
            record["generation"]["terminal_reason"] = "sentence_boundary"
            record["generation"]["sentence_finish_token_positions_zero_based"] = [0]
            record["recovery"]["figure3_execution_gate"] = False
            records.append(record)
    assert core.validate_shard_records(records, point=arithmetic_point, replicate=0) == []

    records[49]["generation"]["sentence_guard"]["final_guard_tokens"] = 4096
    errors = core.validate_shard_records(records, point=arithmetic_point, replicate=0)
    assert any("8192-token guard" in error for error in errors)


def test_point_aggregation_excludes_termination_failures_from_sentence_metrics() -> None:
    core = _core()
    point = {
        "id": "arithmetic_t0.4_k300", "method": "arithmetic",
        "temperature": 0.4, "topk": 300, "precision": 26,
    }
    records = []
    failures = {(1, 49), (1, 50), (1, 58)}
    for replicate in range(3):
        for rank in range(80):
            if (replicate, rank) in failures:
                records.append(_termination_failure_record(rank, replicate))
                continue
            record = _record(rank, replicate, point_id="arithmetic_t0.4_k300", value=1.0)
            record["method"] = "arithmetic"
            record["generation"]["terminal_reason"] = "sentence_boundary"
            record["generation"]["sentence_finish_token_positions_zero_based"] = [0]
            record["recovery"]["figure3_execution_gate"] = False
            records.append(record)

    summary = core.aggregate_point(point, records)
    assert summary["scheduled_run_count"] == 240
    assert summary["terminated_sentence_run_count"] == 237
    assert summary["metric_run_count"] == 237
    assert summary["termination_failure_count"] == 3
    assert summary["termination_success_rate"] == 237 / 240
    assert summary["termination_diagnostic"]["failure_selection_ranks"] == [49, 50, 58]
    # Partial failure metrics are deliberately extreme and must not affect the mean.
    assert summary["metrics"]["bits_per_word_author"]["mean"] == 1.0
    assert summary["metrics"]["kl_q_stego_to_p_lm_bits_author"]["mean"] == 0.1
    assert summary["transport_recovery"]["not_applicable_termination_failure_count"] == 3
    assert summary["sentence_guard_diagnostic"]["final_guard_exhaustion_count"] == 3
