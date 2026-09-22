"""Pure helpers for the Stage-3 full Figure-3 author-compatible run.

This module deliberately avoids importing torch/transformers so that the frozen
execution plan, shard validation, resume semantics and aggregation can be unit
-tested without a GPU.
"""

from __future__ import annotations

from collections import defaultdict
import math
import statistics
from typing import Any, Iterable

EXPECTED_CONFIG_SCHEMA = "stage3.figure3_full_run.v1"
EXPECTED_MATRIX_SCHEMA = "stage3.paper_reproduction_matrix.v1"
EXPECTED_REFERENCE_COMMIT = "14e982564aeaf9a33f7b4de440deda2184d17f12"
EXPECTED_MODEL_ID = "gpt2-medium"


def expected_points_from_matrix(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    if matrix.get("schema_version") != EXPECTED_MATRIX_SCHEMA:
        raise ValueError("unexpected paper reproduction matrix schema")
    sweeps = {str(item["method"]): item for item in matrix.get("information_theoretic_sweeps", [])}
    if set(sweeps) != {"bins", "huffman", "arithmetic"}:
        raise ValueError("matrix must contain bins, huffman and arithmetic sweeps")

    points: list[dict[str, Any]] = []
    for value in sweeps["bins"]["values"]:
        b = int(value)
        points.append({"id": f"bins_b{b}", "method": "bins", "block_size_bits": b})
    for value in sweeps["huffman"]["values"]:
        e = int(value)
        points.append({"id": f"huffman_e{e}", "method": "huffman", "candidate_pool_exponent": e})

    fixed = sweeps["arithmetic"].get("fixed") or {}
    topk = int(fixed["topk"])
    precision = int(fixed["precision"])
    for value in sweeps["arithmetic"]["values"]:
        temp = float(value)
        points.append({
            "id": f"arithmetic_t{temp:.1f}_k{topk}",
            "method": "arithmetic",
            "temperature": temp,
            "topk": topk,
            "precision": precision,
        })

    special = matrix.get("special_points") or []
    if len(special) != 1:
        raise ValueError("matrix must contain exactly one special point")
    item = special[0]
    points.append({
        "id": "arithmetic_t1.0_k50256",
        "method": "arithmetic",
        "temperature": float(item["temperature"]),
        "topk": int(item["topk"]),
        "precision": int(item["precision"]),
        "special_point": str(item["id"]),
    })
    return points


def _strip_runtime_fields(point: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in point.items() if k != "compatibility_profile"}


def validate_full_config(config: dict[str, Any], matrix: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema_version") != EXPECTED_CONFIG_SCHEMA:
        raise ValueError("unexpected full-run schema")
    if config.get("status") != "frozen_before_full_run":
        raise ValueError("full-run config must remain frozen_before_full_run")
    if config.get("reference_commit") != EXPECTED_REFERENCE_COMMIT:
        raise ValueError("unexpected reference commit")
    if config.get("model_id") != EXPECTED_MODEL_ID:
        raise ValueError("full Figure-3 run must use gpt2-medium")
    if int(config.get("context_count", 0)) != 80:
        raise ValueError("full Figure-3 run must use all 80 frozen contexts")
    if config.get("replicates") != [0, 1, 2]:
        raise ValueError("full Figure-3 run must use replicates [0,1,2]")

    provenance = config.get("model_provenance") or {}
    if provenance.get("canonical_hub_repository") != "openai-community/gpt2-medium":
        raise ValueError("unexpected canonical GPT-2 Medium repository")
    if provenance.get("requested_identifier") != "gpt2-medium":
        raise ValueError("author helper identifier must remain gpt2-medium")
    if provenance.get("resolved_hf_revision_required") is not True:
        raise ValueError("resolved Hugging Face revision must be recorded")
    if provenance.get("cross_phase_revision_equality_required") is not True:
        raise ValueError("legacy and Arithmetic phases must resolve the same model revision")

    stream = config.get("secret_stream") or {}
    if int(stream.get("bit_count", 0)) != 16384:
        raise ValueError("full run must reuse the validated 16384-bit sentence stream")
    if stream.get("generator_id") != "paper_sentence_sha256_bitstream_v1":
        raise ValueError("unexpected secret stream generator")
    if stream.get("paired_across_all_points_for_same_context_and_replicate") is not True:
        raise ValueError("secret streams must be paired across points")

    stop = config.get("sentence_stop") or {}
    if stop.get("stop_at_first_boundary") is not True:
        raise ValueError("full run must stop at first sentence boundary")
    if stop.get("predicate") != "pinned_reference_utils.is_sent_finish":
        raise ValueError("full run must use pinned author sentence predicate")
    if int(stop.get("pilot_validated_safety_cap_tokens", 0)) != 256:
        raise ValueError("Step-3.9 validated pilot safety cap must remain recorded as 256 tokens")
    if int(stop.get("max_generated_tokens", 0)) != 1024:
        raise ValueError("full-run emergency cap must be 1024 tokens after the sentence-cap amendment")
    if stop.get("cap_role") != "emergency_runaway_guard_not_paper_parameter":
        raise ValueError("full-run cap must remain an emergency guard, not a paper parameter")
    adaptive_guard = stop.get("adaptive_arithmetic_guard") or {}
    if adaptive_guard.get("enabled") is not True:
        raise ValueError("Arithmetic adaptive sentence guard must be enabled after the 1024-token cap finding")
    if adaptive_guard.get("apply_to_methods") != ["arithmetic"]:
        raise ValueError("adaptive sentence guard must apply only to Arithmetic")
    if adaptive_guard.get("cap_sequence_tokens") != [1024, 2048, 4096, 8192]:
        raise ValueError("unexpected Arithmetic adaptive guard sequence")
    if adaptive_guard.get("rerun_from_start_on_cap_hit") is not True:
        raise ValueError("Arithmetic cap escalation must deterministically rerun from the start")
    if adaptive_guard.get("trigger_terminal_reason") != "max_generated_tokens":
        raise ValueError("Arithmetic cap escalation must trigger only on max_generated_tokens")

    termination_failure = stop.get("termination_failure_policy") or {}
    if termination_failure.get("enabled") is not True:
        raise ValueError("Arithmetic final-guard sentence-termination failure policy must be enabled")
    if termination_failure.get("applies_to_methods") != ["arithmetic"]:
        raise ValueError("sentence_termination_failure must apply only to Arithmetic")
    if termination_failure.get("trigger_terminal_reason") != "max_generated_tokens":
        raise ValueError("sentence_termination_failure must trigger only on max_generated_tokens")
    if int(termination_failure.get("trigger_only_after_final_adaptive_guard_tokens", 0)) != 8192:
        raise ValueError("sentence_termination_failure must require exhaustion of the final 8192-token guard")
    if termination_failure.get("record_status") != "sentence_termination_failure":
        raise ValueError("unexpected sentence termination failure record status")
    if termination_failure.get("execution_gate_role") != "accepted_scheduled_outcome_not_completed_sentence":
        raise ValueError("sentence termination failures must remain accepted scheduled outcomes, not completed sentences")

    cap_amendment = config.get("sentence_stop_amendment") or {}
    if int(cap_amendment.get("new_emergency_cap_tokens", 0)) != 1024:
        raise ValueError("sentence-stop amendment must record the 1024-token emergency cap")
    if "bins_b4" not in str(cap_amendment.get("trigger", "")):
        raise ValueError("sentence-stop amendment trigger must record the observed bins_b4 cap hit")

    escalation = config.get("sentence_guard_escalation_amendment") or {}
    if int(escalation.get("max_escalated_guard_tokens", 0)) != 8192:
        raise ValueError("Arithmetic sentence-guard escalation must record an 8192-token hard ceiling")
    if "arithmetic_t0.4_k300" not in str(escalation.get("trigger", "")):
        raise ValueError("Arithmetic sentence-guard escalation trigger must record the observed low-temperature cap hit")

    termination_amendment = config.get("sentence_termination_failure_amendment") or {}
    if "49, 50 and 58" not in str(termination_amendment.get("trigger", "")):
        raise ValueError("sentence-termination amendment must record the observed ranks 49, 50 and 58")
    if termination_amendment.get("normalized_benchmark_unchanged") is not True:
        raise ValueError("sentence-termination failure policy must not change normalized benchmark")
    if termination_amendment.get("paper_sentence_core_unchanged") is not True:
        raise ValueError("sentence-termination failure policy must not change the validated Step-3.9 core")

    arithmetic_cache = config.get("arithmetic_long_context_compatibility") or {}
    if int(arithmetic_cache.get("max_cache_tokens", 0)) != 1022:
        raise ValueError("Arithmetic long-context compatibility must retain 1022 cache tokens")
    if arithmetic_cache.get("normalized_benchmark_unchanged") is not True:
        raise ValueError("Arithmetic cache compatibility must not change the normalized benchmark")
    if arithmetic_cache.get("paper_sentence_core_unchanged") is not True:
        raise ValueError("validated Step-3.9 paper-sentence core must remain unchanged")
    if "sequence axis" not in str(arithmetic_cache.get("decision", "")):
        raise ValueError("Arithmetic long-context compatibility decision must state sequence-axis trimming")

    transport = config.get("transport_recovery") or {}
    if transport.get("role") != "diagnostic_not_figure3_execution_gate":
        raise ValueError("transport recovery must remain a diagnostic, not a Figure-3 execution gate")
    if transport.get("record_exact_prefix") is not True:
        raise ValueError("transport recovery exact-prefix outcome must still be recorded")
    if transport.get("retain_generation_sample_on_recovery_failure") is not True:
        raise ValueError("Figure-3 generation samples must not be dropped on author decoder BPE-repair failure")
    if transport.get("paper_sentence_pilot_remains_strict") is not True:
        raise ValueError("Step-3.9 strict recovery gate must remain unchanged")

    expected = expected_points_from_matrix(matrix)
    actual = [_strip_runtime_fields(dict(item)) for item in config.get("points", [])]
    if actual != expected:
        raise ValueError("full-run points do not exactly match the frozen paper matrix")

    plan = matrix.get("execution_plan", {}).get("phase_2_information_theoretic_curve", {})
    if int(plan.get("contexts_per_point", 0)) != 80 or int(plan.get("replicates", 0)) != 3:
        raise ValueError("frozen matrix execution plan changed")

    execution = config.get("execution") or {}
    expected_total = len(expected) * 80 * 3
    if int(execution.get("expected_point_count", 0)) != len(expected):
        raise ValueError("expected point count mismatch")
    if int(execution.get("expected_shard_count", 0)) != len(expected) * 3:
        raise ValueError("expected shard count mismatch")
    if int(execution.get("expected_runs_per_point", 0)) != 240:
        raise ValueError("expected runs per point mismatch")
    if int(execution.get("expected_total_run_count", 0)) != expected_total:
        raise ValueError("expected total run count mismatch")
    if execution.get("scientific_claims_are_not_execution_gates") is not True:
        raise ValueError("scientific claims must not be hard execution gates")
    if execution.get("accepted_record_statuses") != ["ok", "sentence_termination_failure"]:
        raise ValueError("full-run accepted record statuses must distinguish completed sentences from termination failures")
    if execution.get("metric_eligible_status") != "ok":
        raise ValueError("only completed sentences may contribute to Figure-3 sentence metrics")

    continuity = config.get("pilot_continuity") or {}
    if continuity.get("required") is not True:
        raise ValueError("Step-3.9 continuity check must remain required")
    if int(continuity.get("replicate", -1)) != 0:
        raise ValueError("pilot continuity must use replicate 0")
    if continuity.get("selection_ranks") != list(range(8)):
        raise ValueError("pilot continuity must use ranks 0..7")
    expected_pilot_points = [
        "bins_b3", "huffman_e3", "arithmetic_t0.9_k300", "arithmetic_t1.0_k50256"
    ]
    if continuity.get("point_ids") != expected_pilot_points:
        raise ValueError("pilot continuity point set changed")

    return {
        "point_count": len(expected),
        "shard_count": len(expected) * 3,
        "runs_per_point": 240,
        "total_runs": expected_total,
    }


def validate_paper_sentence_pilot_result(result: dict[str, Any]) -> None:
    if result.get("schema_version") != "stage3.paper_sentence_pilot_result.v1":
        raise ValueError("unexpected Step-3.9 pilot result schema")
    if result.get("status") != "ok":
        raise ValueError("Step-3.9 pilot result is not ok")
    gate = result.get("gate") or {}
    required_true = [
        "run_count_ok",
        "all_four_fixed_message_parity_checks_passed",
        "all_runs_stop_at_first_sentence_boundary",
        "all_payload_counts_positive",
        "all_confirmed_payload_prefixes_recovered",
        "all_measured_steps_free_of_implicit_zero_lookahead",
        "no_secret_stream_exhaustion",
        "no_safety_cap_hits",
        "all_record_status_ok",
        "reference_worktree_unchanged",
        "ready_for_full_figure3_runner_implementation",
    ]
    missing = [name for name in required_true if gate.get(name) is not True]
    if missing:
        raise ValueError("Step-3.9 pilot gate failed: " + ", ".join(missing))
    if int(gate.get("actual_run_count", 0)) != 32:
        raise ValueError("Step-3.9 pilot must contain 32 runs")


def validate_shard_records(
    records: list[dict[str, Any]], *, point: dict[str, Any], replicate: int,
    expected_context_count: int = 80,
) -> list[str]:
    errors: list[str] = []
    if len(records) != expected_context_count:
        errors.append(f"expected {expected_context_count} records, got {len(records)}")
        return errors

    ranks = [int(record.get("selection_rank", -1)) for record in records]
    if sorted(ranks) != list(range(expected_context_count)):
        errors.append("selection ranks are not exactly 0..79")
    if len(set(ranks)) != expected_context_count:
        errors.append("selection ranks are not unique")

    point_id = str(point["id"])
    method = str(point["method"])
    for idx, record in enumerate(records):
        prefix = f"record[{idx}]"
        if record.get("point_id") != point_id:
            errors.append(f"{prefix}: point_id mismatch")
        if record.get("method") != method:
            errors.append(f"{prefix}: method mismatch")
        if int(record.get("replicate", -1)) != int(replicate):
            errors.append(f"{prefix}: replicate mismatch")

        status = record.get("status")
        if status not in {"ok", "sentence_termination_failure"}:
            errors.append(f"{prefix}: invalid record status")
            continue

        generation = record.get("generation") or {}
        recovery = record.get("recovery") or {}
        secret = record.get("secret_stream") or {}
        metrics = record.get("author_metrics") or {}
        if secret.get("used_implicit_zero_lookahead") is not False:
            errors.append(f"{prefix}: implicit zero look-ahead present")
        if int(metrics.get("carrier_tokens", 0)) <= 0:
            errors.append(f"{prefix}: non-positive carrier length")
        payload_bits = int(metrics.get("payload_bits_confirmed", -1))
        if payload_bits < 0:
            errors.append(f"{prefix}: negative payload")
        elif status == "ok" and method != "arithmetic" and payload_bits == 0:
            errors.append(f"{prefix}: zero payload is only valid for Arithmetic")

        if status == "ok":
            if generation.get("terminal_reason") != "sentence_boundary":
                errors.append(f"{prefix}: terminal_reason is not sentence_boundary")
            if generation.get("first_sentence_boundary_is_final_token") is not True:
                errors.append(f"{prefix}: first boundary is not final token")
            if not isinstance(recovery.get("exact_confirmed_payload_prefix_recovery"), bool):
                errors.append(f"{prefix}: transport recovery diagnostic is missing")
            if recovery.get("figure3_execution_gate") is not False:
                errors.append(f"{prefix}: transport recovery was incorrectly promoted to Figure-3 gate")
            if recovery.get("status") not in {"exact_prefix", "prefix_mismatch", "decoder_exception"}:
                errors.append(f"{prefix}: invalid transport recovery status")
            continue

        # Final-guard non-termination is a valid scheduled outcome only for Arithmetic.
        if method != "arithmetic":
            errors.append(f"{prefix}: sentence_termination_failure is valid only for Arithmetic")
        if generation.get("terminal_reason") != "max_generated_tokens":
            errors.append(f"{prefix}: termination failure did not end at max_generated_tokens")
        if generation.get("first_sentence_boundary_is_final_token") is not False:
            errors.append(f"{prefix}: termination failure unexpectedly reports a sentence boundary")
        positions = generation.get("sentence_finish_token_positions_zero_based")
        if positions not in ([], None):
            errors.append(f"{prefix}: termination failure contains a detected sentence boundary")
        guard = generation.get("sentence_guard") or {}
        if int(guard.get("final_guard_tokens", 0)) != 8192:
            errors.append(f"{prefix}: termination failure must exhaust final 8192-token guard")
        if guard.get("reached_real_boundary") is not False:
            errors.append(f"{prefix}: termination failure incorrectly reports a real boundary")
        if guard.get("final_guard_exhausted") is not True:
            errors.append(f"{prefix}: final guard exhaustion diagnostic is missing")
        if recovery.get("figure3_execution_gate") is not False:
            errors.append(f"{prefix}: termination-failure recovery was incorrectly promoted to Figure-3 gate")
        if recovery.get("status") != "not_applicable_termination_failure":
            errors.append(f"{prefix}: termination-failure recovery must be not-applicable")
        if recovery.get("exact_confirmed_payload_prefix_recovery") is not None:
            errors.append(f"{prefix}: termination-failure recovery exactness must be null")
    return errors


def _mean_sem(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot aggregate empty values")
    mean = statistics.fmean(values)
    if len(values) == 1:
        return mean, 0.0
    sem = statistics.stdev(values) / math.sqrt(len(values))
    return mean, sem


def _clustered_sem(records: list[dict[str, Any]], metric_key: str) -> float:
    by_context: dict[int, list[float]] = defaultdict(list)
    for record in records:
        by_context[int(record["selection_rank"])].append(float(record["author_metrics"][metric_key]))
    context_means = [statistics.fmean(by_context[rank]) for rank in sorted(by_context)]
    if len(context_means) <= 1:
        return 0.0
    return statistics.stdev(context_means) / math.sqrt(len(context_means))


def aggregate_point(point: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != 240:
        raise ValueError(f"point {point['id']} requires 240 scheduled records, got {len(records)}")
    accepted = {"ok", "sentence_termination_failure"}
    if any(record.get("status") not in accepted for record in records):
        raise ValueError(f"point {point['id']} contains invalid record statuses")

    completed = [record for record in records if record.get("status") == "ok"]
    failures = [record for record in records if record.get("status") == "sentence_termination_failure"]
    if not completed:
        raise ValueError(f"point {point['id']} has no successfully terminated sentences to aggregate")

    metric_keys = [
        "bits_per_word_author",
        "kl_q_stego_to_p_lm_bits_author",
        "avg_nll_nats_author",
        "carrier_tokens",
        "payload_bits_confirmed",
    ]
    metrics: dict[str, Any] = {}
    for key in metric_keys:
        values = [float(record["author_metrics"][key]) for record in completed]
        mean, sem = _mean_sem(values)
        metrics[key] = {
            "mean": mean,
            "standard_error_runs": sem,
            "standard_error_context_clustered": _clustered_sem(completed, key),
            "min": min(values),
            "max": max(values),
            "n_successful_sentences": len(completed),
            "conditioning": "successfully terminated sentences only",
        }

    cache_records = [r.get("arithmetic_cache_compatibility") or {} for r in records]
    cache_diag = {
        "applicable": str(point["method"]) == "arithmetic",
        "runs_with_encode_trim": sum(int(item.get("encode_trim_events", 0)) > 0 for item in cache_records),
        "total_encode_trim_events": sum(int(item.get("encode_trim_events", 0)) for item in cache_records),
        "max_encode_seen_sequence_tokens": max((int(item.get("encode_max_seen_sequence_tokens", 0)) for item in cache_records), default=0),
        "runs_with_decode_trim": sum(int(item.get("decode_trim_events", 0)) > 0 for item in cache_records),
        "total_decode_trim_events": sum(int(item.get("decode_trim_events", 0)) for item in cache_records),
        "max_decode_seen_sequence_tokens": max((int(item.get("decode_max_seen_sequence_tokens", 0)) for item in cache_records), default=0),
        "max_cache_tokens": 1022 if str(point["method"]) == "arithmetic" else None,
    }
    guard_records = [((r.get("generation") or {}).get("sentence_guard") or {}) for r in records]
    sentence_guard_diag = {
        "applicable": str(point["method"]) == "arithmetic",
        "runs_with_escalation": sum(int(item.get("escalation_count", 0)) > 0 for item in guard_records),
        "max_final_guard_tokens": max((int(item.get("final_guard_tokens", 0)) for item in guard_records), default=0),
        "total_escalations": sum(int(item.get("escalation_count", 0)) for item in guard_records),
        "final_guard_exhaustion_count": sum(bool(item.get("final_guard_exhausted")) for item in guard_records),
    }

    positive_payload_completed = [r for r in completed if int(r["author_metrics"]["payload_bits_confirmed"]) > 0]
    zero_payload_completed = [r for r in completed if int(r["author_metrics"]["payload_bits_confirmed"]) == 0]
    termination_success_rate = len(completed) / len(records)
    return {
        "point_id": str(point["id"]),
        "method": str(point["method"]),
        "point_parameters": {
            k: v for k, v in point.items()
            if k not in {"id", "method", "compatibility_profile"}
        },
        "run_count": len(records),
        "scheduled_run_count": len(records),
        "terminated_sentence_run_count": len(completed),
        "metric_run_count": len(completed),
        "termination_failure_count": len(failures),
        "termination_success_rate": termination_success_rate,
        "termination_diagnostic": {
            "scheduled_run_count": len(records),
            "terminated_sentence_run_count": len(completed),
            "termination_failure_count": len(failures),
            "termination_success_rate": termination_success_rate,
            "failure_selection_ranks": sorted(int(r["selection_rank"]) for r in failures),
            "metric_conditioning": "conditional_on_successful_sentence_termination",
        },
        "context_count": len({int(r["selection_rank"]) for r in records}),
        "replicate_count": len({int(r["replicate"]) for r in records}),
        "all_payload_recovered": all(bool(r["recovery"]["exact_confirmed_payload_prefix_recovery"]) for r in positive_payload_completed),
        "transport_recovery": {
            "applicable_positive_payload_count": len(positive_payload_completed),
            "not_applicable_zero_payload_count": len(zero_payload_completed),
            "not_applicable_termination_failure_count": len(failures),
            "exact_prefix_count": sum(bool(r["recovery"]["exact_confirmed_payload_prefix_recovery"]) for r in positive_payload_completed),
            "failure_count": sum(not bool(r["recovery"]["exact_confirmed_payload_prefix_recovery"]) for r in positive_payload_completed),
            "exact_prefix_rate": (
                sum(bool(r["recovery"]["exact_confirmed_payload_prefix_recovery"]) for r in positive_payload_completed)
                / len(positive_payload_completed) if positive_payload_completed else None
            ),
            "decoder_exception_count": sum(r["recovery"].get("status") == "decoder_exception" for r in positive_payload_completed),
            "decoder_warning_count": sum(bool(r["recovery"].get("decoder_stdout")) for r in positive_payload_completed),
            "figure3_execution_gate": False,
        },
        "arithmetic_cache_compatibility": cache_diag,
        "sentence_guard_diagnostic": sentence_guard_diag,
        "zero_payload_diagnostic": {
            "count": len(zero_payload_completed),
            "rate_among_completed_sentences": len(zero_payload_completed) / len(completed),
            "valid_for_figure3": str(point["method"]) == "arithmetic",
        },
        "all_first_boundary_stop": all(bool(r["generation"]["first_sentence_boundary_is_final_token"]) for r in completed),
        "sentence_length_diagnostic": {
            "over_step39_pilot_cap_256_count": sum(int(r["author_metrics"]["carrier_tokens"]) > 256 for r in completed),
            "max_carrier_tokens": max(int(r["author_metrics"]["carrier_tokens"]) for r in completed),
            "max_observed_partial_tokens_including_termination_failures": max(int(r["author_metrics"]["carrier_tokens"]) for r in records),
        },
        "all_no_implicit_zero_lookahead": all(not bool(r["secret_stream"]["used_implicit_zero_lookahead"]) for r in records),
        "text_token_roundtrip_exact_count": sum(bool(r["generation"].get("text_token_roundtrip_exact")) for r in completed),
        "metrics": metrics,
    }


def pilot_continuity_differences(
    full_records: Iterable[dict[str, Any]], pilot_result: dict[str, Any], *, point_ids: list[str],
    selection_ranks: list[int], replicate: int, metric_abs_tol: float,
) -> list[str]:
    pilot_lookup = {
        (str(r["point_id"]), int(r["selection_rank"])): r
        for r in pilot_result.get("records", [])
        if str(r.get("point_id")) in point_ids and int(r.get("selection_rank", -1)) in selection_ranks
    }
    full_lookup = {
        (str(r["point_id"]), int(r["selection_rank"])): r
        for r in full_records
        if str(r.get("point_id")) in point_ids
        and int(r.get("selection_rank", -1)) in selection_ranks
        and int(r.get("replicate", -1)) == replicate
    }
    errors: list[str] = []
    expected_keys = {(point_id, rank) for point_id in point_ids for rank in selection_ranks}
    if set(pilot_lookup) != expected_keys:
        errors.append("pilot result does not contain the expected 32 overlap records")
        return errors
    if set(full_lookup) != expected_keys:
        errors.append("full run does not contain the expected 32 overlap records")
        return errors

    for key in sorted(expected_keys):
        pilot = pilot_lookup[key]
        full = full_lookup[key]
        pgen = pilot["generation"]
        fgen = full["generation"]
        if [int(x) for x in pgen["sender_token_ids"]] != [int(x) for x in fgen["sender_token_ids"]]:
            errors.append(f"{key}: generated token IDs differ from Step-3.9")
        pmet = pilot["author_metrics"]
        fmet = full["author_metrics"]
        if int(pmet["payload_bits_confirmed"]) != int(fmet["payload_bits_confirmed"]):
            errors.append(f"{key}: payload bits differ from Step-3.9")
        for metric in ("bits_per_word_author", "kl_q_stego_to_p_lm_bits_author"):
            if not math.isclose(float(pmet[metric]), float(fmet[metric]), rel_tol=0.0, abs_tol=metric_abs_tol):
                errors.append(f"{key}: {metric} differs from Step-3.9")
    return errors
