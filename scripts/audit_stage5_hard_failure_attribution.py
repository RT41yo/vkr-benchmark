#!/usr/bin/env python3
"""Final Stage-5 Pilot-B attribution audit for the 38 hard failures.

This is a read-only/model-backed diagnostic. It does not modify benchmark
method implementations or canonical Pilot-B artifacts.

For each selected hard-failure run it:
  1. requires the completed direct-token audit to report ``direct_exact``;
  2. deterministically re-encodes the sender token sequence from canonical config;
  3. transmits that sequence through the ordinary text channel;
  4. replays receiver-side decoding until either the source hard failure is
     reproduced or decoding finishes;
  5. checkpoints one JSON record immediately after the run.

Important limitation: Pilot B did not persist sender token IDs for hard failures.
Therefore the sender sequence here is a deterministic reconstruction and cannot
be compared byte-for-byte with the lost original Pilot-B sender IDs. The strong
classification name explicitly preserves that limitation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vkr_benchmark.config import LocalModelConfig
from vkr_benchmark.diagnostics import load_hard_failure_records
from vkr_benchmark.distributions import ReferenceDistributionBuilder, StepContext
from vkr_benchmark.inputs import PromptRegistry, create_secret_source
from vkr_benchmark.lm import HFCausalLMAdapter
from vkr_benchmark.methods.arithmetic import ArithmeticConfig, _build_partition
from vkr_benchmark.methods.dair import DAIRConfig, _allocate
from vkr_benchmark.methods.huffman import HuffmanConfig, _build_codebook
from vkr_benchmark.runner.experiment import create_method_runtime
from vkr_benchmark.runner.streaming import (
    encode_fixed_carrier_tokens,
    method_environment_from_builder,
    warm_up_streaming_path,
)
from vkr_benchmark.sweeps import PlannedRun, SweepManifest
from vkr_benchmark.transport import TextChannel

DEFAULT_MANIFEST = ROOT / "configs/pilots/stage5/pilot_b_streaming_llama.json"
DEFAULT_EXECUTION = (
    ROOT
    / "results/stage5/sweeps/stage5-pilot-b-streaming-llama-v1/execution.jsonl"
)
DEFAULT_DIRECT = ROOT / "decoder_audit_local_full/direct_token_replay.jsonl"
DEFAULT_OUTPUT = ROOT / "results/stage5/decoder_audit/hard_failure_attribution"
HARD_METHODS = ("huffman", "arithmetic_coding", "discop", "dairstega")
EXPECTED_HARD_COUNTS = {
    "huffman": 20,
    "arithmetic_coding": 3,
    "discop": 2,
    "dairstega": 13,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def first_mismatch(left: Iterable[int], right: Iterable[int]) -> int | None:
    a = tuple(left)
    b = tuple(right)
    for index, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return index
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def _rank(reference_order: np.ndarray, token_id: int | None) -> int | None:
    if token_id is None:
        return None
    matches = np.flatnonzero(reference_order == int(token_id))
    return None if matches.size == 0 else int(matches[0])


def _token_window(
    sender: tuple[int, ...],
    receiver: tuple[int, ...],
    center: int | None,
    *,
    radius: int = 4,
) -> dict[str, Any] | None:
    if center is None:
        return None
    start = max(0, center - radius)
    stop = min(max(len(sender), len(receiver)), center + radius + 1)
    return {
        "start": start,
        "stop": stop,
        "sender": list(sender[start:stop]),
        "receiver": list(receiver[start:stop]),
    }


def load_direct_gate(path: Path, execution_sha: str) -> tuple[dict[str, dict[str, Any]], str]:
    if not path.is_file():
        raise FileNotFoundError(f"direct-token audit JSONL not found: {path}")
    summary_path = path.parent / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"direct-token audit summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("source_execution_sha256") != execution_sha:
        raise ValueError("direct-token audit belongs to a different execution log")
    rows = read_jsonl(path)
    mapped = {str(row["run_id"]): row for row in rows}
    if len(mapped) != len(rows):
        raise ValueError("duplicate run IDs in direct-token audit")
    return mapped, sha256(path)


def classify_replay(row: dict[str, Any]) -> str:
    """Conservative attribution classification for reconstructed hard failures."""
    if row.get("direct_audit_exact") is not True:
        return "direct_gate_not_exact"
    mismatch = row.get("first_transport_mismatch")
    failure_step = row.get("failure_step")
    if not isinstance(mismatch, int):
        return "no_transport_token_change"
    if not isinstance(failure_step, int):
        return "source_hard_failure_not_reproduced"
    if failure_step < mismatch:
        return "failure_precedes_transport_mismatch"
    if row.get("replayed_error_matches_source") is not True:
        return "transport_failure_reproduced_but_source_error_differs"
    return "text_transport_reproduced_on_reconstructed_sender"


def _method_state(
    planned: PlannedRun,
    *,
    context: StepContext,
    environment: Any,
    decoder: Any,
    observed_token_id: int,
    sender_token_id: int | None,
) -> dict[str, Any]:
    method_id = planned.config.method.method_id
    order = context.reference.token_order
    observed = int(observed_token_id)
    sender = None if sender_token_id is None else int(sender_token_id)
    common: dict[str, Any] = {
        "observed_token_id": observed,
        "sender_token_id_same_index": sender,
        "observed_reference_rank": _rank(order, observed),
        "sender_reference_rank": _rank(order, sender),
        "observed_reference_probability": float(context.reference.probabilities[observed]),
        "sender_reference_probability": (
            None if sender is None else float(context.reference.probabilities[sender])
        ),
        "observed_in_reference_support": _rank(order, observed) is not None,
        "sender_in_reference_support": _rank(order, sender) is not None,
    }

    if method_id == "huffman":
        config = HuffmanConfig.from_mapping(planned.config.method.params)
        codebook = _build_codebook(context=context, config=config, environment=environment)
        candidates = tuple(int(item) for item in codebook.candidate_token_ids)
        candidate_set = set(candidates)
        common.update({
            "candidate_count": len(candidates),
            "observed_in_candidate_set": observed in candidate_set,
            "sender_in_candidate_set": sender is not None and sender in candidate_set,
        })
    elif method_id == "arithmetic_coding":
        config = ArithmeticConfig.from_mapping(planned.config.method.params)
        lower, upper = decoder.interval
        partition = _build_partition(
            context=context,
            config=config,
            environment=environment,
            interval_lower=lower,
            interval_upper=upper,
        )
        candidates = tuple(int(item) for item in partition.candidate_token_ids)
        candidate_set = set(candidates)
        common.update({
            "decoder_interval_before_failure": [int(lower), int(upper)],
            "candidate_count": len(candidates),
            "observed_in_candidate_set": observed in candidate_set,
            "sender_in_candidate_set": sender is not None and sender in candidate_set,
        })
    elif method_id == "dairstega":
        config = DAIRConfig.from_mapping(planned.config.method.params)
        intervals = _allocate(context, config)
        interval_tokens = {int(item.token_id) for item in intervals}
        common.update({
            "positive_interval_count": len(intervals),
            "observed_has_positive_interval": observed in interval_tokens,
            "sender_has_positive_interval": sender is not None and sender in interval_tokens,
            "decoder_window_lower_before_failure": int(getattr(decoder, "_window_lower")),
            "decoder_window_upper_before_failure": int(getattr(decoder, "_window_upper")),
        })
    return common


def replay_one(
    *,
    planned: PlannedRun,
    source_error: Any,
    direct_row: dict[str, Any],
    prompt_registry: PromptRegistry,
    lm: HFCausalLMAdapter,
    execution_sha: str,
    direct_sha: str,
) -> dict[str, Any]:
    config = planned.config
    prompt = prompt_registry.get(config.prompt_id)
    builder = ReferenceDistributionBuilder(
        token_space=lm.token_space,
        policy=config.generation.to_policy(),
    )
    environment = method_environment_from_builder(builder)

    warm_up_streaming_path(
        lm_adapter=lm,
        reference_builder=builder,
        environment=environment,
        prompt_text=prompt.text,
    )

    if config.termination.mode != "fixed_carrier_tokens":
        raise ValueError("hard-failure attribution expects fixed_carrier_tokens runs")
    carrier_tokens = config.termination.target_carrier_tokens
    if carrier_tokens is None:
        raise ValueError("fixed_carrier_tokens run has no target_carrier_tokens")

    encode_runtime = create_method_runtime(config)
    encoded = encode_fixed_carrier_tokens(
        lm_adapter=lm,
        reference_builder=builder,
        method=encode_runtime.method,
        method_config=config.method.params,
        environment=environment,
        prompt_text=prompt.text,
        carrier_tokens=carrier_tokens,
        secret_source=create_secret_source(config.secret_id),
        method_random_source=encode_runtime.encoder_random_source,
        key=config.method.key,
    )
    sender = tuple(int(item) for item in encoded.carrier_token_ids)
    transport = TextChannel(lm).transmit(sender)
    receiver = tuple(int(item) for item in transport.receiver_token_ids)

    decoder_runtime = create_method_runtime(config)
    decoder = decoder_runtime.method.create_decoder(
        config=config.method.params,
        environment=environment,
        random_source=decoder_runtime.decoder_random_source,
        key=config.method.key,
        expected_payload_bits=encoded.payload_bits,
    )
    state = lm.prefill(lm.encode_prompt(prompt.text))

    incremental: list[int] = []
    first_secret_bit_mismatch: int | None = None
    first_secret_bit_mismatch_step: int | None = None
    failure_step: int | None = None
    replayed_error_type: str | None = None
    replayed_error_message: str | None = None
    method_state: dict[str, Any] | None = None

    for step_index, observed_token_id in enumerate(receiver):
        raw_logits, state = lm.next_logits(state)
        reference = builder.build(raw_logits)
        context = StepContext(step_index=step_index, reference=reference)
        sender_token_id = sender[step_index] if step_index < len(sender) else None
        try:
            progress = decoder.observe(context, observed_token_id)
        except Exception as exc:
            failure_step = step_index
            replayed_error_type = type(exc).__name__
            replayed_error_message = str(exc)
            method_state = _method_state(
                planned,
                context=context,
                environment=environment,
                decoder=decoder,
                observed_token_id=observed_token_id,
                sender_token_id=sender_token_id,
            )
            break

        incremental.extend(int(bit) for bit in progress.recovered_bits)
        if first_secret_bit_mismatch is None:
            first_secret_bit_mismatch = first_mismatch(
                encoded.payload_secret_bits[: len(incremental)], incremental
            )
            if first_secret_bit_mismatch is not None:
                first_secret_bit_mismatch_step = step_index
        if step_index + 1 < len(receiver):
            _, state = lm.next_logits(state, observed_token_id)

    final_complete: bool | None = None
    final_recovered_bits: tuple[int, ...] | None = None
    transported_roundtrip_exact: bool | None = None
    transported_raw_recovered_length = len(incremental)
    finalize_error_type: str | None = None
    finalize_error_message: str | None = None
    if failure_step is None:
        try:
            finalization = decoder.finalize()
            final_complete = bool(finalization.complete)
            final_recovered_bits = tuple(int(bit) for bit in finalization.recovered_bits)
            transported_raw_recovered_length = max(
                transported_raw_recovered_length, len(final_recovered_bits)
            )
            transported_roundtrip_exact = (
                first_mismatch(encoded.payload_secret_bits, final_recovered_bits) is None
                and len(final_recovered_bits) == len(encoded.payload_secret_bits)
                and transported_raw_recovered_length == len(encoded.payload_secret_bits)
                and final_complete
            )
        except Exception as exc:
            finalize_error_type = type(exc).__name__
            finalize_error_message = str(exc)

    mismatch = transport.first_token_mismatch
    row: dict[str, Any] = {
        **planned.metadata_record(),
        "schema_version": "stage5.hard_failure_attribution.record.v1",
        "source_execution_sha256": execution_sha,
        "direct_audit_sha256": direct_sha,
        "source_error_type": source_error.error_type,
        "source_error_message": source_error.error_message,
        "direct_audit_status": direct_row.get("audit_status"),
        "direct_audit_exact": direct_row.get("direct_roundtrip_exact") is True,
        "sender_token_source": "deterministic_reencode_original_sender_ids_not_persisted",
        "sender_token_count": len(sender),
        "receiver_token_count": len(receiver),
        "token_count_delta": len(receiver) - len(sender),
        "transport_token_sequence_exact": mismatch is None,
        "first_transport_mismatch": mismatch,
        "first_transport_mismatch_window": _token_window(sender, receiver, mismatch),
        "failure_step": failure_step,
        "replayed_error_type": replayed_error_type,
        "replayed_error_message": replayed_error_message,
        "replayed_error_type_matches_source": replayed_error_type == source_error.error_type,
        "replayed_error_matches_source": (
            replayed_error_type == source_error.error_type
            and replayed_error_message == source_error.error_message
        ),
        "failure_at_or_after_first_transport_mismatch": (
            isinstance(mismatch, int)
            and isinstance(failure_step, int)
            and failure_step >= mismatch
        ),
        "first_secret_bit_mismatch_before_failure": first_secret_bit_mismatch,
        "first_secret_bit_mismatch_step": first_secret_bit_mismatch_step,
        "recovered_bits_before_failure": len(incremental),
        "method_state_at_failure": method_state,
        "failure_window": _token_window(sender, receiver, failure_step),
        "transported_finalize_complete": final_complete,
        "transported_final_recovered_length_bits": (
            None if final_recovered_bits is None else len(final_recovered_bits)
        ),
        "transported_raw_recovered_length_bits": transported_raw_recovered_length,
        "transported_roundtrip_exact_if_no_hard_failure": transported_roundtrip_exact,
        "finalize_error_type": finalize_error_type,
        "finalize_error_message": finalize_error_message,
    }
    row["classification"] = classify_replay(row)
    return row


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, path)


def load_checkpoint_records(
    records_dir: Path,
    *,
    execution_sha: str,
    direct_sha: str,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not records_dir.is_dir():
        return records
    for path in sorted(records_dir.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        run_id = str(row.get("run_id", ""))
        if not run_id:
            raise ValueError(f"checkpoint has no run_id: {path}")
        if row.get("source_execution_sha256") != execution_sha:
            raise ValueError(f"checkpoint belongs to different execution log: {path}")
        if row.get("direct_audit_sha256") != direct_sha:
            raise ValueError(f"checkpoint belongs to different direct audit: {path}")
        if run_id in records:
            raise ValueError(f"duplicate checkpoint run_id: {run_id}")
        records[run_id] = row
    return records


def summarize(
    rows: list[dict[str, Any]],
    *,
    execution_sha: str,
    direct_sha: str,
    total_source_hard_failures: int,
) -> dict[str, Any]:
    by_method: dict[str, dict[str, int]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method_id"])].append(row)
    for method_id, method_rows in sorted(grouped.items()):
        classifications = Counter(str(row["classification"]) for row in method_rows)
        by_method[method_id] = {
            "checkpointed": len(method_rows),
            "transport_changed": sum(
                isinstance(row.get("first_transport_mismatch"), int) for row in method_rows
            ),
            "source_error_reproduced_exactly": sum(
                row.get("replayed_error_matches_source") is True for row in method_rows
            ),
            "text_transport_reproduced_on_reconstructed_sender": classifications.get(
                "text_transport_reproduced_on_reconstructed_sender", 0
            ),
        }
    counts = Counter(str(row["classification"]) for row in rows)
    attributed = counts.get("text_transport_reproduced_on_reconstructed_sender", 0)
    return {
        "schema_version": "stage5.hard_failure_attribution.summary.v1",
        "source_execution_sha256": execution_sha,
        "direct_audit_sha256": direct_sha,
        "source_hard_failure_count": total_source_hard_failures,
        "checkpointed_count": len(rows),
        "remaining_count": total_source_hard_failures - len(rows),
        "classification_counts": dict(sorted(counts.items())),
        "by_method": by_method,
        "direct_gate_exact_count": sum(row.get("direct_audit_exact") is True for row in rows),
        "transport_mismatch_count": sum(
            isinstance(row.get("first_transport_mismatch"), int) for row in rows
        ),
        "source_error_exact_reproduction_count": sum(
            row.get("replayed_error_matches_source") is True for row in rows
        ),
        "attributed_count": attributed,
        "all_checkpointed_attributed": bool(rows) and attributed == len(rows),
        "all_38_attributed": len(rows) == total_source_hard_failures == 38 and attributed == 38,
        "interpretation_caveat": (
            "Original sender token IDs were not persisted for Pilot-B hard failures; "
            "attribution is based on deterministic re-encoding of canonical configs."
        ),
    }


def write_checkpoint_outputs(
    output_dir: Path,
    records: dict[str, dict[str, Any]],
    *,
    execution_sha: str,
    direct_sha: str,
    total_source_hard_failures: int,
) -> dict[str, Any]:
    rows = [records[run_id] for run_id in sorted(records)]
    _atomic_jsonl(output_dir / "hard_failure_attribution.jsonl", rows)
    summary = summarize(
        rows,
        execution_sha=execution_sha,
        direct_sha=direct_sha,
        total_source_hard_failures=total_source_hard_failures,
    )
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def _parse_run_ids(values: list[str]) -> set[str]:
    result: set[str] = set()
    for value in values:
        result.update(item.strip() for item in value.split(",") if item.strip())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--execution-log", type=Path, default=DEFAULT_EXECUTION)
    parser.add_argument("--direct-audit", type=Path, default=DEFAULT_DIRECT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-id", action="append", default=[], help="Repeat or comma-separate IDs")
    parser.add_argument("--method", action="append", choices=HARD_METHODS, default=[])
    parser.add_argument("--inventory-only", action="store_true", help="No model/GPU loading")
    parser.add_argument("--resume", action="store_true", help="Skip valid completed checkpoints")
    args = parser.parse_args()

    execution_log = args.execution_log.expanduser().resolve()
    execution_sha = sha256(execution_log)
    failures = list(load_hard_failure_records(execution_log, methods=HARD_METHODS))
    source_counts = Counter(item.method_id for item in failures)
    if len(failures) != 38 or dict(source_counts) != EXPECTED_HARD_COUNTS:
        raise SystemExit(
            f"Pilot-B hard-failure inventory changed: total={len(failures)} by_method={dict(source_counts)}"
        )

    direct_rows, direct_sha = load_direct_gate(args.direct_audit.expanduser().resolve(), execution_sha)
    missing_direct = [item.run_id for item in failures if item.run_id not in direct_rows]
    non_exact_direct = [
        item.run_id
        for item in failures
        if item.run_id in direct_rows
        and not (
            direct_rows[item.run_id].get("audit_status") == "direct_exact"
            and direct_rows[item.run_id].get("direct_roundtrip_exact") is True
        )
    ]
    if missing_direct or non_exact_direct:
        raise SystemExit(
            f"direct-token gate failed: missing={missing_direct} non_exact={non_exact_direct}"
        )

    requested_ids = _parse_run_ids(args.run_id)
    if requested_ids:
        known = {item.run_id for item in failures}
        unknown = requested_ids - known
        if unknown:
            raise SystemExit(f"requested IDs are not Pilot-B hard failures: {sorted(unknown)}")
        failures = [item for item in failures if item.run_id in requested_ids]
    if args.method:
        selected_methods = set(args.method)
        failures = [item for item in failures if item.method_id in selected_methods]
    if not failures:
        raise SystemExit("no hard failures selected")

    manifest = SweepManifest.from_json(args.manifest, project_root=ROOT)
    plans = {plan.run_id: plan for plan in manifest.expand()}
    missing_plans = [item.run_id for item in failures if item.run_id not in plans]
    if missing_plans:
        raise SystemExit(f"selected hard failures absent from manifest: {missing_plans}")

    print("Stage 5 Pilot-B hard-failure attribution audit")
    print("source hard failures: 38", dict(EXPECTED_HARD_COUNTS))
    print("selected:", len(failures), dict(Counter(item.method_id for item in failures)))
    print("direct-token gate: 38/38 direct_exact")
    print("sender IDs for source hard failures: reconstructed; originals were not persisted")

    if args.inventory_only:
        for item in failures:
            print(item.run_id, item.method_id, item.method_point_id, item.prompt_id, item.replicate_id)
        return

    output_dir = args.output_dir.expanduser().resolve()
    records_dir = output_dir / "records"
    if output_dir == (ROOT / "results/runs").resolve() or (ROOT / "results/runs").resolve() in output_dir.parents:
        raise SystemExit("refusing to write inside canonical results/runs")

    existing = load_checkpoint_records(
        records_dir,
        execution_sha=execution_sha,
        direct_sha=direct_sha,
    )
    selected_ids = {item.run_id for item in failures}
    selected_existing = selected_ids & set(existing)
    if selected_existing and not args.resume:
        raise SystemExit(
            "selected checkpoints already exist; rerun with --resume to skip them: "
            f"{sorted(selected_existing)}"
        )
    pending = [item for item in failures if item.run_id not in existing]
    if not pending:
        summary = write_checkpoint_outputs(
            output_dir,
            existing,
            execution_sha=execution_sha,
            direct_sha=direct_sha,
            total_source_hard_failures=38,
        )
        print("All selected runs already checkpointed; model not loaded.")
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return

    prompts = PromptRegistry.from_jsonl(manifest.prompts_path)
    model_config = LocalModelConfig.from_json(manifest.model_config_path, project_root=ROOT)
    print("pending:", len(pending))
    print("Loading local model once...")
    lm = HFCausalLMAdapter.from_local_config(model_config)

    records = dict(existing)
    for index, failure in enumerate(pending, start=1):
        planned = plans[failure.run_id]
        print(
            f"[{index}/{len(pending)}] {failure.run_id} "
            f"{failure.method_id}/{failure.method_point_id} "
            f"{failure.prompt_id} {failure.replicate_id}",
            flush=True,
        )
        try:
            row = replay_one(
                planned=planned,
                source_error=failure,
                direct_row=direct_rows[failure.run_id],
                prompt_registry=prompts,
                lm=lm,
                execution_sha=execution_sha,
                direct_sha=direct_sha,
            )
        except Exception as exc:
            row = {
                **planned.metadata_record(),
                "schema_version": "stage5.hard_failure_attribution.record.v1",
                "source_execution_sha256": execution_sha,
                "direct_audit_sha256": direct_sha,
                "source_error_type": failure.error_type,
                "source_error_message": failure.error_message,
                "direct_audit_status": direct_rows[failure.run_id].get("audit_status"),
                "direct_audit_exact": direct_rows[failure.run_id].get("direct_roundtrip_exact") is True,
                "sender_token_source": "deterministic_reencode_original_sender_ids_not_persisted",
                "diagnostic_error_type": type(exc).__name__,
                "diagnostic_error_message": str(exc),
                "classification": "diagnostic_error_requires_investigation",
            }

        record_path = records_dir / f"{failure.run_id}.json"
        _atomic_json(record_path, row)
        records[failure.run_id] = row
        summary = write_checkpoint_outputs(
            output_dir,
            records,
            execution_sha=execution_sha,
            direct_sha=direct_sha,
            total_source_hard_failures=38,
        )
        print(
            "  class=", row["classification"],
            " mismatch=", row.get("first_transport_mismatch"),
            " failure_step=", row.get("failure_step"),
            " source_error_match=", row.get("replayed_error_matches_source"),
            " checkpointed=", summary["checkpointed_count"],
            "/38",
            sep="",
            flush=True,
        )

    final = write_checkpoint_outputs(
        output_dir,
        records,
        execution_sha=execution_sha,
        direct_sha=direct_sha,
        total_source_hard_failures=38,
    )
    print()
    print(json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True))
    print("records:", records_dir)
    print("jsonl:", output_dir / "hard_failure_attribution.jsonl")
    print("summary:", output_dir / "summary.json")


if __name__ == "__main__":
    main()
