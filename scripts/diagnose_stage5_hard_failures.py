#!/usr/bin/env python3
"""Replay Stage-5 hard failures with text-channel diagnostics.

This script does not change method semantics.  For each Huffman/DAIRstega
error recorded in a sweep execution log it:

1. reproduces the deterministic sender token sequence;
2. verifies decoding directly from sender token IDs (no text transport);
3. sends the carrier through the ordinary text channel;
4. replays receiver-side decoding until the original hard failure;
5. records the first sender/receiver token mismatch and method-local state at
   the failing decoder step.

The language model is loaded once for the whole diagnostic batch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vkr_benchmark.config import LocalModelConfig
from vkr_benchmark.diagnostics import (
    classify_hard_failure,
    load_hard_failure_records,
    summarize_hard_failure_diagnostics,
)
from vkr_benchmark.distributions import ReferenceDistributionBuilder, StepContext
from vkr_benchmark.inputs import PromptRegistry, create_secret_source
from vkr_benchmark.lm import HFCausalLMAdapter
from vkr_benchmark.methods.dair import DAIRConfig, _allocate
from vkr_benchmark.methods.huffman import HuffmanConfig, _build_codebook
from vkr_benchmark.runner.experiment import create_method_runtime
from vkr_benchmark.runner.streaming import (
    decode_streaming_tokens,
    encode_fixed_carrier_tokens,
    method_environment_from_builder,
    warm_up_streaming_path,
)
from vkr_benchmark.sweeps import PlannedRun, SweepManifest
from vkr_benchmark.transport import TextChannel


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_bit_mismatch(a: tuple[int, ...], b: tuple[int, ...]) -> int | None:
    for index, (left, right) in enumerate(zip(a, b, strict=False)):
        if left != right:
            return index
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def _rank(reference_order: np.ndarray, token_id: int | None) -> int | None:
    if token_id is None:
        return None
    matches = np.flatnonzero(reference_order == int(token_id))
    if matches.size == 0:
        return None
    return int(matches[0])


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
    common = {
        "observed_token_id": int(observed_token_id),
        "sender_token_id_same_index": (
            None if sender_token_id is None else int(sender_token_id)
        ),
        "observed_reference_rank": _rank(order, int(observed_token_id)),
        "sender_reference_rank": _rank(order, sender_token_id),
        "observed_reference_probability": float(
            context.reference.probabilities[int(observed_token_id)]
        ),
        "sender_reference_probability": (
            None
            if sender_token_id is None
            else float(context.reference.probabilities[int(sender_token_id)])
        ),
    }

    if method_id == "huffman":
        config = HuffmanConfig.from_mapping(planned.config.method.params)
        codebook = _build_codebook(
            context=context,
            config=config,
            environment=environment,
        )
        candidates = tuple(int(item) for item in codebook.candidate_token_ids)
        candidate_set = set(candidates)
        common.update(
            {
                "candidate_count": len(candidates),
                "candidate_token_ids": list(candidates),
                "observed_in_candidate_set": int(observed_token_id) in candidate_set,
                "sender_in_candidate_set": (
                    sender_token_id is not None and int(sender_token_id) in candidate_set
                ),
                "sender_huffman_code": (
                    list(codebook.codes[int(sender_token_id)])
                    if sender_token_id is not None
                    and int(sender_token_id) in codebook.codes
                    else None
                ),
            }
        )
        return common

    if method_id == "dairstega":
        config = DAIRConfig.from_mapping(planned.config.method.params)
        intervals = _allocate(context, config)
        interval_tokens = tuple(int(item.token_id) for item in intervals)
        interval_set = set(interval_tokens)
        common.update(
            {
                "positive_interval_count": len(intervals),
                "positive_interval_token_ids": list(interval_tokens),
                "observed_has_positive_interval": int(observed_token_id) in interval_set,
                "sender_has_positive_interval": (
                    sender_token_id is not None and int(sender_token_id) in interval_set
                ),
                "decoder_window_lower_before_failure": int(
                    getattr(decoder, "_window_lower")
                ),
                "decoder_window_upper_before_failure": int(
                    getattr(decoder, "_window_upper")
                ),
            }
        )
        return common

    return common


def _replay_one(
    *,
    planned: PlannedRun,
    source_error: Any,
    prompt_registry: PromptRegistry,
    lm: HFCausalLMAdapter,
) -> dict[str, Any]:
    config = planned.config
    prompt = prompt_registry.get(config.prompt_id)
    builder = ReferenceDistributionBuilder(
        token_space=lm.token_space,
        policy=config.generation.to_policy(),
    )
    environment = method_environment_from_builder(builder)
    runtime = create_method_runtime(config)

    warm_up_streaming_path(
        lm_adapter=lm,
        reference_builder=builder,
        environment=environment,
        prompt_text=prompt.text,
    )

    if config.termination.mode != "fixed_carrier_tokens":
        raise ValueError(
            "hard-failure diagnostic currently expects fixed_carrier_tokens runs"
        )
    carrier_tokens = config.termination.target_carrier_tokens
    if carrier_tokens is None:
        raise ValueError("fixed_carrier_tokens run has no target_carrier_tokens")

    encoded = encode_fixed_carrier_tokens(
        lm_adapter=lm,
        reference_builder=builder,
        method=runtime.method,
        method_config=config.method.params,
        environment=environment,
        prompt_text=prompt.text,
        carrier_tokens=carrier_tokens,
        secret_source=create_secret_source(config.secret_id),
        method_random_source=runtime.encoder_random_source,
        key=config.method.key,
    )

    # Control path: direct token IDs must decode exactly if the method itself is
    # symmetric and the hard failure is induced only by ordinary-text transport.
    direct_runtime = create_method_runtime(config)
    direct_error_type: str | None = None
    direct_error_message: str | None = None
    direct_exact = False
    direct_first_bit_mismatch: int | None = None
    try:
        direct = decode_streaming_tokens(
            lm_adapter=lm,
            reference_builder=builder,
            method=direct_runtime.method,
            method_config=config.method.params,
            environment=environment,
            prompt_text=prompt.text,
            observed_token_ids=encoded.carrier_token_ids,
            method_random_source=direct_runtime.decoder_random_source,
            expected_payload_bits=encoded.payload_bits,
            key=config.method.key,
        )
        direct_first_bit_mismatch = _first_bit_mismatch(
            encoded.payload_secret_bits,
            direct.recovered_bits,
        )
        direct_exact = (
            direct_first_bit_mismatch is None
            and len(direct.recovered_bits) == len(encoded.payload_secret_bits)
            and direct.finalization.complete
        )
    except Exception as exc:  # diagnostics must record, not hide, control failure
        direct_error_type = type(exc).__name__
        direct_error_message = str(exc)

    transport = TextChannel(lm).transmit(encoded.carrier_token_ids)

    transported_runtime = create_method_runtime(config)
    decoder = transported_runtime.method.create_decoder(
        config=config.method.params,
        environment=environment,
        random_source=transported_runtime.decoder_random_source,
        key=config.method.key,
        expected_payload_bits=encoded.payload_bits,
    )
    prompt_token_ids = lm.encode_prompt(prompt.text)
    state = lm.prefill(prompt_token_ids)

    failure_step: int | None = None
    replayed_error_type: str | None = None
    replayed_error_message: str | None = None
    method_state: dict[str, Any] | None = None
    recovered_before_failure: list[int] = []

    observed = transport.receiver_token_ids
    for step_index, observed_token_id in enumerate(observed):
        raw_logits, state = lm.next_logits(state)
        reference = builder.build(raw_logits)
        context = StepContext(step_index=step_index, reference=reference)
        sender_token_id = (
            encoded.carrier_token_ids[step_index]
            if step_index < len(encoded.carrier_token_ids)
            else None
        )
        try:
            progress = decoder.observe(context, int(observed_token_id))
        except Exception as exc:
            failure_step = step_index
            replayed_error_type = type(exc).__name__
            replayed_error_message = str(exc)
            method_state = _method_state(
                planned,
                context=context,
                environment=environment,
                decoder=decoder,
                observed_token_id=int(observed_token_id),
                sender_token_id=sender_token_id,
            )
            break
        recovered_before_failure.extend(progress.recovered_bits)
        if step_index + 1 < len(observed):
            _, state = lm.next_logits(state, int(observed_token_id))

    result: dict[str, Any] = {
        **planned.metadata_record(),
        "source_error_type": source_error.error_type,
        "source_error_message": source_error.error_message,
        "sender_token_count": len(encoded.carrier_token_ids),
        "receiver_token_count": len(transport.receiver_token_ids),
        "token_count_delta": transport.token_count_delta,
        "first_transport_mismatch": transport.first_token_mismatch,
        "transport_token_sequence_exact": transport.token_sequence_roundtrip_exact,
        "first_transport_mismatch_window": _token_window(
            encoded.carrier_token_ids,
            transport.receiver_token_ids,
            transport.first_token_mismatch,
        ),
        "direct_token_decode_exact": direct_exact,
        "direct_token_decode_first_bit_mismatch": direct_first_bit_mismatch,
        "direct_token_decode_error_type": direct_error_type,
        "direct_token_decode_error_message": direct_error_message,
        "failure_step": failure_step,
        "replayed_error_type": replayed_error_type,
        "replayed_error_message": replayed_error_message,
        "replayed_error_matches_source": (
            replayed_error_type == source_error.error_type
            and replayed_error_message == source_error.error_message
        ),
        "failure_at_or_after_first_transport_mismatch": (
            isinstance(transport.first_token_mismatch, int)
            and isinstance(failure_step, int)
            and failure_step >= transport.first_token_mismatch
        ),
        "recovered_bits_before_failure": len(recovered_before_failure),
        "method_state_at_failure": method_state,
        "failure_window": _token_window(
            encoded.carrier_token_ids,
            transport.receiver_token_ids,
            failure_step,
        ),
    }
    result["classification"] = classify_hard_failure(result)
    return result


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--execution-log", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Default: hard_failure_diagnostics.jsonl beside --execution-log.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Default: hard_failure_diagnostics_summary.json beside --execution-log.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=None,
        help="Fail before loading the model unless this many hard failures are selected.",
    )
    parser.add_argument(
        "--run-id",
        help="Optional comma-separated subset of hard-failure run ids.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Validate and print selected hard failures without loading prompts/model.",
    )
    args = parser.parse_args()

    execution_log = args.execution_log.expanduser().resolve()
    failures = list(load_hard_failure_records(execution_log))
    requested_ids = None
    if args.run_id:
        requested_ids = {item.strip() for item in args.run_id.split(",") if item.strip()}
        failures = [item for item in failures if item.run_id in requested_ids]
        missing = requested_ids - {item.run_id for item in failures}
        if missing:
            raise SystemExit(f"requested run ids are not hard failures in source log: {sorted(missing)}")

    if args.expected_count is not None and len(failures) != args.expected_count:
        raise SystemExit(
            f"expected {args.expected_count} hard failures, found {len(failures)}"
        )
    if not failures:
        raise SystemExit("no Huffman/DAIRstega hard failures selected")

    manifest = SweepManifest.from_json(args.manifest, project_root=REPO_ROOT)
    planned_by_id = {item.run_id: item for item in manifest.expand()}
    missing_plans = [item.run_id for item in failures if item.run_id not in planned_by_id]
    if missing_plans:
        raise SystemExit(f"hard-failure run ids absent from manifest: {missing_plans}")

    if args.list_only:
        print("selected hard failures:", len(failures))
        for item in failures:
            print(
                item.run_id,
                item.method_id,
                item.method_point_id,
                item.prompt_id,
                item.replicate_id,
            )
        return

    prompt_registry = PromptRegistry.from_jsonl(manifest.prompts_path)
    model_config = LocalModelConfig.from_json(
        manifest.model_config_path,
        project_root=REPO_ROOT,
    )

    output = (
        args.output.expanduser().resolve()
        if args.output
        else execution_log.with_name("hard_failure_diagnostics.jsonl")
    )
    summary_output = (
        args.summary_output.expanduser().resolve()
        if args.summary_output
        else execution_log.with_name("hard_failure_diagnostics_summary.json")
    )

    print("Stage-5 hard-failure diagnostic")
    print("source sweep:", manifest.sweep_id)
    print("selected hard failures:", len(failures))
    print("source log:", execution_log)
    print("output:", output)
    print("Loading local model once...")
    lm = HFCausalLMAdapter.from_local_config(model_config)

    rows: list[dict[str, Any]] = []
    for index, failure in enumerate(failures, start=1):
        planned = planned_by_id[failure.run_id]
        print(
            f"[{index}/{len(failures)}] {failure.run_id} "
            f"{failure.method_id}/{failure.method_point_id} "
            f"{failure.prompt_id} {failure.replicate_id}"
        )
        row = _replay_one(
            planned=planned,
            source_error=failure,
            prompt_registry=prompt_registry,
            lm=lm,
        )
        rows.append(row)
        print(
            "  direct_exact=", row["direct_token_decode_exact"],
            " transport_mismatch=", row["first_transport_mismatch"],
            " failure_step=", row["failure_step"],
            " class=", row["classification"],
            sep="",
        )

    _write_jsonl(output, rows)
    summary = {
        "schema_version": "stage5.hard_failure_diagnostics.v1",
        "source_sweep_id": manifest.sweep_id,
        "source_execution_log": str(execution_log),
        "source_execution_log_sha256": _sha256(execution_log),
        **summarize_hard_failure_diagnostics(rows),
        "all_replayed_errors_match_source": all(
            row["replayed_error_matches_source"] for row in rows
        ),
    }
    summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print()
    print("diagnosed_runs:", summary["diagnosed_runs"])
    print("by_method:", summary["by_method"])
    print("by_classification:", summary["by_classification"])
    print("direct_token_decode_exact_count:", summary["direct_token_decode_exact_count"])
    print("transport_mismatch_count:", summary["transport_mismatch_count"])
    print("all_replayed_errors_match_source:", summary["all_replayed_errors_match_source"])
    print("written:", output)
    print("written:", summary_output)


if __name__ == "__main__":
    main()
