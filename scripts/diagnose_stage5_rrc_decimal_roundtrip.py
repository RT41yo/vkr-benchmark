#!/usr/bin/env python3
"""Attribute the single inexact RRC Decimal-boundary validation run.

The diagnostic never changes production RRC. It compares persisted sender and
receiver token IDs, then decodes the persisted sender IDs directly (bypassing
text serialization/retokenization) for the selected validation run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vkr_benchmark.config import LocalModelConfig
from vkr_benchmark.distributions import ReferenceDistributionBuilder
from vkr_benchmark.inputs import PromptRegistry, create_secret_source
from vkr_benchmark.lm import HFCausalLMAdapter
from vkr_benchmark.runner.experiment import create_method_runtime
from vkr_benchmark.runner.streaming import decode_streaming_tokens, method_environment_from_builder
from vkr_benchmark.sweeps import SweepManifest

DEFAULT_MANIFEST = ROOT / "configs/pilots/stage5/rrc_decimal_boundary_validation_llama.json"
DEFAULT_EXECUTION = ROOT / "results/stage5/sweeps/stage5-rrc-decimal-boundary-validation-llama-v1/execution.jsonl"
DEFAULT_RUN_ID = "9967d088525fcf46"
DEFAULT_OUTPUT = ROOT / "results/stage5/rrc_decimal_boundary_validation/roundtrip_attribution.json"


def first_mismatch(a: tuple[int, ...], b: tuple[int, ...]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def bit_error_rate(expected: tuple[int, ...], recovered: tuple[int, ...]) -> float:
    length = max(len(expected), len(recovered))
    if length == 0:
        return 0.0
    common = min(len(expected), len(recovered))
    errors = sum(expected[i] != recovered[i] for i in range(common))
    errors += abs(len(expected) - len(recovered))
    return errors / length


def direct_exact(expected: tuple[int, ...], decoded) -> tuple[bool, int | None, int]:
    recovered = tuple(decoded.recovered_bits)
    raw_len = max(len(decoded.incremental_recovered_bits), len(recovered))
    mismatch = first_mismatch(expected, recovered)
    exact = (
        mismatch is None
        and len(recovered) == len(expected)
        and raw_len == len(expected)
        and decoded.finalization.complete
    )
    return exact, mismatch, raw_len


def load_execution_row(path: Path, run_id: str) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            row = json.loads(raw)
            if row.get("run_id") == run_id:
                rows.append(row)
    if not rows:
        raise ValueError(f"run_id {run_id} is absent from {path}")
    return rows[-1]


def load_tokens(run_root: Path, run_id: str) -> tuple[tuple[int, ...], tuple[int, ...], dict[str, object]]:
    path = run_root / run_id / "token_ids.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    sender = tuple(int(x) for x in value["sender_token_ids"])
    receiver = tuple(int(x) for x in value["receiver_token_ids"])
    return sender, receiver, value


def build_inventory(*, source: dict[str, object], sender: tuple[int, ...], receiver: tuple[int, ...], token_record: dict[str, object]) -> dict[str, object]:
    mismatch = first_mismatch(sender, receiver)
    return {
        "source_status": source.get("status"),
        "source_roundtrip_exact": source.get("roundtrip_exact"),
        "source_ber": source.get("ber"),
        "sender_token_count": len(sender),
        "receiver_token_count": len(receiver),
        "token_sequence_roundtrip_exact": sender == receiver,
        "first_token_mismatch": mismatch,
        "token_count_delta": len(receiver) - len(sender),
        "persisted_first_token_mismatch": token_record.get("first_token_mismatch"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--execution-log", type=Path, default=DEFAULT_EXECUTION)
    parser.add_argument("--run-root", type=Path, default=ROOT / "results/runs")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()

    manifest = SweepManifest.from_json(args.manifest, project_root=ROOT)
    plans = {plan.run_id: plan for plan in manifest.expand()}
    if args.run_id not in plans:
        parser.error(f"run_id {args.run_id} is not in the validation manifest")
    plan = plans[args.run_id]
    config = plan.config
    if config.method.method_id != "rrc" or config.termination.mode != "fixed_payload_bits":
        parser.error("selected run must be an RRC fixed_payload_bits validation run")
    if config.termination.target_payload_bits is None:
        parser.error("target_payload_bits is missing")

    source = load_execution_row(args.execution_log, args.run_id)
    if source.get("status") != "ok":
        parser.error("selected source run is not process-successful")
    sender, receiver, token_record = load_tokens(args.run_root, args.run_id)
    inventory = build_inventory(source=source, sender=sender, receiver=receiver, token_record=token_record)
    print(json.dumps({"run_id": args.run_id, **inventory}, indent=2, sort_keys=True))
    if args.inventory_only:
        return 0

    prompts = PromptRegistry.from_jsonl(manifest.prompts_path)
    prompt = prompts.get(config.prompt_id)
    model_conf = LocalModelConfig.from_json(manifest.model_config_path, project_root=ROOT)
    lm = HFCausalLMAdapter.from_local_config(model_conf)
    builder = ReferenceDistributionBuilder(token_space=lm.token_space, policy=config.generation.to_policy())
    environment = method_environment_from_builder(builder)

    payload_bits = int(config.termination.target_payload_bits)
    expected = tuple(create_secret_source(config.secret_id).read_bits(payload_bits))
    runtime = create_method_runtime(config)
    decoded = decode_streaming_tokens(
        lm_adapter=lm,
        reference_builder=builder,
        method=runtime.method,
        method_config=config.method.params,
        environment=environment,
        prompt_text=prompt.text,
        observed_token_ids=sender,
        method_random_source=runtime.decoder_random_source,
        expected_payload_bits=payload_bits,
        key=config.method.key,
    )
    recovered = tuple(decoded.recovered_bits)
    exact, bit_mismatch, raw_len = direct_exact(expected, decoded)
    transport_changed = sender != receiver
    source_inexact = source.get("roundtrip_exact") is False
    attribution = source_inexact and transport_changed and exact

    result = {
        "schema_version": "stage5.rrc_decimal_roundtrip_attribution.v1",
        "run_id": args.run_id,
        "method_point_id": plan.method_point_id,
        **inventory,
        "expected_payload_bits": payload_bits,
        "direct_sender_decode_exact": exact,
        "direct_sender_decode_ber": bit_error_rate(expected, recovered),
        "direct_sender_first_bit_mismatch": bit_mismatch,
        "direct_sender_recovered_length_bits": len(recovered),
        "direct_sender_raw_recovered_length_bits": raw_len,
        "direct_sender_decoder_complete": decoded.finalization.complete,
        "text_transport_attributed": attribution,
        "classification": (
            "text_transport_attributed"
            if attribution
            else "not_attributed_to_text_transport"
        ),
        "interpretation": (
            "Direct sender-token decoding bypasses text serialization and retokenization. "
            "Attribution requires the source run to be inexact, sender/receiver token IDs to differ, "
            "and direct decoding of the persisted sender IDs to recover the payload exactly."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("RRC Decimal-boundary roundtrip attribution")
    print("direct sender decode exact:", exact)
    print("direct sender BER:", result["direct_sender_decode_ber"])
    print("text transport attributed:", attribution)
    print("json:", args.output)
    return 0 if attribution else 2


if __name__ == "__main__":
    raise SystemExit(main())
