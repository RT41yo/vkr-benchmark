#!/usr/bin/env python3
"""Read-only Stage-5 Pilot-B direct-token audit (no text transport).

For successful runs the persisted sender token IDs are authoritative; deterministic
re-encoding must match them before decoding. Failed runs lack stored sender IDs,
so their regenerated carriers are explicitly labelled reconstructed, not original.
This diagnostic never modifies benchmark algorithms or original run artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
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
from vkr_benchmark.runner.streaming import (
    decode_streaming_tokens, encode_fixed_carrier_tokens,
    method_environment_from_builder, warm_up_streaming_path,
)
from vkr_benchmark.sweeps import SweepManifest

DEFAULT_MANIFEST = ROOT / "configs/pilots/stage5/pilot_b_streaming_llama.json"
DEFAULT_LOG = ROOT / "results/stage5/sweeps/stage5-pilot-b-streaming-llama-v1/execution.jsonl"


def mismatch(a, b):
    return next((i for i in range(min(len(a), len(b))) if a[i] != b[i]),
                min(len(a), len(b)) if len(a) != len(b) else None)


def exact_result(expected, decoded):
    """Mirror normative streaming.roundtrip_exact, including incremental extras."""
    recovered = decoded.recovered_bits
    raw_len = max(len(decoded.incremental_recovered_bits), len(recovered))
    bit_mismatch = mismatch(expected, recovered)
    return {
        "direct_roundtrip_exact": (
            bit_mismatch is None and len(recovered) == len(expected)
            and raw_len == len(expected) and decoded.finalization.complete
        ),
        "first_secret_bit_mismatch": bit_mismatch,
        "expected_length_bits": len(expected),
        "recovered_final_length_bits": len(recovered),
        "recovered_raw_length_bits": raw_len,
        "recovered_extra_bits": max(0, raw_len - len(expected)),
        "decoder_complete": decoded.finalization.complete,
    }


def read_execution(path):
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    ids = [row["run_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate run IDs in execution log")
    return rows


def inventory(rows, run_root):
    result = Counter()
    for row in rows:
        if row["status"] == "ok":
            token_path = run_root / row["run_id"] / "token_ids.json"
            if token_path.is_file():
                try:
                    tokens = json.loads(token_path.read_text(encoding="utf-8"))["sender_token_ids"]
                    if not isinstance(tokens, list) or not all(type(x) is int for x in tokens):
                        raise ValueError("invalid sender IDs")
                    result["ok_with_sender_ids"] += 1
                except (OSError, ValueError, KeyError, json.JSONDecodeError):
                    result["ok_invalid_sender_ids"] += 1
            else:
                result["ok_missing_sender_ids"] += 1
        else:
            result["non_ok_reconstruction_needed"] += 1
    return dict(result)


def protected_output(path, protected):
    resolved = path.resolve()
    for directory in protected:
        base = directory.resolve()
        if resolved == base or base in resolved.parents:
            raise ValueError(f"Output must not be in protected benchmark artifacts: {resolved}")
    if resolved.exists():
        raise ValueError(f"Refusing to overwrite existing output: {resolved}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--execution-log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--run-root", type=Path, default=ROOT / "results/runs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "decoder_audit_local")
    parser.add_argument("--expected-count", type=int, default=504)
    parser.add_argument("--run-id", action="append", default=[], help="Repeat to select runs")
    parser.add_argument("--inventory-only", action="store_true", help="No model loading")
    args = parser.parse_args()

    rows = read_execution(args.execution_log)
    if len(rows) != args.expected_count:
        parser.error(f"Execution log contains {len(rows)} rows, expected {args.expected_count}")
    manifest = SweepManifest.from_json(args.manifest, project_root=ROOT)
    plans = {plan.run_id: plan for plan in manifest.expand()}
    if set(plans) != {row["run_id"] for row in rows}:
        parser.error("Manifest and execution log have different run IDs")
    stats = inventory(rows, args.run_root)
    print(json.dumps({"execution_count": len(rows), "inventory": stats}, indent=2))
    if args.inventory_only:
        return

    selected = set(args.run_id)
    if selected and not selected <= set(plans):
        parser.error(f"Unknown run IDs: {sorted(selected - set(plans))}")
    entries = [row for row in rows if not selected or row["run_id"] in selected]
    protected = [ROOT / "results", args.run_root, ROOT / "configs", ROOT / "src"]
    out_jsonl = args.output_dir / "direct_token_replay.jsonl"
    out_summary = args.output_dir / "summary.json"
    for path in (out_jsonl, out_summary):
        protected_output(path, protected)
    if any(row["status"] == "ok" and not (args.run_root / row["run_id"] / "token_ids.json").is_file() for row in entries):
        parser.error("Some successful runs lack sender IDs. Restore local results/runs before replay")

    prompts = PromptRegistry.from_jsonl(manifest.prompts_path)
    model_conf = LocalModelConfig.from_json(manifest.model_config_path, project_root=ROOT)
    lm = HFCausalLMAdapter.from_local_config(model_conf)
    output_rows = []
    for i, row in enumerate(entries, 1):
        plan = plans[row["run_id"]]
        config = plan.config
        outcome = {"run_id": plan.run_id, "method_id": config.method.method_id,
                   "method_point_id": plan.method_point_id, "source_status": row["status"],
                   "source_roundtrip_exact": row.get("roundtrip_exact"),
                   "sender_token_source": "persisted" if row["status"] == "ok" else "deterministic_reencode"}
        try:
            prompt = prompts.get(config.prompt_id)
            builder = ReferenceDistributionBuilder(token_space=lm.token_space, policy=config.generation.to_policy())
            env = method_environment_from_builder(builder)
            warm_up_streaming_path(lm_adapter=lm, reference_builder=builder, environment=env, prompt_text=prompt.text)
            if config.termination.mode != "fixed_carrier_tokens":
                raise ValueError("Only streaming fixed_carrier_tokens supported")
            encode_runtime = create_method_runtime(config)
            encoded = encode_fixed_carrier_tokens(
                lm_adapter=lm, reference_builder=builder, method=encode_runtime.method,
                method_config=config.method.params, environment=env, prompt_text=prompt.text,
                carrier_tokens=config.termination.target_carrier_tokens,
                secret_source=create_secret_source(config.secret_id),
                method_random_source=encode_runtime.encoder_random_source, key=config.method.key)
            regenerated = tuple(encoded.carrier_token_ids)
            if row["status"] == "ok":
                stored = json.loads((args.run_root / plan.run_id / "token_ids.json").read_text(encoding="utf-8"))
                sender = tuple(stored["sender_token_ids"])
                outcome["reencode_sender_ids_exact"] = sender == regenerated
                outcome["first_reencode_token_mismatch"] = mismatch(sender, regenerated)
                if sender != regenerated:
                    outcome["audit_status"] = "replay_nonreproducible"
                    output_rows.append(outcome)
                    print(f"[{i}/{len(entries)}] {plan.run_id}: replay_nonreproducible", flush=True)
                    continue
            else:
                sender = regenerated
                outcome["reencode_sender_ids_exact"] = None
                outcome["first_reencode_token_mismatch"] = None
            outcome["sender_token_count"] = len(sender)
            outcome["payload_bits"] = encoded.payload_bits
            decoder_runtime = create_method_runtime(config)
            decoded = decode_streaming_tokens(
                lm_adapter=lm, reference_builder=builder, method=decoder_runtime.method,
                method_config=config.method.params, environment=env, prompt_text=prompt.text,
                observed_token_ids=sender,
                method_random_source=decoder_runtime.decoder_random_source,
                expected_payload_bits=encoded.payload_bits, key=config.method.key)
            outcome.update(exact_result(encoded.payload_secret_bits, decoded))
            outcome["audit_status"] = "direct_exact" if outcome["direct_roundtrip_exact"] else "direct_inexact"
        except Exception as exc:
            outcome.update(audit_status="audit_error", error_type=type(exc).__name__, error_message=str(exc))
        output_rows.append(outcome)
        print(f"[{i}/{len(entries)}] {plan.run_id}: {outcome['audit_status']}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # exclusive create: cannot accidentally overwrite a previous checkpoint
    with out_jsonl.open("x", encoding="utf-8") as f:
        for result in output_rows:
            f.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "schema_version": "stage5.direct_token_audit.v1",
        "source_execution_sha256": hashlib.sha256(args.execution_log.read_bytes()).hexdigest(),
        "selected_count": len(entries),
        "status_counts": dict(Counter(x["audit_status"] for x in output_rows)),
        "by_method": {method: dict(Counter(x["audit_status"] for x in output_rows if x["method_id"] == method))
                      for method in sorted({x["method_id"] for x in output_rows})},
        "all_selected_direct_exact": all(x["audit_status"] == "direct_exact" for x in output_rows),
        "source_reconstructed_hard_failures_not_verifiable_against_original_ids": True,
    }
    with out_summary.open("x", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    print(json.dumps(summary, indent=2))
    print("Audit output:", args.output_dir)

if __name__ == "__main__":
    main()
