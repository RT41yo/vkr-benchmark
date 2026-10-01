#!/usr/bin/env python3
"""Probe the Discop paper characteristic across the paper's top-p grid.

This is a normalized-model characteristic probe, not a numerical reproduction
of Table II: the paper used GPT-2 over 100 IMDb contexts, whereas this script
uses the benchmark LM and the explicitly supplied prompt.  It checks whether
zero distribution distortion and high entropy use can be observed under a
controlled longer run and records the result for the Stage-4 report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vkr_benchmark.config import LocalModelConfig
from vkr_benchmark.distributions import GenerationPolicy, ReferenceDistributionBuilder
from vkr_benchmark.lm import HFCausalLMAdapter
from vkr_benchmark.methods import DiscopMethod
from vkr_benchmark.randomness import MethodRandomSource, Shake256SecretSource
from vkr_benchmark.runner import (
    method_environment_from_builder,
    run_streaming_text_roundtrip,
)
from vkr_benchmark.runner.streaming import warm_up_streaming_path

PAPER_TARGETS = {
    0.80: {"capacity_bits_per_token": 3.48, "entropy_bits_per_token": 3.79, "utilization": 0.92},
    0.92: {"capacity_bits_per_token": 4.55, "entropy_bits_per_token": 4.86, "utilization": 0.94},
    0.95: {"capacity_bits_per_token": 4.84, "entropy_bits_per_token": 5.18, "utilization": 0.94},
    0.98: {"capacity_bits_per_token": 5.29, "entropy_bits_per_token": 5.59, "utilization": 0.95},
    1.00: {"capacity_bits_per_token": 5.76, "entropy_bits_per_token": 6.08, "utilization": 0.95},
}


def _row(result, top_p: float) -> dict[str, object]:
    payload = result.encode.payload_bits
    carriers = result.encode.carrier_tokens
    entropy_sum = float(sum(result.encode.step_reference_entropy_bits))
    entropy_mean = entropy_sum / carriers
    utilization = payload / entropy_sum if entropy_sum else 0.0
    distortions = result.encode.step_distribution_distortion
    encode_stego_ms = result.encode.timing.stego_algorithm_ms
    decode_stego_ms = result.decode.timing.stego_algorithm_ms
    return {
        "top_p": top_p,
        "carrier_tokens": carriers,
        "payload_bits": payload,
        "capacity_bits_per_token": payload / carriers,
        "entropy_bits_per_token": entropy_mean,
        "entropy_sum_bits": entropy_sum,
        "utilization": utilization,
        "kl_ref_to_stego_max_bits": max(x.kl_ref_to_stego_bits for x in distortions),
        "kl_stego_to_ref_max_bits": max(x.kl_stego_to_ref_bits for x in distortions),
        "tvd_max": max(x.tvd for x in distortions),
        "token_sequence_roundtrip_exact": result.transport.token_sequence_roundtrip_exact,
        "secret_roundtrip_exact": result.roundtrip_exact,
        "decoder_complete": result.decode.finalization.complete,
        "encode_stego_ms_per_token": encode_stego_ms / carriers,
        "decode_stego_ms_per_token": decode_stego_ms / max(1, len(result.decode.observed_token_ids)),
        "paper_table_ii_gpt2_reference": PAPER_TARGETS.get(round(top_p, 2)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_config", type=Path)
    parser.add_argument("--carrier-tokens", type=int, default=100)
    parser.add_argument("--secret-id", default="stage4-discop-characteristic")
    parser.add_argument("--key", type=int, default=12345)
    parser.add_argument(
        "--prompt",
        default="The history of artificial intelligence began",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        nargs="+",
        default=[0.80, 0.92, 0.95, 0.98, 1.00],
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.carrier_tokens <= 0:
        parser.error("--carrier-tokens must be positive")

    config = LocalModelConfig.from_json(args.model_config)
    print("Stage 4 Discop characteristic probe")
    print("model:", config.model_id)
    print("revision:", config.revision)
    print("carrier tokens per point:", args.carrier_tokens)
    print("top-p grid:", args.top_p)
    print("Loading local model...")
    lm = HFCausalLMAdapter.from_local_config(config)

    rows: list[dict[str, object]] = []
    for top_p in args.top_p:
        builder = ReferenceDistributionBuilder(
            token_space=lm.token_space,
            policy=GenerationPolicy(top_p=float(top_p)),
        )
        environment = method_environment_from_builder(builder)
        warm_up_streaming_path(
            lm_adapter=lm,
            reference_builder=builder,
            environment=environment,
            prompt_text=args.prompt,
        )
        result = run_streaming_text_roundtrip(
            lm_adapter=lm,
            reference_builder=builder,
            method=DiscopMethod(),
            method_config={},
            environment=environment,
            prompt_text=args.prompt,
            carrier_tokens=args.carrier_tokens,
            secret_source=Shake256SecretSource(args.secret_id),
            encoder_random_source=MethodRandomSource(args.key),
            decoder_random_source=MethodRandomSource(args.key),
            key=args.key,
        )
        row = _row(result, float(top_p))
        rows.append(row)
        print(
            f"p={top_p:.2f}  BPT={row['capacity_bits_per_token']:.4f}  "
            f"H={row['entropy_bits_per_token']:.4f}  util={100*float(row['utilization']):.2f}%  "
            f"KLmax={float(row['kl_ref_to_stego_max_bits']):.3g}/"
            f"{float(row['kl_stego_to_ref_max_bits']):.3g}  "
            f"TVDmax={float(row['tvd_max']):.3g}  "
            f"tokenRT={row['token_sequence_roundtrip_exact']}  "
            f"secretRT={row['secret_roundtrip_exact']}"
        )

    all_exact = all(
        bool(row["token_sequence_roundtrip_exact"])
        and bool(row["secret_roundtrip_exact"])
        and bool(row["decoder_complete"])
        for row in rows
    )
    zero_distortion = all(
        float(row["kl_ref_to_stego_max_bits"]) == 0.0
        and float(row["kl_stego_to_ref_max_bits"]) == 0.0
        and float(row["tvd_max"]) == 0.0
        for row in rows
    )
    payload = {
        "status": "pass" if all_exact and zero_distortion else "fail",
        "probe_kind": "normalized-characteristic-not-table-ii-numerical-reproduction",
        "model_id": config.model_id,
        "model_revision": config.revision,
        "prompt": args.prompt,
        "secret_id": args.secret_id,
        "key": args.key,
        "carrier_tokens_per_point": args.carrier_tokens,
        "rows": rows,
        "aggregate": {
            "all_text_and_secret_roundtrips_exact": all_exact,
            "all_reported_distribution_distortions_zero": zero_distortion,
            "mean_utilization": mean(float(row["utilization"]) for row in rows),
        },
        "paper_context": {
            "source": "Ding et al., IEEE S&P 2023, Table II",
            "paper_model": "GPT-2",
            "paper_contexts": "100 IMDb texts; first three sentences as context; 100 generated tokens",
            "paper_claim": "Discop utilization 0.92-0.95 over p=0.80..1.00; Ave/Max KLD 0",
            "comparison_limitation": (
                "Absolute BPT/entropy/utilization are not expected to numerically match Table II "
                "because this probe uses a different model and one explicit benchmark prompt."
            ),
        },
    }

    print("status:", payload["status"].upper())
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print("json:", args.json_output)
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
