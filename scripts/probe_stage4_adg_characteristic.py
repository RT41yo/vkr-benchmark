#!/usr/bin/env python3
"""Probe the ADG publication characteristic on the normalized benchmark LM.

This is intentionally *not* a numerical reproduction of Zhang et al. Table 1.
The paper used a word-level two-layer LSTM trained separately on Movie/News/Tweet
corpora, while the normalized Stage-4 benchmark uses the pinned modern LM and
canonical P_reference.  The probe asks whether the characteristic ADG behavior
is present under the benchmark conditions: useful payload with small but
non-zero distribution distortion, measured from an independently enumerated
Q_stego, plus exact secret recovery.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from statistics import mean, median

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vkr_benchmark.config import LocalModelConfig
from vkr_benchmark.distributions import GenerationPolicy, ReferenceDistributionBuilder
from vkr_benchmark.lm import HFCausalLMAdapter
from vkr_benchmark.methods import ADGMethod
from vkr_benchmark.randomness import MethodRandomSource, Shake256SecretSource
from vkr_benchmark.runner import method_environment_from_builder, run_streaming_text_roundtrip
from vkr_benchmark.runner.streaming import warm_up_streaming_path

REFERENCE_REPOSITORY = "https://github.com/Mhzzzzz/ADG-steganography"
REFERENCE_COMMIT = "b4a7e802a97bc3a3b84d07e63f24627b16c51a10"

PAPER_TABLE_1 = {
    "Movie": {"embedding_rate_bits_per_token": 5.147, "kld1_bits_per_token": 0.033},
    "News": {"embedding_rate_bits_per_token": 5.650, "kld1_bits_per_token": 0.027},
    "Tweet": {"embedding_rate_bits_per_token": 5.411, "kld1_bits_per_token": 0.048},
}


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute p95 of an empty sequence")
    # Nearest-rank percentile: deterministic and dependency-free.
    rank = max(1, math.ceil(0.95 * len(ordered)))
    return float(ordered[rank - 1])


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    return {
        "mean": float(mean(values)),
        "median": float(median(values)),
        "p95": _p95(values),
        "max": float(max(values)),
    }


def _run_row(result, *, run_index: int, secret_id: str, method_seed: int) -> dict[str, object]:
    carriers = result.encode.carrier_tokens
    payload = result.encode.payload_bits
    entropy_values = [float(x) for x in result.encode.step_reference_entropy_bits]
    entropy_sum = math.fsum(entropy_values)
    distortions = list(result.encode.step_distribution_distortion)
    kl_forward = [float(x.kl_ref_to_stego_bits) for x in distortions]
    kl_reverse = [float(x.kl_stego_to_ref_bits) for x in distortions]
    tvd = [float(x.tvd) for x in distortions]

    q_modes = sorted({x.q_mode.value for x in distortions})
    q_sources = sorted({x.q_source.value for x in distortions})
    all_finite = all(math.isfinite(x) for x in kl_forward + kl_reverse + tvd)

    return {
        "run_index": run_index,
        "secret_id": secret_id,
        "method_seed": method_seed,
        "carrier_tokens": carriers,
        "payload_bits": payload,
        "bits_per_token": payload / carriers,
        "reference_entropy_mean_bits_per_token": entropy_sum / carriers,
        "reference_entropy_sum_bits": entropy_sum,
        "entropy_utilization": payload / entropy_sum if entropy_sum else None,
        "q_modes": q_modes,
        "q_sources": q_sources,
        "kl_ref_to_stego_bits_per_token": _summary(kl_forward),
        "kl_stego_to_ref_bits_per_token": _summary(kl_reverse),
        "tvd": _summary(tvd),
        "all_distortion_metrics_finite": all_finite,
        "token_sequence_roundtrip_exact": result.transport.token_sequence_roundtrip_exact,
        "secret_roundtrip_exact": result.roundtrip_exact,
        "decoder_complete": result.decode.finalization.complete,
        "encode_stego_algorithm_ms": float(result.encode.timing.stego_algorithm_ms),
        "decode_stego_algorithm_ms": float(result.decode.timing.stego_algorithm_ms),
        "metric_instrumentation_ms": float(
            result.encode.timing.metric_instrumentation_ms
            + result.decode.timing.metric_instrumentation_ms
        ),
        "stegotext": result.transport.text,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_config", type=Path)
    parser.add_argument("--carrier-tokens", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--secret-id", default="stage4-adg-characteristic")
    parser.add_argument("--method-seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--prompt",
        default="The history of artificial intelligence began",
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    if args.carrier_tokens <= 0:
        parser.error("--carrier-tokens must be positive")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")

    config = LocalModelConfig.from_json(args.model_config)
    print("Stage 4 ADG publication-characteristic probe")
    print("reference:", f"{REFERENCE_REPOSITORY}@{REFERENCE_COMMIT}")
    print("model:", config.model_id)
    print("revision:", config.revision)
    print("carrier tokens per run:", args.carrier_tokens)
    print("repeats:", args.repeats)
    print("common generation top-k:", args.top_k)
    print("common generation top-p:", args.top_p)
    print("Loading local model...")

    lm = HFCausalLMAdapter.from_local_config(config)
    builder = ReferenceDistributionBuilder(
        token_space=lm.token_space,
        policy=GenerationPolicy(top_k=args.top_k, top_p=args.top_p),
    )
    environment = method_environment_from_builder(builder)
    warm_up_streaming_path(
        lm_adapter=lm,
        reference_builder=builder,
        environment=environment,
        prompt_text=args.prompt,
    )

    rows: list[dict[str, object]] = []
    for run_index in range(args.repeats):
        secret_id = (
            args.secret_id
            if args.repeats == 1
            else f"{args.secret_id}-{run_index:03d}"
        )
        method_seed = args.method_seed + run_index
        result = run_streaming_text_roundtrip(
            lm_adapter=lm,
            reference_builder=builder,
            method=ADGMethod(),
            method_config={},
            environment=environment,
            prompt_text=args.prompt,
            carrier_tokens=args.carrier_tokens,
            secret_source=Shake256SecretSource(secret_id),
            encoder_random_source=MethodRandomSource(method_seed),
            decoder_random_source=None,
        )
        row = _run_row(
            result,
            run_index=run_index,
            secret_id=secret_id,
            method_seed=method_seed,
        )
        rows.append(row)
        print(
            f"run={run_index}  BPT={float(row['bits_per_token']):.4f}  "
            f"KL(P||Q)mean={float(row['kl_ref_to_stego_bits_per_token']['mean']):.6f}  "
            f"KL(Q||P)mean={float(row['kl_stego_to_ref_bits_per_token']['mean']):.6f}  "
            f"TVDmean={float(row['tvd']['mean']):.6f}  "
            f"tokenRT={row['token_sequence_roundtrip_exact']}  "
            f"secretRT={row['secret_roundtrip_exact']}"
        )

    total_tokens = sum(int(row["carrier_tokens"]) for row in rows)
    total_payload = sum(int(row["payload_bits"]) for row in rows)
    all_exact = all(
        bool(row["token_sequence_roundtrip_exact"])
        and bool(row["secret_roundtrip_exact"])
        and bool(row["decoder_complete"])
        for row in rows
    )
    all_finite = all(bool(row["all_distortion_metrics_finite"]) for row in rows)
    explicit_q = all(
        row["q_modes"] == ["exact_enumeration"]
        and row["q_sources"] == ["independent_enumeration"]
        for row in rows
    )
    forward_means = [
        float(row["kl_ref_to_stego_bits_per_token"]["mean"]) for row in rows
    ]
    reverse_means = [
        float(row["kl_stego_to_ref_bits_per_token"]["mean"]) for row in rows
    ]
    tvd_means = [float(row["tvd"]["mean"]) for row in rows]
    nonzero_forward_distortion = any(value > 1e-12 for value in forward_means)

    status = "pass" if all_exact and all_finite and explicit_q else "fail"
    payload = {
        "status": status,
        "probe_kind": "normalized-publication-characteristic-not-table-1-numerical-reproduction",
        "reference_repository": REFERENCE_REPOSITORY,
        "reference_commit": REFERENCE_COMMIT,
        "model_id": config.model_id,
        "model_revision": config.revision,
        "prompt": args.prompt,
        "carrier_tokens_per_run": args.carrier_tokens,
        "repeats": args.repeats,
        "common_generation": {"top_k": args.top_k, "top_p": args.top_p},
        "rows": rows,
        "aggregate": {
            "total_carrier_tokens": total_tokens,
            "total_payload_bits": total_payload,
            "aggregate_bits_per_token": total_payload / total_tokens,
            "mean_run_kl_ref_to_stego_bits_per_token": float(mean(forward_means)),
            "mean_run_kl_stego_to_ref_bits_per_token": float(mean(reverse_means)),
            "mean_run_tvd": float(mean(tvd_means)),
            "all_text_and_secret_roundtrips_exact": all_exact,
            "all_distortion_metrics_finite": all_finite,
            "all_q_exact_independent_enumeration": explicit_q,
            "nonzero_forward_distortion_observed": nonzero_forward_distortion,
        },
        "paper_context": {
            "source": "Zhang et al., Findings of ACL-IJCNLP 2021, Table 1",
            "paper_setup": (
                "word-level two-layer LSTM; Movie/News/Tweet corpora; "
                "1000 stegotexts per experimental case"
            ),
            "reported_adg": PAPER_TABLE_1,
            "reported_er_range_bits_per_token": [5.147, 5.650],
            "reported_kld1_range_bits_per_token": [0.027, 0.048],
            "characteristic_claim": (
                "high embedding rate together with small but generally non-zero "
                "KLD1 = D_KL(p_LM || q)"
            ),
            "comparison_limitation": (
                "Absolute ER/KLD1 values are not expected to numerically match Table 1: "
                "this probe uses the pinned modern benchmark LM, subword tokens, the benchmark "
                "prompt and canonical P_reference rather than the paper's separately trained "
                "word-level LSTMs and corpora."
            ),
        },
    }

    print("aggregate BPT:", f"{payload['aggregate']['aggregate_bits_per_token']:.6f}")
    print(
        "aggregate mean KL P_ref->Q:",
        f"{payload['aggregate']['mean_run_kl_ref_to_stego_bits_per_token']:.6f}",
    )
    print(
        "aggregate mean KL Q->P_ref:",
        f"{payload['aggregate']['mean_run_kl_stego_to_ref_bits_per_token']:.6f}",
    )
    print("aggregate mean TVD:", f"{payload['aggregate']['mean_run_tvd']:.6f}")
    print("nonzero forward distortion observed:", nonzero_forward_distortion)
    print("status:", status.upper())

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print("json:", args.json_output)

    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
