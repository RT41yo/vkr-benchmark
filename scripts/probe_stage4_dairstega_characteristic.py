#!/usr/bin/env python3
"""Probe the DAIRstega publication characteristic on the normalized benchmark LM.

This is not a numerical reproduction of Wang et al. Table 3.  The publication
reports bit-per-word and perplexity under its LLaMA-2/instruction-conditioned
setup, whereas Stage 4 uses the pinned benchmark LM, subword tokens, a fixed
prompt, canonical P_reference, and benchmark-native distortion metrics.

The probe asks whether the same qualitative family of operating points appears:
a capacity-oriented setting should carry more payload, while paying a quality
and/or distribution-distortion cost relative to a quality-oriented setting.
"""

from __future__ import annotations

import argparse
import json
import math
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
from vkr_benchmark.methods import DAIRMethod
from vkr_benchmark.metrics import compute_raw_lm_quality_metrics
from vkr_benchmark.randomness import Shake256SecretSource
from vkr_benchmark.runner import method_environment_from_builder, run_streaming_text_roundtrip
from vkr_benchmark.runner.streaming import warm_up_streaming_path

REFERENCE_REPOSITORY = "https://github.com/WangYH-BUPT/DAIRstega"
REFERENCE_COMMIT = "8d85edf98d48c3efa827a125b6d4e90f88141ea2"

# Publication context (Table 3 values are Bpw, not benchmark BPT).
PAPER_TABLE_3 = {
    "alpha8_beta1": {"alpha": 8, "beta": 1.0, "bpw": 1.10, "ppl": 7.686},
    "alpha32_beta1": {"alpha": 32, "beta": 1.0, "bpw": 1.11, "ppl": 8.184},
    "alpha8_beta05": {"alpha": 8, "beta": 0.5, "bpw": 1.89, "ppl": 13.775},
    "alpha32_beta05": {"alpha": 32, "beta": 0.5, "bpw": 2.56, "ppl": 20.876},
}

POINTS = (
    ("quality_a8_b1", 8, 1.0),
    ("beta_effect_a8_b05", 8, 0.5),
    ("alpha_effect_a32_b1", 32, 1.0),
    ("capacity_a32_b05", 32, 0.5),
)


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return float(mean(values))


def _row(result, *, label: str, alpha: int, beta: float, run_index: int, secret_id: str) -> dict[str, object]:
    carriers = result.encode.carrier_tokens
    payload = result.encode.payload_bits
    distortion = list(result.encode.step_distribution_distortion)
    forward = [float(x.kl_ref_to_stego_bits) for x in distortion]
    reverse = [float(x.kl_stego_to_ref_bits) for x in distortion]
    tvd = [float(x.tvd) for x in distortion]
    quality = compute_raw_lm_quality_metrics(result.encode.step_raw_lm_nll_nats)

    q_modes = sorted({x.q_mode.value for x in distortion})
    q_sources = sorted({x.q_source.value for x in distortion})
    finite_forward = [x for x in forward if math.isfinite(x)]

    return {
        "label": label,
        "run_index": run_index,
        "secret_id": secret_id,
        "alpha": alpha,
        "beta": beta,
        "carrier_tokens": carriers,
        "payload_bits": payload,
        "secret_bits_read": result.encode.secret_bits_read,
        "bits_per_token": payload / carriers,
        "raw_lm_nll_nats_per_token": quality.nll_raw_lm_nats_per_token,
        "raw_lm_ppl": quality.ppl_raw_lm,
        "kl_ref_to_stego_infinite_steps": sum(not math.isfinite(x) for x in forward),
        "kl_ref_to_stego_finite_mean_bits_per_token": (
            _mean(finite_forward) if finite_forward else None
        ),
        "kl_stego_to_ref_mean_bits_per_token": _mean(reverse),
        "tvd_mean": _mean(tvd),
        "q_modes": q_modes,
        "q_sources": q_sources,
        "token_sequence_roundtrip_exact": result.transport.token_sequence_roundtrip_exact,
        "secret_roundtrip_exact": result.roundtrip_exact,
        "decoder_complete": result.decode.finalization.complete,
        "stegotext": result.transport.text,
    }


def _aggregate(rows: list[dict[str, object]]) -> dict[str, float]:
    total_tokens = sum(int(row["carrier_tokens"]) for row in rows)
    total_payload = sum(int(row["payload_bits"]) for row in rows)
    return {
        "bits_per_token": total_payload / total_tokens,
        "raw_lm_nll_nats_per_token": _mean(
            [float(row["raw_lm_nll_nats_per_token"]) for row in rows]
        ),
        "raw_lm_ppl": _mean([float(row["raw_lm_ppl"]) for row in rows]),
        "kl_stego_to_ref_mean_bits_per_token": _mean(
            [float(row["kl_stego_to_ref_mean_bits_per_token"]) for row in rows]
        ),
        "tvd_mean": _mean([float(row["tvd_mean"]) for row in rows]),
    }


def _strictly_more(left: float, right: float, *, tolerance: float = 1e-12) -> bool:
    return left > right + tolerance


def _tradeoff_summary(points: dict[str, dict[str, float]]) -> dict[str, object]:
    quality = points["quality_a8_b1"]
    capacity = points["capacity_a32_b05"]

    capacity_gain = _strictly_more(capacity["bits_per_token"], quality["bits_per_token"])
    quality_cost = _strictly_more(capacity["raw_lm_ppl"], quality["raw_lm_ppl"])
    reverse_kl_cost = _strictly_more(
        capacity["kl_stego_to_ref_mean_bits_per_token"],
        quality["kl_stego_to_ref_mean_bits_per_token"],
    )
    tvd_cost = _strictly_more(capacity["tvd_mean"], quality["tvd_mean"])

    beta8 = _strictly_more(
        points["beta_effect_a8_b05"]["bits_per_token"],
        quality["bits_per_token"],
    )
    beta32 = _strictly_more(
        capacity["bits_per_token"],
        points["alpha_effect_a32_b1"]["bits_per_token"],
    )

    concealment_or_quality_cost = quality_cost or reverse_kl_cost or tvd_cost
    characteristic_observed = capacity_gain and concealment_or_quality_cost
    return {
        "capacity_oriented_point_has_higher_bpt": capacity_gain,
        "capacity_oriented_point_has_higher_raw_lm_ppl": quality_cost,
        "capacity_oriented_point_has_higher_reverse_kl": reverse_kl_cost,
        "capacity_oriented_point_has_higher_tvd": tvd_cost,
        "beta_0_5_increases_bpt_at_alpha_8": beta8,
        "beta_0_5_increases_bpt_at_alpha_32": beta32,
        "concealment_or_quality_cost_observed": concealment_or_quality_cost,
        "publication_characteristic_observed": characteristic_observed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_config", type=Path)
    parser.add_argument("--carrier-tokens", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--secret-id", default="stage4-dairstega-characteristic")
    parser.add_argument("--method-top-k", type=int, default=100)
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
    if args.method_top_k < 2:
        parser.error("--method-top-k must be >= 2")

    config = LocalModelConfig.from_json(args.model_config)
    print("Stage 4 DAIRstega publication-characteristic probe")
    print("reference:", f"{REFERENCE_REPOSITORY}@{REFERENCE_COMMIT}")
    print("model:", config.model_id)
    print("revision:", config.revision)
    print("carrier tokens per point/run:", args.carrier_tokens)
    print("repeats per point:", args.repeats)
    print("method-local top-k:", args.method_top_k)
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
    by_point: dict[str, list[dict[str, object]]] = {label: [] for label, _, _ in POINTS}
    for label, alpha, beta in POINTS:
        for run_index in range(args.repeats):
            secret_id = (
                f"{args.secret_id}-{label}"
                if args.repeats == 1
                else f"{args.secret_id}-{label}-{run_index:03d}"
            )
            result = run_streaming_text_roundtrip(
                lm_adapter=lm,
                reference_builder=builder,
                method=DAIRMethod(),
                method_config={"alpha": alpha, "beta": beta, "top_k": args.method_top_k},
                environment=environment,
                prompt_text=args.prompt,
                carrier_tokens=args.carrier_tokens,
                secret_source=Shake256SecretSource(secret_id),
                encoder_random_source=None,
                decoder_random_source=None,
            )
            row = _row(
                result,
                label=label,
                alpha=alpha,
                beta=beta,
                run_index=run_index,
                secret_id=secret_id,
            )
            rows.append(row)
            by_point[label].append(row)
            print(
                f"point={label} run={run_index} alpha={alpha} beta={beta:.1f}  "
                f"BPT={float(row['bits_per_token']):.4f}  "
                f"PPL={float(row['raw_lm_ppl']):.4f}  "
                f"KL(Q||P)mean={float(row['kl_stego_to_ref_mean_bits_per_token']):.6f}  "
                f"TVDmean={float(row['tvd_mean']):.6f}  "
                f"infKL(P||Q)={int(row['kl_ref_to_stego_infinite_steps'])}/{int(row['carrier_tokens'])}  "
                f"tokenRT={row['token_sequence_roundtrip_exact']}  "
                f"secretRT={row['secret_roundtrip_exact']}"
            )

    point_aggregates = {label: _aggregate(point_rows) for label, point_rows in by_point.items()}
    tradeoff = _tradeoff_summary(point_aggregates)

    exact_roundtrip = all(
        bool(row["token_sequence_roundtrip_exact"])
        and bool(row["secret_roundtrip_exact"])
        and bool(row["decoder_complete"])
        for row in rows
    )
    exact_independent_q = all(
        row["q_modes"] == ["analytic_exact"]
        and row["q_sources"] == ["independent_analytic"]
        for row in rows
    )
    reverse_and_tvd_finite = all(
        math.isfinite(float(row["kl_stego_to_ref_mean_bits_per_token"]))
        and math.isfinite(float(row["tvd_mean"]))
        for row in rows
    )

    status = (
        "pass"
        if exact_roundtrip
        and exact_independent_q
        and reverse_and_tvd_finite
        and bool(tradeoff["publication_characteristic_observed"])
        else "fail"
    )

    payload = {
        "status": status,
        "probe_kind": "normalized-publication-characteristic-not-table-3-numerical-reproduction",
        "reference_repository": REFERENCE_REPOSITORY,
        "reference_commit": REFERENCE_COMMIT,
        "model_id": config.model_id,
        "model_revision": config.revision,
        "prompt": args.prompt,
        "carrier_tokens_per_point_run": args.carrier_tokens,
        "repeats_per_point": args.repeats,
        "method_local_top_k": args.method_top_k,
        "common_generation": {"top_k": args.top_k, "top_p": args.top_p},
        "rows": rows,
        "point_aggregates": point_aggregates,
        "validation": {
            "all_text_and_secret_roundtrips_exact": exact_roundtrip,
            "all_q_analytic_exact_independent": exact_independent_q,
            "all_reverse_kl_and_tvd_finite": reverse_and_tvd_finite,
            **tradeoff,
        },
        "paper_context": {
            "source": "Wang et al., Applied Soft Computing 176 (2025), Table 3",
            "reported_points": PAPER_TABLE_3,
            "reported_unit": "Bpw (bit per word), not benchmark BPT",
            "characteristic_claim": (
                "lower beta / capacity-oriented settings increase payload while "
                "paying a perceptual-quality and/or concealment cost"
            ),
            "comparison_limitation": (
                "Absolute Bpw/PPL values are not expected to match numerically: the "
                "publication and Stage-4 benchmark use different model/tokenization/"
                "prompting conditions, and Bpw is not the same unit as benchmark BPT."
            ),
        },
    }

    print()
    for label, aggregate in point_aggregates.items():
        print(
            f"aggregate {label}: "
            f"BPT={aggregate['bits_per_token']:.6f}  "
            f"PPL={aggregate['raw_lm_ppl']:.6f}  "
            f"KL(Q||P)={aggregate['kl_stego_to_ref_mean_bits_per_token']:.6f}  "
            f"TVD={aggregate['tvd_mean']:.6f}"
        )
    print("capacity-oriented point has higher BPT:", tradeoff["capacity_oriented_point_has_higher_bpt"])
    print("capacity-oriented point has higher PPL:", tradeoff["capacity_oriented_point_has_higher_raw_lm_ppl"])
    print("capacity-oriented point has higher reverse KL:", tradeoff["capacity_oriented_point_has_higher_reverse_kl"])
    print("capacity-oriented point has higher TVD:", tradeoff["capacity_oriented_point_has_higher_tvd"])
    print("beta=0.5 increases BPT at alpha=8:", tradeoff["beta_0_5_increases_bpt_at_alpha_8"])
    print("beta=0.5 increases BPT at alpha=32:", tradeoff["beta_0_5_increases_bpt_at_alpha_32"])
    print("publication characteristic observed:", tradeoff["publication_characteristic_observed"])
    print("status:", status.upper())

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print("json:", args.json_output)

    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
