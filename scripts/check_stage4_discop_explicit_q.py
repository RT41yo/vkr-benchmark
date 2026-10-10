#!/usr/bin/env python3
"""Independently construct Discop Q_stego and evaluate KL/TVD.

The checker exercises the same independent explicit-Q calculator used by
normalized Discop benchmark runs.  That calculator is separate from the
encoder/decoder path: it rebuilds the pinned Huffman decomposition, integrates
both rotated distribution copies at every internal node, obtains a full
Q_stego vector, and evaluates it with the benchmark's KL/TVD formulas.

Without a model configuration the script runs only deterministic synthetic
cases.  When a local model configuration is supplied it additionally validates
several real P_reference distributions along an actual Discop trajectory for
each requested top-p value.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vkr_benchmark.config import LocalModelConfig
from vkr_benchmark.distributions import (
    GenerationPolicy,
    ReferenceDistribution,
    ReferenceDistributionBuilder,
    StepContext,
)
from vkr_benchmark.lm import HFCausalLMAdapter
from vkr_benchmark.methods import DiscopMethod
from vkr_benchmark.randomness import MethodRandomSource, Shake256SecretSource
from vkr_benchmark.runner import method_environment_from_builder
from vkr_benchmark.validation import evaluate_explicit_discop_q

DEFAULT_VALIDATION_CONFIG = (
    REPO_ROOT / "configs" / "reproducibility" / "stage4_discop_explicit_q.json"
)


def _load_validation_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("method") != "discop":
        raise ValueError("explicit-Q validation config must target Discop")
    return payload


def _metric_value(value: float | None) -> float:
    if value is None:
        raise RuntimeError("explicit-Q validation unexpectedly produced unavailable metric")
    return float(value)


def _diagnostic_row(reference: ReferenceDistribution) -> dict[str, Any]:
    validation = evaluate_explicit_discop_q(reference)
    distortion = validation.distortion
    return {
        "support_size": reference.support_size,
        "reference_storage_sum": validation.reference_storage_sum,
        "q_sum": validation.q_sum,
        "max_abs_q_minus_normalized_p": validation.max_abs_difference,
        "l1_q_minus_normalized_p": validation.l1_difference,
        "kl_ref_to_stego_bits": _metric_value(distortion.kl_ref_to_stego_bits),
        "kl_stego_to_ref_bits": _metric_value(distortion.kl_stego_to_ref_bits),
        "tvd": _metric_value(distortion.tvd),
        "q_mode": distortion.q_mode.value,
        "q_source": distortion.q_source.value,
    }


def _passes(row: dict[str, Any], tolerances: dict[str, float]) -> bool:
    return (
        float(row["max_abs_q_minus_normalized_p"])
        <= float(tolerances["max_abs_q_minus_p"])
        and float(row["kl_ref_to_stego_bits"]) <= float(tolerances["kl_bits"])
        and float(row["kl_stego_to_ref_bits"]) <= float(tolerances["kl_bits"])
        and float(row["tvd"]) <= float(tolerances["tvd"])
    )


def _synthetic_references(random_cases: int) -> list[tuple[str, ReferenceDistribution]]:
    cases: list[tuple[str, ReferenceDistribution]] = [
        ("two_balanced", ReferenceDistribution(np.array([0.5, 0.5], dtype=np.float32))),
        (
            "paper_toy",
            ReferenceDistribution(np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32)),
        ),
        (
            "dominant",
            ReferenceDistribution(np.array([0.9, 0.1], dtype=np.float32)),
        ),
        (
            "sparse_support",
            ReferenceDistribution(
                np.array([0.0, 0.55, 0.0, 0.25, 0.15, 0.05, 0.0], dtype=np.float32)
            ),
        ),
        (
            "leaf_ties",
            ReferenceDistribution(np.array([0.4, 0.2, 0.2, 0.2], dtype=np.float32)),
        ),
    ]

    rng = random.Random(20261002)
    for index in range(random_cases):
        size = rng.randint(2, 32)
        weights = np.array([rng.randint(1, 1000) for _ in range(size)], dtype=np.float64)
        probs = np.asarray(weights / np.sum(weights, dtype=np.float64), dtype=np.float32)
        # Reproduce canonical FP32 normalization rather than constructing an
        # artificially exact FP64 distribution.
        probs = np.asarray(probs / np.sum(probs, dtype=np.float32), dtype=np.float32)
        if index % 4 == 0 and size >= 4:
            # Include support truncation cases while keeping at least two leaves.
            zero_count = min(size - 2, 1 + index % 3)
            probs[:zero_count] = np.float32(0.0)
            probs = np.asarray(probs / np.sum(probs, dtype=np.float32), dtype=np.float32)
        cases.append((f"random_{index:03d}", ReferenceDistribution(probs)))
    return cases


def evaluate_synthetic(
    *, random_cases: int, tolerances: dict[str, float]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case_id, reference in _synthetic_references(random_cases):
        row = _diagnostic_row(reference)
        row["case_id"] = case_id
        row["pass"] = _passes(row, tolerances)
        rows.append(row)

    return {
        "case_count": len(rows),
        "failure_count": sum(not bool(row["pass"]) for row in rows),
        "max_abs_q_minus_normalized_p": max(
            float(row["max_abs_q_minus_normalized_p"]) for row in rows
        ),
        "max_kl_ref_to_stego_bits": max(
            float(row["kl_ref_to_stego_bits"]) for row in rows
        ),
        "max_kl_stego_to_ref_bits": max(
            float(row["kl_stego_to_ref_bits"]) for row in rows
        ),
        "max_tvd": max(float(row["tvd"]) for row in rows),
        "rows": rows,
    }


def evaluate_real_model(
    *,
    model_config: Path,
    prompt: str,
    secret_id: str,
    key: int,
    top_p_values: list[float],
    steps_per_point: int,
    tolerances: dict[str, float],
) -> dict[str, Any]:
    config = LocalModelConfig.from_json(model_config)
    print("Loading local model...")
    lm = HFCausalLMAdapter.from_local_config(config)

    points: list[dict[str, Any]] = []
    for top_p in top_p_values:
        builder = ReferenceDistributionBuilder(
            token_space=lm.token_space,
            policy=GenerationPolicy(top_p=float(top_p)),
        )
        environment = method_environment_from_builder(builder)
        prompt_token_ids = lm.encode_prompt(prompt)
        state = lm.prefill(prompt_token_ids)
        encoder = DiscopMethod().create_encoder(
            config={},
            environment=environment,
            secret_source=Shake256SecretSource(f"{secret_id}:p={top_p:.8f}"),
            random_source=MethodRandomSource(key),
            key=key,
        )

        rows: list[dict[str, Any]] = []
        generated: list[int] = []
        for step_index in range(steps_per_point):
            raw_logits, state = lm.next_logits(state)
            reference = builder.build(raw_logits)
            row = _diagnostic_row(reference)
            row["step_index"] = step_index
            row["pass"] = _passes(row, tolerances)
            rows.append(row)

            decision = encoder.step(StepContext(step_index=step_index, reference=reference))
            token_id = int(decision.token_id)
            generated.append(token_id)
            if step_index + 1 < steps_per_point:
                _, state = lm.next_logits(state, token_id)

        encoder.finalize()
        point = {
            "top_p": float(top_p),
            "steps": rows,
            "generated_token_ids": generated,
            "failure_count": sum(not bool(row["pass"]) for row in rows),
            "max_abs_q_minus_normalized_p": max(
                float(row["max_abs_q_minus_normalized_p"]) for row in rows
            ),
            "max_kl_ref_to_stego_bits": max(
                float(row["kl_ref_to_stego_bits"]) for row in rows
            ),
            "max_kl_stego_to_ref_bits": max(
                float(row["kl_stego_to_ref_bits"]) for row in rows
            ),
            "max_tvd": max(float(row["tvd"]) for row in rows),
        }
        points.append(point)
        print(
            f"p={top_p:.2f}  steps={steps_per_point}  "
            f"max|Q-P|={point['max_abs_q_minus_normalized_p']:.3g}  "
            f"KLmax={point['max_kl_ref_to_stego_bits']:.3g}/"
            f"{point['max_kl_stego_to_ref_bits']:.3g}  "
            f"TVDmax={point['max_tvd']:.3g}  "
            f"failures={point['failure_count']}"
        )

    return {
        "model_id": config.model_id,
        "model_revision": config.revision,
        "prompt": prompt,
        "secret_id": secret_id,
        "key": key,
        "steps_per_point": steps_per_point,
        "points": points,
        "failure_count": sum(int(point["failure_count"]) for point in points),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_config",
        type=Path,
        nargs="?",
        help="optional local model config; omit to run synthetic validation only",
    )
    parser.add_argument(
        "--validation-config",
        type=Path,
        default=DEFAULT_VALIDATION_CONFIG,
    )
    parser.add_argument("--random-cases", type=int)
    parser.add_argument("--steps-per-point", type=int)
    parser.add_argument("--top-p", type=float, nargs="+")
    parser.add_argument(
        "--prompt",
        default="The history of artificial intelligence began",
    )
    parser.add_argument("--secret-id", default="stage4-discop-explicit-q")
    parser.add_argument("--key", type=int, default=12345)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    validation_config = _load_validation_config(args.validation_config)
    tolerances = {
        key: float(value)
        for key, value in validation_config["tolerances"].items()
    }
    random_cases = (
        int(args.random_cases)
        if args.random_cases is not None
        else int(validation_config["synthetic_random_cases"])
    )
    steps_per_point = (
        int(args.steps_per_point)
        if args.steps_per_point is not None
        else int(validation_config["real_steps_per_point"])
    )
    top_p_values = (
        [float(value) for value in args.top_p]
        if args.top_p is not None
        else [float(value) for value in validation_config["top_p"]]
    )
    if random_cases < 0:
        parser.error("--random-cases must be non-negative")
    if steps_per_point <= 0:
        parser.error("--steps-per-point must be positive")

    print("Stage 4 Discop independent explicit-Q validation")
    print("reference commit:", validation_config["reference_commit"])
    print("synthetic random cases:", random_cases)
    synthetic = evaluate_synthetic(random_cases=random_cases, tolerances=tolerances)
    print(
        "synthetic:",
        f"cases={synthetic['case_count']}",
        f"failures={synthetic['failure_count']}",
        f"max|Q-P|={synthetic['max_abs_q_minus_normalized_p']:.3g}",
        f"KLmax={synthetic['max_kl_ref_to_stego_bits']:.3g}/"
        f"{synthetic['max_kl_stego_to_ref_bits']:.3g}",
        f"TVDmax={synthetic['max_tvd']:.3g}",
    )

    real_model = None
    if args.model_config is not None:
        print("model config:", args.model_config)
        print("top-p grid:", top_p_values)
        print("steps per point:", steps_per_point)
        real_model = evaluate_real_model(
            model_config=args.model_config,
            prompt=args.prompt,
            secret_id=args.secret_id,
            key=args.key,
            top_p_values=top_p_values,
            steps_per_point=steps_per_point,
            tolerances=tolerances,
        )

    failure_count = int(synthetic["failure_count"])
    if real_model is not None:
        failure_count += int(real_model["failure_count"])
    payload = {
        "status": "pass" if failure_count == 0 else "fail",
        "validation_kind": "independent-explicit-q",
        "reference_commit": validation_config["reference_commit"],
        "tolerances": tolerances,
        "interpretation": {
            "q_is_constructed_explicitly": True,
            "uses_discop_equality_certificate": False,
            "uses_benchmark_kl_tvd_formulas": True,
            "secret_model": validation_config["notes"]["secret_model"],
            "reference_normalization": validation_config["notes"]["reference_normalization"],
            "main_adapter_changed": False,
        },
        "synthetic": synthetic,
        "real_model": real_model,
    }

    print("status:", payload["status"].upper())
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print("json:", args.json_output)
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
