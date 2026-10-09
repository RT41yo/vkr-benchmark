#!/usr/bin/env python3
"""Validate explicit Q_stego construction and numerical preservation for RRC.

Synthetic small-support cases use near-machine-precision equality checks.  A
real full-vocabulary LM trajectory is judged by the benchmark distortion
metrics themselves (dual KL and TVD), while max |Q-P| remains a diagnostic.
This distinction is intentional: RRC builds sorted FP64 cumulative boundaries,
whereas canonical P_reference is stored in FP32 and normalized independently by
the common metric layer.  With O(1e5) support, mathematically equivalent sums
are not bit-identical even though the resulting distributional distortion is
at the FP64 numerical-noise scale.
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
from vkr_benchmark.methods import RRCMethod
from vkr_benchmark.randomness import MethodRandomSource, Shake256SecretSource
from vkr_benchmark.runner import method_environment_from_builder
from vkr_benchmark.validation import evaluate_explicit_rrc_q

DEFAULT_CONFIG = REPO_ROOT / "configs" / "reproducibility" / "stage4_rrc_conformance.json"
_EPS64 = float(np.finfo(np.float64).eps)


def _metric(value: float | None) -> float:
    if value is None:
        raise RuntimeError("RRC explicit-Q validation produced unavailable metric")
    return float(value)


def _row(reference: ReferenceDistribution) -> dict[str, Any]:
    result = evaluate_explicit_rrc_q(reference)
    d = result.distortion
    return {
        "support_size": reference.support_size,
        "fp64_support_roundoff_scale": reference.support_size * _EPS64,
        "reference_storage_sum": result.reference_storage_sum,
        "q_sum": result.q_sum,
        "max_abs_q_minus_normalized_p": result.max_abs_difference,
        "l1_q_minus_normalized_p": result.l1_difference,
        "kl_ref_to_stego_bits": _metric(d.kl_ref_to_stego_bits),
        "kl_stego_to_ref_bits": _metric(d.kl_stego_to_ref_bits),
        "tvd": _metric(d.tvd),
        "q_mode": d.q_mode.value,
        "q_source": d.q_source.value,
    }


def _synthetic_failure_reasons(row: dict[str, Any], tol: dict[str, float]) -> list[str]:
    reasons: list[str] = []
    if float(row["max_abs_q_minus_normalized_p"]) > tol["max_abs_q_minus_p"]:
        reasons.append("max_abs_q_minus_p")
    if float(row["kl_ref_to_stego_bits"]) > tol["kl_bits"]:
        reasons.append("kl_ref_to_stego_bits")
    if float(row["kl_stego_to_ref_bits"]) > tol["kl_bits"]:
        reasons.append("kl_stego_to_ref_bits")
    if float(row["tvd"]) > tol["tvd"]:
        reasons.append("tvd")
    return reasons


def _real_failure_reasons(row: dict[str, Any], tol: dict[str, float]) -> list[str]:
    """Gate real LM rows on benchmark distortion, not coordinate equality.

    max |Q-P| is deliberately diagnostic here.  On a large support the actual
    RRC partition is obtained from a sequential FP64 cumulative sum in sorted
    probability order, while the common metric layer normalizes canonical FP32
    P_reference independently.  Their coordinate-wise difference can therefore
    exceed a small-support ulp threshold without implying material distortion.
    """

    reasons: list[str] = []
    for name in ("kl_ref_to_stego_bits", "kl_stego_to_ref_bits", "tvd"):
        value = float(row[name])
        if not np.isfinite(value):
            reasons.append(f"{name}_non_finite")
    if reasons:
        return reasons
    if float(row["kl_ref_to_stego_bits"]) > tol["kl_bits"]:
        reasons.append("kl_ref_to_stego_bits")
    if float(row["kl_stego_to_ref_bits"]) > tol["kl_bits"]:
        reasons.append("kl_stego_to_ref_bits")
    if float(row["tvd"]) > tol["tvd"]:
        reasons.append("tvd")
    return reasons


def _synthetic_references(random_cases: int):
    yield "balanced", ReferenceDistribution(np.asarray([0.5, 0.5], dtype=np.float32))
    yield "paper_toy", ReferenceDistribution(
        np.asarray([0.4, 0.3, 0.2, 0.1], dtype=np.float32)
    )
    yield "sparse", ReferenceDistribution(
        np.asarray([0.0, 0.55, 0.0, 0.25, 0.15, 0.05], dtype=np.float32)
    )
    rng = random.Random(20261009)
    for index in range(random_cases):
        size = rng.randint(2, 64)
        weights = np.asarray([rng.randint(1, 10000) for _ in range(size)], dtype=np.float64)
        if size >= 5 and index % 4 == 0:
            weights[rng.randrange(size)] = 0.0
        probs = np.asarray(weights / np.sum(weights, dtype=np.float64), dtype=np.float32)
        # Preserve the benchmark's ordinary FP32 storage behavior.
        probs = np.asarray(probs / np.sum(probs, dtype=np.float32), dtype=np.float32)
        yield f"random_{index:03d}", ReferenceDistribution(probs)


def _large_vocab_stress_reference(vocab_size: int = 128_256) -> ReferenceDistribution:
    """Deterministic full-vocabulary case exposing cumulative FP64 residue."""

    if vocab_size < 2:
        raise ValueError("vocab_size must be at least 2")
    probs = np.full(vocab_size, 0.1 / (vocab_size - 1), dtype=np.float64)
    probs[0] = 0.9
    probs = np.asarray(probs, dtype=np.float32)
    probs = np.asarray(probs / np.sum(probs, dtype=np.float32), dtype=np.float32)
    return ReferenceDistribution(probs)


def evaluate_synthetic(random_cases: int, tol: dict[str, float]) -> dict[str, Any]:
    rows = []
    for case_id, reference in _synthetic_references(random_cases):
        row = _row(reference)
        row["case_id"] = case_id
        row["failure_reasons"] = _synthetic_failure_reasons(row, tol)
        row["pass"] = not row["failure_reasons"]
        rows.append(row)
    return {
        "case_count": len(rows),
        "failure_count": sum(not row["pass"] for row in rows),
        "max_abs_q_minus_normalized_p": max(row["max_abs_q_minus_normalized_p"] for row in rows),
        "max_kl_ref_to_stego_bits": max(row["kl_ref_to_stego_bits"] for row in rows),
        "max_kl_stego_to_ref_bits": max(row["kl_stego_to_ref_bits"] for row in rows),
        "max_tvd": max(row["tvd"] for row in rows),
        "rows": rows,
    }


def evaluate_large_vocab_stress(tol: dict[str, float]) -> dict[str, Any]:
    row = _row(_large_vocab_stress_reference())
    row["failure_reasons"] = _real_failure_reasons(row, tol)
    row["pass"] = not row["failure_reasons"]
    return row


def evaluate_real_model(
    model_config: Path,
    *,
    steps: int,
    payload_bits: int,
    key: int,
    prompt: str,
    secret_id: str,
    tol: dict[str, float],
    max_abs_warning: float,
) -> dict[str, Any]:
    config = LocalModelConfig.from_json(model_config)
    print("Loading local model...")
    lm = HFCausalLMAdapter.from_local_config(config)
    builder = ReferenceDistributionBuilder(
        token_space=lm.token_space,
        policy=GenerationPolicy(),
    )
    environment = method_environment_from_builder(builder)
    state = lm.prefill(lm.encode_prompt(prompt))
    encoder = RRCMethod().create_encoder(
        config={"precision_margin": 40, "verify_termination": True},
        environment=environment,
        secret_source=Shake256SecretSource(secret_id),
        random_source=MethodRandomSource(key),
        key=key,
        target_payload_bits=payload_bits,
    )

    rows: list[dict[str, Any]] = []
    generated: list[int] = []
    for step_index in range(steps):
        raw_logits, state = lm.next_logits(state)
        reference = builder.build(raw_logits)
        row = _row(reference)
        row["step_index"] = step_index
        row["failure_reasons"] = _real_failure_reasons(row, tol)
        row["max_abs_warning"] = (
            float(row["max_abs_q_minus_normalized_p"]) > max_abs_warning
        )
        row["pass"] = not row["failure_reasons"]
        rows.append(row)

        decision = encoder.step(StepContext(step_index, reference))
        token_id = int(decision.token_id)
        generated.append(token_id)
        if encoder.done:
            break
        _, state = lm.next_logits(state, token_id)

    criterion_failures: dict[str, int] = {}
    for row in rows:
        for reason in row["failure_reasons"]:
            criterion_failures[reason] = criterion_failures.get(reason, 0) + 1

    return {
        "model_id": config.model_id,
        "model_revision": config.revision,
        "payload_bits": payload_bits,
        "steps_checked": len(rows),
        "encoder_terminated": encoder.done,
        "generated_token_ids": generated,
        "failure_count": sum(not row["pass"] for row in rows),
        "criterion_failures": criterion_failures,
        "max_abs_warning_count": sum(bool(row["max_abs_warning"]) for row in rows),
        "support_size_min": min(row["support_size"] for row in rows),
        "support_size_max": max(row["support_size"] for row in rows),
        "max_fp64_support_roundoff_scale": max(row["fp64_support_roundoff_scale"] for row in rows),
        "max_abs_q_minus_normalized_p": max(row["max_abs_q_minus_normalized_p"] for row in rows),
        "max_kl_ref_to_stego_bits": max(row["kl_ref_to_stego_bits"] for row in rows),
        "max_kl_stego_to_ref_bits": max(row["kl_stego_to_ref_bits"] for row in rows),
        "max_tvd": max(row["tvd"] for row in rows),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_config", type=Path, nargs="?")
    parser.add_argument("--validation-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--random-cases", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--payload-bits", type=int, default=128)
    parser.add_argument("--key", type=int, default=42)
    parser.add_argument("--secret-id", default="stage4-rrc-explicit-q")
    parser.add_argument(
        "--prompt", default="The history of artificial intelligence began"
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    cfg = json.loads(args.validation_config.read_text(encoding="utf-8"))
    q_cfg = cfg["explicit_q"]
    random_cases = args.random_cases if args.random_cases is not None else int(q_cfg["synthetic_random_cases"])
    steps = args.steps if args.steps is not None else int(q_cfg["real_steps"])
    synthetic_tol = {name: float(value) for name, value in q_cfg["tolerances"].items()}
    real_tol = {name: float(value) for name, value in q_cfg["real_model_tolerances"].items()}
    max_abs_warning = float(q_cfg["real_model_diagnostics"]["max_abs_q_minus_p_warning"])

    synthetic = evaluate_synthetic(random_cases, synthetic_tol)
    large_vocab = evaluate_large_vocab_stress(real_tol)
    print("Stage 4 RRC independent explicit-Q validation")
    print("reference commit:", cfg["reference_commit"])
    print(
        "synthetic:",
        f"cases={synthetic['case_count']}",
        f"failures={synthetic['failure_count']}",
        f"max|Q-P|={synthetic['max_abs_q_minus_normalized_p']:.3g}",
        f"KLmax={synthetic['max_kl_ref_to_stego_bits']:.3g}/"
        f"{synthetic['max_kl_stego_to_ref_bits']:.3g}",
        f"TVDmax={synthetic['max_tvd']:.3g}",
    )
    print(
        "large-vocab stress:",
        f"support={large_vocab['support_size']}",
        f"max|Q-P|={large_vocab['max_abs_q_minus_normalized_p']:.3g}",
        f"KL={large_vocab['kl_ref_to_stego_bits']:.3g}/"
        f"{large_vocab['kl_stego_to_ref_bits']:.3g}",
        f"TVD={large_vocab['tvd']:.3g}",
        f"status={'PASS' if large_vocab['pass'] else 'FAIL'}",
    )

    real = None
    if args.model_config is not None:
        real = evaluate_real_model(
            args.model_config,
            steps=steps,
            payload_bits=args.payload_bits,
            key=args.key,
            prompt=args.prompt,
            secret_id=args.secret_id,
            tol=real_tol,
            max_abs_warning=max_abs_warning,
        )
        print(
            "real:",
            f"steps={real['steps_checked']}",
            f"failures={real['failure_count']}",
            f"support={real['support_size_min']}..{real['support_size_max']}",
            f"max|Q-P|={real['max_abs_q_minus_normalized_p']:.3g}",
            f"KLmax={real['max_kl_ref_to_stego_bits']:.3g}/"
            f"{real['max_kl_stego_to_ref_bits']:.3g}",
            f"TVDmax={real['max_tvd']:.3g}",
            f"max|Q-P| warnings={real['max_abs_warning_count']}",
            f"criterion_failures={real['criterion_failures']}",
        )

    failures = int(synthetic["failure_count"]) + (0 if large_vocab["pass"] else 1)
    failures += 0 if real is None else int(real["failure_count"])
    result = {
        "status": "pass" if failures == 0 else "fail",
        "validation_kind": "independent-interval-measure-explicit-q",
        "reference_commit": cfg["reference_commit"],
        "synthetic_tolerances": synthetic_tol,
        "real_model_tolerances": real_tol,
        "real_model_max_abs_warning": max_abs_warning,
        "interpretation": {
            "q_is_explicit_array": True,
            "uses_equality_certificate": False,
            "uses_common_kl_tvd_formulas": True,
            "construction": "normalized interval widths under fresh uniform rotation",
            "small_support_check": "near-machine-precision coordinate equality",
            "large_support_check": "finite dual KL and TVD within a conservative numerical envelope",
            "max_abs_real_role": "diagnostic only; not an acceptance gate",
            "why": "sorted sequential FP64 interval accumulation is not bit-identical to independent normalization of canonical FP32 P_reference over a large vocabulary",
        },
        "synthetic": synthetic,
        "large_vocab_stress": large_vocab,
        "real_model": real,
    }
    print("status:", result["status"].upper())
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("json:", args.json_output)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
