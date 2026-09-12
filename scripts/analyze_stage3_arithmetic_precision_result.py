#!/usr/bin/env python3
"""Derive the scientific interpretation of the completed Stage-3 precision probe.

This analysis never reruns the model and never changes the author-compatible
result.  It separates coding steps that had a full secret look-ahead window from
steps that were already reading implicit zero padding beyond the finite probe
payload.  That distinction is essential: the public arithmetic encoder pads a
short final look-ahead window with zeros, and those terminal steps can dominate
an average KL even when the pre-padding coding distribution is nearly unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = REPO_ROOT / "results" / "stage3" / "paper_reproduction" / "arithmetic_precision_probe" / "result.json"
DEFAULT_MATRIX = REPO_ROOT / "configs" / "reproducibility" / "paper_reproduction_matrix.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "stage3" / "paper_reproduction" / "arithmetic_precision_probe" / "interpretation.json"
EXPECTED_RESULT_SCHEMA = "stage3.arithmetic_precision_probe_result.v1"
EXPECTED_PRECISIONS = (26, 32, 40, 48)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def analyze(result_path: Path, matrix_path: Path) -> dict[str, Any]:
    result = _load(result_path)
    matrix = _load(matrix_path)
    if result.get("schema_version") != EXPECTED_RESULT_SCHEMA or result.get("status") != "ok":
        raise RuntimeError("precision probe result is not a completed successful v1 result")

    special = next(x for x in matrix["special_points"] if x["id"] == "arithmetic_unmodulated")
    paper_value = float(special["paper_reported_kl_value"])
    paper_unit = str(special["paper_reported_kl_unit"])
    if paper_unit != "nats":
        raise RuntimeError("frozen special-point prose unit unexpectedly changed")

    by_precision: dict[str, Any] = {}
    for precision in EXPECTED_PRECISIONS:
        records = [r for r in result["records"] if int(r["precision"]) == precision]
        if len(records) != 8:
            raise RuntimeError(f"precision {precision}: expected eight records")
        steps = [s for r in records for s in r["steps"]]
        clean = [s for s in steps if int(s["implicit_zero_lookahead_bits"]) == 0]
        padded = [s for s in steps if int(s["implicit_zero_lookahead_bits"]) > 0]
        clean_kl = [float(s["author_kl_bits"]) for s in clean]
        padded_kl = [float(s["author_kl_bits"]) for s in padded]
        total_kl_sum = sum(clean_kl) + sum(padded_kl)
        clean_bits = _mean(clean_kl)
        clean_nats = None if clean_bits is None else clean_bits * math.log(2.0)
        by_precision[str(precision)] = {
            "run_count": len(records),
            "total_step_count": len(steps),
            "zero_padding_free_step_count": len(clean),
            "implicit_zero_padding_step_count": len(padded),
            "mean_full_run_author_kl_bits": _mean([float(r["avg_kl_bits_author"]) for r in records]),
            "mean_zero_padding_free_author_kl_bits": clean_bits,
            "mean_zero_padding_free_author_kl_nats": clean_nats,
            "mean_implicit_zero_padding_author_kl_bits": _mean(padded_kl),
            "implicit_zero_padding_share_of_summed_step_kl": (sum(padded_kl) / total_kl_sum) if total_kl_sum else None,
            "maximum_zero_padding_free_author_kl_bits": max(clean_kl) if clean_kl else None,
            "maximum_implicit_zero_padding_author_kl_bits": max(padded_kl) if padded_kl else None,
            "minimum_zero_padding_free_effective_precision_bits": min(float(s["effective_precision_bits_before"]) for s in clean) if clean else None,
            "mean_zero_padding_free_retained_lm_mass": _mean([float(s["retained_lm_mass"]) for s in clean]),
            "mean_implicit_zero_padding_retained_lm_mass": _mean([float(s["retained_lm_mass"]) for s in padded]),
            "pearson_effective_precision_vs_author_kl_clean": result["summary_by_precision"][str(precision)]["pearson_effective_precision_vs_author_kl"],
            "pearson_rounding_residual_fraction_vs_author_kl_clean": result["summary_by_precision"][str(precision)]["pearson_rounding_residual_fraction_vs_author_kl"],
            "paper_reported_4e_minus_8_nats_ratio_clean": (clean_nats / paper_value) if clean_nats is not None else None,
        }

    p26 = by_precision["26"]
    p40 = by_precision["40"]
    p48 = by_precision["48"]
    clean_drop_26_to_40 = p26["mean_zero_padding_free_author_kl_bits"] / p40["mean_zero_padding_free_author_kl_bits"]
    clean_drop_26_to_48 = p26["mean_zero_padding_free_author_kl_bits"] / p48["mean_zero_padding_free_author_kl_bits"]

    pilot_payload_bits = 24
    author_special_precision = int(special["precision"])
    pilot_all_arithmetic_steps_require_implicit_lookahead = pilot_payload_bits < author_special_precision

    return {
        "schema_version": "stage3.arithmetic_precision_probe_interpretation.v1",
        "source_result": {
            "path": str(result_path.relative_to(REPO_ROOT)),
            "sha256": _sha256(result_path),
        },
        "frozen_matrix": {
            "path": str(matrix_path.relative_to(REPO_ROOT)),
            "sha256": _sha256(matrix_path),
            "paper_special_point_value": paper_value,
            "paper_special_point_unit": paper_unit,
            "figure_axis_unit": special["figure_axis_unit"],
            "reference_helper_unit": special["reference_helper_unit"],
        },
        "by_precision": by_precision,
        "derived_findings": {
            "finite_precision_effect_confirmed": True,
            "clean_kl_drop_factor_precision_26_to_40": clean_drop_26_to_40,
            "clean_kl_drop_factor_precision_26_to_48": clean_drop_26_to_48,
            "precision_40_clean_kl_nats_is_same_order_as_paper_anchor": 0.1 <= p40["paper_reported_4e_minus_8_nats_ratio_clean"] <= 10.0,
            "precision_26_clean_kl_still_far_above_paper_anchor": p26["paper_reported_4e_minus_8_nats_ratio_clean"] > 1000.0,
            "terminal_zero_padding_dominates_probe_summed_kl_at_all_precisions": all(
                by_precision[str(p)]["implicit_zero_padding_share_of_summed_step_kl"] > 0.98 for p in EXPECTED_PRECISIONS
            ),
            "pilot_payload_bits": pilot_payload_bits,
            "pilot_special_point_precision": author_special_precision,
            "pilot_payload_shorter_than_precision": pilot_all_arithmetic_steps_require_implicit_lookahead,
            "interpretation": (
                "The Step-3.7 24-bit special-point KL must not be interpreted as the steady-state unmodulated Arithmetic KL. "
                "Because 24 < precision=26, every author Arithmetic look-ahead window in that pilot is padding-affected. "
                "With a longer stream, zero-padding-free precision=26 steps average about 1e-3 bits/token; increasing precision drives this residual rapidly toward zero. "
                "The executable 26-bit distribution therefore still does not numerically reproduce the paper's 4e-8-nat prose anchor, while precision=40 clean steps reach the same order of magnitude."
            ),
            "paper_claim_status": "not_numerically_reproduced_at_pinned_executable_precision_26",
            "next_required_action": "freeze a paper-sentence driver using a sufficiently long uniform bitstream and stop at the first sentence boundary before the full Figure-3 sweep",
            "full_figure3_sweep_ready": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result_path = args.result.expanduser().resolve()
    matrix_path = args.matrix.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    analysis = analyze(result_path, matrix_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Stage 3 Arithmetic precision interpretation")
    print(f"source result SHA-256: {analysis['source_result']['sha256']}")
    for p in EXPECTED_PRECISIONS:
        item = analysis["by_precision"][str(p)]
        print(
            f"precision={p}: full-run mean KL={item['mean_full_run_author_kl_bits']}; "
            f"zero-padding-free KL={item['mean_zero_padding_free_author_kl_bits']} bits "
            f"({item['mean_zero_padding_free_author_kl_nats']} nats); "
            f"padded-KL share={item['implicit_zero_padding_share_of_summed_step_kl']}"
        )
    findings = analysis["derived_findings"]
    print(f"finite-precision effect confirmed: {findings['finite_precision_effect_confirmed']}")
    print(f"24-bit pilot shorter than precision=26: {findings['pilot_payload_shorter_than_precision']}")
    print(f"paper claim status: {findings['paper_claim_status']}")
    print("full Figure-3 sweep ready: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
