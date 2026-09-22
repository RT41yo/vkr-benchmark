#!/usr/bin/env python3
"""Interpret the completed Stage-3 Figure-3 reproduction without rerunning a language model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from stage3_figure3_interpretation import (
    build_interpretation,
    load_points_csv,
    render_svg,
    write_claims_csv,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "reproducibility" / "figure3_interpretation.json"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(config_path: Path, *, write: bool = True) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    config = _load_json(config_path)
    inputs = config["inputs"]
    points_path = (REPO_ROOT / inputs["figure3_points_csv"]).resolve()
    summary_path = (REPO_ROOT / inputs["figure3_summary_json"]).resolve()
    precision_path = (REPO_ROOT / inputs["precision_interpretation_json"]).resolve()

    for path in (points_path, summary_path, precision_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    points = load_points_csv(points_path)
    summary = _load_json(summary_path)
    precision = _load_json(precision_path)
    interpretation = build_interpretation(points, summary, precision, config)

    if write:
        outputs = config["outputs"]
        interpretation_path = (REPO_ROOT / outputs["interpretation_json"]).resolve()
        claims_path = (REPO_ROOT / outputs["claim_assessment_csv"]).resolve()
        figure_path = (REPO_ROOT / outputs["figure_svg"]).resolve()
        interpretation_path.parent.mkdir(parents=True, exist_ok=True)
        interpretation_path.write_text(
            json.dumps(interpretation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        write_claims_csv(claims_path, interpretation)
        render_svg(figure_path, points)

    overall = interpretation["overall_assessment"]
    minimum = interpretation["curves"]["arithmetic_k300"]["minimum"]
    special = interpretation["curves"]["arithmetic_k50256"]
    reliability = interpretation["reliability_diagnostics"]
    print("Stage 3 Step 3.11 Figure-3 interpretation")
    print(f"overall reproduction status: {overall['status']}")
    print(
        "Arithmetic k=300 minimum: "
        f"tau={minimum['temperature']:.1f}; "
        f"bits/word={minimum['bits_per_word']:.6f}; KL={minimum['kl_bits']:.9f} bits"
    )
    print(
        "Arithmetic unmodulated: "
        f"bits/word={special['bits_per_word']:.6f}; KL={special['kl_bits']:.9f} bits; "
        f"KL={special['kl_nats']:.9g} nats; paper-anchor-ratio={special['ratio_to_paper_anchor']:.1f}x"
    )
    print(
        "termination: "
        f"{interpretation['execution_input']['terminated_sentence_runs']}/"
        f"{interpretation['execution_input']['scheduled_runs']} successful"
    )
    print(
        "Arithmetic zero-payload: "
        f"{reliability['zero_payload_completed_arithmetic_count']}/"
        f"{reliability['arithmetic_completed_runs']} completed"
    )
    print("Stage 3 Step 3.11: READY FOR MATCHED AUTHOR VS NORMALIZED COMPARISON")
    return interpretation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    run(args.config, write=not args.check_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
