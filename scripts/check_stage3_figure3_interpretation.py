#!/usr/bin/env python3
"""Check the committed Step-3.11 interpretation artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "reproducibility" / "figure3_interpretation.json"


def main() -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    outputs = config["outputs"]
    interpretation_path = REPO_ROOT / outputs["interpretation_json"]
    claims_path = REPO_ROOT / outputs["claim_assessment_csv"]
    figure_path = REPO_ROOT / outputs["figure_svg"]

    errors: list[str] = []
    for path in (interpretation_path, claims_path, figure_path):
        if not path.is_file():
            errors.append(f"missing output: {path.relative_to(REPO_ROOT)}")

    interpretation = {}
    if interpretation_path.is_file():
        interpretation = json.loads(interpretation_path.read_text(encoding="utf-8"))
        if interpretation.get("scientific_claims_evaluated") is not True:
            errors.append("scientific_claims_evaluated is not true")
        if interpretation.get("overall_assessment", {}).get("status") != "partial_reproduction":
            errors.append("overall assessment is not partial_reproduction")
        if interpretation.get("overall_assessment", {}).get("ready_for_step_3_12_matched_author_vs_normalized") is not True:
            errors.append("Step-3.12 readiness flag is not true")
        minimum = interpretation.get("curves", {}).get("arithmetic_k300", {}).get("minimum", {})
        if minimum.get("temperature") != 1.0:
            errors.append("Arithmetic k=300 minimum is not at temperature=1.0")
        special = interpretation.get("curves", {}).get("arithmetic_k50256", {})
        if not (float(special.get("ratio_to_paper_anchor", 0)) > 1000):
            errors.append("special-point discrepancy ratio unexpectedly small")
        dominance = interpretation.get("curve_dominance_diagnostic", {})
        for key in ("arithmetic_vs_huffman", "arithmetic_vs_bins"):
            if dominance.get(key, {}).get("arithmetic_lower_at_every_grid_point") is not True:
                errors.append(f"curve dominance failed: {key}")

    if claims_path.is_file():
        with claims_path.open("r", encoding="utf-8", newline="") as handle:
            claims = list(csv.DictReader(handle))
        if len(claims) != 8:
            errors.append(f"expected 8 claim rows, found {len(claims)}")
        allowed = set(config["policy"]["allowed_reproduction_statuses"])
        bad = [row["status"] for row in claims if row["status"] not in allowed]
        if bad:
            errors.append(f"invalid claim statuses: {bad}")

    if figure_path.is_file():
        text = figure_path.read_text(encoding="utf-8")
        if "Stage 3.11" not in text or "Arithmetic k=50256" not in text:
            errors.append("SVG does not contain expected title/legend")

    if errors:
        print("Stage 3 Step 3.11 interpretation: NOT READY")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Stage 3 Step 3.11 interpretation summary")
    print("claims: 8/8 classified")
    print("Figure-3 trend reproduction: PASS")
    print("exact 4e-8-nat special-point anchor: NOT NUMERICALLY REPRODUCED at pinned precision=26")
    print("historical Figure-3 orchestration: PARTIAL (exact MC driver/sample count unavailable)")
    print("Stage 3 Step 3.11: READY FOR STEP 3.12 MATCHED AUTHOR VS NORMALIZED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
