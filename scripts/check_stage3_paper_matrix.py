#!/usr/bin/env python3
"""Validate the frozen Stage-3 paper-reproduction matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = REPO_ROOT / "configs" / "reproducibility" / "paper_reproduction_matrix.json"
EXPECTED_REFERENCE_COMMIT = "14e982564aeaf9a33f7b4de440deda2184d17f12"
EXPECTED_MODEL = "gpt2-medium"


def _fail(message: str) -> None:
    raise ValueError(message)


def validate_matrix(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != "stage3.paper_reproduction_matrix.v1":
        _fail("unexpected schema_version")
    if payload.get("status") != "frozen_before_paper_runs":
        _fail("matrix must be frozen before paper runs")

    reference = payload.get("reference") or {}
    if reference.get("commit") != EXPECTED_REFERENCE_COMMIT:
        _fail("reference commit is not the pinned Harvard commit")

    setup = payload.get("common_paper_setup") or {}
    if setup.get("model_id") != EXPECTED_MODEL:
        _fail("paper-level historical model must be gpt2-medium")
    if setup.get("dataset_name") != "CNN/DailyMail":
        _fail("paper reproduction must use CNN/DailyMail")
    if setup.get("message_rule") != "uniform random binary message":
        _fail("paper reproduction must use uniform random bits")
    if setup.get("kl_target_direction") != "Q_stego || P_LM":
        _fail("paper-compatible KL direction must be Q_stego || P_LM")

    sweeps = payload.get("information_theoretic_sweeps")
    if not isinstance(sweeps, list) or len(sweeps) != 3:
        _fail("expected exactly three primary method sweeps")
    by_method = {str(item.get("method")): item for item in sweeps}
    if set(by_method) != {"bins", "huffman", "arithmetic"}:
        _fail("primary sweeps must cover bins, huffman, arithmetic")

    if by_method["bins"].get("values") != [1, 2, 3, 4, 5]:
        _fail("Bins sweep must be exponents 1..5")
    if by_method["huffman"].get("values") != list(range(1, 9)):
        _fail("Huffman sweep must be exponents 1..8")
    arithmetic_values = by_method["arithmetic"].get("values")
    expected_temps = [round(0.4 + 0.1 * i, 1) for i in range(9)]
    if arithmetic_values != expected_temps:
        _fail("Arithmetic temperature grid must be 0.4..1.2 by 0.1")
    arithmetic_fixed = by_method["arithmetic"].get("fixed") or {}
    if arithmetic_fixed.get("topk") != 300 or arithmetic_fixed.get("precision") != 26:
        _fail("Arithmetic sweep must pin topk=300 and precision=26")

    special = payload.get("special_points")
    if not isinstance(special, list) or len(special) != 1:
        _fail("expected one unmodulated Arithmetic special point")
    unmod = special[0]
    if unmod.get("id") != "arithmetic_unmodulated":
        _fail("unexpected special point id")
    if (unmod.get("temperature"), unmod.get("topk"), unmod.get("precision")) != (1.0, 50256, 26):
        _fail("unmodulated Arithmetic point must be tau=1, topk=50256, precision=26")
    if unmod.get("paper_reported_kl_value") != 4e-8:
        _fail("paper-reported near-zero KL anchor must remain explicit")
    if unmod.get("paper_reported_kl_unit") != "nats":
        _fail("paper prose unit for the 4e-8 anchor must remain nats")
    if unmod.get("figure_axis_unit") != "bits":
        _fail("Figure-3 axis unit ambiguity must remain explicit")

    plan = payload.get("execution_plan") or {}
    pilot = plan.get("phase_1_gpt2_medium_pilot") or {}
    full = plan.get("phase_2_information_theoretic_curve") or {}
    if (pilot.get("contexts"), pilot.get("replicates")) != (8, 1):
        _fail("pilot must remain 8 contexts x 1 replicate")
    if (full.get("contexts_per_point"), full.get("replicates")) != (80, 3):
        _fail("full curve must remain 80 contexts/point x 3 replicates")

    go_no_go = payload.get("go_no_go") or {}
    if go_no_go.get("technical_smoke_gate_complete") is not True:
        _fail("technical smoke gate must be marked complete")
    if go_no_go.get("matrix_frozen") is not True:
        _fail("matrix_frozen must be true")
    if go_no_go.get("ready_for_gpt2_medium_pilot") is not False:
        _fail("pilot must remain blocked until CNN/DailyMail artifact is pinned")

    primary_points = sum(len(item["values"]) for item in sweeps)
    total_points = primary_points + len(special)
    return {
        "model": setup["model_id"],
        "primary_points": primary_points,
        "special_points": len(special),
        "total_points": total_points,
        "pilot_contexts": pilot["contexts"],
        "full_contexts_per_point": full["contexts_per_point"],
        "full_replicates": full["replicates"],
        "ready_for_gpt2_medium_pilot": go_no_go["ready_for_gpt2_medium_pilot"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    args = parser.parse_args()

    path = args.matrix.expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = validate_matrix(payload)
    except Exception as exc:
        print(f"Stage 3 paper reproduction matrix: INVALID ({type(exc).__name__}: {exc})")
        return 1

    print("Stage 3 paper reproduction matrix")
    print(f"matrix: {path}")
    print(f"model: {summary['model']}")
    print(f"parameter points: {summary['total_points']} ({summary['primary_points']} sweep + {summary['special_points']} special)")
    print(f"pilot: {summary['pilot_contexts']} contexts x 1 replicate")
    print(
        "full curve: "
        f"{summary['full_contexts_per_point']} contexts/point x {summary['full_replicates']} replicates"
    )
    print("gpt2-medium pilot blocked until CNN/DailyMail artifact/revision is pinned: True")
    print("Stage 3 paper reproduction matrix: FROZEN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
