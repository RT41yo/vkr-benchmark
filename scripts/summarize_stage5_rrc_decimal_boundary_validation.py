#!/usr/bin/env python3
"""Summarize the Stage-5 RRC Decimal-boundary validation sweep."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXECUTION = (
    REPO_ROOT
    / "results/stage5/sweeps/stage5-rrc-decimal-boundary-validation-llama-v1/execution.jsonl"
)
DEFAULT_SOURCE = (
    REPO_ROOT
    / "results/stage5/sweeps/stage5-rrc-precision-margin-calibration-llama-v1/execution.jsonl"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "results/stage5/rrc_decimal_boundary_validation/summary.json"
)
EXPECTED = 8
EXPECTED_SWEEP_ID = "stage5-rrc-decimal-boundary-validation-llama-v1"
EXPECTED_POINT_ID = "decimal-margin40"


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _latest_ok_by_run(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    selected: dict[str, dict[str, object]] = {}
    for row in rows:
        run_id = row.get("run_id")
        if isinstance(run_id, str) and row.get("status") == "ok":
            selected[run_id] = row
    return selected


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def summarize(execution_path: Path, source_path: Path | None = None) -> dict[str, object]:
    rows = [
        row
        for row in _latest_ok_by_run(_read_jsonl(execution_path)).values()
        if row.get("sweep_id") == EXPECTED_SWEEP_ID
        and row.get("method_point_id") == EXPECTED_POINT_ID
    ]
    rows.sort(key=lambda r: (str(r.get("prompt_id")), str(r.get("replicate_id"))))

    exact = sum(row.get("roundtrip_exact") is True for row in rows)
    zero_ber = sum(_finite_number(row.get("ber")) and float(row["ber"]) == 0.0 for row in rows)
    finite_forward = sum(_finite_number(row.get("kl_ref_to_stego_mean_bits")) for row in rows)
    finite_reverse = sum(_finite_number(row.get("kl_stego_to_ref_mean_bits")) for row in rows)
    forward_values = [float(row["kl_ref_to_stego_mean_bits"]) for row in rows if _finite_number(row.get("kl_ref_to_stego_mean_bits"))]
    reverse_values = [float(row["kl_stego_to_ref_mean_bits"]) for row in rows if _finite_number(row.get("kl_stego_to_ref_mean_bits"))]
    tvd_values = [float(row["tvd_mean"]) for row in rows if _finite_number(row.get("tvd_mean"))]

    by_prompt_replicate: dict[str, dict[str, object]] = {}
    for row in rows:
        bpt = float(row["bits_per_token"]) if _finite_number(row.get("bits_per_token")) else None
        carrier_tokens = round(1024.0 / bpt) if bpt and bpt > 0 else None
        key = f"{row.get('prompt_id')}:{row.get('replicate_id')}"
        by_prompt_replicate[key] = {
            "run_id": row.get("run_id"),
            "carrier_tokens": carrier_tokens,
            "bits_per_token": bpt,
            "roundtrip_exact": row.get("roundtrip_exact"),
            "ber": row.get("ber"),
            "kl_ref_to_stego_mean_bits": row.get("kl_ref_to_stego_mean_bits"),
            "kl_stego_to_ref_mean_bits": row.get("kl_stego_to_ref_mean_bits"),
            "tvd_mean": row.get("tvd_mean"),
        }

    source_margin40_inf = None
    source_margin40_count = None
    if source_path is not None and source_path.exists():
        source_rows = [
            row
            for row in _latest_ok_by_run(_read_jsonl(source_path)).values()
            if row.get("method_point_id") == "margin40"
        ]
        source_margin40_count = len(source_rows)
        source_margin40_inf = sum(
            not _finite_number(row.get("kl_ref_to_stego_mean_bits"))
            for row in source_rows
        )

    max_forward = max((abs(value) for value in forward_values), default=None)
    max_reverse = max((abs(value) for value in reverse_values), default=None)
    max_tvd = max(tvd_values, default=None)
    validation_gate = (
        len(rows) == EXPECTED
        and exact == EXPECTED
        and zero_ber == EXPECTED
        and finite_forward == EXPECTED
        and finite_reverse == EXPECTED
        and len(tvd_values) == EXPECTED
        and max_forward is not None
        and max_reverse is not None
        and max_tvd is not None
        and max_forward <= 1e-12
        and max_reverse <= 1e-12
        and max_tvd <= 1e-12
    )

    return {
        "schema_version": "stage5.rrc_decimal_boundary_validation.summary.v1",
        "selected_count": EXPECTED,
        "completed_ok_count": len(rows),
        "exact_count": exact,
        "zero_ber_count": zero_ber,
        "finite_forward_kl_count": finite_forward,
        "finite_reverse_kl_count": finite_reverse,
        "max_abs_forward_kl_bits": max_forward,
        "max_abs_reverse_kl_bits": max_reverse,
        "max_tvd": max_tvd,
        "source_margin40_count": source_margin40_count,
        "source_margin40_infinite_forward_kl_count": source_margin40_inf,
        "validation_gate": validation_gate,
        "by_prompt_replicate": by_prompt_replicate,
        "interpretation": (
            "The validation gate checks the Decimal-boundary RRC implementation only. "
            "The new sweep uses a new Stage-5 sweep id, so synchronized RRC randomness "
            "is newly derived rather than paired bit-for-bit with the earlier margin40 sweep."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution", type=Path, default=DEFAULT_EXECUTION)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    result = summarize(args.execution, args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Stage 5 RRC Decimal-boundary validation")
    print("completed ok:", f"{result['completed_ok_count']}/{result['selected_count']}")
    print("exact:", f"{result['exact_count']}/{result['selected_count']}")
    print("BER=0:", f"{result['zero_ber_count']}/{result['selected_count']}")
    print("finite forward KL:", f"{result['finite_forward_kl_count']}/{result['selected_count']}")
    print("max |forward KL| bits:", result["max_abs_forward_kl_bits"])
    print("max TVD:", result["max_tvd"])
    print("old margin40 +inf runs:", result["source_margin40_infinite_forward_kl_count"])
    print("validation gate:", result["validation_gate"])
    print("json:", args.output)
    return 0 if result["validation_gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
