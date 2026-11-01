#!/usr/bin/env python3
"""Summarize the Stage-5 RRC fixed-payload calibration."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP_ID = "stage5-rrc-payload-calibration-llama-v1"
DEFAULT_EXECUTION = REPO_ROOT / "results/stage5/sweeps" / SWEEP_ID / "execution.jsonl"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_OUTPUT = REPO_ROOT / "results/stage5/rrc_payload_calibration/summary.json"
PAYLOAD_BY_POINT = {
    "payload128": 128,
    "payload256": 256,
    "payload512": 512,
    "payload1024": 1024,
}
EXPECTED_PER_PAYLOAD = 8
TARGET_COMPARISON_TOKENS = 256


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


def _latest_terminal_by_run(rows: Iterable[dict[str, object]]) -> dict[str, dict[str, object]]:
    selected: dict[str, dict[str, object]] = {}
    for row in rows:
        run_id = row.get("run_id")
        if isinstance(run_id, str) and row.get("status") in {"ok", "error"}:
            selected[run_id] = row
    return selected


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _numbers(rows: list[dict[str, object]], field: str) -> list[float]:
    return [float(row[field]) for row in rows if _finite(row.get(field))]


def _summary_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "mean": None, "max": None}
    return {
        "min": min(values),
        "median": median(values),
        "mean": mean(values),
        "max": max(values),
    }


def _read_result(results_root: Path, run_id: str) -> dict[str, object] | None:
    path = results_root / "runs" / run_id / "result.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def summarize(execution_path: Path, results_root: Path) -> dict[str, object]:
    terminal = _latest_terminal_by_run(_read_jsonl(execution_path))
    rows = [
        row for row in terminal.values()
        if row.get("sweep_id") == SWEEP_ID and row.get("method_point_id") in PAYLOAD_BY_POINT
    ]

    by_payload: dict[str, dict[str, object]] = {}
    all_complete = True
    all_numeric_clean = True

    for point_id, payload_bits in PAYLOAD_BY_POINT.items():
        point_rows = [row for row in rows if row.get("method_point_id") == point_id]
        ok_rows = [row for row in point_rows if row.get("status") == "ok"]
        error_rows = [row for row in point_rows if row.get("status") == "error"]
        ok_rows.sort(key=lambda row: (str(row.get("prompt_id")), str(row.get("replicate_id"))))

        carrier_tokens: list[float] = []
        transport_exact = 0
        result_artifact_count = 0
        detailed: dict[str, dict[str, object]] = {}
        for row in ok_rows:
            run_id = str(row["run_id"])
            result = _read_result(results_root, run_id)
            if result is not None:
                result_artifact_count += 1
                carrier = result.get("carrier_tokens")
                if _finite(carrier):
                    carrier_tokens.append(float(carrier))
                if result.get("token_sequence_roundtrip_exact") is True:
                    transport_exact += 1
            elif _finite(row.get("bits_per_token")) and float(row["bits_per_token"]) > 0:
                carrier_tokens.append(round(payload_bits / float(row["bits_per_token"])))

            key = f"{row.get('prompt_id')}:{row.get('replicate_id')}"
            detailed[key] = {
                "run_id": run_id,
                "status": row.get("status"),
                "carrier_tokens": result.get("carrier_tokens") if result else None,
                "bits_per_token": row.get("bits_per_token"),
                "entropy_utilization": row.get("entropy_utilization"),
                "roundtrip_exact": row.get("roundtrip_exact"),
                "ber": row.get("ber"),
                "token_sequence_roundtrip_exact": result.get("token_sequence_roundtrip_exact") if result else None,
                "first_token_roundtrip_mismatch": result.get("first_token_roundtrip_mismatch") if result else None,
                "kl_ref_to_stego_mean_bits": row.get("kl_ref_to_stego_mean_bits"),
                "kl_stego_to_ref_mean_bits": row.get("kl_stego_to_ref_mean_bits"),
                "tvd_mean": row.get("tvd_mean"),
                "ppl_raw_lm": row.get("ppl_raw_lm"),
                "elapsed_wall_seconds": row.get("elapsed_wall_seconds"),
            }

        forward = _numbers(ok_rows, "kl_ref_to_stego_mean_bits")
        reverse = _numbers(ok_rows, "kl_stego_to_ref_mean_bits")
        tvd = _numbers(ok_rows, "tvd_mean")
        exact_count = sum(row.get("roundtrip_exact") is True for row in ok_rows)
        zero_ber_count = sum(_finite(row.get("ber")) and float(row["ber"]) == 0.0 for row in ok_rows)
        numeric_clean = (
            len(ok_rows) == EXPECTED_PER_PAYLOAD
            and len(forward) == EXPECTED_PER_PAYLOAD
            and len(reverse) == EXPECTED_PER_PAYLOAD
            and len(tvd) == EXPECTED_PER_PAYLOAD
            and max((abs(v) for v in forward), default=math.inf) <= 1e-12
            and max((abs(v) for v in reverse), default=math.inf) <= 1e-12
            and max(tvd, default=math.inf) <= 1e-12
        )
        completed = len(ok_rows) + len(error_rows)
        if completed != EXPECTED_PER_PAYLOAD:
            all_complete = False
        if not numeric_clean:
            all_numeric_clean = False

        by_payload[str(payload_bits)] = {
            "point_id": point_id,
            "expected_count": EXPECTED_PER_PAYLOAD,
            "completed_count": completed,
            "ok_count": len(ok_rows),
            "error_count": len(error_rows),
            "roundtrip_exact_count": exact_count,
            "zero_ber_count": zero_ber_count,
            "transport_exact_count": transport_exact if result_artifact_count == len(ok_rows) else None,
            "result_artifact_count": result_artifact_count,
            "numeric_clean": numeric_clean,
            "carrier_tokens": _summary_stats(carrier_tokens),
            "bits_per_token": _summary_stats(_numbers(ok_rows, "bits_per_token")),
            "entropy_utilization": _summary_stats(_numbers(ok_rows, "entropy_utilization")),
            "elapsed_wall_seconds": _summary_stats(_numbers(ok_rows, "elapsed_wall_seconds")),
            "encode_ms_per_token": _summary_stats(_numbers(ok_rows, "encode_ms_per_token")),
            "decode_ms_per_token": _summary_stats(_numbers(ok_rows, "decode_ms_per_token")),
            "stego_algorithm_total_ms": _summary_stats(_numbers(ok_rows, "stego_algorithm_total_ms")),
            "metric_instrumentation_total_ms": _summary_stats(_numbers(ok_rows, "metric_instrumentation_total_ms")),
            "ppl_raw_lm": _summary_stats(_numbers(ok_rows, "ppl_raw_lm")),
            "max_abs_forward_kl_bits": max((abs(v) for v in forward), default=None),
            "max_abs_reverse_kl_bits": max((abs(v) for v in reverse), default=None),
            "max_tvd": max(tvd, default=None),
            "by_prompt_replicate": detailed,
        }

    medians = {
        int(payload): details["carrier_tokens"]["median"]
        for payload, details in by_payload.items()
        if details["carrier_tokens"]["median"] is not None
    }
    closest_payload = None
    if medians:
        closest_payload = min(
            medians,
            key=lambda payload: (abs(float(medians[payload]) - TARGET_COMPARISON_TOKENS), -payload),
        )

    return {
        "schema_version": "stage5.rrc_payload_calibration.summary.v1",
        "sweep_id": SWEEP_ID,
        "payload_candidates_bits": list(PAYLOAD_BY_POINT.values()),
        "expected_total_count": EXPECTED_PER_PAYLOAD * len(PAYLOAD_BY_POINT),
        "terminal_count": len(rows),
        "all_runs_terminal": all_complete,
        "all_payloads_numeric_clean": all_numeric_clean,
        "comparison_reference_carrier_tokens": TARGET_COMPARISON_TOKENS,
        "closest_median_carrier_to_reference_payload_bits": closest_payload,
        "by_payload_bits": by_payload,
        "interpretation": (
            "The calibration compares fixed RRC payload sizes under paired prompt/replicate randomness. "
            "The 256-token value is only a comparison reference inherited from Pilot B streaming runs; "
            "it is not an automatic Stage-6 selection rule. End-to-end reliability and text-transport "
            "stability must be considered separately from numerical RRC correctness."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution", type=Path, default=DEFAULT_EXECUTION)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    result = summarize(args.execution, args.results_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Stage 5 RRC payload calibration")
    print("terminal:", f"{result['terminal_count']}/{result['expected_total_count']}")
    for payload in result["payload_candidates_bits"]:
        item = result["by_payload_bits"][str(payload)]
        carrier = item["carrier_tokens"]
        print(
            f"payload={payload:4d}: ok={item['ok_count']}/8 "
            f"exact={item['roundtrip_exact_count']}/8 "
            f"transport_exact={item['transport_exact_count']}/8 "
            f"carrier median={carrier['median']} range={carrier['min']}..{carrier['max']} "
            f"numeric_clean={item['numeric_clean']}"
        )
    print("all numeric clean:", result["all_payloads_numeric_clean"])
    print("closest median carrier to 256 tokens:", result["closest_median_carrier_to_reference_payload_bits"])
    print("json:", args.output)
    return 0 if result["all_runs_terminal"] and result["all_payloads_numeric_clean"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
