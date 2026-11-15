#!/usr/bin/env python3
"""Aggregate the frozen Stage-6 main experiment logs.

The script is intentionally analysis-only.  It does not rerun methods, recompute
Q_stego, or change benchmark semantics.  It consumes the four Stage-6
``execution.jsonl`` logs and produces descriptive operating-point summaries.

Hard failures remain failures.  Numerical metrics are summarized only over
``status=ok`` runs because failed runs do not have a complete benchmark vector.
End-to-end exact reliability uses *all planned runs* as the denominator, so a
hard failure counts as non-exact.

Forward KL is handled according to benchmark semantics: if a run-level mean is
+inf, it is never silently replaced by a finite-only value.  The output reports
how many successful runs have infinite forward KL and, separately, finite-only
statistics for diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True, slots=True)
class SweepSource:
    model_label: str
    family: str
    expected_runs: int
    relative_path: str


SWEEP_SOURCES: tuple[SweepSource, ...] = (
    SweepSource(
        "Llama-3.2-3B",
        "streaming",
        1008,
        "results/stage6/sweeps/stage6-main-streaming-llama-v1/execution.jsonl",
    ),
    SweepSource(
        "Llama-3.2-3B",
        "rrc",
        72,
        "results/stage6/sweeps/stage6-main-rrc-llama-v1/execution.jsonl",
    ),
    SweepSource(
        "Qwen3-4B-Base",
        "streaming",
        672,
        "results/stage6/sweeps/stage6-main-streaming-qwen-v1/execution.jsonl",
    ),
    SweepSource(
        "Qwen3-4B-Base",
        "rrc",
        48,
        "results/stage6/sweeps/stage6-main-rrc-qwen-v1/execution.jsonl",
    ),
)


NUMERIC_METRICS: tuple[str, ...] = (
    "bits_per_token",
    "entropy_utilization",
    "reference_entropy_mean_bits",
    "kl_stego_to_ref_mean_bits",
    "tvd_mean",
    "nll_raw_lm_nats_per_token",
    "ppl_raw_lm",
    "encode_ms_per_token",
    "decode_ms_per_token",
    "stego_algorithm_total_ms",
    "metric_instrumentation_total_ms",
    "elapsed_wall_seconds",
    "ber",
    "carrier_tokens",
)


CSV_COLUMNS: tuple[str, ...] = (
    "model",
    "family",
    "method_id",
    "method_point_id",
    "n_total",
    "n_ok",
    "n_hard_failure",
    "n_soft_roundtrip_failure",
    "n_exact_roundtrip",
    "exact_roundtrip_rate",
    "bits_per_token_mean",
    "bits_per_token_median",
    "entropy_utilization_mean",
    "entropy_utilization_median",
    "reference_entropy_mean_bits_mean",
    "reference_entropy_mean_bits_median",
    "kl_ref_to_stego_infinite_runs",
    "kl_ref_to_stego_infinite_rate_ok",
    "kl_ref_to_stego_finite_mean_bits",
    "kl_ref_to_stego_finite_median_bits",
    "kl_stego_to_ref_mean_bits_mean",
    "kl_stego_to_ref_mean_bits_median",
    "tvd_mean_mean",
    "tvd_mean_median",
    "nll_raw_lm_nats_per_token_mean",
    "nll_raw_lm_nats_per_token_median",
    "ppl_raw_lm_mean",
    "ppl_raw_lm_median",
    "encode_ms_per_token_mean",
    "encode_ms_per_token_median",
    "decode_ms_per_token_mean",
    "decode_ms_per_token_median",
    "stego_algorithm_total_ms_mean",
    "stego_algorithm_total_ms_median",
    "metric_instrumentation_total_ms_mean",
    "metric_instrumentation_total_ms_median",
    "elapsed_wall_seconds_mean",
    "elapsed_wall_seconds_median",
    "carrier_tokens_mean",
    "carrier_tokens_median",
    "ber_mean_ok",
    "ber_median_ok",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row is not an object")
            row = dict(row)
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def _choose_canonical_attempt(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose one canonical outcome from a log history for the same run_id.

    A successful attempt dominates later ``skipped_existing_ok`` records.  A
    failed run may have been retried after a resumed sweep; if it never
    succeeded, the last error record is retained.
    """

    ok = [row for row in attempts if row.get("status") == "ok"]
    if ok:
        return ok[-1]
    error = [row for row in attempts if row.get("status") == "error"]
    if error:
        return error[-1]
    raise ValueError(
        "run history contains neither status=ok nor status=error: "
        + str(attempts[0].get("run_id"))
    )


def load_stage6_runs(project_root: Path) -> list[dict[str, Any]]:
    all_runs: list[dict[str, Any]] = []
    global_ids: set[str] = set()

    for source in SWEEP_SOURCES:
        path = project_root / source.relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Stage-6 execution log not found: {path}")

        histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in _read_jsonl(path):
            run_id = row.get("run_id")
            if not isinstance(run_id, str) or not run_id:
                raise ValueError(f"missing run_id in {path}:{row['_line_number']}")
            histories[run_id].append(row)

        if len(histories) != source.expected_runs:
            raise ValueError(
                f"{path}: expected {source.expected_runs} unique run_ids, "
                f"got {len(histories)}"
            )

        for run_id, attempts in histories.items():
            if run_id in global_ids:
                raise ValueError(f"duplicate run_id across Stage-6 sweeps: {run_id}")
            global_ids.add(run_id)

            row = dict(_choose_canonical_attempt(attempts))
            row["model"] = source.model_label
            row["family"] = source.family
            row["attempt_count"] = len(attempts)

            if row.get("status") == "ok":
                _add_derived_metrics(row)

            all_runs.append(row)

    if len(all_runs) != 1800:
        raise ValueError(f"expected 1800 canonical Stage-6 runs, got {len(all_runs)}")
    return all_runs


def _add_derived_metrics(row: dict[str, Any]) -> None:
    bpt = float(row["bits_per_token"])
    utilization = float(row["entropy_utilization"])
    ppl = float(row["ppl_raw_lm"])

    row["reference_entropy_mean_bits"] = (
        bpt / utilization if utilization != 0.0 else math.nan
    )
    row["nll_raw_lm_nats_per_token"] = math.log(ppl) if ppl > 0.0 else math.nan

    termination = row.get("termination") or {}
    if termination.get("mode") == "fixed_carrier_tokens":
        row["carrier_tokens"] = int(termination["target_carrier_tokens"])
    elif termination.get("mode") == "fixed_payload_bits":
        target_payload = int(termination["target_payload_bits"])
        # BPT is persisted as payload_bits / carrier_tokens.  Recovering the
        # integer carrier length this way is exact up to stored float rounding.
        row["carrier_tokens"] = int(round(target_payload / bpt))
    else:
        row["carrier_tokens"] = None


def _finite_numbers(rows: Iterable[Mapping[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            values.append(number)
    return values


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _stdev(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def _stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "mean": _mean(values),
        "median": _median(values),
        "stdev": _stdev(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def aggregate_operating_points(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        key = (
            str(row["model"]),
            str(row["family"]),
            str(row["method_id"]),
            str(row["method_point_id"]),
        )
        grouped[key].append(row)

    output: list[dict[str, Any]] = []
    for (model, family, method_id, point_id), rows in sorted(grouped.items()):
        ok = [row for row in rows if row.get("status") == "ok"]
        hard = [row for row in rows if row.get("status") == "error"]
        exact = [row for row in ok if row.get("roundtrip_exact") is True]
        soft_fail = [row for row in ok if row.get("roundtrip_exact") is False]

        metrics = {metric: _stats(_finite_numbers(ok, metric)) for metric in NUMERIC_METRICS}

        forward_values = [float(row["kl_ref_to_stego_mean_bits"]) for row in ok]
        forward_inf = sum(math.isinf(value) and value > 0 for value in forward_values)
        forward_finite = [value for value in forward_values if math.isfinite(value)]

        record: dict[str, Any] = {
            "model": model,
            "family": family,
            "method_id": method_id,
            "method_point_id": point_id,
            "n_total": len(rows),
            "n_ok": len(ok),
            "n_hard_failure": len(hard),
            "n_soft_roundtrip_failure": len(soft_fail),
            "n_exact_roundtrip": len(exact),
            "exact_roundtrip_rate": len(exact) / len(rows),
            "kl_ref_to_stego_infinite_runs": forward_inf,
            "kl_ref_to_stego_infinite_rate_ok": forward_inf / len(ok) if ok else None,
            "kl_ref_to_stego_finite": _stats(forward_finite),
            "metrics": metrics,
        }
        output.append(record)
    return output


def flatten_operating_point(record: Mapping[str, Any]) -> dict[str, Any]:
    metrics = record["metrics"]
    fwd = record["kl_ref_to_stego_finite"]
    flat: dict[str, Any] = {
        key: record.get(key)
        for key in (
            "model",
            "family",
            "method_id",
            "method_point_id",
            "n_total",
            "n_ok",
            "n_hard_failure",
            "n_soft_roundtrip_failure",
            "n_exact_roundtrip",
            "exact_roundtrip_rate",
            "kl_ref_to_stego_infinite_runs",
            "kl_ref_to_stego_infinite_rate_ok",
        )
    }
    flat["kl_ref_to_stego_finite_mean_bits"] = fwd["mean"]
    flat["kl_ref_to_stego_finite_median_bits"] = fwd["median"]

    for metric in NUMERIC_METRICS:
        stat = metrics[metric]
        flat[f"{metric}_mean"] = stat["mean"]
        flat[f"{metric}_median"] = stat["median"]

    # CSV uses shorter names for the BER columns to make it explicit that hard
    # failures do not have BER values.
    flat["ber_mean_ok"] = flat.pop("ber_mean")
    flat["ber_median_ok"] = flat.pop("ber_median")
    return flat


def build_execution_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    scopes: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        scopes[(str(row["model"]), str(row["family"]))].append(row)
        scopes[(str(row["model"]), "all")].append(row)
        scopes[("ALL", "all")].append(row)

    scope_rows: list[dict[str, Any]] = []
    for (model, family), rows in sorted(scopes.items()):
        ok = [row for row in rows if row.get("status") == "ok"]
        hard = [row for row in rows if row.get("status") == "error"]
        exact = sum(row.get("roundtrip_exact") is True for row in ok)
        soft = sum(row.get("roundtrip_exact") is False for row in ok)
        wall = sum(float(row.get("elapsed_wall_seconds") or 0.0) for row in rows)
        scope_rows.append(
            {
                "model": model,
                "family": family,
                "n_total": len(rows),
                "n_ok": len(ok),
                "n_hard_failure": len(hard),
                "n_soft_roundtrip_failure": soft,
                "n_exact_roundtrip": exact,
                "exact_roundtrip_rate": exact / len(rows),
                "canonical_run_wall_seconds": wall,
                "canonical_run_wall_hours": wall / 3600.0,
            }
        )

    hard = [row for row in runs if row.get("status") == "error"]
    error_types = Counter(str(row.get("error_type", "<missing>")) for row in hard)

    return {
        "stage": 6,
        "canonical_runs": len(runs),
        "status_counts": dict(Counter(str(row.get("status")) for row in runs)),
        "error_types": dict(error_types),
        "scopes": scope_rows,
    }


def parse_final_integrity_audit(project_root: Path) -> dict[str, Any] | None:
    path = project_root / "results/stage6/stage6_final_audit.txt"
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")

    def integer(pattern: str) -> int | None:
        match = re.search(pattern, text)
        return int(match.group(1)) if match else None

    return {
        "source": str(path.relative_to(project_root)),
        "hard_failures_total": integer(r"hard failures total:\s*(\d+)"),
        "hard_failures_token_roundtrip_false": integer(
            r"token_sequence_roundtrip_exact=False:\s*(\d+)"
        ),
        "hard_failures_token_roundtrip_true": integer(
            r"token_sequence_roundtrip_exact=True\s*:\s*(\d+)"
        ),
        "missing_result_json": integer(r"missing result\.json:\s*(\d+)"),
        "missing_failure_diagnostics_json": integer(
            r"missing failure_diagnostics\.json:\s*(\d+)"
        ),
        "successful_missing_dual_kl_tvd": integer(
            r"successful runs missing dual-KL/TVD fields:\s*(\d+)"
        ),
        "audit_ok": "FINAL STAGE 6 INTEGRITY AUDIT: OK" in text,
    }


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in CSV_COLUMNS})


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _format_float(value: Any, *, digits: int = 4) -> str:
    if value is None:
        return "—"
    number = float(value)
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    if math.isnan(number):
        return "—"
    if abs(number) != 0 and (abs(number) < 1e-4 or abs(number) >= 1e5):
        return f"{number:.3e}"
    return f"{number:.{digits}f}"


def build_markdown_table(flat_records: list[dict[str, Any]]) -> str:
    lines = [
        "# Stage 6 — предварительная сводка по рабочим точкам",
        "",
        "Численные метрики агрегированы только по `status=ok`; hard failures не имеют полного benchmark vector. "
        "`Exact` использует все запланированные runs в знаменателе, поэтому hard failure считается неточным end-to-end round-trip.",
        "",
        "Forward KL не заменяется finite-only оценкой: колонка `KLf=inf` показывает число successful runs с `+inf`; "
        "finite-only значения ниже являются только диагностикой.",
        "",
        "| Model | Method | Point | ok/total | Exact | BPT mean | Entropy util. mean | KLf=inf | KLrev mean | TVD mean | PPL median | Encode ms/tok median | Decode ms/tok median |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in flat_records:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model"]),
                    str(row["method_id"]),
                    str(row["method_point_id"]),
                    f'{row["n_ok"]}/{row["n_total"]}',
                    f'{100.0 * float(row["exact_roundtrip_rate"]):.2f}%',
                    _format_float(row["bits_per_token_mean"]),
                    _format_float(row["entropy_utilization_mean"]),
                    str(row["kl_ref_to_stego_infinite_runs"]),
                    _format_float(row["kl_stego_to_ref_mean_bits_mean"], digits=6),
                    _format_float(row["tvd_mean_mean"], digits=6),
                    _format_float(row["ppl_raw_lm_median"], digits=3),
                    _format_float(row["encode_ms_per_token_median"], digits=2),
                    _format_float(row["decode_ms_per_token_median"], digits=2),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def build_cross_model(flat_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_point: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in flat_records:
        by_point[(str(row["method_id"]), str(row["method_point_id"]))][
            str(row["model"])
        ] = row

    output: list[dict[str, Any]] = []
    for (method_id, point_id), model_rows in sorted(by_point.items()):
        llama = model_rows.get("Llama-3.2-3B")
        qwen = model_rows.get("Qwen3-4B-Base")
        if llama is None or qwen is None:
            continue

        def delta_pp(key: str) -> float:
            return 100.0 * (float(qwen[key]) - float(llama[key]))

        output.append(
            {
                "method_id": method_id,
                "method_point_id": point_id,
                "llama_exact_rate": llama["exact_roundtrip_rate"],
                "qwen_exact_rate": qwen["exact_roundtrip_rate"],
                "exact_rate_delta_qwen_minus_llama_pp": delta_pp("exact_roundtrip_rate"),
                "llama_bpt_mean": llama["bits_per_token_mean"],
                "qwen_bpt_mean": qwen["bits_per_token_mean"],
                "llama_entropy_utilization_mean": llama["entropy_utilization_mean"],
                "qwen_entropy_utilization_mean": qwen["entropy_utilization_mean"],
                "llama_reverse_kl_mean": llama["kl_stego_to_ref_mean_bits_mean"],
                "qwen_reverse_kl_mean": qwen["kl_stego_to_ref_mean_bits_mean"],
                "llama_tvd_mean": llama["tvd_mean_mean"],
                "qwen_tvd_mean": qwen["tvd_mean_mean"],
                "llama_ppl_median": llama["ppl_raw_lm_median"],
                "qwen_ppl_median": qwen["ppl_raw_lm_median"],
                "llama_encode_ms_per_token_median": llama["encode_ms_per_token_median"],
                "qwen_encode_ms_per_token_median": qwen["encode_ms_per_token_median"],
            }
        )
    return output


def _write_dynamic_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(records[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate frozen Stage-6 main experiment logs.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: inferred from this script)",
    )
    args = parser.parse_args()

    root = args.project_root.expanduser().resolve()
    out = root / "results/stage6"

    runs = load_stage6_runs(root)
    operating_points = aggregate_operating_points(runs)
    flat = [flatten_operating_point(row) for row in operating_points]
    execution_summary = build_execution_summary(runs)
    execution_summary["final_integrity_audit"] = parse_final_integrity_audit(root)
    cross_model = build_cross_model(flat)

    _write_csv(out / "stage6_operating_point_summary.csv", flat)
    _json_dump(out / "stage6_operating_point_summary.json", operating_points)
    _json_dump(out / "stage6_execution_summary.json", execution_summary)
    _write_dynamic_csv(out / "stage6_cross_model_summary.csv", cross_model)
    (out / "stage6_operating_point_summary.md").write_text(
        build_markdown_table(flat), encoding="utf-8"
    )

    print("Stage 6 aggregation complete")
    print(f"canonical runs: {len(runs)}")
    print(f"operating-point rows: {len(flat)}")
    print(f"cross-model rows: {len(cross_model)}")
    print(f"output directory: {out}")


if __name__ == "__main__":
    main()
