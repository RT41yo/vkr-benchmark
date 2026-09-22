from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import json
from typing import Any, Iterable


EXPECTED_REFERENCE_COMMIT = "14e982564aeaf9a33f7b4de440deda2184d17f12"
EXPECTED_MODEL_REVISION = "6dcaa7a952f72f9298047fd5137cd6e4f05f41da"
EXPECTED_CONTEXT_MANIFEST_SHA256 = "cc25b492b26e102496137bc2f8eb78320c7f7768d83ff7e10af9b2e04a14aebc"

COMPARISON_COLUMNS = (
    "point_id",
    "classification",
    "core_principle_preserved",
    "pair_count",
    "author_mean_bpt",
    "normalized_mean_bpt",
    "mean_delta_bpt",
    "author_mean_reverse_kl_bits",
    "normalized_mean_reverse_kl_bits",
    "mean_delta_reverse_kl_bits",
    "mean_normalized_tvd",
    "exact_token_sequence_pairs",
    "mean_token_agreement_rate",
    "normalized_exact_decode_pairs",
    "benchmark_native_kl_infinite_pairs",
)


@dataclass(frozen=True)
class Stage3CloseoutSummary:
    schema_version: str
    stage: int
    figure3_scheduled_runs: int
    figure3_terminated_runs: int
    figure3_termination_failures: int
    figure3_point_count: int
    figure3_overall_assessment: str
    exact_unmodulated_anchor_reproduced: bool
    matched_pair_count: int
    matched_exact_normalized_decode_count: int
    conformance_point_count: int
    core_principle_preserved_count: int
    benchmark_native_kl_infinite_pair_count: int
    overall_conformance: str
    reference_commit: str
    model_revision: str
    context_manifest_sha256: str
    ready_for_stage4: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_comparison_rows(conformance: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in conformance["point_results"]:
        stats = item["statistics"]
        rows.append(
            {
                "point_id": item["point_id"],
                "classification": item["classification"]["status"],
                "core_principle_preserved": item["classification"]["core_principle_preserved"],
                "pair_count": stats["pair_count"],
                "author_mean_bpt": stats["capacity"]["author_mean_bits_per_token"],
                "normalized_mean_bpt": stats["capacity"]["normalized_mean_bits_per_token"],
                "mean_delta_bpt": stats["capacity"]["mean_delta_normalized_minus_author"],
                "author_mean_reverse_kl_bits": stats["reverse_kl"]["author_mean_bits"],
                "normalized_mean_reverse_kl_bits": stats["reverse_kl"]["normalized_mean_bits"],
                "mean_delta_reverse_kl_bits": stats["reverse_kl"]["mean_delta_normalized_minus_author_bits"],
                "mean_normalized_tvd": stats["normalized_tvd"]["mean"],
                "exact_token_sequence_pairs": stats["token_sequence"]["exact_pair_count"],
                "mean_token_agreement_rate": stats["token_sequence"]["mean_agreement_rate"],
                "normalized_exact_decode_pairs": stats["pair_count"],
                "benchmark_native_kl_infinite_pairs": stats["benchmark_native_kl"]["infinite_pair_count"],
            }
        )
    return rows


def write_comparison_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_COLUMNS)
        writer.writeheader()
        for row in materialized:
            writer.writerow({column: row[column] for column in COMPARISON_COLUMNS})
    return output


def write_comparison_parquet(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "pyarrow is required for results/stage3/comparison.parquet; "
            "install the project storage extra (pip install -e '.[storage]')."
        ) from exc

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(rows))
    pq.write_table(table, output, compression="zstd")
    return output


def build_closeout_summary(
    figure3_summary: dict[str, Any],
    interpretation: dict[str, Any],
    matched_summary: dict[str, Any],
    conformance: dict[str, Any],
) -> Stage3CloseoutSummary:
    claims = {item["claim_id"]: item for item in interpretation["claims"]}
    special_anchor = claims["unmodulated_4e_minus_8_nats"]
    preserved = sum(
        bool(item["classification"]["core_principle_preserved"])
        for item in conformance["point_results"]
    )
    ready = (
        figure3_summary.get("run_count") == 5520
        and figure3_summary.get("point_count") == 23
        and interpretation.get("overall_assessment", {}).get("status") == "partial_reproduction"
        and matched_summary.get("pair_count") == 32
        and matched_summary.get("normalized_exact_decode_count") == 32
        and preserved == 4
        and conformance.get("cross_method_findings", {}).get("benchmark_native_kl_infinite_pairs") == 32
        and conformance.get("ready_for_step_3_14_finalization") is True
    )
    return Stage3CloseoutSummary(
        schema_version="stage3.closeout_summary.v1",
        stage=3,
        figure3_scheduled_runs=int(figure3_summary["scheduled_run_count"]),
        figure3_terminated_runs=int(figure3_summary["terminated_sentence_run_count"]),
        figure3_termination_failures=int(figure3_summary["termination_failure_count"]),
        figure3_point_count=int(figure3_summary["point_count"]),
        figure3_overall_assessment=str(interpretation["overall_assessment"]["status"]),
        exact_unmodulated_anchor_reproduced=(special_anchor["status"] == "reproduced"),
        matched_pair_count=int(matched_summary["pair_count"]),
        matched_exact_normalized_decode_count=int(matched_summary["normalized_exact_decode_count"]),
        conformance_point_count=len(conformance["point_results"]),
        core_principle_preserved_count=preserved,
        benchmark_native_kl_infinite_pair_count=int(
            conformance["cross_method_findings"]["benchmark_native_kl_infinite_pairs"]
        ),
        overall_conformance=str(conformance["overall_classification"]),
        reference_commit=str(matched_summary["reference_commit"]),
        model_revision=str(matched_summary["model_revision"]),
        context_manifest_sha256=str(matched_summary["context_manifest_sha256"]),
        ready_for_stage4=ready,
    )
