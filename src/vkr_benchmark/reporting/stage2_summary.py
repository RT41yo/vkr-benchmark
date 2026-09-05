"""Deterministic human-readable summaries for Stage-2 run tables.

This module is deliberately independent of pandas/pyarrow. The CLI is responsible
for reading Parquet and passes ordinary mapping records here. Keeping formatting
separate makes reporting deterministic and unit-testable without the optional
storage backend.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class SummaryColumn:
    """One output column in the compact Stage-2 comparison table."""

    source: str
    heading: str
    digits: int | None = None
    percent: bool = False


DEFAULT_COLUMNS: tuple[SummaryColumn, ...] = (
    SummaryColumn("method_id", "Method"),
    SummaryColumn("bits_per_token", "BPT", digits=4),
    SummaryColumn("entropy_utilization", "Entropy util. (%)", digits=2, percent=True),
    SummaryColumn("kl_mean_bits", "KL ref→stego (bits/token)", digits=4),
    SummaryColumn("tvd_mean", "TVD mean", digits=4),
    SummaryColumn("nll_raw_lm_nats_per_token", "Raw-LM NLL", digits=4),
    SummaryColumn("ppl_raw_lm", "Raw-LM PPL", digits=4),
    SummaryColumn("ber", "BER", digits=6),
    SummaryColumn("encode_ms_per_token", "Encode ms/token", digits=3),
    SummaryColumn("decode_ms_per_token", "Decode ms/token", digits=3),
    SummaryColumn("status", "Status"),
)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _format_value(value: Any, column: SummaryColumn) -> str:
    if _is_missing(value):
        return "—"
    if isinstance(value, float):
        if value == math.inf:
            return "inf"
        if value == -math.inf:
            return "-inf"
    if column.percent:
        value = float(value) * 100.0
    if column.digits is not None and isinstance(value, (int, float)):
        return f"{float(value):.{column.digits}f}"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def normalize_summary_records(
    records: Iterable[Mapping[str, Any]],
    *,
    status: str | None = "ok",
    model_id: str | None = None,
    prompt_id: str | None = None,
) -> list[dict[str, Any]]:
    """Filter and deterministically order summary records.

    Ordering is by method id, method-parameter hash, secret id and run id so the
    same Parquet contents always produce the same Markdown table.
    """

    normalized: list[dict[str, Any]] = []
    for record in records:
        row = dict(record)
        if status is not None and row.get("status") != status:
            continue
        if model_id is not None and row.get("model_id") != model_id:
            continue
        if prompt_id is not None and row.get("prompt_id") != prompt_id:
            continue
        normalized.append(row)

    normalized.sort(
        key=lambda row: (
            str(row.get("method_id", "")),
            str(row.get("method_params_hash", "")),
            str(row.get("secret_id", "")),
            str(row.get("run_id", "")),
        )
    )
    return normalized


def build_markdown_table(
    records: Sequence[Mapping[str, Any]],
    *,
    columns: Sequence[SummaryColumn] = DEFAULT_COLUMNS,
) -> str:
    """Render a compact Markdown table from normalized run records."""

    if not records:
        return "_No matching runs._\n"

    headings = [column.heading for column in columns]
    lines = [
        "| " + " | ".join(headings) + " |",
        "| " + " | ".join("---" for _ in headings) + " |",
    ]
    for record in records:
        cells = [_format_value(record.get(column.source), column) for column in columns]
        cells = [cell.replace("|", "\\|") for cell in cells]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
