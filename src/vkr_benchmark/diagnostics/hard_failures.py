"""Pure helpers for Stage-5 hard-failure diagnosis.

The GPU replay lives in ``scripts/diagnose_stage5_hard_failures.py``.  This
module deliberately contains only parsing/classification logic so it can be
unit-tested without loading a language model.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True, slots=True)
class HardFailureRecord:
    run_id: str
    method_id: str
    method_point_id: str
    prompt_id: str
    replicate_id: str
    error_type: str
    error_message: str
    raw: Mapping[str, Any]


def load_hard_failure_records(
    path: str | Path,
    *,
    methods: Iterable[str] = ("huffman", "dairstega"),
) -> tuple[HardFailureRecord, ...]:
    """Load error records for the requested methods from a sweep JSONL log."""

    selected_methods = {str(item) for item in methods}
    records: list[HardFailureRecord] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"line {line_number} is not a JSON object")
        if raw.get("status") != "error":
            continue
        method_id = str(raw.get("method_id", ""))
        if method_id not in selected_methods:
            continue
        required = (
            "run_id",
            "method_point_id",
            "prompt_id",
            "replicate_id",
            "error_type",
            "error_message",
        )
        missing = [field for field in required if not raw.get(field)]
        if missing:
            raise ValueError(
                f"line {line_number} hard-failure record missing fields: {missing}"
            )
        records.append(
            HardFailureRecord(
                run_id=str(raw["run_id"]),
                method_id=method_id,
                method_point_id=str(raw["method_point_id"]),
                prompt_id=str(raw["prompt_id"]),
                replicate_id=str(raw["replicate_id"]),
                error_type=str(raw["error_type"]),
                error_message=str(raw["error_message"]),
                raw=raw,
            )
        )
    return tuple(records)


def classify_hard_failure(record: Mapping[str, Any]) -> str:
    """Classify one replay without hiding unresolved cases.

    A text-transport diagnosis is accepted only when all three observations are
    present: sender-token direct decoding is exact, ordinary-text transport
    changes token IDs, and the transported decoder fails at or after the first
    token mismatch.
    """

    direct_exact = record.get("direct_token_decode_exact") is True
    mismatch = record.get("first_transport_mismatch")
    failure_step = record.get("failure_step")
    if (
        direct_exact
        and isinstance(mismatch, int)
        and isinstance(failure_step, int)
        and failure_step >= mismatch
    ):
        return "text_transport_induced"
    return "requires_investigation"


def summarize_hard_failure_diagnostics(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = list(records)
    by_method: dict[str, int] = {}
    by_classification: dict[str, int] = {}
    direct_exact = 0
    transport_mismatch = 0
    for row in rows:
        method_id = str(row.get("method_id", "unknown"))
        by_method[method_id] = by_method.get(method_id, 0) + 1
        classification = str(
            row.get("classification") or classify_hard_failure(row)
        )
        by_classification[classification] = (
            by_classification.get(classification, 0) + 1
        )
        if row.get("direct_token_decode_exact") is True:
            direct_exact += 1
        if isinstance(row.get("first_transport_mismatch"), int):
            transport_mismatch += 1
    return {
        "diagnosed_runs": len(rows),
        "by_method": dict(sorted(by_method.items())),
        "by_classification": dict(sorted(by_classification.items())),
        "direct_token_decode_exact_count": direct_exact,
        "transport_mismatch_count": transport_mismatch,
    }
