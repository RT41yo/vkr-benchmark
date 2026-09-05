#!/usr/bin/env python3
"""Render results/summary.parquet as a compact Stage-2 Markdown table."""

from __future__ import annotations

import argparse
from pathlib import Path

from vkr_benchmark.reporting import build_markdown_table, normalize_summary_records


def _read_parquet_records(path: Path) -> list[dict[str, object]]:
    """Read summary.parquet using the same pyarrow dependency as storage.

    The ``storage`` extra already installs pyarrow, so reporting must not add an
    unrelated pandas dependency just to convert the table into ordinary records.
    """

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            'Parquet reporting requires the storage extra: pip install -e ".[storage]"'
        ) from exc

    table = pq.read_table(path)
    return table.to_pylist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "summary",
        type=Path,
        nargs="?",
        default=Path("results/summary.parquet"),
        help="Path to summary.parquet (default: results/summary.parquet)",
    )
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--prompt-id", default=None)
    parser.add_argument(
        "--all-statuses",
        action="store_true",
        help="Include failed runs as well as status=ok runs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional Markdown output path; stdout is always printed",
    )
    args = parser.parse_args()

    summary_path = args.summary.expanduser().resolve()
    if not summary_path.is_file():
        raise SystemExit(f"summary parquet not found: {summary_path}")

    rows = _read_parquet_records(summary_path)
    rows = normalize_summary_records(
        rows,
        status=None if args.all_statuses else "ok",
        model_id=args.model_id,
        prompt_id=args.prompt_id,
    )
    markdown = build_markdown_table(rows)
    print(markdown, end="")

    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
        print(f"\nwritten: {output_path}")


if __name__ == "__main__":
    main()
