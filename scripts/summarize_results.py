#!/usr/bin/env python3
"""Сформировать компактную Markdown-таблицу из results/summary.parquet."""

from __future__ import annotations

import argparse
from pathlib import Path

from vkr_benchmark.reporting import build_markdown_table, normalize_summary_records


def _read_parquet_records(path: Path) -> list[dict[str, object]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            'Для чтения Parquet установите storage-зависимости: pip install -e ".[storage]"'
        ) from exc

    table = pq.read_table(path)
    return table.to_pylist()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Сформировать компактную Markdown-сводку результатов Этапа 2."
    )
    parser.add_argument(
        "summary",
        type=Path,
        nargs="?",
        default=Path("results/summary.parquet"),
        help="Путь к summary.parquet (по умолчанию results/summary.parquet)",
    )
    parser.add_argument("--model-id", default=None, help="Фильтр по идентификатору модели")
    parser.add_argument("--prompt-id", default=None, help="Фильтр по идентификатору промпта")
    parser.add_argument(
        "--all-statuses",
        action="store_true",
        help="Показывать не только успешные, но и неуспешные запуски",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Необязательный путь для записи Markdown-файла; таблица также выводится в stdout",
    )
    args = parser.parse_args()

    summary_path = args.summary.expanduser().resolve()
    if not summary_path.is_file():
        raise SystemExit(f"Файл summary.parquet не найден: {summary_path}")

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
        print(f"\nЗаписано: {output_path}")


if __name__ == "__main__":
    main()
