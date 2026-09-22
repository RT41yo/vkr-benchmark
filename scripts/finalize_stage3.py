#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from stage3_closeout import (
    build_closeout_summary,
    build_comparison_rows,
    load_json,
    write_comparison_csv,
    write_comparison_parquet,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic Stage-3 closeout artifacts.")
    parser.add_argument(
        "--skip-parquet",
        action="store_true",
        help="Write JSON/CSV only. Intended for development environments without pyarrow; final closeout requires Parquet.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = REPO_ROOT / "results" / "stage3"
    figure3 = results / "paper_reproduction" / "figure3_full"
    matched = results / "matched_author_normalized"

    figure3_summary = load_json(figure3 / "summary.json")
    interpretation = load_json(figure3 / "interpretation.json")
    matched_summary = load_json(matched / "summary.json")
    conformance = load_json(matched / "conformance_analysis.json")

    rows = build_comparison_rows(conformance)
    csv_path = write_comparison_csv(results / "comparison.csv", rows)
    parquet_path = results / "comparison.parquet"
    if not args.skip_parquet:
        try:
            write_comparison_parquet(parquet_path, rows)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    closeout = build_closeout_summary(figure3_summary, interpretation, matched_summary, conformance)
    summary_path = results / "stage3_closeout_summary.json"
    summary_path.write_text(
        json.dumps(closeout.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("Stage 3 closeout artifacts")
    print(f"comparison rows: {len(rows)}")
    print(f"comparison csv: {csv_path.relative_to(REPO_ROOT)}")
    if args.skip_parquet:
        print("comparison parquet: SKIPPED (--skip-parquet)")
    else:
        print(f"comparison parquet: {parquet_path.relative_to(REPO_ROOT)}")
    print(f"closeout summary: {summary_path.relative_to(REPO_ROOT)}")
    print(f"ready for Stage 4: {closeout.ready_for_stage4}")
    return 0 if closeout.ready_for_stage4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
