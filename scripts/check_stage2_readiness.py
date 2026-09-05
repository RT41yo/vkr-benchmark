#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vkr_benchmark.validation import evaluate_stage2_readiness, format_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the technical completion criteria for Stage 2 of VKR Benchmark."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=REPO_ROOT / "results" / "summary.parquet",
        help="Path to the Stage-2 summary.parquet file.",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Also execute the full pytest suite as part of the readiness gate.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path for a machine-readable readiness report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate_stage2_readiness(
        REPO_ROOT,
        summary_path=args.summary,
        include_git_check=True,
        include_tests=args.run_tests,
    )
    print(format_report(report), end="")

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(report.to_json(), encoding="utf-8")
        print(f"readiness json: {args.json_output}")

    return 0 if report.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
