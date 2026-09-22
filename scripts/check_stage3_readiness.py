#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vkr_benchmark.validation.stage3 import evaluate_stage3_readiness, format_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Stage-3 reproducibility/conformance closeout criteria.")
    parser.add_argument("--run-tests", action="store_true", help="Execute the complete pytest suite as part of the closeout gate.")
    parser.add_argument("--json-output", type=Path, help="Optional path for a machine-readable readiness report.")
    parser.add_argument(
        "--allow-missing-parquet",
        action="store_true",
        help="Development-only mode for environments without pyarrow. Final Stage-3 closeout must not use this flag.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate_stage3_readiness(
        REPO_ROOT,
        include_tests=args.run_tests,
        require_parquet=not args.allow_missing_parquet,
    )
    print(format_report(report), end="")
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(report.to_json(), encoding="utf-8")
        print(f"readiness json: {args.json_output}")
    return 0 if report.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
