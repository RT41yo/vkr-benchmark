#!/usr/bin/env python3
"""Compare normalized RRC with an independent pinned-source Python oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stage4_rrc_conformance import (
    REFERENCE_COMMIT,
    REFERENCE_REPOSITORY,
    fixture_suite,
    normalized_run,
    reference_run,
)


def evaluate(random_cases: int = 64) -> dict[str, object]:
    fixtures = fixture_suite(random_cases)
    failures: list[dict[str, object]] = []
    max_tokens = 0
    max_retries = 0
    for fixture in fixtures:
        reference = reference_run(fixture)
        normalized = normalized_run(fixture)
        checks = {
            "token_ids": normalized.token_ids == reference.token_ids,
            "reference_roundtrip": reference.recovered_bits == fixture.secret_bits,
            "normalized_roundtrip": normalized.recovered_bits == fixture.secret_bits,
            "decoded_bits": normalized.recovered_bits == reference.recovered_bits,
            "precision_digits": normalized.precision_digits == reference.precision_digits,
            "termination_retries": (
                normalized.termination_retries == reference.termination_retries
            ),
        }
        max_tokens = max(max_tokens, len(reference.token_ids))
        max_retries = max(max_retries, reference.termination_retries)
        if not all(checks.values()):
            failures.append(
                {
                    "fixture_id": fixture.fixture_id,
                    "payload_bits": fixture.payload_bits,
                    "seed": fixture.seed,
                    "checks": checks,
                    "reference": {
                        "token_ids": reference.token_ids,
                        "recovered_bits": reference.recovered_bits,
                        "precision_digits": reference.precision_digits,
                        "termination_retries": reference.termination_retries,
                    },
                    "normalized": {
                        "token_ids": normalized.token_ids,
                        "recovered_bits": normalized.recovered_bits,
                        "precision_digits": normalized.precision_digits,
                        "termination_retries": normalized.termination_retries,
                    },
                }
            )

    return {
        "status": "pass" if not failures else "fail",
        "comparison_kind": "pinned-source-oracle-vs-normalized",
        "reference_repository": REFERENCE_REPOSITORY,
        "reference_commit": REFERENCE_COMMIT,
        "fixture_count": len(fixtures),
        "random_fixture_count": random_cases,
        "failure_count": len(failures),
        "max_generated_tokens": max_tokens,
        "max_termination_retries": max_retries,
        "failures": failures,
        "limitation": (
            "The oracle is an independent Python transcription of the pinned "
            "rrc_core arithmetic and does not execute the original author file. "
            "Use run_stage4_rrc_author_conformance.py for that direct check."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-cases", type=int, default=64)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.random_cases < 0:
        parser.error("--random-cases must be non-negative")

    result = evaluate(args.random_cases)
    print("Stage 4 RRC core conformance")
    print("reference:", f"{result['reference_repository']}@{result['reference_commit']}")
    print("fixtures:", result["fixture_count"])
    print("failures:", result["failure_count"])
    print("max generated tokens:", result["max_generated_tokens"])
    print("max termination retries:", result["max_termination_retries"])
    print("status:", str(result["status"]).upper())
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("json:", args.json_output)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
