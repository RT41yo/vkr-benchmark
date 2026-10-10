#!/usr/bin/env python3
"""Compare normalized Discop with an independent pinned-source Python oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stage4_discop_conformance import (
    REFERENCE_COMMIT,
    REFERENCE_REPOSITORY,
    fixture_suite,
    normalized_step,
    reference_decode_step,
    reference_encode_step,
)


def evaluate(random_cases: int = 128) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    fixtures = fixture_suite(random_cases)
    for fixture in fixtures:
        reference_encoded = reference_encode_step(fixture)
        reference_decoded = reference_decode_step(fixture, reference_encoded.token_id)
        normalized_encoded, normalized_decoded = normalized_step(fixture)

        checks = {
            "encoded_token": normalized_encoded.token_id == reference_encoded.token_id,
            "encoded_bits": normalized_encoded.bits == reference_encoded.bits,
            "encoder_rng_draws": normalized_encoded.rng_draws == reference_encoded.rng_draws,
            "reference_roundtrip": reference_decoded.bits == reference_encoded.bits,
            "decoded_bits": normalized_decoded.bits == reference_decoded.bits,
            "decoder_rng_draws": normalized_decoded.rng_draws == reference_decoded.rng_draws,
        }
        if not all(checks.values()):
            failures.append(
                {
                    "fixture_id": fixture.fixture_id,
                    "checks": checks,
                    "reference_encoded": {
                        "token_id": reference_encoded.token_id,
                        "bits": reference_encoded.bits,
                        "rng_draws": reference_encoded.rng_draws,
                    },
                    "reference_decoded": {
                        "bits": reference_decoded.bits,
                        "rng_draws": reference_decoded.rng_draws,
                    },
                    "normalized_encoded": {
                        "token_id": normalized_encoded.token_id,
                        "bits": normalized_encoded.bits,
                        "rng_draws": normalized_encoded.rng_draws,
                    },
                    "normalized_decoded": {
                        "bits": normalized_decoded.bits,
                        "rng_draws": normalized_decoded.rng_draws,
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
        "failures": failures,
        "limitation": (
            "The oracle is an independent Python translation of the pinned Cython "
            "step logic; this check does not execute the original compiled Cython module."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-cases", type=int, default=128)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.random_cases < 0:
        parser.error("--random-cases must be non-negative")

    result = evaluate(args.random_cases)
    print("Stage 4 Discop core conformance")
    print("reference:", result["reference_repository"])
    print("commit:", result["reference_commit"])
    print("comparison:", result["comparison_kind"])
    print("fixtures:", result["fixture_count"])
    print("failures:", result["failure_count"])
    print("status:", str(result["status"]).upper())
    print("note:", result["limitation"])

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("json:", args.json_output)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
