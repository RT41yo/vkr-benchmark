#!/usr/bin/env python3
"""Execute the pinned original RRC ``rrc_core.py`` against normalized fixtures.

No author file is modified.  The runner expects a clean checkout at
``external/RRC_steganography`` with HEAD exactly equal to the Stage-4 pinned
commit.  A fake language-model stepper supplies already-final P_reference rows,
so the comparison isolates RRC itself and requires no author model download.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stage4_rrc_conformance import (
    AuthorFixtureStepper,
    REFERENCE_COMMIT,
    REFERENCE_REPOSITORY,
    fixture_suite,
    normalized_run,
)

DEFAULT_REFERENCE_DIR = REPO_ROOT / "external" / "RRC_steganography"


def _git(checkout: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "git command failed")
    return proc.stdout.strip()


def _validate_checkout(reference_dir: Path) -> dict[str, object]:
    if not reference_dir.is_dir():
        raise FileNotFoundError(f"reference checkout not found: {reference_dir}")
    for required in ("rrc_core.py", "RRC_embed.py", "RRC_extract.py", "test_roundtrip.py"):
        if not (reference_dir / required).is_file():
            raise FileNotFoundError(f"missing pinned reference file: {required}")
    head = _git(reference_dir, "rev-parse", "HEAD")
    if head != REFERENCE_COMMIT:
        raise RuntimeError(f"expected RRC HEAD {REFERENCE_COMMIT}, got {head}")
    tracked_diff = _git(reference_dir, "diff", "--name-only")
    if tracked_diff:
        raise RuntimeError(
            "tracked files in external/RRC_steganography are modified before "
            "conformance run: " + tracked_diff.replace("\n", ", ")
        )
    return {"head": head, "tracked_diff": tracked_diff}


def _load_author_core(reference_dir: Path) -> Any:
    path = reference_dir / "rrc_core.py"
    spec = importlib.util.spec_from_file_location("stage4_pinned_author_rrc_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned author core: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _author_run(core: Any, fixture) -> dict[str, object]:
    # The pinned implementation uses the process Decimal context.  Set it by
    # its own helper before each independent sender/receiver execution.
    precision = int(core.set_precision(fixture.payload_bits, fixture.precision_margin))
    secret_text = "".join(str(bit) for bit in fixture.secret_bits)
    embedded = core.embed_message(
        AuthorFixtureStepper(fixture),
        None,
        secret_text,
        random.Random(fixture.seed),
        fixture.vocab_size,
        max_tokens=fixture.max_tokens,
        verify=fixture.verify_termination,
    )
    core.set_precision(fixture.payload_bits, fixture.precision_margin)
    recovered_text = core.extract_message(
        AuthorFixtureStepper(fixture),
        None,
        embedded["token_ids"],
        fixture.payload_bits,
        random.Random(fixture.seed),
        fixture.vocab_size,
    )
    return {
        "token_ids": tuple(int(x) for x in embedded["token_ids"]),
        "recovered_bits": tuple(int(bit) for bit in recovered_text),
        "precision_digits": precision,
        "termination_retries": int(embedded["extra_tokens"]),
    }


def evaluate(reference_dir: Path, random_cases: int = 64) -> dict[str, object]:
    checkout = _validate_checkout(reference_dir)
    core = _load_author_core(reference_dir)
    fixtures = fixture_suite(random_cases)
    failures: list[dict[str, object]] = []
    for fixture in fixtures:
        author = _author_run(core, fixture)
        normalized = normalized_run(fixture)
        normalized_payload = {
            "token_ids": normalized.token_ids,
            "recovered_bits": normalized.recovered_bits,
            "precision_digits": normalized.precision_digits,
            "termination_retries": normalized.termination_retries,
        }
        checks = {
            "token_ids": normalized_payload["token_ids"] == author["token_ids"],
            "author_roundtrip": author["recovered_bits"] == fixture.secret_bits,
            "normalized_roundtrip": normalized.recovered_bits == fixture.secret_bits,
            "decoded_bits": normalized_payload["recovered_bits"] == author["recovered_bits"],
            "precision_digits": (
                normalized_payload["precision_digits"] == author["precision_digits"]
            ),
            "termination_retries": (
                normalized_payload["termination_retries"] == author["termination_retries"]
            ),
        }
        if not all(checks.values()):
            failures.append(
                {
                    "fixture_id": fixture.fixture_id,
                    "payload_bits": fixture.payload_bits,
                    "checks": checks,
                    "author": author,
                    "normalized": normalized_payload,
                }
            )

    tracked_diff_after = _git(reference_dir, "diff", "--name-only")
    if tracked_diff_after:
        raise RuntimeError(
            "author checkout changed during conformance run: "
            + tracked_diff_after.replace("\n", ", ")
        )

    return {
        "status": "pass" if not failures else "fail",
        "comparison_kind": "original-author-rrc-core-vs-normalized",
        "reference_repository": REFERENCE_REPOSITORY,
        "reference_commit": REFERENCE_COMMIT,
        "reference_dir": str(reference_dir),
        "checkout": checkout,
        "fixture_count": len(fixtures),
        "random_fixture_count": random_cases,
        "failure_count": len(failures),
        "failures": failures,
        "author_checkout_unchanged": tracked_diff_after == "",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--random-cases", type=int, default=64)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.random_cases < 0:
        parser.error("--random-cases must be non-negative")

    result = evaluate(args.reference_dir.resolve(), args.random_cases)
    print("Stage 4 RRC direct author conformance")
    print("reference:", f"{result['reference_repository']}@{result['reference_commit']}")
    print("fixtures:", result["fixture_count"])
    print("failures:", result["failure_count"])
    print("author checkout unchanged:", result["author_checkout_unchanged"])
    print("status:", str(result["status"]).upper())
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("json:", args.json_output)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
