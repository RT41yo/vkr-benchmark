#!/usr/bin/env python3
"""Execute the pinned original ADG helpers and confirm the public ``near`` behavior.

The script is read-only with respect to the author checkout.  It deliberately
checks the source behavior rather than modifying the suspected ``near`` typo.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from stage4_adg_conformance import (
    REFERENCE_COMMIT,
    REFERENCE_REPOSITORY,
    paper_near,
    repository_near,
)

DEFAULT_REFERENCE_DIR = REPO_ROOT / "external" / "ADG-steganography"


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
    required = ("adg-stega.py", "utils.py", "lm.py")
    for name in required:
        if not (reference_dir / name).is_file():
            raise FileNotFoundError(f"missing pinned reference file: {name}")
    head = _git(reference_dir, "rev-parse", "HEAD")
    if head != REFERENCE_COMMIT:
        raise RuntimeError(f"expected ADG HEAD {REFERENCE_COMMIT}, got {head}")
    tracked_diff = _git(reference_dir, "diff", "--name-only")
    if tracked_diff:
        raise RuntimeError(
            "tracked files in author checkout are modified before conformance run: "
            + tracked_diff.replace("\n", ", ")
        )
    return {"head": head, "tracked_diff": tracked_diff}


def _load_author_module(reference_dir: Path) -> Any:
    sys.path.insert(0, str(reference_dir))
    path = reference_dir / "adg-stega.py"
    spec = importlib.util.spec_from_file_location("stage4_pinned_author_adg", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned author ADG: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def evaluate(reference_dir: Path) -> dict[str, object]:
    checkout = _validate_checkout(reference_dir)
    author = _load_author_module(reference_dir)

    near_cases = [
        ([0.18, 0.11], 0.16),
        ([0.21, 0.17, 0.08], 0.19),
        ([0.14, 0.10, 0.06, 0.03], 0.115),
        ([0.30, 0.20, 0.10], 0.20),
        ([0.12], 0.07),
    ]
    case_results = []
    literal_mismatches = 0
    paper_mismatches = 0
    for values, target in near_cases:
        author_index = int(author.near(list(values), float(target)))
        literal_index = repository_near(list(values), float(target))
        paper_index = paper_near(list(values), float(target))
        if author_index != literal_index:
            literal_mismatches += 1
        if author_index != paper_index:
            paper_mismatches += 1
        case_results.append(
            {
                "values_desc": values,
                "target": target,
                "author_index": author_index,
                "literal_repository_index": literal_index,
                "paper_index": paper_index,
            }
        )

    bit_failures = []
    for width in range(1, 9):
        for value in range(1 << width):
            bits = author.int2bits(value, width)
            recovered = int(author.bits2int(bits))
            if recovered != value:
                bit_failures.append(
                    {"width": width, "value": value, "bits": list(bits), "recovered": recovered}
                )
                break

    tracked_diff_after = _git(reference_dir, "diff", "--name-only")
    if tracked_diff_after:
        raise RuntimeError(
            "author checkout changed during conformance run: "
            + tracked_diff_after.replace("\n", ", ")
        )

    status = (
        "pass"
        if literal_mismatches == 0
        and paper_mismatches > 0
        and not bit_failures
        and tracked_diff_after == ""
        else "fail"
    )
    return {
        "status": status,
        "comparison_kind": "direct-original-author-helper-vs-paper-rule",
        "reference_repository": REFERENCE_REPOSITORY,
        "reference_commit": REFERENCE_COMMIT,
        "reference_dir": str(reference_dir),
        "checkout": checkout,
        "near_case_count": len(near_cases),
        "author_vs_literal_repository_mismatches": literal_mismatches,
        "author_vs_paper_mismatches": paper_mismatches,
        "near_cases": case_results,
        "bit_conversion_failure_count": len(bit_failures),
        "bit_conversion_failures": bit_failures,
        "author_checkout_unchanged": tracked_diff_after == "",
        "interpretation": (
            "PASS confirms that the pinned author file executes the literal public "
            "near() behavior, that at least one diagnostic case differs from the "
            "Algorithm-1 nearest-probability rule, and that little-endian bit helpers "
            "round-trip exactly."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.reference_dir.resolve())
    print("Stage 4 ADG direct author near() conformance")
    print("reference:", f"{result['reference_repository']}@{result['reference_commit']}")
    print("near cases:", result["near_case_count"])
    print(
        "author vs literal repository mismatches:",
        result["author_vs_literal_repository_mismatches"],
    )
    print("author vs paper mismatches:", result["author_vs_paper_mismatches"])
    print("bit conversion failures:", result["bit_conversion_failure_count"])
    print("author checkout unchanged:", result["author_checkout_unchanged"])
    print("status:", str(result["status"]).upper())
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("json:", args.json_output)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
