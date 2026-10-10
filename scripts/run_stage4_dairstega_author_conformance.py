#!/usr/bin/env python3
"""Direct checks against the pinned original DAIRstega source without editing it.

The public encoder keeps its interval allocation inline inside a nested function,
so the direct check executes the original top-level bit/prefix helpers and also
verifies the exact source fragments that define candidate restriction,
linear/sqrt mapping, integer allocation, residual handling and point selection.
The synthetic paper-vs-repository checker separately executes a literal
transcription of those frozen allocation statements on many distributions.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from stage4_dairstega_conformance import (
    REFERENCE_COMMIT,
    REFERENCE_REPOSITORY,
    repository_num_same_from_beg,
)

DEFAULT_REFERENCE_DIR = REPO_ROOT / "external" / "DAIRstega"


def _git(checkout: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(checkout), *args], check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "git command failed")
    return proc.stdout.strip()


def _validate_checkout(reference_dir: Path) -> dict[str, object]:
    if not reference_dir.is_dir():
        raise FileNotFoundError(f"reference checkout not found: {reference_dir}")
    source = reference_dir / "DAIRstega_t.py"
    manuscript = reference_dir / "Manuscript.pdf"
    if not source.is_file():
        raise FileNotFoundError(f"missing pinned reference file: {source.name}")
    if not manuscript.is_file():
        raise FileNotFoundError(f"missing pinned reference file: {manuscript.name}")
    head = _git(reference_dir, "rev-parse", "HEAD")
    if head != REFERENCE_COMMIT:
        raise RuntimeError(f"expected DAIRstega HEAD {REFERENCE_COMMIT}, got {head}")
    tracked_diff = _git(reference_dir, "diff", "--name-only")
    if tracked_diff:
        raise RuntimeError("tracked author files are modified before conformance run: " + tracked_diff.replace("\n", ", "))
    return {"head": head, "tracked_diff": tracked_diff}


def _load_original_helpers(source_path: Path):
    source_text = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source_text, filename=str(source_path))
    wanted = {"bits2int", "int2bits", "num_same_from_beg"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    found = {node.name for node in nodes}
    if found != wanted:
        raise RuntimeError(f"missing original helpers: {sorted(wanted - found)}")
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {}
    exec(compile(module, str(source_path), "exec"), namespace, namespace)
    return SimpleNamespace(**{name: namespace[name] for name in wanted}), source_text


def evaluate(reference_dir: Path) -> dict[str, object]:
    checkout = _validate_checkout(reference_dir)
    helpers, source_text = _load_original_helpers(reference_dir / "DAIRstega_t.py")

    bit_failures = []
    for width in range(1, 13):
        for value in range(1 << min(width, 8)):
            bits = helpers.int2bits(value, width)
            recovered = int(helpers.bits2int(bits))
            if recovered != value:
                bit_failures.append({"width": width, "value": value, "bits": list(bits), "recovered": recovered})
                break

    prefix_cases = [
        ([0, 0, 0, 0], [0, 0, 1, 1]),
        ([1, 0, 1, 0], [1, 0, 1, 1]),
        ([1, 1, 0, 1], [1, 1, 0, 1]),
    ]
    helper_mismatches = 0
    singleton_paper_mismatches = 0
    prefix_results = []
    for left, right in prefix_cases:
        author = int(helpers.num_same_from_beg(list(left), list(right)))
        literal = repository_num_same_from_beg(list(left), list(right))
        paper = len(left)
        for i, (a, b) in enumerate(zip(left, right)):
            if a != b:
                paper = i
                break
        if author != literal:
            helper_mismatches += 1
        if left == right and author != paper:
            singleton_paper_mismatches += 1
        prefix_results.append({"left": left, "right": right, "author": author, "literal": literal, "paper": paper})

    required_fragments = [
        "max_val = 2 ** PRECISION",
        "if prob[-1] < cur_threshold:",
        "k = max(2, (prob < cur_threshold).nonzero()[0].item())",
        "prob = prob[:top_k]",
        'if map == "sqrt":',
        "torch.round(torch.sqrt(prob), decimals=4)",
        "prob = prob / prob.sum()",
        "prob *= cur_int_range",
        "prob = prob.round().long()",
        "cum_probs += cur_int_range - cum_probs[-1]",
        "message_idx = bits2int(reversed(message_bits))",
        "selection = (cum_probs > message_idx).nonzero()[0].item()",
    ]
    missing_fragments = [fragment for fragment in required_fragments if fragment not in source_text]

    tracked_diff_after = _git(reference_dir, "diff", "--name-only")
    if tracked_diff_after:
        raise RuntimeError("author checkout changed during conformance run: " + tracked_diff_after.replace("\n", ", "))

    status = "pass" if not bit_failures and helper_mismatches == 0 and singleton_paper_mismatches > 0 and not missing_fragments and tracked_diff_after == "" else "fail"
    return {
        "status": status,
        "comparison_kind": "direct-original-author-helpers-and-source-guard",
        "reference_repository": REFERENCE_REPOSITORY,
        "reference_commit": REFERENCE_COMMIT,
        "reference_dir": str(reference_dir),
        "checkout": checkout,
        "bit_conversion_failure_count": len(bit_failures),
        "bit_conversion_failures": bit_failures,
        "prefix_case_count": len(prefix_cases),
        "author_vs_literal_prefix_mismatch_count": helper_mismatches,
        "singleton_author_vs_paper_mismatch_count": singleton_paper_mismatches,
        "prefix_cases": prefix_results,
        "required_source_fragment_count": len(required_fragments),
        "missing_source_fragments": missing_fragments,
        "author_checkout_unchanged": tracked_diff_after == "",
        "interpretation": (
            "PASS confirms the directly executed pinned bit/prefix helpers, the singleton-prefix "
            "off-by-one behavior, and the exact source statements used by the separate literal "
            "allocation conformance implementation."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.reference_dir.resolve())
    print("Stage 4 DAIRstega direct author conformance")
    print("reference:", f"{result['reference_repository']}@{result['reference_commit']}")
    print("bit conversion failures:", result["bit_conversion_failure_count"])
    print("author vs literal prefix mismatches:", result["author_vs_literal_prefix_mismatch_count"])
    print("singleton author vs paper mismatches:", result["singleton_author_vs_paper_mismatch_count"])
    print("missing source fragments:", len(result["missing_source_fragments"]))
    print("author checkout unchanged:", result["author_checkout_unchanged"])
    print("status:", str(result["status"]).upper())
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("json:", args.json_output)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
