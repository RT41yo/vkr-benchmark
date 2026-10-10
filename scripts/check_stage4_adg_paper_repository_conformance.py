#!/usr/bin/env python3
"""Characterize paper-faithful ADG vs the literal pinned repository grouping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from stage4_adg_conformance import (
    REFERENCE_COMMIT,
    REFERENCE_REPOSITORY,
    explicit_q,
    fixture_suite,
    kl_bits,
    paper_near,
    repository_near,
    tvd,
)
from vkr_benchmark.distributions.adg_q import explicit_adg_q
from vkr_benchmark.distributions.types import ReferenceDistribution


def _reference_distribution(probabilities: tuple[float, ...]) -> ReferenceDistribution:
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum(dtype=np.float64)
    # ReferenceDistribution stores a full vector and a canonical token order.
    order = np.lexsort((np.arange(len(probs), dtype=np.int64), -probs))
    return ReferenceDistribution(
        probabilities=probs,
        token_order=tuple(int(x) for x in order),
    )


def evaluate(random_cases: int = 128) -> dict[str, object]:
    fixtures = fixture_suite(random_cases)
    normalized_failures: list[dict[str, object]] = []
    repository_divergences: list[dict[str, object]] = []
    max_normalized_abs = 0.0
    max_repo_tvd = 0.0
    max_repo_kl_paper_to_repo = 0.0
    max_repo_kl_repo_to_paper = 0.0

    for fixture in fixtures:
        reference = _reference_distribution(fixture.probabilities)
        # Conformance must compare algorithms on the exact same canonical
        # FP32 P_reference seen by normalized benchmark code.  ADG grouping is
        # discrete, so even tiny pre-canonicalization differences can change a
        # nearest-neighbour decision and cascade through the recursion tree.
        canonical = tuple(float(x) for x in reference.probabilities)
        paper_q = explicit_q(canonical, repository_semantics=False)
        repo_q = explicit_q(canonical, repository_semantics=True)
        normalized_q = explicit_adg_q(reference)
        max_abs = float(np.max(np.abs(normalized_q - paper_q)))
        max_normalized_abs = max(max_normalized_abs, max_abs)
        if max_abs > 1e-12:
            normalized_failures.append(
                {"fixture_id": fixture.fixture_id, "max_abs_q": max_abs}
            )

        repo_tvd = tvd(paper_q, repo_q)
        if repo_tvd > 1e-15:
            kl_pr = kl_bits(paper_q, repo_q)
            kl_rp = kl_bits(repo_q, paper_q)
            max_repo_tvd = max(max_repo_tvd, repo_tvd)
            max_repo_kl_paper_to_repo = max(max_repo_kl_paper_to_repo, kl_pr)
            max_repo_kl_repo_to_paper = max(max_repo_kl_repo_to_paper, kl_rp)
            if len(repository_divergences) < 12:
                repository_divergences.append(
                    {
                        "fixture_id": fixture.fixture_id,
                        "tvd": repo_tvd,
                        "kl_paper_to_repository_bits": kl_pr,
                        "kl_repository_to_paper_bits": kl_rp,
                    }
                )

    # A small direct witness for the exact near() discrepancy.
    near_values = [0.18, 0.11]
    near_target = 0.16
    paper_index = paper_near(near_values, near_target)
    repository_index = repository_near(near_values, near_target)

    divergent_count = 0
    for fixture in fixtures:
        reference = _reference_distribution(fixture.probabilities)
        canonical = tuple(float(x) for x in reference.probabilities)
        if tvd(
            explicit_q(canonical, repository_semantics=False),
            explicit_q(canonical, repository_semantics=True),
        ) > 1e-15:
            divergent_count += 1

    status = (
        "pass"
        if not normalized_failures
        and paper_index == 0
        and repository_index == 1
        and divergent_count > 0
        else "fail"
    )
    return {
        "status": status,
        "comparison_kind": "paper-algorithm1-vs-public-repository-near",
        "reference_repository": REFERENCE_REPOSITORY,
        "reference_commit": REFERENCE_COMMIT,
        "fixture_count": len(fixtures),
        "random_fixture_count": random_cases,
        "normalized_paper_failure_count": len(normalized_failures),
        "normalized_paper_max_abs_q": max_normalized_abs,
        "repository_divergent_fixture_count": divergent_count,
        "repository_divergent_fixture_fraction": divergent_count / len(fixtures),
        "repository_max_tvd_vs_paper": max_repo_tvd,
        "repository_max_kl_paper_to_repository_bits": max_repo_kl_paper_to_repo,
        "repository_max_kl_repository_to_paper_bits": max_repo_kl_repo_to_paper,
        "near_witness": {
            "values_desc": near_values,
            "target": near_target,
            "paper_index": paper_index,
            "paper_value": near_values[paper_index],
            "repository_index": repository_index,
            "repository_value": near_values[repository_index],
        },
        "normalized_failures": normalized_failures,
        "repository_divergence_examples": repository_divergences,
        "interpretation": (
            "PASS means normalized explicit-Q matches an independent Algorithm-1 "
            "implementation while the literal public-repository near() discrepancy "
            "is observed and quantified; repository divergence is expected here, not "
            "treated as a normalized-method failure."
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
    print("Stage 4 ADG paper-vs-repository conformance")
    print("reference:", f"{result['reference_repository']}@{result['reference_commit']}")
    print("fixtures:", result["fixture_count"])
    print("normalized paper failures:", result["normalized_paper_failure_count"])
    print("normalized max|Q-paperQ|:", f"{result['normalized_paper_max_abs_q']:.3e}")
    print(
        "repository divergences:",
        f"{result['repository_divergent_fixture_count']}/{result['fixture_count']}",
        f"({100.0 * result['repository_divergent_fixture_fraction']:.1f}%)",
    )
    print("repository max TVD vs paper:", f"{result['repository_max_tvd_vs_paper']:.6f}")
    witness = result["near_witness"]
    print(
        "near witness:",
        f"target={witness['target']} paper->{witness['paper_value']} "
        f"repository->{witness['repository_value']}",
    )
    print("status:", str(result["status"]).upper())
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("json:", args.json_output)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
