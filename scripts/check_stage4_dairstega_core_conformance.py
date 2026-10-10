#!/usr/bin/env python3
"""Compare normalized DAIRstega with frozen paper and public-repository semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from stage4_dairstega_conformance import (
    REFERENCE_COMMIT,
    REFERENCE_REPOSITORY,
    canonical_fixture_probabilities,
    fixture_suite,
    paper_allocate,
    point_selection,
    prefix_length_paper,
    prefix_length_repository,
    repository_allocate,
)
from vkr_benchmark.distributions import ReferenceDistribution, StepContext
from vkr_benchmark.methods.dair import DAIRConfig, _allocate, _common_prefix


def _normalized_intervals(fixture):
    reference = ReferenceDistribution(canonical_fixture_probabilities(fixture.probabilities))
    context = StepContext(0, reference)
    config = DAIRConfig(alpha=fixture.alpha, beta=fixture.beta, top_k=fixture.top_k)
    intervals = _allocate(context, config)
    return tuple((int(x.token_id), int(x.start), int(x.stop)) for x in intervals)


def _interval_payload(intervals):
    return tuple((int(x.token_id), int(x.start), int(x.stop)) for x in intervals)


def evaluate(random_cases: int = 128) -> dict[str, object]:
    fixtures = fixture_suite(random_cases)
    normalized_failures: list[dict[str, object]] = []
    repository_allocation_divergences = 0
    repository_selection_divergences = 0
    sqrt_allocation_divergences = 0
    linear_allocation_divergences = 0
    max_boundary_delta = 0

    for fixture in fixtures:
        paper = paper_allocate(fixture)
        repository = repository_allocate(fixture)
        normalized = _normalized_intervals(fixture)
        paper_payload = _interval_payload(paper)
        repository_payload = _interval_payload(repository)
        if normalized != paper_payload:
            normalized_failures.append(
                {
                    "fixture_id": fixture.fixture_id,
                    "alpha": fixture.alpha,
                    "beta": fixture.beta,
                    "normalized": normalized,
                    "paper": paper_payload,
                }
            )
        if repository_payload != paper_payload:
            repository_allocation_divergences += 1
            if fixture.beta == 0.5:
                sqrt_allocation_divergences += 1
            else:
                linear_allocation_divergences += 1
            repo_bounds = [x.stop for x in repository]
            paper_bounds = [x.stop for x in paper]
            for a, b in zip(repo_bounds, paper_bounds):
                max_boundary_delta = max(max_boundary_delta, abs(int(a) - int(b)))

        total_codes = 1 << fixture.alpha
        diagnostic_points = sorted(set([0, total_codes // 7, total_codes // 3, total_codes // 2, total_codes - 1]))
        for point in diagnostic_points:
            if point_selection(repository, point) != point_selection(paper, point):
                repository_selection_divergences += 1

    # Prefix semantics are compared independently so that allocation differences
    # do not hide the single-width edge case.
    from stage4_dairstega_conformance import Interval

    prefix_cases = [
        Interval(0, 0b00000000, 0b00100000),
        Interval(1, 0b01000000, 0b01100000),
        Interval(2, 0b10110000, 0b10111000),
        Interval(3, 0b11100110, 0b11100111),  # singleton
    ]
    prefix_case_results = []
    prefix_repository_mismatches = 0
    normalized_prefix_failures = 0
    for interval in prefix_cases:
        paper_len = prefix_length_paper(interval, 8)
        repo_len = prefix_length_repository(interval, 8)
        # Use normalized private helper only as the system under test.
        class Obj:
            start = interval.start
            stop = interval.stop
        normalized_len = len(_common_prefix(Obj(), alpha=8))
        if normalized_len != paper_len:
            normalized_prefix_failures += 1
        if repo_len != paper_len:
            prefix_repository_mismatches += 1
        prefix_case_results.append(
            {
                "start": interval.start,
                "stop": interval.stop,
                "paper_prefix_bits": paper_len,
                "repository_prefix_bits": repo_len,
                "normalized_prefix_bits": normalized_len,
            }
        )

    status = "pass" if not normalized_failures and normalized_prefix_failures == 0 and prefix_repository_mismatches > 0 else "fail"
    return {
        "status": status,
        "comparison_kind": "paper-normalized-vs-literal-public-repository",
        "reference_repository": REFERENCE_REPOSITORY,
        "reference_commit": REFERENCE_COMMIT,
        "fixture_count": len(fixtures),
        "random_fixture_count": random_cases,
        "normalized_paper_failure_count": len(normalized_failures),
        "normalized_paper_failures": normalized_failures[:20],
        "repository_allocation_divergence_count": repository_allocation_divergences,
        "repository_allocation_divergence_fraction": repository_allocation_divergences / len(fixtures),
        "repository_linear_allocation_divergence_count": linear_allocation_divergences,
        "repository_sqrt_allocation_divergence_count": sqrt_allocation_divergences,
        "repository_selection_divergence_count_on_diagnostic_points": repository_selection_divergences,
        "repository_max_boundary_delta_codes": max_boundary_delta,
        "prefix_case_count": len(prefix_cases),
        "normalized_prefix_failure_count": normalized_prefix_failures,
        "repository_prefix_mismatch_count": prefix_repository_mismatches,
        "prefix_cases": prefix_case_results,
        "interpretation": (
            "PASS requires normalized allocation/prefix semantics to match the independently "
            "implemented paper/ADR rule on every fixture. Repository divergences are diagnostic: "
            "sqrt uses four-decimal rounding and the public prefix helper under-consumes a singleton."
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
    print("Stage 4 DAIRstega paper-vs-repository conformance")
    print("reference:", f"{result['reference_repository']}@{result['reference_commit']}")
    print("fixtures:", result["fixture_count"])
    print("normalized paper failures:", result["normalized_paper_failure_count"])
    print(
        "repository allocation divergences:",
        f"{result['repository_allocation_divergence_count']}/{result['fixture_count']} "
        f"({100.0 * result['repository_allocation_divergence_fraction']:.1f}%)",
    )
    print("  linear divergences:", result["repository_linear_allocation_divergence_count"])
    print("  sqrt divergences:", result["repository_sqrt_allocation_divergence_count"])
    print("repository selection divergences on diagnostic points:", result["repository_selection_divergence_count_on_diagnostic_points"])
    print("repository max boundary delta (codes):", result["repository_max_boundary_delta_codes"])
    print("repository prefix mismatches:", result["repository_prefix_mismatch_count"])
    print("status:", str(result["status"]).upper())
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("json:", args.json_output)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
