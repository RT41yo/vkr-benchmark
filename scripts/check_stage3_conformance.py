#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = REPO_ROOT / "results" / "stage3" / "matched_author_normalized"

EXPECTED_STATUS = {
    "bins_b3": "core_principle_preserved_expected_partition_divergence",
    "huffman_e3": "strong_conformance_with_localized_candidate_tree_divergence",
    "arithmetic_t0.9_k300": "core_principle_preserved_with_expected_reference_and_numeric_divergence",
    "arithmetic_t1.0_k50256": "strong_distributional_conformance_with_sequence_sensitivity",
}


def main() -> int:
    analysis_path = RESULT_DIR / "conformance_analysis.json"
    table_path = RESULT_DIR / "conformance_table.csv"
    if not analysis_path.exists() or not table_path.exists():
        raise SystemExit("Step 3.13 outputs missing; run scripts/analyze_stage3_conformance.py")

    result = json.loads(analysis_path.read_text(encoding="utf-8"))
    if result.get("input_pair_count") != 32:
        raise SystemExit("expected 32 matched input pairs")
    if result.get("exact_normalized_decode_count") != 32:
        raise SystemExit("normalized exact decode gate failed")
    if result.get("carrier_lengths_matched") is not True:
        raise SystemExit("carrier alignment gate failed")
    if result.get("same_secret_stream_verified") is not True:
        raise SystemExit("secret stream pairing gate failed")
    if result.get("cross_method_findings", {}).get("benchmark_native_kl_infinite_pairs") != 32:
        raise SystemExit("expected benchmark-native KL support mismatch in 32/32 matched runs")
    if result.get("cross_method_findings", {}).get("core_principle_preserved_for_all_representative_points") is not True:
        raise SystemExit("core-principle conformance conclusion missing")

    statuses = {
        item["point_id"]: item["classification"]["status"]
        for item in result.get("point_results", [])
    }
    if statuses != EXPECTED_STATUS:
        raise SystemExit(f"unexpected point classifications: {statuses}")

    with table_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 4:
        raise SystemExit("conformance_table.csv must contain four point rows")

    # Strong Huffman localization evidence: six pairs are exactly identical.
    huffman = next(item for item in result["point_results"] if item["point_id"] == "huffman_e3")
    if huffman["statistics"]["token_sequence"]["exact_pair_count"] != 6:
        raise SystemExit("expected 6/8 exact Huffman token sequences")
    if huffman["statistics"]["exact_sequence_subgroup"]["max_absolute_capacity_delta"] != 0.0:
        raise SystemExit("exact Huffman token sequences must have identical capacity")

    # Bins capacity is a structural invariant of b=3 despite different partition identity.
    bins = next(item for item in result["point_results"] if item["point_id"] == "bins_b3")
    if bins["statistics"]["capacity"]["max_absolute_pair_delta"] != 0.0:
        raise SystemExit("Bins b=3 must preserve exactly 3 bits/token in all matched pairs")

    if result.get("ready_for_step_3_14_finalization") is not True:
        raise SystemExit("Step 3.13 result not marked ready for Step 3.14")

    print("Stage 3 Step 3.13 conformance summary")
    print("matched pairs: 32/32")
    print("normalized exact token-ID decode: 32/32")
    print("core method principle preserved: 4/4 representative points")
    print("Huffman exact token sequences: 6/8")
    print("Bins b=3 capacity parity: exact 8/8")
    print("benchmark-native D_KL(P_reference || Q_stego): +inf in 32/32")
    print("dual-KL recommendation: retain support diagnostic; pair with reverse KL + TVD")
    print("Stage 3 Step 3.13: READY FOR STEP 3.14 FINALIZATION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
