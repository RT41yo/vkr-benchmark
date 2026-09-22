#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage3_conformance_analysis import analyze, load_csv, write_csv

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage-3.13 matched conformance")
    parser.add_argument(
        "--config",
        default="configs/reproducibility/conformance_analysis.json",
    )
    args = parser.parse_args()

    config_path = REPO_ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    inputs = config["inputs"]
    matched_summary = json.loads((REPO_ROOT / inputs["matched_summary"]).read_text(encoding="utf-8"))
    rows = load_csv(REPO_ROOT / inputs["paired_comparison"])
    result = analyze(rows, matched_summary, config)

    out_dir = REPO_ROOT / "results" / "stage3" / "matched_author_normalized"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "conformance_analysis.json"
    table_path = out_dir / "conformance_table.csv"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    write_csv(result, table_path)

    print("Stage 3 Step 3.13 conformance/discrepancy analysis")
    for item in result["point_results"]:
        stats = item["statistics"]
        print(
            f"{item['point_id']}: {item['classification']['status']}; "
            f"ΔBPT={stats['capacity']['mean_delta_normalized_minus_author']:+.6f}; "
            f"ΔKLrev={stats['reverse_kl']['mean_delta_normalized_minus_author_bits']:+.6g} bits; "
            f"exact-seq={stats['token_sequence']['exact_pair_count']}/8"
        )
    cross = result["cross_method_findings"]
    print(
        "benchmark-native KL support diagnostic: "
        f"+inf in {cross['benchmark_native_kl_infinite_pairs']}/"
        f"{cross['benchmark_native_kl_pair_count']} matched runs"
    )
    print("overall:", result["overall_classification"])
    print("Stage 3 Step 3.13: READY FOR STEP 3.14 FINALIZATION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
