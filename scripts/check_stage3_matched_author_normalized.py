#!/usr/bin/env python3
"""Readiness checker for Stage-3 Step 3.12 matched comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from stage3_matched_comparison_core import (  # noqa: E402
    EXPECTED_PAIR_COUNT,
    load_author_records,
    load_json,
    load_jsonl,
    sha256_file,
    validate_config,
    validate_context_records,
)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "reproducibility" / "matched_author_normalized.json"


def _check_dual_kl_contract() -> None:
    from vkr_benchmark.metrics.distribution_distortion import (
        DistributionDistortionMetrics,
        StepDistributionDistortion,
    )

    step_fields = set(StepDistributionDistortion.__dataclass_fields__)
    aggregate_fields = set(DistributionDistortionMetrics.__dataclass_fields__)
    if "kl_stego_to_ref_bits" not in step_fields:
        raise RuntimeError("ADR-0012 reverse KL is absent from per-step metric contract")
    required = {
        "kl_stego_to_ref_mean_bits",
        "kl_stego_to_ref_median_bits",
        "kl_stego_to_ref_p95_bits",
        "kl_stego_to_ref_max_bits",
        "kl_stego_to_ref_infinite_steps",
        "kl_stego_to_ref_finite_steps",
    }
    if not required.issubset(aggregate_fields):
        raise RuntimeError("ADR-0012 reverse KL is incomplete in aggregate metric contract")


def preflight(config_path: Path) -> dict:
    config = load_json(config_path)
    validate_config(config)
    manifest_path = REPO_ROOT / config["context_manifest_path"]
    if sha256_file(manifest_path) != config["context_manifest_sha256"]:
        raise RuntimeError("context manifest SHA-256 mismatch")
    manifest = load_json(manifest_path)
    local_context_path = REPO_ROOT / config["local_context_text_path"]
    if not local_context_path.is_file():
        raise FileNotFoundError(
            f"missing local frozen context text: {local_context_path}; regenerate it from the pinned dataset artifact"
        )
    validate_context_records(load_jsonl(local_context_path), manifest=manifest)
    author = load_author_records(repo_root=REPO_ROOT, config=config)
    if len(author) != EXPECTED_PAIR_COUNT:
        raise RuntimeError("author-side pair count mismatch")

    figure3 = load_json(REPO_ROOT / config["figure3_summary_path"])
    if figure3.get("execution_gate_ready_for_step_3_11_interpretation") is not True:
        raise RuntimeError("Figure-3 author sweep is not complete")
    if int(figure3.get("scheduled_run_count", -1)) != 5520:
        raise RuntimeError("unexpected committed Figure-3 scheduled run count")
    _check_dual_kl_contract()
    return config


def check_outputs(config: dict) -> dict:
    out_dir = REPO_ROOT / config["execution"]["output_directory"]
    records_path = out_dir / "records.jsonl"
    summary_path = out_dir / "summary.json"
    csv_path = out_dir / "paired_comparison.csv"
    for path in (records_path, summary_path, csv_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    records = load_jsonl(records_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if len(records) != EXPECTED_PAIR_COUNT:
        raise RuntimeError(f"expected {EXPECTED_PAIR_COUNT} matched records, got {len(records)}")
    if int(summary.get("pair_count", -1)) != EXPECTED_PAIR_COUNT:
        raise RuntimeError("summary pair count mismatch")
    if summary.get("all_carrier_lengths_matched") is not True:
        raise RuntimeError("matched carrier-length gate failed")
    if summary.get("all_normalized_token_id_decodes_exact") is not True:
        raise RuntimeError("normalized token-id recovery gate failed")
    dual = summary.get("dual_kl") or {}
    if dual.get("all_reverse_kl_present") is not True:
        raise RuntimeError("reverse KL accounting gate failed")
    if dual.get("all_forward_kl_accounted") is not True:
        raise RuntimeError("benchmark-native KL accounting gate failed")
    if summary.get("same_secret_stream_verified") is not True:
        raise RuntimeError("secret stream pairing gate failed")
    if summary.get("scientific_closeness_evaluated") is not False:
        raise RuntimeError("Step 3.12 must not pre-judge scientific closeness")
    if summary.get("ready_for_step_3_13_conformance_analysis") is not True:
        raise RuntimeError("Step 3.12 output is not ready for Step 3.13")

    for record in records:
        if (record.get("carrier_alignment") or {}).get("equal") is not True:
            raise RuntimeError(f"carrier mismatch in {record.get('pair_key')}")
        normalized = record.get("normalized") or {}
        if normalized.get("exact_token_id_decode") is not True:
            raise RuntimeError(f"decode mismatch in {record.get('pair_key')}")
        if normalized.get("kl_stego_to_ref_mean_bits") is None:
            raise RuntimeError(f"reverse KL missing in {record.get('pair_key')}")
        if normalized.get("kl_ref_to_stego_mean_bits") is None and int(
            normalized.get("kl_ref_to_stego_infinite_steps", 0)
        ) == 0:
            raise RuntimeError(f"forward KL not accounted in {record.get('pair_key')}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = preflight(config_path)

    print("Stage 3 Step 3.12 matched-comparison preflight")
    print(f"author pairs: {EXPECTED_PAIR_COUNT}/{EXPECTED_PAIR_COUNT}")
    print("carrier alignment: normalized fixed carrier length = author sentence token length")
    print("dual KL contract: PASS")
    print("metric closeness is execution gate: False")
    if args.preflight:
        print("Stage 3 Step 3.12 preflight: READY TO RUN")
        return 0

    summary = check_outputs(config)
    print(f"matched pairs: {summary['pair_count']}/{summary['expected_pair_count']}")
    print(
        "normalized exact token-ID decode: "
        f"{summary['normalized_exact_decode_count']}/{summary['pair_count']}"
    )
    print("scientific closeness evaluated: False")
    print("Stage 3 Step 3.12: READY FOR STEP 3.13 CONFORMANCE ANALYSIS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
