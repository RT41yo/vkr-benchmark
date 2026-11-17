#!/usr/bin/env python3
"""Extend the pinned Stage-3 CNN/DM selection to ranks 32..159 (Stage 7).

This DOES NOT write Stage-3 or Stage-6 files. Full prompt text remains local;
only manifests with identifiers/hashes may be committed to Git.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_stage3_cnndm_contexts import (  # noqa: E402
    _iter_rows, _load_config, _validate_config, build_context_records,
    ensure_artifact, sha256_file,
)
from vkr_benchmark.errors import ConfigurationError  # noqa: E402
from vkr_benchmark.steganalysis.protocol import grouped_split  # noqa: E402

FIRST_RANK = 32
TOTAL_SELECTION_COUNT = 160
SOURCE = "abisee/cnn_dailymail|config=3.0.0|split=test"
SOURCE_CONFIG = ROOT / "configs/reproducibility/cnndm_context_source.json"
STAGE3_MANIFEST = ROOT / "results/stage3/paper_reproduction/cnndm_context_manifest.json"
STAGE6_PROMPT_MANIFEST = ROOT / "results/stage5/stage6_prompt_manifest.json"
OUTPUT_PROMPTS = ROOT / "data/stage7/prompts.jsonl"
OUTPUT_SPLITS = ROOT / "results/stage7/splits"


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _validate_frozen_prefix(
    candidates: list[dict[str, Any]],
    *,
    stage3_manifest: dict[str, Any],
    stage6_manifest: dict[str, Any],
) -> None:
    historical = stage3_manifest["contexts"]
    if len(historical) != 80:
        raise ConfigurationError("Stage-3 frozen manifest must contain 80 records")
    for expected, actual in zip(historical, candidates[:80], strict=True):
        for field in ("selection_rank", "row_index", "article_id", "context_sha256"):
            if expected[field] != actual[field]:
                raise ConfigurationError(f"pinned Stage-3 prefix mismatch in {field}")
    stage6_rows = stage6_manifest["prompt_records"]
    if len(stage6_rows) != 24:
        raise ConfigurationError("Stage-6 prompt manifest must contain 24 prompts")
    for index, frozen in enumerate(stage6_rows, start=8):
        actual = candidates[index]
        if (
            frozen["stage3_selection_rank"] != index
            or frozen["row_index"] != actual["row_index"]
            or frozen["article_id"] != actual["article_id"]
            or frozen["context_sha256"] != actual["context_sha256"]
        ):
            raise ConfigurationError("Stage-6 held-out prompt differs from pinned selection")


def build_stage7_records(
    candidates: list[dict[str, Any]],
    *,
    frozen_stage6_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(candidates) != TOTAL_SELECTION_COUNT:
        raise ConfigurationError("Stage-7 preparation requires precisely 160 accepted contexts")
    if [int(item["selection_rank"]) for item in candidates] != list(range(160)):
        raise ConfigurationError("Stage-7 selection ranks must be 0..159 without gaps")
    if len(frozen_stage6_records) != 24:
        raise ConfigurationError("Stage-7 external test requires 24 frozen Stage-6 prompts")

    selected = candidates[FIRST_RANK:]
    all_articles = [str(row["article_id"]) for row in candidates]
    all_hashes = [str(row["context_sha256"]) for row in candidates]
    if len(set(all_articles)) != len(all_articles) or len(set(all_hashes)) != len(all_hashes):
        raise ConfigurationError("duplicate article or exact context in Stage-5/6/7 selection")

    prompts: list[dict[str, Any]] = []
    new_manifest_rows: list[dict[str, Any]] = []
    for index, record in enumerate(selected, start=1):
        pid = f"s7p{index:04d}"
        context = record["context"]
        if _hash_text(context) != record["context_sha256"]:
            raise ConfigurationError(f"context hash mismatch at Stage-7 {pid}")
        prompts.append({
            "prompt_id": pid, "source": SOURCE,
            "source_revision": "3adf6249f0cc8409a97a4d38471529ef5f7dc496",
            "text": context, "language": "en",
        })
        new_manifest_rows.append({
            "prompt_id": pid,
            "article_id": str(record["article_id"]),
            "row_index": int(record["row_index"]),
            "selection_rank": int(record["selection_rank"]),
            "context_sha256": record["context_sha256"],
        })

    split = grouped_split((row["prompt_id"], row["article_id"]) for row in new_manifest_rows)
    for row in new_manifest_rows:
        row["partition"] = split[row["prompt_id"]]

    test_rows = []
    for row in frozen_stage6_records:
        test_rows.append({
            "prompt_id": row["prompt_id"],
            "article_id": row["article_id"],
            "row_index": row["row_index"],
            "selection_rank": row["stage3_selection_rank"],
            "context_sha256": row["context_sha256"],
            "partition": "test",
        })

    rows = sorted(new_manifest_rows + test_rows, key=lambda row: (row["partition"], row["prompt_id"]))
    if len({row["article_id"] for row in rows}) != 152:
        raise ConfigurationError("train/validation/test article leakage or missing group")
    if len({row["context_sha256"] for row in rows}) != 152:
        raise ConfigurationError("train/validation/test exact prompt leakage")
    if len({row["prompt_id"] for row in rows}) != 152:
        raise ConfigurationError("train/validation/test prompt_id collision")
    if sum(x["partition"] == "train" for x in rows) != 96 or sum(x["partition"] == "validation" for x in rows) != 32:
        raise ConfigurationError("unexpected train/validation split")
    manifest = {
        "schema_version": "stage7.prompt_split.v1",
        "status": "proposed_before_training",
        "selection": {"source": SOURCE, "seed": 1234, "algorithm": "sha256_rank_v1", "stage7_ranks": [32,159]},
        "split": {"seed": 701, "algorithm": "sha256_order_article_group_v1", "train": 96, "validation": 32, "test": 24},
        "rows": rows,
    }
    return prompts, manifest


def prepare(*, artifact: Path | None, output_prompts: Path = OUTPUT_PROMPTS,
            output_splits: Path = OUTPUT_SPLITS, dry_run: bool = False,
            force: bool = False) -> dict[str, Any]:
    config, source_config_sha = _load_config(SOURCE_CONFIG)
    _validate_config(config)
    cache = ROOT / config["local_paths"]["artifact_cache"]
    pinned = ensure_artifact(config, cache, artifact)
    row_count, articles, article_ids = _iter_rows(pinned)
    if row_count != config["dataset"]["split_rows"]:
        raise ConfigurationError("pinned CNN/DailyMail split row count changed")
    candidates, rejected = build_context_records(
        articles, article_ids, seed=config["selection"]["seed"],
        full_count=TOTAL_SELECTION_COUNT, pilot_count=8,
        minimum_sentences=config["context_extraction"]["minimum_sentences_required"],
    )
    stage3 = json.loads(STAGE3_MANIFEST.read_text(encoding="utf-8"))
    stage6 = json.loads(STAGE6_PROMPT_MANIFEST.read_text(encoding="utf-8"))
    _validate_frozen_prefix(candidates, stage3_manifest=stage3, stage6_manifest=stage6)
    prompts, manifest = build_stage7_records(
        candidates, frozen_stage6_records=stage6["prompt_records"]
    )
    manifest["sources"] = {
        "source_config_sha256": source_config_sha,
        "dataset_parquet_sha256": sha256_file(pinned),
        "stage3_manifest_sha256": sha256_file(STAGE3_MANIFEST),
        "stage6_prompt_manifest_sha256": sha256_file(STAGE6_PROMPT_MANIFEST),
    }
    text = "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in prompts)
    manifest["local_prompt_text_sha256"] = _hash_text(text)
    if not dry_run:
        outputs = [output_prompts, output_splits / "split_manifest.json"] + [
            output_splits / f"{partition}_prompt_ids.json" for partition in ("train", "validation", "test")
        ]
        if not force and any(path.exists() for path in outputs):
            raise ConfigurationError("Stage-7 split exists; use --force only after deliberate review")
        output_prompts.parent.mkdir(parents=True, exist_ok=True)
        output_prompts.write_text(text, encoding="utf-8")
        _write_json(output_splits / "split_manifest.json", manifest)
        for partition in ("train", "validation", "test"):
            _write_json(
                output_splits / f"{partition}_prompt_ids.json",
                {"partition": partition, "prompt_ids": sorted(row["prompt_id"] for row in manifest["rows"] if row["partition"] == partition)},
            )
    return {"train": 96, "validation": 32, "test": 24, "invalid_ranked_rows_skipped": rejected,
            "artifact_sha256": manifest["sources"]["dataset_parquet_sha256"], "dry_run": dry_run}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, help="Locally available pinned CNN/DailyMail Parquet")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Explicitly replace local Stage-7 prompt/split outputs")
    args = parser.parse_args()
    try:
        result = prepare(artifact=args.artifact, dry_run=args.dry_run, force=args.force)
    except Exception as exc:
        print(f"Stage-7 prompt preparation: ERROR ({type(exc).__name__}: {exc})")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
