#!/usr/bin/env python3
"""Validate the pinned CNN/DailyMail source and frozen Stage-3 context manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "reproducibility" / "cnndm_context_source.json"
EXPECTED_DATASET_REVISION = "3adf6249f0cc8409a97a4d38471529ef5f7dc496"
EXPECTED_ARTIFACT_SHA256 = "04e322d2634a96dba76bf9a6294fbbe48e0b36abeae43f13d86ba2c3bebffe4e"
EXPECTED_MATRIX_SHA256 = "67b376590c7bb5a61dfb7565fd4699567880978c0a5b5bc4d741dafde5ff200b"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _selection_digest(row_index: int, seed: int) -> str:
    material = f"stage3-cnndm-test-v1|seed={seed}|row={row_index}".encode("ascii")
    return hashlib.sha256(material).hexdigest()


def validate_source_config(config: dict[str, Any], *, matrix_path: Path) -> None:
    if config.get("schema_version") != "stage3.cnndm_context_source.v1":
        raise ValueError("unexpected source schema")
    if config.get("status") != "pinned_before_gpt2_medium_pilot":
        raise ValueError("source config is not pinned")

    paper_basis = config.get("paper_basis") or {}
    if paper_basis.get("paper_split_reported") is not False:
        raise ValueError("paper split uncertainty must remain explicit")
    if paper_basis.get("split_operationalization") != "test":
        raise ValueError("Stage-3 operationalization must pin the test split")

    dataset = config.get("dataset") or {}
    expected = {
        "repository": "abisee/cnn_dailymail",
        "revision": EXPECTED_DATASET_REVISION,
        "config": "3.0.0",
        "split": "test",
        "split_rows": 11490,
        "artifact_path": "3.0.0/test-00000-of-00001.parquet",
        "artifact_size_bytes": 30000471,
        "artifact_sha256": EXPECTED_ARTIFACT_SHA256,
    }
    for key, value in expected.items():
        if dataset.get(key) != value:
            raise ValueError(f"dataset pin mismatch for {key}: {dataset.get(key)!r}")

    extraction = config.get("context_extraction") or {}
    if extraction.get("sentence_count") != 3:
        raise ValueError("paper context must contain exactly three extracted sentences")
    if extraction.get("splitter_id") != "stage3_news_sentence_splitter_v1":
        raise ValueError("unexpected splitter id")
    if extraction.get("minimum_sentences_required") != 4:
        raise ValueError("four sentences must remain available for context+next-sentence diagnostics")

    selection = config.get("selection") or {}
    if selection.get("algorithm_id") != "sha256_rank_v1" or selection.get("seed") != 1234:
        raise ValueError("selection algorithm/seed changed")
    if selection.get("full_context_count") != 80 or selection.get("pilot_context_count") != 8:
        raise ValueError("context counts changed")

    paper_matrix = config.get("paper_matrix") or {}
    if paper_matrix.get("sha256") != EXPECTED_MATRIX_SHA256:
        raise ValueError("source config does not reference the frozen Step-3.5 matrix")
    if not matrix_path.is_file():
        raise FileNotFoundError(matrix_path)
    if _sha256_file(matrix_path) != EXPECTED_MATRIX_SHA256:
        raise ValueError("frozen paper-reproduction matrix changed after Step 3.5")


def validate_manifest(config: dict[str, Any], config_path: Path, manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != "stage3.cnndm_context_manifest.v1":
        raise ValueError("unexpected manifest schema")
    if manifest.get("status") != "frozen":
        raise ValueError("context manifest is not frozen")

    expected_config_sha = _sha256_bytes(config_path.read_bytes())
    if manifest.get("source_config_sha256") != expected_config_sha:
        raise ValueError("manifest was not generated from the current pinned source config")

    dataset = manifest.get("dataset") or {}
    source_dataset = config["dataset"]
    for key in ("repository", "revision", "config", "split", "artifact_path", "artifact_size_bytes", "artifact_sha256", "split_rows"):
        if dataset.get(key) != source_dataset.get(key):
            raise ValueError(f"manifest dataset mismatch for {key}")

    selection = manifest.get("selection") or {}
    if selection.get("algorithm_id") != "sha256_rank_v1" or selection.get("seed") != 1234:
        raise ValueError("manifest selection policy mismatch")
    if selection.get("full_context_count") != 80 or selection.get("pilot_context_count") != 8:
        raise ValueError("manifest context counts mismatch")

    contexts = manifest.get("contexts")
    if not isinstance(contexts, list) or len(contexts) != 80:
        raise ValueError("manifest must contain exactly 80 context metadata records")
    if sum(bool(item.get("pilot")) for item in contexts) != 8:
        raise ValueError("manifest must mark exactly 8 pilot contexts")

    seen_rows: set[int] = set()
    seen_ids: set[str] = set()
    for rank, item in enumerate(contexts):
        if item.get("selection_rank") != rank:
            raise ValueError("selection ranks must be contiguous and ordered")
        if bool(item.get("pilot")) != (rank < 8):
            raise ValueError("pilot flags must mark the first eight selected contexts")
        row_index = item.get("row_index")
        if not isinstance(row_index, int) or not 0 <= row_index < 11490:
            raise ValueError("invalid row index")
        if row_index in seen_rows:
            raise ValueError("duplicate row index")
        seen_rows.add(row_index)
        article_id = item.get("article_id")
        if not isinstance(article_id, str) or re.fullmatch(r"[0-9a-f]{40}", article_id) is None:
            raise ValueError("article id must be a 40-character lowercase hex source identifier")
        if article_id in seen_ids:
            raise ValueError("duplicate article id")
        seen_ids.add(article_id)
        if item.get("selection_digest") != _selection_digest(row_index, 1234):
            raise ValueError("selection digest mismatch")
        for hash_field in ("context_sha256", "human_next_sentence_sha256"):
            value = item.get(hash_field)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"invalid {hash_field}")
        if item.get("segmented_sentence_count", 0) < 4:
            raise ValueError("selected article has fewer than four segmented sentences")

    go = manifest.get("go_no_go") or {}
    for field in (
        "dataset_artifact_verified",
        "context_manifest_frozen",
        "local_context_text_generated",
        "ready_for_gpt2_medium_pilot",
    ):
        if go.get(field) is not True:
            raise ValueError(f"manifest go/no-go field {field} must be true")


def validate_local_text(manifest: dict[str, Any], text_path: Path) -> None:
    if not text_path.is_file():
        raise FileNotFoundError(f"local generated context text is missing: {text_path}")
    lines = [line for line in text_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 80:
        raise ValueError("local context JSONL must contain exactly 80 records")
    manifest_by_rank = {int(item["selection_rank"]): item for item in manifest["contexts"]}
    for rank, line in enumerate(lines):
        record = json.loads(line)
        meta = manifest_by_rank[rank]
        if record.get("selection_rank") != rank or record.get("row_index") != meta["row_index"]:
            raise ValueError("local context order does not match manifest")
        context = record.get("context")
        next_sentence = record.get("human_next_sentence")
        if not isinstance(context, str) or not isinstance(next_sentence, str):
            raise ValueError("local context record is missing text")
        if _sha256_bytes(context.encode("utf-8")) != meta["context_sha256"]:
            raise ValueError("local context hash mismatch")
        if _sha256_bytes(next_sentence.encode("utf-8")) != meta["human_next_sentence_sha256"]:
            raise ValueError("local next-sentence hash mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    try:
        config = _read_json(config_path)
        matrix_path = (REPO_ROOT / config["paper_matrix"]["path"]).resolve()
        validate_source_config(config, matrix_path=matrix_path)

        local_paths = config["local_paths"]
        manifest_path = (REPO_ROOT / local_paths["committed_manifest"]).resolve()
        text_path = (REPO_ROOT / local_paths["generated_context_text"]).resolve()
        artifact_path = (REPO_ROOT / local_paths["artifact_cache"]).resolve()

        manifest = _read_json(manifest_path)
        validate_manifest(config, config_path, manifest)
        validate_local_text(manifest, text_path)

        if not artifact_path.is_file():
            raise FileNotFoundError(f"pinned dataset artifact cache is missing: {artifact_path}")
        if artifact_path.stat().st_size != config["dataset"]["artifact_size_bytes"]:
            raise ValueError("cached dataset artifact size mismatch")
        if _sha256_file(artifact_path) != config["dataset"]["artifact_sha256"]:
            raise ValueError("cached dataset artifact SHA-256 mismatch")
    except Exception as exc:
        print(f"Stage 3 CNN/DailyMail context gate: NOT READY ({type(exc).__name__}: {exc})")
        return 1

    print("Stage 3 CNN/DailyMail context gate")
    print(f"dataset: {config['dataset']['repository']}@{config['dataset']['revision']}")
    print(f"config/split: {config['dataset']['config']} / {config['dataset']['split']}")
    print(f"artifact SHA-256: {config['dataset']['artifact_sha256']}")
    print(f"split rows: {config['dataset']['split_rows']}")
    print(f"splitter: {config['context_extraction']['splitter_id']}")
    print("frozen contexts: 80")
    print("pilot contexts: 8")
    print("paper-reported split known: False")
    print("selected split is Stage-3 operationalization: test")
    print("frozen Step-3.5 matrix unchanged: True")
    print("ready for gpt2-medium pilot: True")
    print("Stage 3 CNN/DailyMail context gate: READY")
    return 0


if __name__ == "__main__":
    sys.exit(main())
