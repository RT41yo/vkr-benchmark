#!/usr/bin/env python3
"""Prepare the frozen Stage-5 pilot prompt registry from Stage-3 CNN/DM contexts.

No network access is used here.  The input is the already-pinned Stage-3 local
artifact ``data/stage3/cnndm/generated_contexts.jsonl``.  The script verifies
the stored context hashes and selects exactly the eight Stage-3 pilot records
(selection ranks 0..7), preserving their exact text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vkr_benchmark.errors import ConfigurationError

DATASET_REVISION = "3adf6249f0cc8409a97a4d38471529ef5f7dc496"
SOURCE_LABEL = "abisee/cnn_dailymail|config=3.0.0|split=test"
PILOT_COUNT = 8


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prepare_records(source_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        lines = source_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConfigurationError(
            f"cannot read Stage-3 generated contexts {source_path}: {exc}"
        ) from exc

    pilot: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConfigurationError(
                f"invalid JSON at {source_path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(raw, dict):
            raise ConfigurationError(
                f"context record at {source_path}:{line_number} is not an object"
            )
        if not raw.get("pilot", False):
            continue
        context = raw.get("context")
        expected_hash = raw.get("context_sha256")
        if not isinstance(context, str) or not context:
            raise ConfigurationError(
                f"pilot context at line {line_number} has no non-empty context"
            )
        if not isinstance(expected_hash, str) or _sha256_text(context) != expected_hash:
            raise ConfigurationError(
                f"pilot context hash mismatch at line {line_number}"
            )
        pilot.append(raw)

    pilot.sort(key=lambda item: int(item["selection_rank"]))
    if len(pilot) != PILOT_COUNT:
        raise ConfigurationError(
            f"expected exactly {PILOT_COUNT} Stage-3 pilot contexts, got {len(pilot)}"
        )
    expected_ranks = list(range(PILOT_COUNT))
    actual_ranks = [int(item["selection_rank"]) for item in pilot]
    if actual_ranks != expected_ranks:
        raise ConfigurationError(
            f"Stage-3 pilot selection ranks must be {expected_ranks}, got {actual_ranks}"
        )

    prompts: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for index, raw in enumerate(pilot, start=1):
        prompt_id = f"s5p{index:04d}"
        context = str(raw["context"])
        context_hash = str(raw["context_sha256"])
        prompts.append(
            {
                "prompt_id": prompt_id,
                "source": SOURCE_LABEL,
                "source_revision": DATASET_REVISION,
                "text": context,
                "language": "en",
            }
        )
        manifest_rows.append(
            {
                "prompt_id": prompt_id,
                "stage3_selection_rank": int(raw["selection_rank"]),
                "row_index": int(raw["row_index"]),
                "article_id": str(raw["article_id"]),
                "context_sha256": context_hash,
                "context_char_count": int(raw["context_char_count"]),
            }
        )

    manifest = {
        "schema_version": "stage5.pilot_prompts.v1",
        "source": SOURCE_LABEL,
        "source_revision": DATASET_REVISION,
        "source_file": "data/stage3/cnndm/generated_contexts.jsonl",
        "selection_rule": "reuse exactly the eight Stage-3 records with pilot=true, selection_rank 0..7",
        "prompt_count": PILOT_COUNT,
        "prompt_records": manifest_rows,
    }
    return prompts, manifest


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=REPO_ROOT / "data/stage3/cnndm/generated_contexts.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data/stage5/pilot_prompts.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "results/stage5/pilot_prompt_manifest.json",
    )
    args = parser.parse_args()

    prompts, manifest = prepare_records(args.source.expanduser().resolve())
    _write_jsonl(args.output.expanduser().resolve(), prompts)
    _write_json(args.manifest.expanduser().resolve(), manifest)
    print(f"prepared {len(prompts)} Stage-5 pilot prompts")
    print("prompt registry:", args.output)
    print("prompt manifest:", args.manifest)


if __name__ == "__main__":
    main()
