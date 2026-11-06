#!/usr/bin/env python3
"""Prepare the frozen Stage-6 main prompt registry from Stage-3 CNN/DM contexts.

The Stage-5 pilot used selection ranks 0..7.  The main experiment deliberately
uses the next 24 deterministic Stage-3 records, ranks 8..31, so parameter
selection and final evaluation do not reuse the same prompts.
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
FIRST_SELECTION_RANK = 8
PROMPT_COUNT = 24
LAST_SELECTION_RANK = FIRST_SELECTION_RANK + PROMPT_COUNT - 1


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prepare_records(source_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        lines = source_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConfigurationError(
            f"cannot read Stage-3 generated contexts {source_path}: {exc}"
        ) from exc

    selected: list[dict[str, Any]] = []
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

        try:
            selection_rank = int(raw["selection_rank"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"context record at {source_path}:{line_number} has invalid selection_rank"
            ) from exc
        if not FIRST_SELECTION_RANK <= selection_rank <= LAST_SELECTION_RANK:
            continue

        if raw.get("pilot", False):
            raise ConfigurationError(
                f"Stage-6 main context rank {selection_rank} is marked as Stage-5 pilot"
            )
        context = raw.get("context")
        expected_hash = raw.get("context_sha256")
        if not isinstance(context, str) or not context:
            raise ConfigurationError(
                f"Stage-6 context at line {line_number} has no non-empty context"
            )
        if not isinstance(expected_hash, str) or _sha256_text(context) != expected_hash:
            raise ConfigurationError(
                f"Stage-6 context hash mismatch at line {line_number}"
            )
        selected.append(raw)

    selected.sort(key=lambda item: int(item["selection_rank"]))
    expected_ranks = list(range(FIRST_SELECTION_RANK, LAST_SELECTION_RANK + 1))
    actual_ranks = [int(item["selection_rank"]) for item in selected]
    if actual_ranks != expected_ranks:
        raise ConfigurationError(
            f"Stage-6 selection ranks must be {expected_ranks}, got {actual_ranks}"
        )

    prompts: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for index, raw in enumerate(selected, start=1):
        prompt_id = f"s6p{index:04d}"
        context = str(raw["context"])
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
                "context_sha256": str(raw["context_sha256"]),
                "context_char_count": int(raw["context_char_count"]),
            }
        )

    manifest = {
        "schema_version": "stage6.main_prompts.v1",
        "source": SOURCE_LABEL,
        "source_revision": DATASET_REVISION,
        "source_file": "data/stage3/cnndm/generated_contexts.jsonl",
        "selection_rule": (
            "use the next 24 deterministic Stage-3 records after the Stage-5 pilot: "
            "selection_rank 8..31; none may be marked pilot"
        ),
        "stage5_pilot_selection_ranks": list(range(8)),
        "stage6_selection_ranks": expected_ranks,
        "prompt_count": PROMPT_COUNT,
        "prompt_records": manifest_rows,
    }
    return prompts, manifest


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


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
        default=REPO_ROOT / "data/stage6/main_prompts.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "results/stage5/stage6_prompt_manifest.json",
    )
    args = parser.parse_args()

    prompts, manifest = prepare_records(args.source.expanduser().resolve())
    _write_jsonl(args.output.expanduser().resolve(), prompts)
    _write_json(args.manifest.expanduser().resolve(), manifest)
    print(f"prepared {len(prompts)} Stage-6 main prompts")
    print("prompt registry:", args.output)
    print("prompt manifest:", args.manifest)


if __name__ == "__main__":
    main()
