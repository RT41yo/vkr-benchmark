from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy

import pytest

from vkr_benchmark.errors import ConfigurationError


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/prepare_stage5_pilot_prompts.py"


def _prepare_records():
    namespace = runpy.run_path(str(SCRIPT), run_name="stage5_prompt_prepare_test")
    return namespace["prepare_records"]


def _record(rank: int) -> dict[str, object]:
    text = f"Prompt context {rank}. Second sentence. Third sentence."
    return {
        "article_id": f"article-{rank}",
        "context": text,
        "context_char_count": len(text),
        "context_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "pilot": True,
        "row_index": 100 + rank,
        "selection_rank": rank,
    }


def test_prepare_stage5_prompts_reuses_exact_eight_stage3_pilot_records(tmp_path: Path) -> None:
    source = tmp_path / "contexts.jsonl"
    records = [_record(rank) for rank in reversed(range(8))]
    source.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    prompts, manifest = _prepare_records()(source)

    assert [item["prompt_id"] for item in prompts] == [f"s5p{i:04d}" for i in range(1, 9)]
    assert [item["text"] for item in prompts] == [
        f"Prompt context {rank}. Second sentence. Third sentence." for rank in range(8)
    ]
    assert manifest["prompt_count"] == 8
    assert [row["stage3_selection_rank"] for row in manifest["prompt_records"]] == list(range(8))


def test_prepare_stage5_prompts_rejects_context_hash_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "contexts.jsonl"
    records = [_record(rank) for rank in range(8)]
    records[3]["context_sha256"] = "0" * 64
    source.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="hash mismatch"):
        _prepare_records()(source)
