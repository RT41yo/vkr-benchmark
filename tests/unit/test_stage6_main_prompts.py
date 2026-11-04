from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy

import pytest

from vkr_benchmark.errors import ConfigurationError


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/prepare_stage6_main_prompts.py"


def _prepare_records():
    namespace = runpy.run_path(str(SCRIPT), run_name="stage6_prompt_prepare_test")
    return namespace["prepare_records"]


def _record(rank: int, *, pilot: bool = False) -> dict[str, object]:
    text = f"Stage 6 prompt context {rank}. Second sentence. Third sentence."
    return {
        "article_id": f"article-{rank}",
        "context": text,
        "context_char_count": len(text),
        "context_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "pilot": pilot,
        "row_index": 1000 + rank,
        "selection_rank": rank,
    }


def test_prepare_stage6_prompts_uses_held_out_ranks_8_through_31(tmp_path: Path) -> None:
    source = tmp_path / "contexts.jsonl"
    records = [_record(rank, pilot=rank < 8) for rank in reversed(range(32))]
    source.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    prompts, manifest = _prepare_records()(source)

    assert len(prompts) == 24
    assert [item["prompt_id"] for item in prompts] == [
        f"s6p{i:04d}" for i in range(1, 25)
    ]
    assert manifest["stage5_pilot_selection_ranks"] == list(range(8))
    assert manifest["stage6_selection_ranks"] == list(range(8, 32))
    assert [row["stage3_selection_rank"] for row in manifest["prompt_records"]] == list(
        range(8, 32)
    )


def test_prepare_stage6_prompts_rejects_overlap_with_stage5_pilot(tmp_path: Path) -> None:
    source = tmp_path / "contexts.jsonl"
    records = [_record(rank, pilot=(rank == 12)) for rank in range(8, 32)]
    source.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="marked as Stage-5 pilot"):
        _prepare_records()(source)


def test_prepare_stage6_prompts_rejects_context_hash_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "contexts.jsonl"
    records = [_record(rank) for rank in range(8, 32)]
    records[5]["context_sha256"] = "0" * 64
    source.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="hash mismatch"):
        _prepare_records()(source)
