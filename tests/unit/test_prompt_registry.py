from __future__ import annotations

import json

import pytest

from vkr_benchmark.errors import ConfigurationError
from vkr_benchmark.inputs import PromptRecord, PromptRegistry


def test_prompt_registry_loads_spec_jsonl_and_preserves_exact_text(tmp_path) -> None:
    path = tmp_path / "prompts.jsonl"
    text = "  Exact prompt with trailing space "
    path.write_text(
        json.dumps(
            {
                "prompt_id": "p000001",
                "source": "unit",
                "source_revision": "r1",
                "text": text,
                "language": "en",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    registry = PromptRegistry.from_jsonl(path)

    assert registry.get("p000001").text == text
    assert registry.prompt_ids == ("p000001",)


def test_prompt_registry_rejects_duplicate_ids() -> None:
    prompt = PromptRecord("p1", "unit", "r1", "text", "en")
    with pytest.raises(ConfigurationError, match="duplicate prompt_id"):
        PromptRegistry([prompt, prompt])


def test_prompt_registry_rejects_unknown_id() -> None:
    registry = PromptRegistry([PromptRecord("p1", "unit", "r1", "text", "en")])
    with pytest.raises(ConfigurationError, match="unknown prompt_id"):
        registry.get("missing")


def test_prompt_registry_rejects_missing_required_field(tmp_path) -> None:
    path = tmp_path / "prompts.jsonl"
    path.write_text('{"prompt_id":"p1","text":"x"}\n', encoding="utf-8")
    with pytest.raises(ConfigurationError, match="missing keys"):
        PromptRegistry.from_jsonl(path)


def test_prompt_record_rejects_empty_text() -> None:
    with pytest.raises(ConfigurationError, match="prompt text"):
        PromptRecord("p1", "unit", "r1", "", "en")
