"""Stable prompt records used by benchmark experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from vkr_benchmark.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class PromptRecord:
    """One exact prompt string with a stable benchmark identifier."""

    prompt_id: str
    source: str
    source_revision: str
    text: str
    language: str

    def __post_init__(self) -> None:
        if not self.prompt_id:
            raise ConfigurationError("prompt_id must be non-empty")
        if not self.source:
            raise ConfigurationError("prompt source must be non-empty")
        if not self.source_revision:
            raise ConfigurationError("prompt source_revision must be non-empty")
        if not isinstance(self.text, str) or self.text == "":
            raise ConfigurationError("prompt text must be a non-empty string")
        if not self.language:
            raise ConfigurationError("prompt language must be non-empty")


class PromptRegistry:
    """In-memory registry loaded from the specification-v0.1 JSONL format."""

    def __init__(self, prompts: tuple[PromptRecord, ...] | list[PromptRecord]) -> None:
        by_id: dict[str, PromptRecord] = {}
        for prompt in prompts:
            if prompt.prompt_id in by_id:
                raise ConfigurationError(f"duplicate prompt_id: {prompt.prompt_id}")
            by_id[prompt.prompt_id] = prompt
        if not by_id:
            raise ConfigurationError("prompt registry must contain at least one prompt")
        self._by_id = by_id

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "PromptRegistry":
        jsonl_path = Path(path).expanduser().resolve()
        prompts: list[PromptRecord] = []
        try:
            lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ConfigurationError(f"cannot read prompts file {jsonl_path}: {exc}") from exc

        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConfigurationError(
                    f"invalid JSON in prompts file {jsonl_path} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(raw, dict):
                raise ConfigurationError(
                    f"prompt record at line {line_number} must be a JSON object"
                )
            required = {"prompt_id", "source", "source_revision", "text", "language"}
            missing = sorted(required - raw.keys())
            if missing:
                raise ConfigurationError(
                    f"prompt record at line {line_number} is missing keys: {', '.join(missing)}"
                )
            prompts.append(
                PromptRecord(
                    prompt_id=str(raw["prompt_id"]),
                    source=str(raw["source"]),
                    source_revision=str(raw["source_revision"]),
                    text=raw["text"],
                    language=str(raw["language"]),
                )
            )
        return cls(prompts)

    def get(self, prompt_id: str) -> PromptRecord:
        try:
            return self._by_id[prompt_id]
        except KeyError as exc:
            raise ConfigurationError(f"unknown prompt_id: {prompt_id}") from exc

    @property
    def prompt_ids(self) -> tuple[str, ...]:
        return tuple(self._by_id)
