"""Model configuration used by the LM adapter.

The logical Hugging Face id/revision is kept separate from the local path so
large model weights can stay outside Git while every run still records their
provenance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vkr_benchmark.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class LocalModelConfig:
    """Configuration for a locally available causal language model."""

    model_id: str
    revision: str
    local_path: Path
    dtype: str = "bfloat16"
    prompt_add_special_tokens: bool = False
    device: str = "cuda"
    attn_implementation: str | None = None

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ConfigurationError("model id must be non-empty")
        if not self.revision:
            raise ConfigurationError("model revision must be non-empty")
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ConfigurationError(f"unsupported model dtype: {self.dtype}")
        if not self.device:
            raise ConfigurationError("device must be non-empty")

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        project_root: str | Path | None = None,
    ) -> "LocalModelConfig":
        config_path = Path(path).expanduser().resolve()
        try:
            raw: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"cannot read model config {config_path}: {exc}") from exc

        if project_root is None:
            # Expected repository layout: <root>/configs/models/<name>.json
            try:
                root = config_path.parents[2]
            except IndexError as exc:
                raise ConfigurationError(
                    "cannot infer project root from model config path"
                ) from exc
        else:
            root = Path(project_root).expanduser().resolve()

        required = {"id", "revision", "local_path"}
        missing = sorted(required - raw.keys())
        if missing:
            raise ConfigurationError(
                f"model config {config_path} is missing keys: {', '.join(missing)}"
            )

        local_path = Path(raw["local_path"]).expanduser()
        if not local_path.is_absolute():
            local_path = root / local_path

        return cls(
            model_id=str(raw["id"]),
            revision=str(raw["revision"]),
            local_path=local_path.resolve(),
            dtype=str(raw.get("dtype", "bfloat16")),
            prompt_add_special_tokens=bool(
                raw.get("prompt_add_special_tokens", False)
            ),
            device=str(raw.get("device", "cuda")),
            attn_implementation=(
                None
                if raw.get("attn_implementation") is None
                else str(raw["attn_implementation"])
            ),
        )
