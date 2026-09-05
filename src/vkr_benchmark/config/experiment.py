"""Typed configuration for one normalized Stage-2 experiment run.

This is the executable launcher configuration. The storage layer added later in
Stage 2 will materialize the canonical persisted run configuration and run_id.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

from vkr_benchmark.errors import ConfigurationError

if TYPE_CHECKING:
    from vkr_benchmark.distributions import GenerationPolicy


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"{field} must be a non-empty string")
    return value


def _require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{field} must be a boolean")
    return value


def _require_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{field} must be a number")
    return float(value)


def _optional_positive_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{field} must be a positive integer or null")
    return value


@dataclass(frozen=True, slots=True)
class MethodRunConfig:
    """Method identity, method parameters and reproducible method randomness."""

    method_id: str
    params: Mapping[str, Any]
    implementation_revision: str | None = None
    random_seed: int | str | None = None
    key: int | str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.method_id, str) or not self.method_id:
            raise ConfigurationError("method id must be non-empty")
        if self.implementation_revision is not None and (
            not isinstance(self.implementation_revision, str)
            or not self.implementation_revision
        ):
            raise ConfigurationError("implementation_revision must be non-empty or null")
        if isinstance(self.random_seed, bool) or (
            self.random_seed is not None and not isinstance(self.random_seed, (int, str))
        ):
            raise ConfigurationError("method random_seed must be an integer, string, or null")
        if isinstance(self.key, bool) or (
            self.key is not None and not isinstance(self.key, (int, str))
        ):
            raise ConfigurationError("method key must be an integer, string, or null")
        if not isinstance(self.params, Mapping):
            raise ConfigurationError("method params must be a mapping")
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))


@dataclass(frozen=True, slots=True)
class GenerationRunConfig:
    """Common external policy used to construct P_reference."""

    temperature: float = 1.0
    top_k: int | None = None
    top_p: float = 1.0
    exclude_special_tokens: bool = True
    kv_cache: bool = True

    def __post_init__(self) -> None:
        # Reuse canonical distribution-policy validation so launcher and
        # P_reference construction cannot silently diverge.
        self.to_policy()
        if self.kv_cache is not True:
            raise ConfigurationError(
                "normalized Stage-2 experiments require kv_cache=true"
            )

    def to_policy(self) -> "GenerationPolicy":
        # Local import avoids config -> distributions -> lm -> config cycle.
        from vkr_benchmark.distributions import GenerationPolicy

        return GenerationPolicy(
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            exclude_special_tokens=self.exclude_special_tokens,
        )


@dataclass(frozen=True, slots=True)
class TerminationConfig:
    """Run stopping rule currently supported by the three baseline methods."""

    mode: str
    target_carrier_tokens: int

    def __post_init__(self) -> None:
        if self.mode != "fixed_carrier_tokens":
            raise ConfigurationError(
                "Stage-2 baseline runner currently supports only fixed_carrier_tokens"
            )
        if (
            isinstance(self.target_carrier_tokens, bool)
            or not isinstance(self.target_carrier_tokens, int)
            or self.target_carrier_tokens <= 0
        ):
            raise ConfigurationError("target_carrier_tokens must be a positive integer")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Complete executable configuration for one normalized experiment run."""

    benchmark_version: str
    run_kind: str
    model_config_path: Path
    prompt_id: str
    method: MethodRunConfig
    generation: GenerationRunConfig
    secret_id: str
    termination: TerminationConfig
    source_path: Path | None = None

    def __post_init__(self) -> None:
        if self.benchmark_version != "0.1":
            raise ConfigurationError(
                "Stage-2 launcher currently implements benchmark_version=0.1"
            )
        if self.run_kind != "normalized":
            raise ConfigurationError(
                "Stage-2 unified runner currently supports only run_kind=normalized"
            )
        if not isinstance(self.prompt_id, str) or not self.prompt_id:
            raise ConfigurationError("prompt_id must be non-empty")
        if not isinstance(self.secret_id, str) or not self.secret_id:
            raise ConfigurationError("secret_id must be non-empty")

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        project_root: str | Path | None = None,
    ) -> "ExperimentConfig":
        config_path = Path(path).expanduser().resolve()
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(
                f"cannot read experiment config {config_path}: {exc}"
            ) from exc
        if not isinstance(raw, dict):
            raise ConfigurationError("experiment config must be a JSON object")

        if project_root is None:
            try:
                root = config_path.parents[2]
            except IndexError as exc:
                raise ConfigurationError(
                    "cannot infer project root from experiment config path"
                ) from exc
        else:
            root = Path(project_root).expanduser().resolve()

        required = {
            "benchmark_version",
            "run_kind",
            "model_config",
            "prompt_id",
            "method",
            "generation",
            "secret_id",
            "termination",
        }
        missing = sorted(required - raw.keys())
        if missing:
            raise ConfigurationError(
                f"experiment config {config_path} is missing keys: {', '.join(missing)}"
            )

        method_raw = raw["method"]
        generation_raw = raw["generation"]
        termination_raw = raw["termination"]
        if not isinstance(method_raw, dict):
            raise ConfigurationError("method must be a JSON object")
        if not isinstance(generation_raw, dict):
            raise ConfigurationError("generation must be a JSON object")
        if not isinstance(termination_raw, dict):
            raise ConfigurationError("termination must be a JSON object")

        method_id = method_raw.get("id")
        params = method_raw.get("params")
        if method_id is None or params is None:
            raise ConfigurationError("method must contain id and params")
        if not isinstance(params, dict):
            raise ConfigurationError("method.params must be a JSON object")

        model_config_raw = _require_string(raw["model_config"], "model_config")
        model_config_path = Path(model_config_raw).expanduser()
        if not model_config_path.is_absolute():
            model_config_path = root / model_config_path

        implementation_revision = method_raw.get("implementation_revision")
        if implementation_revision is not None:
            implementation_revision = _require_string(
                implementation_revision,
                "method.implementation_revision",
            )

        temperature = _require_number(
            generation_raw.get("temperature", 1.0),
            "generation.temperature",
        )
        top_p = _require_number(
            generation_raw.get("top_p", 1.0),
            "generation.top_p",
        )
        top_k = _optional_positive_int(
            generation_raw.get("top_k"),
            "generation.top_k",
        )
        exclude_special_tokens = _require_bool(
            generation_raw.get("exclude_special_tokens", True),
            "generation.exclude_special_tokens",
        )
        kv_cache = _require_bool(
            generation_raw.get("kv_cache", True),
            "generation.kv_cache",
        )

        target_carrier_tokens_raw = termination_raw.get("target_carrier_tokens")
        if (
            isinstance(target_carrier_tokens_raw, bool)
            or not isinstance(target_carrier_tokens_raw, int)
        ):
            raise ConfigurationError(
                "termination.target_carrier_tokens must be a positive integer"
            )

        return cls(
            benchmark_version=_require_string(
                raw["benchmark_version"], "benchmark_version"
            ),
            run_kind=_require_string(raw["run_kind"], "run_kind"),
            model_config_path=model_config_path.resolve(),
            prompt_id=_require_string(raw["prompt_id"], "prompt_id"),
            method=MethodRunConfig(
                method_id=_require_string(method_id, "method.id"),
                params=params,
                implementation_revision=implementation_revision,
                random_seed=method_raw.get("random_seed"),
                key=method_raw.get("key"),
            ),
            generation=GenerationRunConfig(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                exclude_special_tokens=exclude_special_tokens,
                kv_cache=kv_cache,
            ),
            secret_id=_require_string(raw["secret_id"], "secret_id"),
            termination=TerminationConfig(
                mode=_require_string(termination_raw.get("mode"), "termination.mode"),
                target_carrier_tokens=target_carrier_tokens_raw,
            ),
            source_path=config_path,
        )
