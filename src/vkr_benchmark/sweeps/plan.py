"""Deterministic sweep planning for Stage-5 pilot experiments.

The planner expands one compact JSON manifest into ordinary ``ExperimentConfig``
objects.  It deliberately does *not* load a language model.  This lets Stage 5
validate the complete experimental matrix, paired replicate identity, and
canonical run ids before spending GPU time.

Method-local parameters, common generation conditions, and technical
termination rules remain distinct in the manifest.  In particular, changing a
common generation point (for example ``top_p``) never creates a new method
operating point.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from vkr_benchmark.config import (
    ExperimentConfig,
    GenerationRunConfig,
    LocalModelConfig,
    MethodRunConfig,
    TerminationConfig,
)
from vkr_benchmark.errors import ConfigurationError
from vkr_benchmark.methods import (
    ADGConfig,
    ArithmeticConfig,
    BinsConfig,
    DAIRConfig,
    DiscopConfig,
    HuffmanConfig,
    RRCConfig,
)
from vkr_benchmark.storage import compute_run_id, materialize_run_config

_SCHEMA_VERSION = "stage5.sweep.v1"
_RANDOMNESS_DOMAIN = "VKR-BENCHMARK-STAGE5-v1"

_METHOD_CONFIG_TYPES = {
    "adg": ADGConfig,
    "arithmetic_coding": ArithmeticConfig,
    "bins": BinsConfig,
    "dairstega": DAIRConfig,
    "discop": DiscopConfig,
    "huffman": HuffmanConfig,
    "rrc": RRCConfig,
}

# Method-internal randomness semantics are already frozen by the normalized
# adapters.  Stage-5 manifests therefore do not get to invent new RNG knobs.
_RANDOMNESS_ROLE = {
    "adg": "random_seed",
    "arithmetic_coding": None,
    "bins": "random_seed",
    "dairstega": None,
    "discop": "key",
    "huffman": None,
    "rrc": "key",
}


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"{field} must be a non-empty string")
    return value


def _require_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{field} must be a JSON object")
    return value


def _require_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ConfigurationError(f"{field} must be a non-empty JSON array")
    return value


def _resolve_project_path(root: Path, value: Any, field: str) -> Path:
    text = _require_nonempty_string(value, field)
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _ensure_unique(values: Iterable[str], field: str) -> tuple[str, ...]:
    items = tuple(values)
    if len(items) != len(set(items)):
        raise ConfigurationError(f"{field} contains duplicate values")
    return items


@dataclass(frozen=True, slots=True)
class ReplicateSpec:
    replicate_id: str
    secret_id: str


@dataclass(frozen=True, slots=True)
class GenerationPoint:
    point_id: str
    config: GenerationRunConfig


@dataclass(frozen=True, slots=True)
class MethodPoint:
    point_id: str
    params: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class MethodSweep:
    method_id: str
    implementation_revision: str | None
    points: tuple[MethodPoint, ...]
    termination: TerminationConfig


@dataclass(frozen=True, slots=True)
class PlannedRun:
    """One fully expanded Stage-5 run before LM execution."""

    sweep_id: str
    generation_point_id: str
    method_point_id: str
    replicate_id: str
    config: ExperimentConfig
    run_id: str

    def metadata_record(self) -> dict[str, Any]:
        return {
            "sweep_id": self.sweep_id,
            "run_id": self.run_id,
            "generation_point_id": self.generation_point_id,
            "method_id": self.config.method.method_id,
            "method_point_id": self.method_point_id,
            "prompt_id": self.config.prompt_id,
            "replicate_id": self.replicate_id,
            "secret_id": self.config.secret_id,
            "method_params": dict(self.config.method.params),
            "method_random_seed": self.config.method.random_seed,
            "method_key": self.config.method.key,
            "generation": {
                "temperature": self.config.generation.temperature,
                "top_k": self.config.generation.top_k,
                "top_p": self.config.generation.top_p,
                "exclude_special_tokens": self.config.generation.exclude_special_tokens,
                "kv_cache": self.config.generation.kv_cache,
            },
            "termination": _termination_record(self.config.termination),
        }


@dataclass(frozen=True, slots=True)
class SweepManifest:
    schema_version: str
    sweep_id: str
    purpose: str
    benchmark_version: str
    run_kind: str
    model_config_path: Path
    prompts_path: Path
    prompt_ids: tuple[str, ...]
    replicates: tuple[ReplicateSpec, ...]
    generation_points: tuple[GenerationPoint, ...]
    methods: tuple[MethodSweep, ...]
    source_path: Path
    project_root: Path

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        project_root: str | Path,
    ) -> "SweepManifest":
        source_path = Path(path).expanduser().resolve()
        root = Path(project_root).expanduser().resolve()
        try:
            raw = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(
                f"cannot read sweep manifest {source_path}: {exc}"
            ) from exc
        if not isinstance(raw, dict):
            raise ConfigurationError("sweep manifest must be a JSON object")

        schema_version = _require_nonempty_string(
            raw.get("schema_version"), "schema_version"
        )
        if schema_version != _SCHEMA_VERSION:
            raise ConfigurationError(
                f"unsupported sweep schema_version {schema_version!r}; "
                f"expected {_SCHEMA_VERSION!r}"
            )

        prompt_ids = _ensure_unique(
            (
                _require_nonempty_string(item, "prompt_ids[]")
                for item in _require_list(raw.get("prompt_ids"), "prompt_ids")
            ),
            "prompt_ids",
        )

        replicates_raw = _require_list(raw.get("replicates"), "replicates")
        replicates: list[ReplicateSpec] = []
        for index, item in enumerate(replicates_raw):
            obj = _require_object(item, f"replicates[{index}]")
            replicates.append(
                ReplicateSpec(
                    replicate_id=_require_nonempty_string(
                        obj.get("id"), f"replicates[{index}].id"
                    ),
                    secret_id=_require_nonempty_string(
                        obj.get("secret_id"), f"replicates[{index}].secret_id"
                    ),
                )
            )
        _ensure_unique((item.replicate_id for item in replicates), "replicate ids")
        _ensure_unique((item.secret_id for item in replicates), "replicate secret_ids")

        generation_raw = _require_list(
            raw.get("generation_points"), "generation_points"
        )
        generation_points: list[GenerationPoint] = []
        for index, item in enumerate(generation_raw):
            obj = _require_object(item, f"generation_points[{index}]")
            point_id = _require_nonempty_string(
                obj.get("id"), f"generation_points[{index}].id"
            )
            generation_points.append(
                GenerationPoint(
                    point_id=point_id,
                    config=GenerationRunConfig(
                        temperature=float(obj.get("temperature", 1.0)),
                        top_k=obj.get("top_k"),
                        top_p=float(obj.get("top_p", 1.0)),
                        exclude_special_tokens=obj.get("exclude_special_tokens", True),
                        kv_cache=obj.get("kv_cache", True),
                    ),
                )
            )
        _ensure_unique(
            (item.point_id for item in generation_points), "generation point ids"
        )

        methods_raw = _require_list(raw.get("methods"), "methods")
        methods: list[MethodSweep] = []
        for method_index, item in enumerate(methods_raw):
            obj = _require_object(item, f"methods[{method_index}]")
            method_id = _require_nonempty_string(
                obj.get("id"), f"methods[{method_index}].id"
            )
            try:
                config_type = _METHOD_CONFIG_TYPES[method_id]
            except KeyError as exc:
                raise ConfigurationError(
                    f"unsupported method in sweep manifest: {method_id!r}"
                ) from exc

            points_raw = _require_list(
                obj.get("points"), f"methods[{method_index}].points"
            )
            points: list[MethodPoint] = []
            for point_index, point_item in enumerate(points_raw):
                point_obj = _require_object(
                    point_item,
                    f"methods[{method_index}].points[{point_index}]",
                )
                point_id = _require_nonempty_string(
                    point_obj.get("id"),
                    f"methods[{method_index}].points[{point_index}].id",
                )
                params = _require_object(
                    point_obj.get("params"),
                    f"methods[{method_index}].points[{point_index}].params",
                )
                # Parse now so impossible pilot points fail before model loading.
                config_type.from_mapping(params)
                points.append(MethodPoint(point_id=point_id, params=dict(params)))
            _ensure_unique(
                (point.point_id for point in points),
                f"method point ids for {method_id}",
            )

            termination_raw = _require_object(
                obj.get("termination"), f"methods[{method_index}].termination"
            )
            termination = TerminationConfig(
                mode=_require_nonempty_string(
                    termination_raw.get("mode"),
                    f"methods[{method_index}].termination.mode",
                ),
                target_carrier_tokens=termination_raw.get("target_carrier_tokens"),
                target_payload_bits=termination_raw.get("target_payload_bits"),
                max_carrier_tokens=termination_raw.get("max_carrier_tokens"),
            )

            implementation_revision = obj.get("implementation_revision")
            if implementation_revision is not None:
                implementation_revision = _require_nonempty_string(
                    implementation_revision,
                    f"methods[{method_index}].implementation_revision",
                )
            methods.append(
                MethodSweep(
                    method_id=method_id,
                    implementation_revision=implementation_revision,
                    points=tuple(points),
                    termination=termination,
                )
            )
        _ensure_unique((item.method_id for item in methods), "method ids")

        return cls(
            schema_version=schema_version,
            sweep_id=_require_nonempty_string(raw.get("sweep_id"), "sweep_id"),
            purpose=_require_nonempty_string(raw.get("purpose"), "purpose"),
            benchmark_version=_require_nonempty_string(
                raw.get("benchmark_version"), "benchmark_version"
            ),
            run_kind=_require_nonempty_string(raw.get("run_kind"), "run_kind"),
            model_config_path=_resolve_project_path(
                root, raw.get("model_config"), "model_config"
            ),
            prompts_path=_resolve_project_path(
                root, raw.get("prompts_file"), "prompts_file"
            ),
            prompt_ids=prompt_ids,
            replicates=tuple(replicates),
            generation_points=tuple(generation_points),
            methods=tuple(methods),
            source_path=source_path,
            project_root=root,
        )

    def planned_run_count(self) -> int:
        method_points = sum(len(method.points) for method in self.methods)
        return (
            len(self.prompt_ids)
            * len(self.replicates)
            * len(self.generation_points)
            * method_points
        )

    def expand(self) -> tuple[PlannedRun, ...]:
        """Expand into canonical ExperimentConfig objects and predicted run ids."""

        model_config = LocalModelConfig.from_json(
            self.model_config_path,
            project_root=self.project_root,
        )
        runs: list[PlannedRun] = []
        for generation in self.generation_points:
            for method in self.methods:
                for method_point in method.points:
                    for prompt_id in self.prompt_ids:
                        for replicate in self.replicates:
                            random_seed, key = _method_random_material(
                                sweep_id=self.sweep_id,
                                replicate_id=replicate.replicate_id,
                                method_id=method.method_id,
                            )
                            config = ExperimentConfig(
                                benchmark_version=self.benchmark_version,
                                run_kind=self.run_kind,
                                model_config_path=self.model_config_path,
                                prompt_id=prompt_id,
                                method=MethodRunConfig(
                                    method_id=method.method_id,
                                    params=method_point.params,
                                    implementation_revision=method.implementation_revision,
                                    random_seed=random_seed,
                                    key=key,
                                ),
                                generation=generation.config,
                                secret_id=replicate.secret_id,
                                termination=method.termination,
                                source_path=self.source_path,
                            )
                            canonical = materialize_run_config(config, model_config)
                            runs.append(
                                PlannedRun(
                                    sweep_id=self.sweep_id,
                                    generation_point_id=generation.point_id,
                                    method_point_id=method_point.point_id,
                                    replicate_id=replicate.replicate_id,
                                    config=config,
                                    run_id=compute_run_id(canonical),
                                )
                            )

        run_ids = [run.run_id for run in runs]
        if len(run_ids) != len(set(run_ids)):
            raise ConfigurationError(
                "sweep expansion produced duplicate canonical run_ids"
            )
        return tuple(runs)


def _method_random_material(
    *, sweep_id: str, replicate_id: str, method_id: str
) -> tuple[str | None, str | None]:
    role = _RANDOMNESS_ROLE[method_id]
    if role is None:
        return None, None
    material = (
        f"{_RANDOMNESS_DOMAIN}|sweep={sweep_id}|replicate={replicate_id}|"
        f"method={method_id}|role={role}"
    )
    if role == "random_seed":
        return material, None
    if role == "key":
        return None, material
    raise AssertionError(f"unexpected randomness role: {role}")


def _termination_record(config: TerminationConfig) -> dict[str, Any]:
    if config.mode == "fixed_carrier_tokens":
        return {
            "mode": config.mode,
            "target_carrier_tokens": config.target_carrier_tokens,
        }
    return {
        "mode": config.mode,
        "target_payload_bits": config.target_payload_bits,
        "max_carrier_tokens": config.max_carrier_tokens,
    }
