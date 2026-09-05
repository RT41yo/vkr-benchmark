"""Canonical run identifiers and persistent Stage-2 result storage.

The storage layer deliberately consumes the already-computed experiment result.
It does not recompute metrics and does not participate in timed encode/decode
sections.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Mapping

from vkr_benchmark.config import ExperimentConfig, LocalModelConfig
from vkr_benchmark.errors import StorageError
from vkr_benchmark.runner.experiment import ExperimentExecution

_RUN_ID_HEX_CHARS = 16
_METHOD_PARAMS_HASH_HEX_CHARS = 16


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    """Filesystem locations materialized for one successfully persisted run."""

    run_id: str
    run_directory: Path
    config_path: Path
    result_path: Path
    stegotext_path: Path | None
    token_ids_path: Path | None
    trace_path: Path | None
    summary_path: Path | None


def _canonicalizable(value: Any) -> Any:
    """Return a JSON-compatible value with deterministic mapping semantics."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StorageError("canonical configuration cannot contain NaN or infinity")
        return value
    if isinstance(value, Enum):
        return _canonicalizable(value.value)
    if isinstance(value, Mapping):
        return {str(key): _canonicalizable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonicalizable(item) for item in value]
    raise StorageError(
        f"value of type {type(value).__name__} cannot appear in canonical JSON"
    )


def canonical_json_text(value: Mapping[str, Any]) -> str:
    """Serialize configuration exactly as required by benchmark v0.1.

    UTF-8 is used by the caller when hashing/writing. Keys are lexicographically
    sorted and insignificant whitespace is omitted.
    """

    normalized = _canonicalizable(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_prefix(text: str, length: int) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def compute_run_id(canonical_config: Mapping[str, Any]) -> str:
    """Return ``SHA256(canonical_config_json)[:16]`` (FIXED v0.1)."""

    return _sha256_prefix(canonical_json_text(canonical_config), _RUN_ID_HEX_CHARS)


def compute_method_params_hash(params: Mapping[str, Any]) -> str:
    """Stable operating-point identifier from method parameters only.

    Secret ids, keys and method RNG seeds are deliberately excluded: they are
    repeat/run identity, not the method operating point itself.
    """

    if not isinstance(params, Mapping):
        raise StorageError("method params must be a mapping")
    return _sha256_prefix(
        canonical_json_text(dict(params)), _METHOD_PARAMS_HASH_HEX_CHARS
    )


def materialize_run_config(
    config: ExperimentConfig,
    model_config: LocalModelConfig,
) -> dict[str, Any]:
    """Materialize the machine-independent configuration used for ``run_id``.

    The local model path and device are intentionally excluded because they are
    machine/environment details. Prompt tokenization policy and attention
    implementation are included because changing either can alter model output.
    """

    method: dict[str, Any] = {
        "id": config.method.method_id,
        "implementation_revision": config.method.implementation_revision,
        "params": dict(config.method.params),
    }
    if config.method.random_seed is not None:
        method["random_seed"] = config.method.random_seed
    if config.method.key is not None:
        method["key"] = config.method.key

    return {
        "benchmark_version": config.benchmark_version,
        "run_kind": config.run_kind,
        "model": {
            "id": model_config.model_id,
            "revision": model_config.revision,
            "dtype": model_config.dtype,
            "prompt_add_special_tokens": model_config.prompt_add_special_tokens,
            "attn_implementation": model_config.attn_implementation,
        },
        "prompt_id": config.prompt_id,
        "method": method,
        "generation": {
            "temperature": config.generation.temperature,
            "top_k": config.generation.top_k,
            "top_p": config.generation.top_p,
            "exclude_special_tokens": config.generation.exclude_special_tokens,
            "kv_cache": config.generation.kv_cache,
        },
        "secret_id": config.secret_id,
        "termination": {
            "mode": config.termination.mode,
            "target_carrier_tokens": config.termination.target_carrier_tokens,
        },
    }


def _strict_json_value(value: Any) -> Any:
    """Convert values to strict JSON while preserving +inf explicitly.

    RFC-style JSON has no numeric infinity. Benchmark-native KL can legitimately
    be +inf, so persisted JSON uses the string ``"inf"`` for that value.
    Parquet keeps IEEE-754 ``+inf`` as a float.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            raise StorageError("NaN is not a valid persisted benchmark result")
        if value == math.inf:
            return "inf"
        if value == -math.inf:
            return "-inf"
        return value
    if isinstance(value, Enum):
        return _strict_json_value(value.value)
    if isinstance(value, Mapping):
        return {str(key): _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_strict_json_value(item) for item in value]
    raise StorageError(f"cannot serialize result value of type {type(value).__name__}")


def _pretty_json_text(value: Mapping[str, Any]) -> str:
    return json.dumps(
        _strict_json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"


def build_result_record(
    execution: ExperimentExecution,
    *,
    run_id: str,
) -> dict[str, Any]:
    """Build the run-level ``result.json`` payload from computed metrics."""

    roundtrip = execution.roundtrip
    capacity = execution.capacity_entropy_metrics
    distortion = execution.distribution_distortion_metrics
    quality = execution.quality_metrics
    reliability = execution.reliability_metrics
    performance = execution.performance_metrics

    return {
        "run_id": run_id,
        "status": "ok",
        "method_params_hash": compute_method_params_hash(execution.config.method.params),
        "payload_bits": capacity.payload_bits,
        "carrier_tokens": capacity.carrier_tokens,
        "secret_bits_read": roundtrip.encode.secret_bits_read,
        "bits_per_token": capacity.bits_per_token,
        "reference_entropy_mean_bits": capacity.reference_entropy_mean_bits,
        "reference_entropy_sum_bits": capacity.reference_entropy_sum_bits,
        "entropy_utilization": capacity.entropy_utilization,
        "entropy_utilization_percent": capacity.entropy_utilization_percent,
        "q_mode": distortion.q_mode.value,
        # Specification v0.1 generic KL fields mean KL(P_reference || Q_stego).
        "kl_mean_bits": distortion.kl_mean_bits,
        "kl_median_bits": distortion.kl_median_bits,
        "kl_p95_bits": distortion.kl_p95_bits,
        "kl_max_bits": distortion.kl_max_bits,
        "kl_infinite_steps": distortion.kl_infinite_steps,
        "kl_finite_steps": distortion.kl_finite_steps,
        "tvd_mean": distortion.tvd_mean,
        "tvd_median": distortion.tvd_median,
        "tvd_p95": distortion.tvd_p95,
        "tvd_max": distortion.tvd_max,
        "nll_raw_lm_nats_per_token": quality.nll_raw_lm_nats_per_token,
        "ppl_raw_lm": quality.ppl_raw_lm,
        "roundtrip_exact": reliability.roundtrip_exact,
        "ber": reliability.ber,
        "bit_errors": reliability.bit_errors,
        "expected_length_bits": reliability.expected_length_bits,
        "recovered_length_bits": reliability.recovered_length_bits,
        "length_delta_bits": reliability.length_delta_bits,
        "recovered_extra_bits": reliability.recovered_extra_bits,
        "first_mismatch_bit": reliability.first_mismatch_bit,
        "token_sequence_roundtrip_exact": reliability.token_sequence_roundtrip_exact,
        "first_token_roundtrip_mismatch": reliability.first_token_roundtrip_mismatch,
        "decoder_complete": roundtrip.decode.finalization.complete,
        "encode_total_ms": performance.encode_total_ms,
        "decode_total_ms": performance.decode_total_ms,
        "encode_ms_per_token": performance.encode_ms_per_token,
        "decode_ms_per_token": performance.decode_ms_per_token,
        "payload_bits_per_second_encode": performance.payload_bits_per_second_encode,
        "payload_bits_per_second_decode": performance.payload_bits_per_second_decode,
        "lm_forward_total_ms": performance.lm_forward_total_ms,
        "distribution_processing_total_ms": performance.distribution_processing_total_ms,
        "stego_algorithm_total_ms": performance.stego_algorithm_total_ms,
        "error": None,
    }


def build_failure_result_record(*, run_id: str, error: Exception) -> dict[str, Any]:
    """Persist a failed configured run without inventing unavailable metrics."""

    return {
        "run_id": run_id,
        "status": "error",
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
    }


def build_token_ids_record(execution: ExperimentExecution) -> dict[str, Any]:
    roundtrip = execution.roundtrip
    return {
        "prompt_token_ids": list(roundtrip.encode.prompt_token_ids),
        "sender_token_ids": list(roundtrip.encode.carrier_token_ids),
        "receiver_token_ids": list(roundtrip.decode.observed_token_ids),
        "token_sequence_roundtrip_exact": (
            roundtrip.transport.token_sequence_roundtrip_exact
        ),
        "first_token_mismatch": roundtrip.transport.first_token_mismatch,
        "token_count_delta": roundtrip.transport.token_count_delta,
    }


def iter_trace_records(execution: ExperimentExecution) -> Iterable[dict[str, Any]]:
    """Yield compact sender-side per-token trace records.

    Full P_reference/Q_stego arrays are intentionally not persisted in normal
    mode. The current Stage-2 runner retains enough scalar diagnostics to audit
    capacity, entropy, distortion, and raw-LM quality per carrier step.
    """

    encode = execution.roundtrip.encode
    cumulative_bits = 0
    for index, token_id in enumerate(encode.carrier_token_ids):
        bits_step = encode.step_bits_consumed[index]
        if bits_step is not None:
            cumulative_bits += int(bits_step)
        distortion = encode.step_distribution_distortion[index]
        yield {
            "step": index,
            "token_id": int(token_id),
            "bits_consumed_step": bits_step,
            "bits_consumed_total": cumulative_bits if bits_step is not None else None,
            "reference_entropy_bits": encode.step_reference_entropy_bits[index],
            "q_mode": distortion.q_mode.value,
            "q_source": distortion.q_source.value,
            "kl_ref_to_stego_bits": distortion.kl_bits,
            "tvd": distortion.tvd,
            "raw_lm_nll_selected_nats": encode.step_raw_lm_nll_nats[index],
        }


def build_summary_row(
    execution: ExperimentExecution,
    *,
    model_config: LocalModelConfig,
    run_id: str,
) -> dict[str, Any]:
    """Build one row of ``summary.parquet`` (one row == one run)."""

    capacity = execution.capacity_entropy_metrics
    distortion = execution.distribution_distortion_metrics
    quality = execution.quality_metrics
    reliability = execution.reliability_metrics
    performance = execution.performance_metrics

    return {
        "run_id": run_id,
        "run_kind": execution.config.run_kind,
        "benchmark_version": execution.config.benchmark_version,
        "model_id": model_config.model_id,
        "model_revision": model_config.revision,
        "prompt_id": execution.config.prompt_id,
        "method_id": execution.config.method.method_id,
        "method_params_hash": compute_method_params_hash(execution.config.method.params),
        "secret_id": execution.config.secret_id,
        "carrier_tokens": capacity.carrier_tokens,
        "payload_bits": capacity.payload_bits,
        "bits_per_token": capacity.bits_per_token,
        "reference_entropy_mean_bits": capacity.reference_entropy_mean_bits,
        "reference_entropy_sum_bits": capacity.reference_entropy_sum_bits,
        "entropy_utilization": capacity.entropy_utilization,
        "q_mode": distortion.q_mode.value,
        "kl_mean_bits": distortion.kl_mean_bits,
        "kl_max_bits": distortion.kl_max_bits,
        "kl_infinite_steps": distortion.kl_infinite_steps,
        "tvd_mean": distortion.tvd_mean,
        "tvd_max": distortion.tvd_max,
        "nll_raw_lm_nats_per_token": quality.nll_raw_lm_nats_per_token,
        "ppl_raw_lm": quality.ppl_raw_lm,
        "roundtrip_exact": reliability.roundtrip_exact,
        "ber": reliability.ber,
        "encode_ms_per_token": performance.encode_ms_per_token,
        "decode_ms_per_token": performance.decode_ms_per_token,
        "payload_bits_per_second_encode": performance.payload_bits_per_second_encode,
        "payload_bits_per_second_decode": performance.payload_bits_per_second_decode,
        "token_sequence_roundtrip_exact": reliability.token_sequence_roundtrip_exact,
        "status": "ok",
    }


def build_failure_summary_row(
    *,
    config: ExperimentConfig,
    model_config: LocalModelConfig,
    run_id: str,
) -> dict[str, Any]:
    """Summary row for an executed configuration that ended with an error."""

    return {
        "run_id": run_id,
        "run_kind": config.run_kind,
        "benchmark_version": config.benchmark_version,
        "model_id": model_config.model_id,
        "model_revision": model_config.revision,
        "prompt_id": config.prompt_id,
        "method_id": config.method.method_id,
        "method_params_hash": compute_method_params_hash(config.method.params),
        "secret_id": config.secret_id,
        "carrier_tokens": None,
        "payload_bits": None,
        "bits_per_token": None,
        "reference_entropy_mean_bits": None,
        "reference_entropy_sum_bits": None,
        "entropy_utilization": None,
        "q_mode": None,
        "kl_mean_bits": None,
        "kl_max_bits": None,
        "kl_infinite_steps": None,
        "tvd_mean": None,
        "tvd_max": None,
        "nll_raw_lm_nats_per_token": None,
        "ppl_raw_lm": None,
        "roundtrip_exact": None,
        "ber": None,
        "encode_ms_per_token": None,
        "decode_ms_per_token": None,
        "payload_bits_per_second_encode": None,
        "payload_bits_per_second_decode": None,
        "token_sequence_roundtrip_exact": None,
        "status": "error",
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temp_path = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def _write_trace_gzip(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        temp_path = Path(handle.name)
        # mtime=0 makes gzip bytes reproducible for the same trace.
        with gzip.GzipFile(fileobj=handle, mode="wb", mtime=0) as compressed:
            for record in records:
                line = json.dumps(
                    _strict_json_value(record),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
                compressed.write(line + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def _import_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise StorageError(
            "summary.parquet requires the storage extra: "
            "pip install -e '.[storage]'"
        ) from exc
    return pa, pq


def ensure_summary_backend_available() -> None:
    """Fail before a costly model run if Parquet persistence is unavailable."""

    _import_pyarrow()


def _summary_schema(pa: Any) -> Any:
    """Stable v0.2 physical schema implementing the v0.1 logical minimum."""

    return pa.schema(
        [
            ("run_id", pa.string()),
            ("run_kind", pa.string()),
            ("benchmark_version", pa.string()),
            ("model_id", pa.string()),
            ("model_revision", pa.string()),
            ("prompt_id", pa.string()),
            ("method_id", pa.string()),
            ("method_params_hash", pa.string()),
            ("secret_id", pa.string()),
            ("carrier_tokens", pa.int64()),
            ("payload_bits", pa.int64()),
            ("bits_per_token", pa.float64()),
            ("reference_entropy_mean_bits", pa.float64()),
            ("reference_entropy_sum_bits", pa.float64()),
            ("entropy_utilization", pa.float64()),
            ("q_mode", pa.string()),
            ("kl_mean_bits", pa.float64()),
            ("kl_max_bits", pa.float64()),
            ("kl_infinite_steps", pa.int64()),
            ("tvd_mean", pa.float64()),
            ("tvd_max", pa.float64()),
            ("nll_raw_lm_nats_per_token", pa.float64()),
            ("ppl_raw_lm", pa.float64()),
            ("roundtrip_exact", pa.bool_()),
            ("ber", pa.float64()),
            ("encode_ms_per_token", pa.float64()),
            ("decode_ms_per_token", pa.float64()),
            ("payload_bits_per_second_encode", pa.float64()),
            ("payload_bits_per_second_decode", pa.float64()),
            ("token_sequence_roundtrip_exact", pa.bool_()),
            ("status", pa.string()),
        ]
    )


def update_summary_parquet(path: str | Path, row: Mapping[str, Any]) -> Path:
    """Upsert one run row by ``run_id`` and atomically rewrite summary Parquet."""

    pa, pq = _import_pyarrow()
    summary_path = Path(path).expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    schema = _summary_schema(pa)

    required = {field.name for field in schema}
    missing = sorted(required - row.keys())
    extra = sorted(row.keys() - required)
    if missing or extra:
        pieces: list[str] = []
        if missing:
            pieces.append("missing: " + ", ".join(missing))
        if extra:
            pieces.append("extra: " + ", ".join(extra))
        raise StorageError("summary row does not match schema (" + "; ".join(pieces) + ")")

    rows: list[dict[str, Any]] = []
    if summary_path.exists():
        try:
            existing = pq.read_table(summary_path, schema=schema)
        except Exception as exc:  # pyarrow exposes several concrete errors
            raise StorageError(f"cannot read existing summary {summary_path}: {exc}") from exc
        rows.extend(existing.to_pylist())

    run_id = str(row["run_id"])
    rows = [existing for existing in rows if existing.get("run_id") != run_id]
    rows.append(dict(row))
    rows.sort(key=lambda item: str(item["run_id"]))

    try:
        table = pa.Table.from_pylist(rows, schema=schema)
    except Exception as exc:
        raise StorageError(f"cannot construct summary table: {exc}") from exc

    with NamedTemporaryFile("wb", dir=summary_path.parent, delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        pq.write_table(table, temp_path, compression="zstd")
        os.replace(temp_path, summary_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return summary_path


def _run_paths(results_root: Path, run_id: str) -> tuple[Path, Path]:
    run_directory = results_root / "runs" / run_id
    summary_path = results_root / "summary.parquet"
    return run_directory, summary_path


def persist_experiment_execution(
    execution: ExperimentExecution,
    *,
    model_config: LocalModelConfig,
    results_root: str | Path,
    update_summary: bool = True,
) -> RunArtifacts:
    """Persist one successful execution using the Stage-2 v0.2 layout."""

    root = Path(results_root).expanduser().resolve()
    canonical_config = materialize_run_config(execution.config, model_config)
    run_id = compute_run_id(canonical_config)
    run_directory, summary_path = _run_paths(root, run_id)
    run_directory.mkdir(parents=True, exist_ok=True)

    config_path = run_directory / "config.json"
    result_path = run_directory / "result.json"
    stegotext_path = run_directory / "stegotext.txt"
    token_ids_path = run_directory / "token_ids.json"
    trace_path = run_directory / "trace.jsonl.gz"

    # config.json bytes are the exact canonical JSON bytes used for run_id.
    _atomic_write_text(config_path, canonical_json_text(canonical_config))
    _atomic_write_text(
        result_path,
        _pretty_json_text(build_result_record(execution, run_id=run_id)),
    )
    # Exact transmitted ordinary text: no synthetic trailing newline is added.
    _atomic_write_text(stegotext_path, execution.roundtrip.transport.text)
    _atomic_write_text(
        token_ids_path,
        _pretty_json_text(build_token_ids_record(execution)),
    )
    _write_trace_gzip(trace_path, iter_trace_records(execution))

    persisted_summary: Path | None = None
    if update_summary:
        persisted_summary = update_summary_parquet(
            summary_path,
            build_summary_row(
                execution,
                model_config=model_config,
                run_id=run_id,
            ),
        )

    return RunArtifacts(
        run_id=run_id,
        run_directory=run_directory,
        config_path=config_path,
        result_path=result_path,
        stegotext_path=stegotext_path,
        token_ids_path=token_ids_path,
        trace_path=trace_path,
        summary_path=persisted_summary,
    )


def persist_failed_run(
    *,
    config: ExperimentConfig,
    model_config: LocalModelConfig,
    error: Exception,
    results_root: str | Path,
    update_summary: bool = True,
) -> RunArtifacts:
    """Persist a configured run that failed after configuration was resolved."""

    root = Path(results_root).expanduser().resolve()
    canonical_config = materialize_run_config(config, model_config)
    run_id = compute_run_id(canonical_config)
    run_directory, summary_path = _run_paths(root, run_id)
    run_directory.mkdir(parents=True, exist_ok=True)

    config_path = run_directory / "config.json"
    result_path = run_directory / "result.json"
    _atomic_write_text(config_path, canonical_json_text(canonical_config))
    _atomic_write_text(
        result_path,
        _pretty_json_text(build_failure_result_record(run_id=run_id, error=error)),
    )

    # Remove success-only artifacts if an identical configuration is being
    # rerun and now fails; the directory must describe the latest attempt only.
    for filename in ("stegotext.txt", "token_ids.json", "trace.jsonl.gz"):
        stale = run_directory / filename
        if stale.exists():
            stale.unlink()

    persisted_summary: Path | None = None
    if update_summary:
        persisted_summary = update_summary_parquet(
            summary_path,
            build_failure_summary_row(
                config=config,
                model_config=model_config,
                run_id=run_id,
            ),
        )

    return RunArtifacts(
        run_id=run_id,
        run_directory=run_directory,
        config_path=config_path,
        result_path=result_path,
        stegotext_path=None,
        token_ids_path=None,
        trace_path=None,
        summary_path=persisted_summary,
    )
