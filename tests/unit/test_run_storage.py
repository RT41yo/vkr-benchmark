from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from vkr_benchmark.config import (
    ExperimentConfig,
    GenerationRunConfig,
    LocalModelConfig,
    MethodRunConfig,
    TerminationConfig,
)
from vkr_benchmark.distributions import QMode, QSource
from vkr_benchmark.errors import StorageError
from vkr_benchmark.metrics import (
    CapacityEntropyMetrics,
    DistributionDistortionMetrics,
    PerformanceMetrics,
    RawLMQualityMetrics,
    ReliabilityMetrics,
    StepDistributionDistortion,
)
from vkr_benchmark.storage import (
    build_result_record,
    build_summary_row,
    canonical_json_text,
    compute_method_params_hash,
    compute_run_id,
    materialize_run_config,
    persist_experiment_execution,
    persist_failed_run,
    update_summary_parquet,
)
import vkr_benchmark.storage.run_store as run_store


def _config(*, params: dict[str, object] | None = None) -> ExperimentConfig:
    return ExperimentConfig(
        benchmark_version="0.1",
        run_kind="normalized",
        model_config_path=Path("/tmp/model.json"),
        prompt_id="p000001",
        method=MethodRunConfig(
            method_id="bins",
            params=params or {"block_size": 2},
            implementation_revision="test-rev",
            random_seed=12345,
        ),
        generation=GenerationRunConfig(
            temperature=1.0,
            top_k=None,
            top_p=1.0,
            exclude_special_tokens=True,
            kv_cache=True,
        ),
        secret_id="000001",
        termination=TerminationConfig(
            mode="fixed_carrier_tokens",
            target_carrier_tokens=2,
        ),
    )


def _model() -> LocalModelConfig:
    return LocalModelConfig(
        model_id="example/model",
        revision="abc123",
        local_path=Path("/machine/local/model"),
        dtype="bfloat16",
        prompt_add_special_tokens=True,
        device="cuda",
        attn_implementation=None,
    )


def _execution() -> SimpleNamespace:
    config = _config()
    step0 = StepDistributionDistortion(
        q_mode=QMode.ANALYTIC_EXACT,
        q_source=QSource.ADAPTER_EXACT,
        kl_bits=math.inf,
        tvd=0.5,
    )
    step1 = StepDistributionDistortion(
        q_mode=QMode.ANALYTIC_EXACT,
        q_source=QSource.ADAPTER_EXACT,
        kl_bits=math.inf,
        tvd=0.25,
    )
    encode = SimpleNamespace(
        prompt_token_ids=(10, 11),
        carrier_token_ids=(20, 21),
        secret_bits_read=4,
        step_bits_consumed=(2, 2),
        step_reference_entropy_bits=(3.0, 4.0),
        step_distribution_distortion=(step0, step1),
        step_raw_lm_nll_nats=(1.5, 2.5),
    )
    decode_finalization = SimpleNamespace(complete=True)
    decode = SimpleNamespace(
        observed_token_ids=(20, 21),
        finalization=decode_finalization,
    )
    transport = SimpleNamespace(
        text="hello world",
        token_sequence_roundtrip_exact=True,
        first_token_mismatch=None,
        token_count_delta=0,
    )
    roundtrip = SimpleNamespace(
        encode=encode,
        decode=decode,
        transport=transport,
    )
    return SimpleNamespace(
        config=config,
        roundtrip=roundtrip,
        capacity_entropy_metrics=CapacityEntropyMetrics(
            payload_bits=4,
            carrier_tokens=2,
            bits_per_token=2.0,
            reference_entropy_mean_bits=3.5,
            reference_entropy_sum_bits=7.0,
            entropy_utilization=4 / 7,
            entropy_utilization_percent=400 / 7,
        ),
        distribution_distortion_metrics=DistributionDistortionMetrics(
            q_mode=QMode.ANALYTIC_EXACT,
            kl_mean_bits=math.inf,
            kl_median_bits=math.inf,
            kl_p95_bits=math.inf,
            kl_max_bits=math.inf,
            kl_infinite_steps=2,
            kl_finite_steps=0,
            tvd_mean=0.375,
            tvd_median=0.375,
            tvd_p95=0.5,
            tvd_max=0.5,
        ),
        quality_metrics=RawLMQualityMetrics(
            nll_raw_lm_nats_per_token=2.0,
            ppl_raw_lm=math.exp(2.0),
        ),
        reliability_metrics=ReliabilityMetrics(
            roundtrip_exact=True,
            ber=0.0,
            bit_errors=0,
            expected_length_bits=4,
            recovered_length_bits=4,
            length_delta_bits=0,
            recovered_extra_bits=0,
            first_mismatch_bit=None,
            first_decode_failure_token=None,
            token_sequence_roundtrip_exact=True,
            first_token_roundtrip_mismatch=None,
        ),
        performance_metrics=PerformanceMetrics(
            encode_total_ms=10.0,
            decode_total_ms=8.0,
            encode_ms_per_token=5.0,
            decode_ms_per_token=4.0,
            payload_bits_per_second_encode=400.0,
            payload_bits_per_second_decode=500.0,
            lm_forward_total_ms=8.0,
            distribution_processing_total_ms=4.0,
            stego_algorithm_total_ms=6.0,
            encode_lm_forward_ms=4.0,
            encode_distribution_processing_ms=2.0,
            encode_stego_algorithm_ms=4.0,
            decode_lm_forward_ms=4.0,
            decode_distribution_processing_ms=2.0,
            decode_stego_algorithm_ms=2.0,
        ),
    )


def test_canonical_json_is_sorted_utf8_and_compact() -> None:
    text = canonical_json_text({"z": 1, "a": "кириллица", "nested": {"b": 2, "a": 1}})
    assert text == '{"a":"кириллица","nested":{"a":1,"b":2},"z":1}'


def test_canonical_json_rejects_nonfinite_configuration_values() -> None:
    with pytest.raises(StorageError, match="NaN or infinity"):
        canonical_json_text({"x": math.inf})


def test_run_id_is_deterministic_and_sensitive_to_config() -> None:
    first = {"b": 2, "a": 1}
    second = {"a": 1, "b": 2}
    assert compute_run_id(first) == compute_run_id(second)
    assert len(compute_run_id(first)) == 16
    assert compute_run_id(first) != compute_run_id({"a": 1, "b": 3})


def test_materialized_config_excludes_machine_local_model_path_and_device() -> None:
    materialized = materialize_run_config(_config(), _model())
    encoded = canonical_json_text(materialized)
    assert "/machine/local/model" not in encoded
    assert '"device"' not in encoded
    assert materialized["model"]["id"] == "example/model"
    assert materialized["model"]["prompt_add_special_tokens"] is True


def test_method_params_hash_is_order_independent() -> None:
    left = compute_method_params_hash({"precision": 16, "top_k": 50000})
    right = compute_method_params_hash({"top_k": 50000, "precision": 16})
    assert left == right
    assert len(left) == 16
    assert left != compute_method_params_hash({"precision": 16, "top_k": 1000})


def test_result_record_retains_benchmark_native_infinite_kl_in_memory() -> None:
    record = build_result_record(_execution(), run_id="0123456789abcdef")
    assert record["kl_mean_bits"] == math.inf
    assert record["status"] == "ok"
    assert record["error"] is None


def test_summary_row_matches_v01_required_identity_and_metrics() -> None:
    row = build_summary_row(
        _execution(), model_config=_model(), run_id="0123456789abcdef"
    )
    assert row["run_id"] == "0123456789abcdef"
    assert row["method_id"] == "bins"
    assert row["carrier_tokens"] == 2
    assert row["kl_mean_bits"] == math.inf
    assert row["status"] == "ok"


def test_success_persistence_writes_canonical_run_artifacts_without_parquet(tmp_path: Path) -> None:
    execution = _execution()
    artifacts = persist_experiment_execution(
        execution,
        model_config=_model(),
        results_root=tmp_path,
        update_summary=False,
    )
    config_text = artifacts.config_path.read_text(encoding="utf-8")
    assert compute_run_id(json.loads(config_text)) == artifacts.run_id
    assert artifacts.stegotext_path is not None
    assert artifacts.stegotext_path.read_text(encoding="utf-8") == "hello world"

    persisted_result = json.loads(artifacts.result_path.read_text(encoding="utf-8"))
    assert persisted_result["kl_mean_bits"] == "inf"
    assert persisted_result["run_id"] == artifacts.run_id

    assert artifacts.token_ids_path is not None
    token_ids = json.loads(artifacts.token_ids_path.read_text(encoding="utf-8"))
    assert token_ids["sender_token_ids"] == [20, 21]
    assert token_ids["receiver_token_ids"] == [20, 21]

    assert artifacts.trace_path is not None
    with gzip.open(artifacts.trace_path, "rt", encoding="utf-8") as handle:
        trace = [json.loads(line) for line in handle]
    assert len(trace) == 2
    assert trace[0]["bits_consumed_total"] == 2
    assert trace[1]["bits_consumed_total"] == 4
    assert trace[0]["kl_ref_to_stego_bits"] == "inf"
    assert artifacts.summary_path is None


def test_success_persistence_is_idempotent_for_same_run_id(tmp_path: Path) -> None:
    first = persist_experiment_execution(
        _execution(), model_config=_model(), results_root=tmp_path, update_summary=False
    )
    second = persist_experiment_execution(
        _execution(), model_config=_model(), results_root=tmp_path, update_summary=False
    )
    assert first.run_id == second.run_id
    assert first.run_directory == second.run_directory
    assert len(list((tmp_path / "runs").iterdir())) == 1


def test_failed_run_persists_config_and_error_and_removes_success_only_files(tmp_path: Path) -> None:
    success = persist_experiment_execution(
        _execution(), model_config=_model(), results_root=tmp_path, update_summary=False
    )
    failed = persist_failed_run(
        config=_config(),
        model_config=_model(),
        error=RuntimeError("boom"),
        results_root=tmp_path,
        update_summary=False,
    )
    assert failed.run_id == success.run_id
    result = json.loads(failed.result_path.read_text(encoding="utf-8"))
    assert result["status"] == "error"
    assert result["error"] == {"message": "boom", "type": "RuntimeError"}
    assert not (failed.run_directory / "stegotext.txt").exists()
    assert not (failed.run_directory / "token_ids.json").exists()
    assert not (failed.run_directory / "trace.jsonl.gz").exists()


class _FakeField:
    def __init__(self, name: str):
        self.name = name


class _FakeSchema:
    def __init__(self, names: list[str]):
        self.names = names

    def __iter__(self):
        return iter([_FakeField(name) for name in self.names])


class _FakePA:
    class _TypeFactory:
        @staticmethod
        def from_pylist(rows, schema):
            return _FakeTable(list(rows))

    Table = _TypeFactory()

    @staticmethod
    def string(): return "string"
    @staticmethod
    def int64(): return "int64"
    @staticmethod
    def float64(): return "float64"
    @staticmethod
    def bool_(): return "bool"

    @staticmethod
    def schema(fields):
        return _FakeSchema([name for name, _ in fields])


class _FakeTable:
    def __init__(self, rows):
        self._rows = rows

    def to_pylist(self):
        return list(self._rows)


class _FakePQ:
    @staticmethod
    def write_table(table, path, compression=None):
        Path(path).write_text(json.dumps(table.to_pylist()), encoding="utf-8")

    @staticmethod
    def read_table(path, schema=None):
        return _FakeTable(json.loads(Path(path).read_text(encoding="utf-8")))


def test_summary_parquet_upsert_is_one_row_per_run_id(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(run_store, "_import_pyarrow", lambda: (_FakePA, _FakePQ))
    path = tmp_path / "summary.parquet"
    row = build_summary_row(_execution(), model_config=_model(), run_id="run-a")
    update_summary_parquet(path, row)
    changed = dict(row)
    changed["bits_per_token"] = 9.0
    update_summary_parquet(path, changed)

    stored = json.loads(path.read_text(encoding="utf-8"))
    assert len(stored) == 1
    assert stored[0]["run_id"] == "run-a"
    assert stored[0]["bits_per_token"] == 9.0


def test_summary_parquet_keeps_distinct_run_ids(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(run_store, "_import_pyarrow", lambda: (_FakePA, _FakePQ))
    path = tmp_path / "summary.parquet"
    first = build_summary_row(_execution(), model_config=_model(), run_id="run-b")
    second = dict(first)
    second["run_id"] = "run-a"
    update_summary_parquet(path, first)
    update_summary_parquet(path, second)
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert [row["run_id"] for row in stored] == ["run-a", "run-b"]


def test_summary_backend_error_is_explicit(monkeypatch) -> None:
    def _missing():
        raise StorageError("summary.parquet requires the storage extra")

    monkeypatch.setattr(run_store, "_import_pyarrow", _missing)
    with pytest.raises(StorageError, match="storage extra"):
        run_store.ensure_summary_backend_available()
