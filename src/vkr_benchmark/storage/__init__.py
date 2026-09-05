"""Persistent run storage and canonical benchmark identifiers."""

from vkr_benchmark.storage.run_store import (
    RunArtifacts,
    build_failure_result_record,
    build_failure_summary_row,
    build_result_record,
    build_summary_row,
    build_token_ids_record,
    canonical_json_text,
    compute_method_params_hash,
    compute_run_id,
    ensure_summary_backend_available,
    iter_trace_records,
    materialize_run_config,
    persist_experiment_execution,
    persist_failed_run,
    update_summary_parquet,
)

__all__ = [
    "RunArtifacts",
    "build_failure_result_record",
    "build_failure_summary_row",
    "build_result_record",
    "build_summary_row",
    "build_token_ids_record",
    "canonical_json_text",
    "compute_method_params_hash",
    "compute_run_id",
    "ensure_summary_backend_available",
    "iter_trace_records",
    "materialize_run_config",
    "persist_experiment_execution",
    "persist_failed_run",
    "update_summary_parquet",
]
