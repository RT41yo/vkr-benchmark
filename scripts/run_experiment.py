#!/usr/bin/env python3
"""Run one Stage-2 normalized experiment from a single JSON config.

Step 7.4 completes the base single-run metric block with raw-LM NLL/PPL,
reliability/BER, and computational-efficiency timing. Step 7.5 persists canonical run artifacts and updates summary.parquet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vkr_benchmark.config import ExperimentConfig, LocalModelConfig
from vkr_benchmark.inputs import PromptRegistry
from vkr_benchmark.lm import HFCausalLMAdapter
from vkr_benchmark.runner import run_experiment
from vkr_benchmark.storage import (
    ensure_summary_backend_available,
    persist_experiment_execution,
    persist_failed_run,
)


def _format_metric(value: float | None, *, digits: int = 6) -> str:
    if value is None:
        return "unavailable"
    if value == float("inf"):
        return "inf"
    return f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_config", type=Path)
    parser.add_argument(
        "--prompts",
        type=Path,
        default=None,
        help="Prompt JSONL path; defaults to <project_root>/data/prompts.jsonl",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Results directory; defaults to <project_root>/results",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run diagnostics without writing run artifacts",
    )
    parser.add_argument(
        "--skip-summary-parquet",
        action="store_true",
        help="Persist per-run files but do not update summary.parquet",
    )
    args = parser.parse_args()

    experiment = ExperimentConfig.from_json(args.experiment_config)
    assert experiment.source_path is not None
    project_root = experiment.source_path.parents[2]
    prompts_path = (
        args.prompts.expanduser().resolve()
        if args.prompts is not None
        else project_root / "data" / "prompts.jsonl"
    )
    prompts = PromptRegistry.from_jsonl(prompts_path)
    results_root = (
        args.results_root.expanduser().resolve()
        if args.results_root is not None
        else project_root / "results"
    )

    model_config = LocalModelConfig.from_json(
        experiment.model_config_path,
        project_root=project_root,
    )
    update_summary = not args.skip_summary_parquet
    if not args.no_save and update_summary:
        # Fail before loading the multi-GB model if the optional storage backend
        # is missing.
        ensure_summary_backend_available()

    print("Unified Stage-2 experiment runner")
    print("experiment config:", experiment.source_path)
    print("model:", model_config.model_id)
    print("revision:", model_config.revision)
    print("prompt_id:", experiment.prompt_id)
    print("method:", experiment.method.method_id)
    print("method params:", dict(experiment.method.params))
    print("secret_id:", experiment.secret_id)
    print(
        "target carrier tokens:",
        experiment.termination.target_carrier_tokens,
    )
    print("Loading local model...")

    try:
        lm = HFCausalLMAdapter.from_local_config(model_config)
        execution = run_experiment(
            config=experiment,
            prompt_registry=prompts,
            lm_adapter=lm,
        )
    except Exception as exc:
        if not args.no_save:
            artifacts = persist_failed_run(
                config=experiment,
                model_config=model_config,
                error=exc,
                results_root=results_root,
                update_summary=update_summary,
            )
            print("run_id:", artifacts.run_id)
            print("failed run saved to:", artifacts.run_directory)
        raise

    artifacts = None
    if not args.no_save:
        artifacts = persist_experiment_execution(
            execution,
            model_config=model_config,
            results_root=results_root,
            update_summary=update_summary,
        )

    result = execution.roundtrip
    capacity = execution.capacity_entropy_metrics
    distortion = execution.distribution_distortion_metrics
    quality = execution.quality_metrics
    reliability = execution.reliability_metrics
    performance = execution.performance_metrics

    print()
    print("generated carrier tokens:", result.encode.carrier_tokens)
    print("payload bits:", result.encode.payload_bits)
    print("secret bits read:", result.encode.secret_bits_read)
    print("bits per token:", f"{capacity.bits_per_token:.6f}")
    print(
        "reference entropy mean (bits/token):",
        f"{capacity.reference_entropy_mean_bits:.6f}",
    )
    print(
        "reference entropy sum (bits):",
        f"{capacity.reference_entropy_sum_bits:.6f}",
    )
    print("entropy utilization:", f"{capacity.entropy_utilization:.6f}")
    print(
        "entropy utilization (%):",
        f"{capacity.entropy_utilization_percent:.3f}",
    )

    print("q_mode:", distortion.q_mode.value)
    print("KL mean (bits/token):", _format_metric(distortion.kl_mean_bits))
    print("KL median (bits/token):", _format_metric(distortion.kl_median_bits))
    print("KL p95 (bits/token):", _format_metric(distortion.kl_p95_bits))
    print("KL max (bits/token):", _format_metric(distortion.kl_max_bits))
    print("KL infinite steps:", distortion.kl_infinite_steps)
    print("KL finite steps:", distortion.kl_finite_steps)
    print("TVD mean:", _format_metric(distortion.tvd_mean))
    print("TVD median:", _format_metric(distortion.tvd_median))
    print("TVD p95:", _format_metric(distortion.tvd_p95))
    print("TVD max:", _format_metric(distortion.tvd_max))

    print(
        "raw-LM NLL (nats/token):",
        _format_metric(quality.nll_raw_lm_nats_per_token),
    )
    print("raw-LM PPL:", _format_metric(quality.ppl_raw_lm))

    print("BER:", _format_metric(reliability.ber))
    print("bit errors:", reliability.bit_errors)
    print("token_sequence_roundtrip_exact:", reliability.token_sequence_roundtrip_exact)
    print("roundtrip_exact:", reliability.roundtrip_exact)
    print("decoder complete:", result.decode.finalization.complete)
    print("expected length bits:", reliability.expected_length_bits)
    print("recovered length bits:", reliability.recovered_length_bits)
    print("length delta bits:", reliability.length_delta_bits)
    print("recovered extra bits:", reliability.recovered_extra_bits)
    print("first_token_mismatch:", reliability.first_token_roundtrip_mismatch)
    print("first_bit_mismatch:", reliability.first_mismatch_bit)

    print("encode total (ms):", _format_metric(performance.encode_total_ms, digits=3))
    print("decode total (ms):", _format_metric(performance.decode_total_ms, digits=3))
    print(
        "encode ms/token:",
        _format_metric(performance.encode_ms_per_token, digits=3),
    )
    print(
        "decode ms/token:",
        _format_metric(performance.decode_ms_per_token, digits=3),
    )
    print(
        "payload bits/s encode:",
        _format_metric(performance.payload_bits_per_second_encode, digits=3),
    )
    print(
        "payload bits/s decode:",
        _format_metric(performance.payload_bits_per_second_decode, digits=3),
    )
    print(
        "LM forward total (ms):",
        _format_metric(performance.lm_forward_total_ms, digits=3),
    )
    print(
        "distribution processing total (ms):",
        _format_metric(performance.distribution_processing_total_ms, digits=3),
    )
    print(
        "stego algorithm total (ms):",
        _format_metric(performance.stego_algorithm_total_ms, digits=3),
    )

    print()
    print("stegotext:")
    print(result.transport.text)
    if artifacts is not None:
        print()
        print("run_id:", artifacts.run_id)
        print("run directory:", artifacts.run_directory)
        if artifacts.summary_path is not None:
            print("summary parquet:", artifacts.summary_path)
        else:
            print("summary parquet: skipped")
    print()
    print("Unified experiment pipeline: COMPLETED")


if __name__ == "__main__":
    main()
