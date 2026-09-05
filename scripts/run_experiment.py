#!/usr/bin/env python3
"""Run one Stage-2 normalized experiment from a single JSON config.

Step 7.2 adds the first common benchmark metric block: useful capacity and
reference-entropy utilization. Persistent run storage and the remaining metric
groups are added in later Stage-2 steps.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vkr_benchmark.config import ExperimentConfig, LocalModelConfig
from vkr_benchmark.inputs import PromptRegistry
from vkr_benchmark.lm import HFCausalLMAdapter
from vkr_benchmark.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_config", type=Path)
    parser.add_argument(
        "--prompts",
        type=Path,
        default=None,
        help="Prompt JSONL path; defaults to <project_root>/data/prompts.jsonl",
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

    model_config = LocalModelConfig.from_json(
        experiment.model_config_path,
        project_root=project_root,
    )

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

    lm = HFCausalLMAdapter.from_local_config(model_config)
    execution = run_experiment(
        config=experiment,
        prompt_registry=prompts,
        lm_adapter=lm,
    )
    result = execution.roundtrip
    metrics = execution.capacity_entropy_metrics

    print()
    print("generated carrier tokens:", result.encode.carrier_tokens)
    print("payload bits:", result.encode.payload_bits)
    print("secret bits read:", result.encode.secret_bits_read)
    print("bits per token:", f"{metrics.bits_per_token:.6f}")
    print(
        "reference entropy mean (bits/token):",
        f"{metrics.reference_entropy_mean_bits:.6f}",
    )
    print(
        "reference entropy sum (bits):",
        f"{metrics.reference_entropy_sum_bits:.6f}",
    )
    print("entropy utilization:", f"{metrics.entropy_utilization:.6f}")
    print(
        "entropy utilization (%):",
        f"{metrics.entropy_utilization_percent:.3f}",
    )
    print("token_sequence_roundtrip_exact:", result.transport.token_sequence_roundtrip_exact)
    print("roundtrip_exact:", result.roundtrip_exact)
    print("decoder complete:", result.decode.finalization.complete)
    print("first_token_mismatch:", result.transport.first_token_mismatch)
    print("first_bit_mismatch:", result.first_bit_mismatch)
    print()
    print("stegotext:")
    print(result.transport.text)
    print()
    print("Unified experiment pipeline: COMPLETED")


if __name__ == "__main__":
    main()
