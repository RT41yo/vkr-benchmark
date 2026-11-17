#!/usr/bin/env python3
"""Generate matched ordinary controls from a validated Stage-7 sample index.

Never touches results/summary.parquet or results/runs/. Loads a given LM once.
Use --limit 1 for first GPU smoke, --all after validating the one-run output.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vkr_benchmark.config import LocalModelConfig  # noqa: E402
from vkr_benchmark.distributions import GenerationPolicy, ReferenceDistributionBuilder  # noqa: E402
from vkr_benchmark.inputs import PromptRegistry  # noqa: E402
from vkr_benchmark.lm import HFCausalLMAdapter  # noqa: E402
from vkr_benchmark.randomness import ControlRandomSource  # noqa: E402
from vkr_benchmark.steganalysis.cover import generate_cover  # noqa: E402
from vkr_benchmark.steganalysis.protocol import cover_seed  # noqa: E402


def _read_plan(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--prompts-file", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/stage7/cover_runs")
    args = parser.parse_args()
    if (args.limit is None) == (not args.all):
        parser.error("choose exactly one of --limit N or --all")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    model_cfg = LocalModelConfig.from_json(args.model_config, project_root=ROOT)
    registry = PromptRegistry.from_jsonl(args.prompts_file)
    raw_plan = _read_plan(args.plan)
    selected = [row for row in raw_plan if row["model_id"] == model_cfg.model_id]
    if args.limit is not None:
        selected = selected[:args.limit]
    if not selected:
        parser.error("no samples for selected model")
    # All metadata checks complete before loading a multi-GB model.
    unique_samples: set[str] = set()
    for row in selected:
        if row["sample_id"] in unique_samples:
            raise ValueError("duplicate sample_id")
        unique_samples.add(row["sample_id"])
        registry.get(row["prompt_id"])
        expected_seed = cover_seed(model_id=row["model_id"], article_id=row["article_id"],
                                   method_point_id=f"{row['method_id']}/{row['method_point_id']}",
                                   replicate_id=row["replicate_id"])
        if row["cover_seed"] != expected_seed:
            raise ValueError("cover seed mismatch; sample index invalid")
        if not isinstance(row["carrier_tokens"], int) or row["carrier_tokens"] < 1:
            raise ValueError("invalid cover carrier length")
    lm = HFCausalLMAdapter.from_local_config(model_cfg)
    builder = ReferenceDistributionBuilder(token_space=lm.token_space,
                                           policy=GenerationPolicy(temperature=1.0, top_k=None,
                                                                   top_p=1.0, exclude_special_tokens=True))
    completed = skipped = 0
    for row in selected:
        folder = args.output_root / row["sample_id"]
        marker = folder / "cover_result.json"
        if marker.exists():
            if args.resume:
                old = json.loads(marker.read_text(encoding="utf-8"))
                if (old["stego_run_id"] == row["stego_run_id"]
                        and old["cover_seed"] == row["cover_seed"]
                        and old["carrier_tokens"] == row["carrier_tokens"]):
                    if (folder / "cover.txt").is_file() and (folder / "token_ids.json").is_file():
                        existing = (folder / "cover.txt").read_text(encoding="utf-8")
                        if hashlib.sha256(existing.encode("utf-8")).hexdigest() == old["text_sha256"]:
                            skipped += 1
                            continue
                raise ValueError(f"existing cover has different parameters or corrupt files: {folder}")
            raise ValueError(f"existing cover would be overwritten: {folder}; use --resume")
        result = generate_cover(lm_adapter=lm, reference_builder=builder,
                                prompt_text=registry.get(row["prompt_id"]).text,
                                carrier_tokens=row["carrier_tokens"],
                                random_source=ControlRandomSource(row["cover_seed"]))
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "cover.txt").write_text(result.text, encoding="utf-8")
        (folder / "token_ids.json").write_text(json.dumps({
            "prompt_token_ids": result.prompt_token_ids,
            "sender_token_ids": result.sender_token_ids,
            "receiver_token_ids": result.receiver_token_ids,
            "token_sequence_roundtrip_exact": result.token_sequence_roundtrip_exact,
            "first_token_mismatch": result.first_token_mismatch,
        }, indent=2) + "\n", encoding="utf-8")
        marker.write_text(json.dumps({
            "schema_version": "stage7.cover_run.v1", "sample_id": row["sample_id"],
            "stego_run_id": row["stego_run_id"], "model_id": row["model_id"],
            "model_revision": lm.revision, "prompt_id": row["prompt_id"],
            "cover_seed": row["cover_seed"], "carrier_tokens": row["carrier_tokens"],
            "text_sha256": hashlib.sha256(result.text.encode("utf-8")).hexdigest(),
            "token_sequence_roundtrip_exact": result.token_sequence_roundtrip_exact,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        completed += 1
        print(f"COVER OK {row['sample_id']} model={row['model_id']} tokens={row['carrier_tokens']}", flush=True)
    print(f"Cover generation complete: {completed} new, {skipped} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
