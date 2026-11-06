#!/usr/bin/env python3
"""Validate and expand a Stage-5 sweep manifest without loading model weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vkr_benchmark.inputs import PromptRegistry
from vkr_benchmark.sweeps import SweepManifest


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSONL path for the expanded run plan.",
    )
    args = parser.parse_args()

    manifest = SweepManifest.from_json(args.manifest, project_root=REPO_ROOT)
    prompt_registry = PromptRegistry.from_jsonl(manifest.prompts_path)
    unknown = [pid for pid in manifest.prompt_ids if pid not in prompt_registry.prompt_ids]
    if unknown:
        raise SystemExit("unknown prompt ids in sweep: " + ", ".join(unknown))

    planned = manifest.expand()
    print("sweep_id:", manifest.sweep_id)
    print("purpose:", manifest.purpose)
    print("model_config:", manifest.model_config_path)
    print("prompts_file:", manifest.prompts_path)
    print("prompt_count:", len(manifest.prompt_ids))
    print("replicate_count:", len(manifest.replicates))
    print("generation_point_count:", len(manifest.generation_points))
    print("method_count:", len(manifest.methods))
    print("method_point_count:", sum(len(item.points) for item in manifest.methods))
    print("planned_run_count:", len(planned))

    by_method: dict[str, int] = {}
    for run in planned:
        by_method[run.config.method.method_id] = by_method.get(run.config.method.method_id, 0) + 1
    print("runs_by_method:")
    for method_id, count in sorted(by_method.items()):
        print(f"  {method_id}: {count}")

    if args.output is not None:
        output = args.output.expanduser().resolve()
        _write_jsonl(output, [run.metadata_record() for run in planned])
        print("expanded_plan:", output)


if __name__ == "__main__":
    main()
