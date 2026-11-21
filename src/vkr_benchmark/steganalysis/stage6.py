"""Read-only inventory of frozen Stage-6 stegotexts for external detector test."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from vkr_benchmark.errors import ConfigurationError
from vkr_benchmark.storage.run_store import compute_method_params_hash, compute_run_id
from vkr_benchmark.steganalysis.protocol import cover_seed
from vkr_benchmark.steganalysis.textio import read_carrier_text


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"cannot read required Stage-6 artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(f"Stage-6 artifact must be a JSON object: {path}")
    return value


def _require_text(value: Any, desc: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"missing/empty {desc}")
    return value


def build_stage6_test_index(
    *, run_root: Path, frozen_manifest: dict[str, Any],
    stage6_prompts: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate all frozen runs, including failed decodes with transport_text.

    No artifacts are ever modified. An incomplete local run set is a hard error;
    the detector must not learn a selectively filtered successful-run subset.
    """
    expected_count = int(frozen_manifest["dataset_scope"]["canonical_run_count"])
    prompt_map = {row["prompt_id"]: row for row in stage6_prompts["prompt_records"]}
    if len(prompt_map) != 24:
        raise ConfigurationError("Stage-6 prompt manifest must have 24 distinct prompt IDs")
    points = {
        (point["method_id"], compute_method_params_hash(point["params"])): point["method_point_id"]
        for point in frozen_manifest["operating_points"]
    }
    expected_models = {value["model_id"] for value in frozen_manifest["inputs"]["models"].values()}
    run_dirs = sorted(p for p in run_root.iterdir() if p.is_dir()) if run_root.exists() else []
    # Other stages may coexist in results/runs; select Stage 6 by prompt and
    # generation configs rather than insisting that all folders belong to it.
    rows: list[dict[str, Any]] = []
    statuses = {"ok": 0, "error": 0}
    seen: set[tuple[str, str, str, str, str]] = set()
    for directory in run_dirs:
        config_path = directory / "config.json"
        if not config_path.is_file():
            continue
        config = _load(config_path)
        if config.get("prompt_id") not in prompt_map or config.get("model", {}).get("id") not in expected_models:
            continue
        method = config.get("method", {})
        if not isinstance(method, dict):
            continue
        point_hash = compute_method_params_hash(method.get("params", {}))
        point_id = points.get((method.get("id"), point_hash))
        if point_id is None or not str(config.get("secret_id", "")).startswith("stage6-main-"):
            continue
        if compute_run_id(config) != directory.name:
            raise ConfigurationError(f"Stage-6 run_id checksum mismatch: {directory.name}")
        result = _load(directory / "result.json")
        status = result.get("status")
        if status not in statuses or result.get("run_id") != directory.name:
            raise ConfigurationError(f"invalid Stage-6 run status/id: {directory.name}")
        prompt = prompt_map[config["prompt_id"]]
        model_id = config["model"]["id"]
        secret_id = config["secret_id"]
        replicate_id = f"r{int(secret_id.removeprefix('stage6-main-')):02d}"
        # Operating-point labels are local to their method: e.g. both ADG and
        # Discop use "default". Include method_id in the stratum identity.
        method_id = method["id"]
        match = (model_id, config["prompt_id"], method_id, point_id, replicate_id)
        if match in seen:
            raise ConfigurationError(f"duplicate Stage-6 experimental stratum: {match}")
        seen.add(match)
        if status == "ok":
            text = read_carrier_text(directory / "stegotext.txt")
            sender_ids = _load(directory / "token_ids.json").get("sender_token_ids")
            text_source = "stegotext.txt"
        else:
            failure = _load(directory / "failure_diagnostics.json")
            text = failure.get("transport_text")
            sender_ids = failure.get("sender_token_ids")
            text_source = "failure_diagnostics.json:transport_text"
        _require_text(text, f"Stage-6 {directory.name} {text_source}")
        if (not isinstance(sender_ids, list) or not sender_ids
                or any(type(item) is not int or item < 0 for item in sender_ids)):
            raise ConfigurationError(f"Stage-6 missing sender carrier tokens: {directory.name}")
        if method_id != "rrc" and len(sender_ids) != 256:
            raise ConfigurationError(f"streaming carrier length must be 256: {directory.name}")
        if method_id == "rrc" and config["termination"].get("mode") != "fixed_payload_bits":
            raise ConfigurationError("Stage-6 RRC semantics were not fixed-payload")
        statuses[status] += 1
        sample_id = hashlib.sha256(("stage7-cover|" + directory.name).encode()).hexdigest()[:24]
        rows.append({
            "sample_id": sample_id, "partition": "test", "stego_run_id": directory.name,
            "stego_status": status, "stego_text_source": text_source,
            "stego_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "model_id": model_id, "prompt_id": config["prompt_id"],
            "article_id": prompt["article_id"], "method_id": method_id,
            "method_point_id": point_id, "replicate_id": replicate_id,
            "carrier_tokens": len(sender_ids),
            "cover_seed": cover_seed(model_id=model_id, article_id=prompt["article_id"],
                                     method_point_id=f"{method_id}/{point_id}", replicate_id=replicate_id),
        })
    if len(rows) != expected_count:
        raise ConfigurationError(
            f"Stage-6 runs incomplete: found {len(rows)} of {expected_count}; "
            "restore canonical results/runs/ before proceeding"
        )
    expected_status = frozen_manifest["dataset_scope"]["canonical_status_counts"]
    if statuses != expected_status:
        raise ConfigurationError(f"Stage-6 status counts changed: {statuses} != {expected_status}")
    rows.sort(key=lambda item: item["stego_run_id"])
    return rows, {"schema_version": "stage7.stage6_source_audit.v1", "status": "ok",
                  "run_count": len(rows), "status_counts": statuses,
                  "models": sorted(expected_models), "operating_point_count": len(points)}
