from __future__ import annotations

import json
from pathlib import Path

import pytest

from vkr_benchmark.errors import ConfigurationError
from vkr_benchmark.steganalysis.stage6 import build_stage6_test_index
from vkr_benchmark.storage.run_store import compute_run_id


def _fixture(root: Path):
    frozen = {
        "dataset_scope": {"canonical_run_count": 2, "canonical_status_counts": {"ok": 1, "error": 1}},
        "operating_points": [{"method_id": "bins", "method_point_id": "b1", "params": {"block_size": 1}}],
        "inputs": {"models": {"Llama": {"model_id": "test/llama"}}},
    }
    prompts = {"prompt_records": [
        {"prompt_id": f"s6p{i:04d}", "article_id": f"article-{i}"} for i in range(1,25)
    ]}
    for pos, status in enumerate(("ok", "error"), start=1):
        config = {"benchmark_version": "1.0", "model": {"id": "test/llama"},
                  "method": {"id": "bins", "params": {"block_size": 1}},
                  "prompt_id": f"s6p{pos:04d}", "secret_id": "stage6-main-0001",
                  "termination": {"mode": "fixed_carrier_tokens", "target_carrier_tokens": 256}}
        run_id = compute_run_id(config)
        folder = root / run_id
        folder.mkdir(parents=True)
        (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (folder / "result.json").write_text(json.dumps({"run_id": run_id, "status": status}), encoding="utf-8")
        sender = [1] * 256
        if status == "ok":
            (folder / "stegotext.txt").write_text("hello", encoding="utf-8")
            (folder / "token_ids.json").write_text(json.dumps({"sender_token_ids": sender}), encoding="utf-8")
        else:
            (folder / "failure_diagnostics.json").write_text(json.dumps({
                "failure_phase": "decode", "transport_text": "failed but visible",
                "sender_token_ids": sender,
            }), encoding="utf-8")
    return frozen, prompts


def test_stage6_index_includes_hard_failures_with_transport_text(tmp_path: Path):
    frozen, prompts = _fixture(tmp_path)
    rows, audit = build_stage6_test_index(run_root=tmp_path, frozen_manifest=frozen,
                                           stage6_prompts=prompts)
    assert audit["run_count"] == 2
    assert audit["status_counts"] == {"ok": 1, "error": 1}
    assert {row["stego_status"] for row in rows} == {"ok", "error"}
    assert all(row["replicate_id"] == "r01" for row in rows)
    assert all(row["carrier_tokens"] == 256 for row in rows)
    assert {row["method_point_id"] for row in rows} == {"b1"}


def test_stage6_index_rejects_missing_failure_text_not_filtering(tmp_path: Path):
    frozen, prompts = _fixture(tmp_path)
    failure = next(p for p in tmp_path.glob("*/failure_diagnostics.json"))
    content = json.loads(failure.read_text())
    content["transport_text"] = ""
    failure.write_text(json.dumps(content))
    with pytest.raises(ConfigurationError, match="transport_text"):
        build_stage6_test_index(run_root=tmp_path, frozen_manifest=frozen,
                                stage6_prompts=prompts)


def test_stage6_index_rejects_incomplete_snapshot(tmp_path: Path):
    frozen, prompts = _fixture(tmp_path)
    frozen["dataset_scope"]["canonical_run_count"] = 1800
    with pytest.raises(ConfigurationError, match="found 2 of 1800"):
        build_stage6_test_index(run_root=tmp_path, frozen_manifest=frozen,
                                stage6_prompts=prompts)


def test_stage6_index_allows_same_point_label_across_different_methods(tmp_path: Path):
    """ADG and Discop both name their operating point 'default'."""
    frozen = {
        "dataset_scope": {"canonical_run_count": 2, "canonical_status_counts": {"ok": 2, "error": 0}},
        "operating_points": [
            {"method_id": "adg", "method_point_id": "default", "params": {}},
            {"method_id": "discop", "method_point_id": "default", "params": {}},
        ],
        "inputs": {"models": {"Llama": {"model_id": "test/llama"}}},
    }
    prompts = {"prompt_records": [
        {"prompt_id": f"s6p{i:04d}", "article_id": f"article-{i}"} for i in range(1, 25)
    ]}
    for method_id in ("adg", "discop"):
        config = {
            "benchmark_version": "1.0", "model": {"id": "test/llama"},
            "method": {"id": method_id, "params": {}},
            "prompt_id": "s6p0016", "secret_id": "stage6-main-0001",
            "termination": {"mode": "fixed_carrier_tokens", "target_carrier_tokens": 256},
        }
        run_id = compute_run_id(config)
        folder = tmp_path / run_id
        folder.mkdir()
        (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (folder / "result.json").write_text(json.dumps({"run_id": run_id, "status": "ok"}), encoding="utf-8")
        (folder / "stegotext.txt").write_text("visible text", encoding="utf-8")
        (folder / "token_ids.json").write_text(json.dumps({"sender_token_ids": [1] * 256}), encoding="utf-8")

    rows, audit = build_stage6_test_index(
        run_root=tmp_path, frozen_manifest=frozen, stage6_prompts=prompts
    )
    assert audit["run_count"] == 2
    assert {(r["method_id"], r["method_point_id"]) for r in rows} == {
        ("adg", "default"), ("discop", "default")
    }
    assert len({r["cover_seed"] for r in rows}) == 2


def test_stage6_index_rejects_actual_duplicate_within_one_method(tmp_path: Path):
    """Distinct run IDs must not silently duplicate the same experimental cell."""
    frozen, prompts = _fixture(tmp_path)
    frozen["dataset_scope"]["canonical_status_counts"] = {"ok": 2, "error": 1}
    frozen["dataset_scope"]["canonical_run_count"] = 3
    sample = next(p for p in tmp_path.glob("*/config.json") if json.loads(p.read_text())["prompt_id"] == "s6p0001")
    config = json.loads(sample.read_text(encoding="utf-8"))
    config["extra_metadata_for_duplicate_run"] = "distinct-run-same-stratum"
    run_id = compute_run_id(config)
    folder = tmp_path / run_id
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (folder / "result.json").write_text(json.dumps({"run_id": run_id, "status": "ok"}), encoding="utf-8")
    (folder / "stegotext.txt").write_text("another text", encoding="utf-8")
    (folder / "token_ids.json").write_text(json.dumps({"sender_token_ids": [1] * 256}), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="duplicate Stage-6 experimental stratum"):
        build_stage6_test_index(run_root=tmp_path, frozen_manifest=frozen,
                                stage6_prompts=prompts)
