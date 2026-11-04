from __future__ import annotations

from pathlib import Path

from vkr_benchmark.sweeps import SweepManifest

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs" / "experiments"

EXPECTED_STREAMING_POINTS = {
    "bins": ["b1", "b4", "b6"],
    "huffman": ["e2", "e4", "e8"],
    "arithmetic_coding": ["p32-k128", "p32-k2048", "p32-k50000"],
    "adg": ["default"],
    "discop": ["default"],
    "dairstega": ["a8-b1-k100", "a16-b075-k100", "a32-b05-k100"],
}


def _load(name: str) -> SweepManifest:
    return SweepManifest.from_json(CONFIG_DIR / name, project_root=REPO_ROOT)


def test_stage6_llama_manifests_have_expected_scale_and_points() -> None:
    streaming = _load("stage6_main_streaming_llama.json")
    rrc = _load("stage6_main_rrc_llama.json")

    assert streaming.benchmark_version == "1.0"
    assert rrc.benchmark_version == "1.0"
    assert streaming.prompt_ids == tuple(f"s6p{i:04d}" for i in range(1, 25))
    assert len(streaming.replicates) == 3
    assert streaming.planned_run_count() == 1008
    assert rrc.planned_run_count() == 72

    actual = {
        method.method_id: [point.point_id for point in method.points]
        for method in streaming.methods
    }
    assert actual == EXPECTED_STREAMING_POINTS

    assert len(rrc.methods) == 1
    rrc_method = rrc.methods[0]
    assert rrc_method.method_id == "rrc"
    assert [point.point_id for point in rrc_method.points] == ["payload1024-margin40"]
    assert rrc_method.termination.mode == "fixed_payload_bits"
    assert rrc_method.termination.target_payload_bits == 1024


def test_stage6_qwen_manifests_use_same_prompts_points_and_first_two_secrets() -> None:
    llama_streaming = _load("stage6_main_streaming_llama.json")
    qwen_streaming = _load("stage6_main_streaming_qwen.json")
    qwen_rrc = _load("stage6_main_rrc_qwen.json")

    assert qwen_streaming.prompt_ids == llama_streaming.prompt_ids
    assert qwen_streaming.replicates == llama_streaming.replicates[:2]
    assert qwen_streaming.planned_run_count() == 672
    assert qwen_rrc.planned_run_count() == 48

    llama_points = {
        method.method_id: [point.point_id for point in method.points]
        for method in llama_streaming.methods
    }
    qwen_points = {
        method.method_id: [point.point_id for point in method.points]
        for method in qwen_streaming.methods
    }
    assert qwen_points == llama_points == EXPECTED_STREAMING_POINTS


def test_stage6_all_four_manifests_total_1800_runs() -> None:
    names = [
        "stage6_main_streaming_llama.json",
        "stage6_main_rrc_llama.json",
        "stage6_main_streaming_qwen.json",
        "stage6_main_rrc_qwen.json",
    ]
    manifests = [_load(name) for name in names]
    assert sum(manifest.planned_run_count() for manifest in manifests) == 1800

    all_run_ids: list[str] = []
    for manifest in manifests:
        all_run_ids.extend(run.run_id for run in manifest.expand())
    assert len(all_run_ids) == 1800
    assert len(set(all_run_ids)) == 1800
