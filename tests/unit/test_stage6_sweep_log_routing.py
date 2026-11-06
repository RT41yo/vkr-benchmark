from __future__ import annotations

from pathlib import Path
import runpy

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_stage5_sweep.py"


def test_stage6_sweep_logs_are_routed_to_stage6() -> None:
    namespace = runpy.run_path(str(SCRIPT), run_name="stage6_sweep_log_routing_test")
    route = namespace["_sweep_log_stage"]
    assert route("stage6-main-streaming-llama-v1") == "stage6"
    assert route("stage6-main-rrc-qwen-v1") == "stage6"
    assert route("stage5-pilot-b-streaming-llama-v1") == "stage5"
