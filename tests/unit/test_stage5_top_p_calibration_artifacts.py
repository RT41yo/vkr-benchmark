from __future__ import annotations

import json
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_DIR = REPO_ROOT / "results/stage5/sweeps/stage5-top-p-calibration-llama-v1"
SUMMARY = REPO_ROOT / "results/stage5/top_p_calibration_summary.json"


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_committed_top_p_calibration_logs_are_complete() -> None:
    execution = _jsonl(SWEEP_DIR / "execution.jsonl")
    support = _jsonl(SWEEP_DIR / "support_probe_execution.jsonl")

    assert len(execution) == 14
    assert all(row["status"] == "ok" for row in execution)
    assert all(row["generation_point_id"] == "tp100" for row in execution)
    assert all(row["roundtrip_exact"] is True for row in execution)
    assert max(float(row["ber"]) for row in execution) == 0.0
    assert {row["method_id"] for row in execution} == {
        "bins",
        "huffman",
        "arithmetic_coding",
        "adg",
        "discop",
        "rrc",
        "dairstega",
    }

    tp098 = next(row for row in support if row["generation_point_id"] == "tp098")
    tp100 = next(row for row in support if row["generation_point_id"] == "tp100")
    assert tp098["status"] == "error"
    assert tp098["error_type"] == "UnsupportedConfigurationError"
    assert "bin(s) [0, 1, 2]" in str(tp098["error_message"])
    assert tp100["status"] == "ok"


def test_top_p_summary_matches_committed_logs() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    execution = _jsonl(SWEEP_DIR / "execution.jsonl")

    assert summary["decision"]["common_top_p"] == 1.0
    assert summary["decision"]["common_top_k"] is None
    assert summary["main_tp100"]["run_count"] == 14
    assert summary["main_tp100"]["ok_count"] == 14
    assert summary["main_tp100"]["failed_count"] == 0
    assert summary["main_tp100"]["all_roundtrip_exact"] is True
    assert summary["main_tp100"]["ber_max"] == 0.0
    assert math.isclose(
        summary["main_tp100"]["sum_elapsed_wall_seconds"],
        sum(float(row["elapsed_wall_seconds"]) for row in execution),
        rel_tol=0.0,
        abs_tol=1e-9,
    )
