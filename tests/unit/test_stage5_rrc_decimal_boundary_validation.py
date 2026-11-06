from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/summarize_stage5_rrc_decimal_boundary_validation.py"


def _load():
    spec = importlib.util.spec_from_file_location("rrc_decimal_validation", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rrc_decimal_boundary_summary_gate_accepts_eight_clean_runs(tmp_path: Path) -> None:
    module = _load()
    execution = tmp_path / "execution.jsonl"
    rows = []
    for prompt_index in range(1, 5):
        for replicate_index in range(1, 3):
            rows.append(
                {
                    "sweep_id": module.EXPECTED_SWEEP_ID,
                    "run_id": f"run-{prompt_index}-{replicate_index}",
                    "method_point_id": module.EXPECTED_POINT_ID,
                    "prompt_id": f"s5p{prompt_index:04d}",
                    "replicate_id": f"r{replicate_index:02d}",
                    "status": "ok",
                    "bits_per_token": 4.0,
                    "roundtrip_exact": True,
                    "ber": 0.0,
                    "kl_ref_to_stego_mean_bits": 0.0,
                    "kl_stego_to_ref_mean_bits": 0.0,
                    "tvd_mean": 0.0,
                }
            )
    execution.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    result = module.summarize(execution, None)

    assert result["completed_ok_count"] == 8
    assert result["finite_forward_kl_count"] == 8
    assert result["validation_gate"] is True
