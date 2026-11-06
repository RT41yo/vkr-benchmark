from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/summarize_stage5_rrc_payload_calibration.py"


def _load():
    spec = importlib.util.spec_from_file_location("rrc_payload_calibration", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rrc_payload_summary_reports_paired_candidate_statistics(tmp_path: Path) -> None:
    module = _load()
    execution = tmp_path / "execution.jsonl"
    results_root = tmp_path / "results"
    rows = []

    for payload in (128, 256, 512, 1024):
        point_id = f"payload{payload}"
        for prompt_index in range(1, 5):
            for replicate_index in range(1, 3):
                run_id = f"run-{payload}-{prompt_index}-{replicate_index}"
                carrier_tokens = payload // 4
                rows.append(
                    {
                        "sweep_id": module.SWEEP_ID,
                        "run_id": run_id,
                        "method_point_id": point_id,
                        "prompt_id": f"s5p{prompt_index:04d}",
                        "replicate_id": f"r{replicate_index:02d}",
                        "status": "ok",
                        "elapsed_wall_seconds": 1.0,
                        "bits_per_token": 4.0,
                        "entropy_utilization": 1.0,
                        "roundtrip_exact": True,
                        "ber": 0.0,
                        "kl_ref_to_stego_mean_bits": 0.0,
                        "kl_stego_to_ref_mean_bits": 0.0,
                        "tvd_mean": 0.0,
                        "ppl_raw_lm": 5.0,
                        "encode_ms_per_token": 1.0,
                        "decode_ms_per_token": 1.0,
                        "stego_algorithm_total_ms": 1.0,
                        "metric_instrumentation_total_ms": 1.0,
                    }
                )
                run_dir = results_root / "runs" / run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "result.json").write_text(
                    json.dumps(
                        {
                            "carrier_tokens": carrier_tokens,
                            "token_sequence_roundtrip_exact": True,
                            "first_token_roundtrip_mismatch": None,
                        }
                    ),
                    encoding="utf-8",
                )

    execution.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    result = module.summarize(execution, results_root)

    assert result["terminal_count"] == 32
    assert result["all_runs_terminal"] is True
    assert result["all_payloads_numeric_clean"] is True
    assert result["closest_median_carrier_to_reference_payload_bits"] == 1024
    assert result["by_payload_bits"]["128"]["carrier_tokens"]["median"] == 32
    assert result["by_payload_bits"]["1024"]["carrier_tokens"]["median"] == 256
    assert result["by_payload_bits"]["1024"]["transport_exact_count"] == 8
