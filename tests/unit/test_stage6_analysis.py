from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "analyze_stage6_main.py"
SPEC = importlib.util.spec_from_file_location("analyze_stage6_main", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


def _ok_row(*, run_id: str, roundtrip_exact: bool, forward_kl: float) -> dict[str, object]:
    return {
        "run_id": run_id,
        "status": "ok",
        "model": "Llama-3.2-3B",
        "family": "streaming",
        "method_id": "bins",
        "method_point_id": "b1",
        "roundtrip_exact": roundtrip_exact,
        "bits_per_token": 1.0,
        "entropy_utilization": 0.25,
        "reference_entropy_mean_bits": 4.0,
        "kl_ref_to_stego_mean_bits": forward_kl,
        "kl_stego_to_ref_mean_bits": 1.5,
        "tvd_mean": 0.5,
        "nll_raw_lm_nats_per_token": 2.0,
        "ppl_raw_lm": math.exp(2.0),
        "encode_ms_per_token": 10.0,
        "decode_ms_per_token": 9.0,
        "stego_algorithm_total_ms": 100.0,
        "metric_instrumentation_total_ms": 50.0,
        "elapsed_wall_seconds": 3.0,
        "ber": 0.0 if roundtrip_exact else 0.25,
        "carrier_tokens": 256,
    }


def test_choose_canonical_attempt_prefers_success_over_skip() -> None:
    attempts = [
        {"run_id": "a", "status": "ok", "value": 1},
        {"run_id": "a", "status": "skipped_existing_ok"},
    ]
    assert analysis._choose_canonical_attempt(attempts)["value"] == 1


def test_choose_canonical_attempt_uses_latest_error_when_never_successful() -> None:
    attempts = [
        {"run_id": "a", "status": "error", "value": 1},
        {"run_id": "a", "status": "error", "value": 2},
    ]
    assert analysis._choose_canonical_attempt(attempts)["value"] == 2


def test_aggregate_keeps_infinite_forward_kl_explicit_and_uses_all_runs_for_reliability() -> None:
    runs = [
        _ok_row(run_id="a", roundtrip_exact=True, forward_kl=math.inf),
        _ok_row(run_id="b", roundtrip_exact=False, forward_kl=math.inf),
        {
            "run_id": "c",
            "status": "error",
            "model": "Llama-3.2-3B",
            "family": "streaming",
            "method_id": "bins",
            "method_point_id": "b1",
        },
    ]

    [result] = analysis.aggregate_operating_points(runs)

    assert result["n_total"] == 3
    assert result["n_ok"] == 2
    assert result["n_hard_failure"] == 1
    assert result["n_soft_roundtrip_failure"] == 1
    assert result["n_exact_roundtrip"] == 1
    assert result["exact_roundtrip_rate"] == 1 / 3
    assert result["kl_ref_to_stego_infinite_runs"] == 2
    assert result["kl_ref_to_stego_infinite_rate_ok"] == 1.0
    assert result["kl_ref_to_stego_finite"]["n"] == 0
    assert result["kl_ref_to_stego_finite"]["mean"] is None


def test_add_derived_metrics_recovers_reference_entropy_and_fixed_payload_length() -> None:
    row = {
        "bits_per_token": 4.0,
        "entropy_utilization": 0.8,
        "ppl_raw_lm": math.exp(2.0),
        "termination": {
            "mode": "fixed_payload_bits",
            "target_payload_bits": 1024,
        },
    }

    analysis._add_derived_metrics(row)

    assert row["reference_entropy_mean_bits"] == 5.0
    assert row["nll_raw_lm_nats_per_token"] == 2.0
    assert row["carrier_tokens"] == 256
