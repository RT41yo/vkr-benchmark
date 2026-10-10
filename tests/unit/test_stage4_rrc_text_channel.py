from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_stage4_rrc_text_channel.py"
CONFIG = REPO_ROOT / "configs" / "reproducibility" / "stage4_rrc_text_channel.json"
EXPECTED_COMMIT = "dae326259e4fca8bc4fcf460dafdbc0e88a0a71a"


def _load():
    spec = importlib.util.spec_from_file_location("stage4_rrc_text_channel", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_rrc_text_channel_config_freezes_series() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert payload["reference_commit"] == EXPECTED_COMMIT
    assert payload["samples"] == 100
    assert payload["payload_bits"] == 128
    assert payload["key"] == 42
    assert payload["success_gate"]["direct_token_id_recovery_rate"] == 1.0
    assert payload["interpretation"]["text_channel_recovery_is_measurement_not_gate"] is True


def test_rrc_text_channel_summary_does_not_fail_on_retokenization_only() -> None:
    module = _load()
    rows = [
        module.SampleResult(
            sample_index=0,
            secret_id="a",
            carrier_tokens=25,
            bits_per_token=5.12,
            termination_retries=0,
            direct_decoder_complete=True,
            direct_secret_exact=True,
            token_sequence_roundtrip_exact=True,
            first_token_mismatch=None,
            token_count_delta=0,
            text_decoder_complete=True,
            text_secret_exact=True,
        ),
        module.SampleResult(
            sample_index=1,
            secret_id="b",
            carrier_tokens=26,
            bits_per_token=128 / 26,
            termination_retries=1,
            direct_decoder_complete=True,
            direct_secret_exact=True,
            token_sequence_roundtrip_exact=False,
            first_token_mismatch=7,
            token_count_delta=1,
            text_decoder_complete=True,
            text_secret_exact=False,
        ),
    ]
    summary = module.summarize(rows)
    assert summary["status"] == "pass"
    assert summary["direct_token_id_recovery"]["rate"] == 1.0
    assert summary["text_transport"]["token_sequence_exact_rate"] == 0.5
    assert summary["text_transport"]["secret_exact_rate"] == 0.5


def test_rrc_text_channel_summary_fails_on_direct_algorithmic_error() -> None:
    module = _load()
    row = module.SampleResult(
        sample_index=0,
        secret_id="a",
        carrier_tokens=25,
        bits_per_token=5.12,
        termination_retries=0,
        direct_decoder_complete=True,
        direct_secret_exact=False,
        token_sequence_roundtrip_exact=True,
        first_token_mismatch=None,
        token_count_delta=0,
        text_decoder_complete=True,
        text_secret_exact=False,
    )
    assert module.summarize([row])["status"] == "fail"
