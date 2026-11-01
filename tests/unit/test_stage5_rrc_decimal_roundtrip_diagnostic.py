from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/diagnose_stage5_rrc_decimal_roundtrip.py"
_spec = importlib.util.spec_from_file_location("rrc_decimal_roundtrip_diag", SCRIPT)
assert _spec is not None and _spec.loader is not None
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_first_mismatch_handles_value_and_length_changes() -> None:
    assert mod.first_mismatch((1, 2, 3), (1, 9, 3)) == 1
    assert mod.first_mismatch((1, 2), (1, 2, 3)) == 2
    assert mod.first_mismatch((1, 2), (1, 2)) is None


def test_bit_error_rate_counts_length_difference() -> None:
    assert mod.bit_error_rate((0, 1, 1, 0), (0, 0, 1, 0)) == 0.25
    assert mod.bit_error_rate((1, 0), (1, 0, 1, 1)) == 0.5
    assert mod.bit_error_rate((), ()) == 0.0


def test_build_inventory_reports_transport_change() -> None:
    source = {"status": "ok", "roundtrip_exact": False, "ber": 0.4}
    token_record = {"first_token_mismatch": 1}
    got = mod.build_inventory(
        source=source,
        sender=(10, 20, 30),
        receiver=(10, 99, 30, 40),
        token_record=token_record,
    )
    assert got["token_sequence_roundtrip_exact"] is False
    assert got["first_token_mismatch"] == 1
    assert got["token_count_delta"] == 1
    assert got["source_roundtrip_exact"] is False
