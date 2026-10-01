from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    "check_discop_e2e.py",
    "check_stage4_discop_core_conformance.py",
    "probe_stage4_discop_characteristic.py",
    "run_stage4_discop_author_cython_conformance.py",
)


@pytest.mark.parametrize("script_name", SCRIPTS)
def test_stage4_discop_script_bootstraps_src_without_pythonpath(script_name: str) -> None:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script_name), "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "usage:" in proc.stdout.lower()


def test_author_cython_runner_accepts_setup_py_inplace_output_location() -> None:
    text = (REPO_ROOT / "scripts" / "run_stage4_discop_author_cython_conformance.py").read_text(
        encoding="utf-8"
    )
    assert 'reference_dir.glob("stega_cy*.so")' in text
    assert "reference_root" in text


def test_author_cython_decoder_uses_direct_decode_step_return_value() -> None:
    text = (REPO_ROOT / "scripts" / "run_stage4_discop_author_cython_conformance.py").read_text(
        encoding="utf-8"
    )
    assert "decoded_bits = stega.decode_step" in text
    assert "decoded.message_decoded_t" not in text
