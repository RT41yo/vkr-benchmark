from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def _help(script_name: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script_name), "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_stage5_plan_script_bootstraps_without_installing_package() -> None:
    completed = _help("plan_stage5_sweep.py")
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()


def test_stage5_run_script_bootstraps_without_installing_package() -> None:
    completed = _help("run_stage5_sweep.py")
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()


def test_stage5_prompt_prepare_script_bootstraps_without_installing_package() -> None:
    completed = _help("prepare_stage5_pilot_prompts.py")
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()


def test_stage5_hard_failure_diagnostic_script_bootstraps_without_installing_package() -> None:
    completed = _help("diagnose_stage5_hard_failures.py")
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()
