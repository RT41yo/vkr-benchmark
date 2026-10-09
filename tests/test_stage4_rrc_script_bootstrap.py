from __future__ import annotations

from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_rrc_smoke_script_help_runs_without_pythonpath() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_rrc_e2e.py"), "--help"],
        cwd=REPO_ROOT,
        env={"PATH": str(Path(sys.executable).parent)},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    assert "--payload-bits" in result.stdout
