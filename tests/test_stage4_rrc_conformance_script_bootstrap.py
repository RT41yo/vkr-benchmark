from __future__ import annotations

from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]


def _help(script: str) -> str:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script), "--help"],
        cwd=REPO_ROOT,
        env={"PATH": str(Path(sys.executable).parent)},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    return result.stdout


def test_rrc_conformance_scripts_bootstrap_without_pythonpath() -> None:
    assert "--random-cases" in _help("check_stage4_rrc_core_conformance.py")
    assert "--reference-dir" in _help("run_stage4_rrc_author_conformance.py")
    assert "--payload-bits" in _help("check_stage4_rrc_explicit_q.py")
    assert "--samples" in _help("check_stage4_rrc_text_channel.py")
