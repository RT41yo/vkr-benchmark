from pathlib import Path
import subprocess
import sys


def test_adg_conformance_scripts_import_without_pythonpath() -> None:
    root = Path(__file__).resolve().parents[1]
    for script in (
        "check_stage4_adg_paper_repository_conformance.py",
        "run_stage4_adg_author_near_conformance.py",
    ):
        proc = subprocess.run(
            [sys.executable, str(root / "scripts" / script), "--help"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
