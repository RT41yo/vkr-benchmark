from pathlib import Path
import subprocess
import sys


def test_dairstega_characteristic_script_imports_without_pythonpath() -> None:
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "probe_stage4_dairstega_characteristic.py"),
            "--help",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
