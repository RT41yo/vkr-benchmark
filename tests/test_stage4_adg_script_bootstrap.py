from pathlib import Path
import subprocess
import sys


def test_adg_smoke_script_imports_without_pythonpath() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "check_adg_e2e.py"), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        env={"PATH": str(Path(sys.executable).parent)},
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--carrier-tokens" in result.stdout
