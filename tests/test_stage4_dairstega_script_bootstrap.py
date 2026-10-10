from __future__ import annotations

import runpy
from pathlib import Path


def test_dairstega_smoke_script_bootstraps_without_installing_package() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_dairstega_e2e.py"
    namespace = runpy.run_path(str(script), run_name="stage4_dairstega_bootstrap_test")
    assert callable(namespace["main"])
