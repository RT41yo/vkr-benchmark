from __future__ import annotations

import runpy
from pathlib import Path


def _namespace() -> dict[str, object]:
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_stage4_compatibility_matrix.py"
    return runpy.run_path(str(script), run_name="stage4_compatibility_matrix_bootstrap_test")


def test_compatibility_matrix_script_bootstraps_without_installing_package() -> None:
    namespace = _namespace()
    assert callable(namespace["main"])


def test_compatibility_status_rule_distinguishes_qwen_style_adaptation() -> None:
    classify = _namespace()["classify_status"]
    assert classify(
        direct_secret_exact=True,
        text_token_exact=True,
        text_secret_exact=True,
        output_only_count=0,
    ) == "PASS"
    assert classify(
        direct_secret_exact=True,
        text_token_exact=True,
        text_secret_exact=True,
        output_only_count=267,
    ) == "PASS_WITH_ADAPTATION"
    assert classify(
        direct_secret_exact=True,
        text_token_exact=False,
        text_secret_exact=True,
        output_only_count=0,
    ) == "FAIL"
