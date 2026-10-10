from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_stage4_reference_revisions_are_frozen() -> None:
    data = json.loads(
        (REPO_ROOT / "reference" / "reference_sources.json").read_text(encoding="utf-8")
    )
    by_method = {
        source["method_family"][0]: source
        for source in data["sources"]
        if source.get("role") == "stage4_algorithm_reference"
    }
    assert by_method["adg"]["commit"] == "b4a7e802a97bc3a3b84d07e63f24627b16c51a10"
    assert by_method["discop"]["commit"] == "3c3a10099a242eae405b49cc4d09fba1abb148ad"
    assert by_method["dairstega"]["commit"] == "8d85edf98d48c3efa827a125b6d4e90f88141ea2"
    assert by_method["rrc"]["commit"] == "dae326259e4fca8bc4fcf460dafdbc0e88a0a71a"
    assert by_method["rrc"]["prior_analysis_commit"] == (
        "8b6a86a66e516e3763c907a1ac640c17bcd3641d"
    )
