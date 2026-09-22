from __future__ import annotations

import json
from pathlib import Path
import tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_REPOSITORY = "https://github.com/harvardnlp/NeuralSteganography"
EXPECTED_COMMIT = "14e982564aeaf9a33f7b4de440deda2184d17f12"


def test_stage3_harvard_reference_is_pinned() -> None:
    payload = json.loads(
        (REPO_ROOT / "reference" / "reference_sources.json").read_text(encoding="utf-8")
    )
    matches = [
        source
        for source in payload["sources"]
        if source.get("role") == "algorithm_reference"
        and {"bins", "huffman", "arithmetic_coding"}.issubset(
            set(source.get("method_family", []))
        )
    ]
    assert len(matches) == 1
    assert matches[0]["repository"] == EXPECTED_REPOSITORY
    assert matches[0]["commit"] == EXPECTED_COMMIT


def test_stage3_reference_extra_pins_bitarray() -> None:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)
    extras = project["project"]["optional-dependencies"]
    assert extras["reference"] == ["bitarray==3.4.2"]


def test_external_reference_checkout_is_gitignored() -> None:
    lines = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "/external/" in lines
