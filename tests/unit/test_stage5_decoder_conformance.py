from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_stage5_decoder_conformance.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("stage5_decoder_conformance", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_stage5_decoder_conformance_is_cpu_only_and_passes_small_suite() -> None:
    module = _load_module()
    result = module.evaluate(random_cases=4)
    assert result["cpu_only"] is True
    assert result["production_method_sources_modified"] is False
    assert result["status"] == "pass"
    assert result["failed_methods"] == []
    assert set(result["methods"]) == {
        "bins",
        "huffman",
        "arithmetic_coding",
        "adg",
        "discop",
        "dairstega",
    }
    for payload in result["methods"].values():
        assert payload["status"] == "pass"
        assert payload["failure_count"] == 0
        assert payload["check_count"] > 0


def test_stage5_decoder_conformance_does_not_import_private_production_helpers() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    forbidden = (
        "from vkr_benchmark.methods.adg import _",
        "from vkr_benchmark.methods.arithmetic import _",
        "from vkr_benchmark.methods.bins import _",
        "from vkr_benchmark.methods.dair import _",
        "from vkr_benchmark.methods.discop import _",
        "from vkr_benchmark.methods.huffman import _",
    )
    for fragment in forbidden:
        assert fragment not in source
