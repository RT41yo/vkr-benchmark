#!/usr/bin/env python3
"""Compare normalized Discop against the pinned original Cython step API.

This runner does not edit the author checkout.  It expects the pinned checkout
under ``external/Discop`` and a previously built ``stega_cy`` extension in its
``src`` directory.  Only synthetic step-level distributions are used, so no
legacy GPT-2 model or dataset download is required.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stage4_discop_conformance import (
    REFERENCE_COMMIT,
    REFERENCE_REPOSITORY,
    fixture_suite,
    normalized_step,
)

DEFAULT_REFERENCE_DIR = REPO_ROOT / "external" / "Discop"


def _git(checkout: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "git command failed")
    return proc.stdout.strip()


def _validate_checkout(reference_dir: Path) -> dict[str, object]:
    if not reference_dir.is_dir():
        raise FileNotFoundError(f"reference checkout not found: {reference_dir}")
    src = reference_dir / "src"
    for required in ("stega_cy.pyx", "config.py", "model.py", "utils.py"):
        if not (src / required).is_file():
            raise FileNotFoundError(f"missing pinned reference file: src/{required}")
    head = _git(reference_dir, "rev-parse", "HEAD")
    if head != REFERENCE_COMMIT:
        raise RuntimeError(f"expected Discop HEAD {REFERENCE_COMMIT}, got {head}")
    tracked_diff_before = _git(reference_dir, "diff", "--name-only")
    if tracked_diff_before:
        raise RuntimeError(
            "tracked files in external/Discop are modified before conformance run: "
            + tracked_diff_before.replace("\n", ", ")
        )
    compiled = (
        sorted(reference_dir.glob("stega_cy*.so"))
        + sorted(reference_dir.glob("stega_cy*.pyd"))
        + sorted(src.glob("stega_cy*.so"))
        + sorted(src.glob("stega_cy*.pyd"))
    )
    if not compiled:
        raise FileNotFoundError(
            "compiled author stega_cy extension not found in external/Discop or "
            "external/Discop/src; build it first with `python src/setup.py "
            "build_ext --inplace` from the author checkout root"
        )
    return {"head": head, "compiled_extension": str(compiled[0])}


def _load_author_modules(reference_dir: Path) -> tuple[Any, Any]:
    """Import the original Cython module with a step-only compatibility shim.

    ``stega_cy.pyx`` imports author LM helpers at module import time even though
    ``encode_step``/``decode_step`` do not use them.  Modern Transformers may no
    longer expose every legacy class imported by those helpers.  To keep the
    pinned Cython file untouched while isolating the step API, provide minimal
    placeholder ``model``/``utils`` modules.  Any accidental attempt to enter an
    LM-dependent author path fails loudly.
    """

    import types

    sys.dont_write_bytecode = True
    reference_root = str(reference_dir.resolve())
    src = str((reference_dir / "src").resolve())
    # setup.py builds the top-level `stega_cy` extension into the checkout root
    # with --inplace, while config.py remains under src/.  Expose both paths.
    for path in (reference_root, src):
        if path not in sys.path:
            sys.path.insert(0, path)
    config = importlib.import_module("config")

    def _unsupported(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            "Stage-4 Discop Cython conformance permits only encode_step/decode_step"
        )

    model_stub = types.ModuleType("model")
    model_stub.get_model = _unsupported
    model_stub.get_tokenizer = _unsupported
    model_stub.get_feature_extractor = _unsupported

    utils_stub = types.ModuleType("utils")
    utils_stub.get_probs_indices_past = _unsupported
    utils_stub.set_seed = _unsupported

    class _SingleExampleOutput:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _unsupported(*args, **kwargs)

    utils_stub.SingleExampleOutput = _SingleExampleOutput

    previous_model = sys.modules.get("model")
    previous_utils = sys.modules.get("utils")
    sys.modules["model"] = model_stub
    sys.modules["utils"] = utils_stub
    try:
        stega = importlib.import_module("stega_cy")
    finally:
        if previous_model is None:
            sys.modules.pop("model", None)
        else:
            sys.modules["model"] = previous_model
        if previous_utils is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = previous_utils
    return config, stega


def _author_step(stega: Any, settings: Any, fixture: Any) -> tuple[int, int, str, int]:
    # Feed the author API exactly the candidate order it receives after sorting.
    import numpy as np
    from vkr_benchmark.distributions import ReferenceDistribution

    reference = ReferenceDistribution(np.asarray(fixture.probabilities, dtype=np.float32))
    indices = [int(x) for x in reference.token_order]
    probs = [float(reference.probabilities[index]) for index in indices]
    message = "".join(str(bit) for bit in fixture.secret_bits)

    random.seed(fixture.seed)
    encoded = stega.encode_step(settings, indices, probs, message)
    token_id = int(encoded.sampled_index)
    n_bits = int(encoded.n_bits)

    random.seed(fixture.seed)
    decoded_bits = stega.decode_step(settings, indices, probs, token_id)
    if isinstance(decoded_bits, bytes):
        decoded_bits = decoded_bits.decode("ascii")
    decoded_bits = str(decoded_bits)
    return token_id, n_bits, decoded_bits, len(decoded_bits)


def evaluate(reference_dir: Path, random_cases: int) -> dict[str, object]:
    provenance = _validate_checkout(reference_dir)
    config_module, stega = _load_author_modules(reference_dir)
    settings = config_module.Settings(task="text", algo="Discop", model_name="gpt2")

    failures: list[dict[str, object]] = []
    fixtures = fixture_suite(random_cases)
    for fixture in fixtures:
        token_id, n_bits, decoded_bits, decoded_len = _author_step(stega, settings, fixture)
        normalized_encoded, normalized_decoded = normalized_step(fixture)
        normalized_bits = "".join(str(bit) for bit in normalized_encoded.bits)
        normalized_decoded_bits = "".join(str(bit) for bit in normalized_decoded.bits)
        checks = {
            "token_id": token_id == normalized_encoded.token_id,
            "bits_consumed": n_bits == len(normalized_encoded.bits),
            "author_decode": decoded_bits == normalized_bits,
            "normalized_decode": normalized_decoded_bits == decoded_bits,
            "decoded_length": decoded_len == n_bits,
        }
        if not all(checks.values()):
            failures.append(
                {
                    "fixture_id": fixture.fixture_id,
                    "checks": checks,
                    "author": {
                        "token_id": token_id,
                        "bits_consumed": n_bits,
                        "decoded_bits": decoded_bits,
                    },
                    "normalized": {
                        "token_id": normalized_encoded.token_id,
                        "bits": normalized_bits,
                        "decoded_bits": normalized_decoded_bits,
                    },
                }
            )

    tracked_diff_after = _git(reference_dir, "diff", "--name-only")
    return {
        "status": "pass" if not failures and not tracked_diff_after else "fail",
        "comparison_kind": "original-cython-vs-normalized",
        "reference_repository": REFERENCE_REPOSITORY,
        "reference_commit": REFERENCE_COMMIT,
        "reference_checkout": str(reference_dir.resolve()),
        "compiled_extension": provenance["compiled_extension"],
        "fixture_count": len(fixtures),
        "failure_count": len(failures),
        "failures": failures,
        "tracked_reference_files_changed": tracked_diff_after.splitlines() if tracked_diff_after else [],
        "compatibility_note": (
            "Original pinned Cython step functions are executed in the benchmark's modern "
            "Python environment; model/dataset code paths are not exercised by this step test."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--random-cases", type=int, default=128)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.random_cases < 0:
        parser.error("--random-cases must be non-negative")

    try:
        result = evaluate(args.reference_dir, args.random_cases)
    except Exception as exc:
        result = {
            "status": "error",
            "comparison_kind": "original-cython-vs-normalized",
            "reference_repository": REFERENCE_REPOSITORY,
            "reference_commit": REFERENCE_COMMIT,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }

    print("Stage 4 Discop original-Cython conformance")
    print("reference:", result["reference_repository"])
    print("commit:", result["reference_commit"])
    print("status:", str(result["status"]).upper())
    if "fixture_count" in result:
        print("fixtures:", result["fixture_count"])
        print("failures:", result["failure_count"])
    if "error" in result:
        print("error:", result["error"])

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("json:", args.json_output)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
