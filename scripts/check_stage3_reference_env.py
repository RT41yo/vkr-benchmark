#!/usr/bin/env python3
"""Audit the local environment and pinned Harvard reference checkout for Stage 3.

This script is deliberately diagnostic-only: it does not download models, clone
repositories, mutate the reference checkout, or execute steganography code.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SOURCES = REPO_ROOT / "reference" / "reference_sources.json"

EXPECTED_REPOSITORY = "https://github.com/harvardnlp/NeuralSteganography"
EXPECTED_COMMIT = "14e982564aeaf9a33f7b4de440deda2184d17f12"
EXPECTED_METHODS = {"bins", "huffman", "arithmetic_coding"}
EXPECTED_FILES = (
    "run_single.py",
    "block_baseline.py",
    "huffman_baseline.py",
    "huffman.py",
    "arithmetic.py",
    "utils.py",
    "requirements.txt",
)
PACKAGE_NAMES = (
    "numpy",
    "torch",
    "transformers",
    "tokenizers",
    "huggingface-hub",
    "safetensors",
    "bitarray",
)

EXPECTED_VERSION_PREFIXES = {
    "numpy": (2,),
    "torch": (2, 7),
    "transformers": (4, 52),
    "tokenizers": (0, 21),
    "huggingface-hub": (0, 32),
    "safetensors": (0, 5),
}


@dataclass(frozen=True)
class CheckItem:
    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class CheckReport:
    ready: bool
    reference_repository: str
    reference_commit: str
    python_version: str
    packages: dict[str, str | None]
    checks: tuple[CheckItem, ...]

    def to_json(self) -> str:
        payload = asdict(self)
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _version_prefix(version: str, parts: int) -> tuple[int, ...] | None:
    base = version.split("+", 1)[0]
    numbers: list[int] = []
    for piece in base.split(".")[:parts]:
        digits = "".join(ch for ch in piece if ch.isdigit())
        if not digits:
            return None
        numbers.append(int(digits))
    if len(numbers) != parts:
        return None
    return tuple(numbers)


def _package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _load_pinned_reference() -> tuple[str, str]:
    payload = json.loads(REFERENCE_SOURCES.read_text(encoding="utf-8"))
    matches: list[dict[str, Any]] = []
    for source in payload.get("sources", []):
        methods = set(source.get("method_family", []))
        if EXPECTED_METHODS.issubset(methods) and source.get("role") == "algorithm_reference":
            matches.append(source)
    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one algorithm_reference covering bins, huffman and "
            f"arithmetic_coding, found {len(matches)}"
        )
    source = matches[0]
    return str(source.get("repository", "")), str(source.get("commit", ""))


def _git_head(checkout: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def evaluate(reference_dir: Path | None) -> CheckReport:
    checks: list[CheckItem] = []

    try:
        repository, commit = _load_pinned_reference()
    except Exception as exc:  # diagnostic boundary
        repository, commit = "", ""
        checks.append(CheckItem("reference_sources", "FAIL", str(exc)))
    else:
        if repository == EXPECTED_REPOSITORY and commit == EXPECTED_COMMIT:
            checks.append(
                CheckItem(
                    "reference_sources",
                    "PASS",
                    f"pinned to {repository}@{commit}",
                )
            )
        else:
            checks.append(
                CheckItem(
                    "reference_sources",
                    "FAIL",
                    f"expected {EXPECTED_REPOSITORY}@{EXPECTED_COMMIT}, got {repository}@{commit}",
                )
            )

    py_ok = sys.version_info[:2] == (3, 12)
    checks.append(
        CheckItem(
            "python",
            "PASS" if py_ok else "FAIL",
            f"{sys.version.split()[0]} (benchmark requires Python 3.12.x)",
        )
    )

    packages = _package_versions()
    for name, expected_prefix in EXPECTED_VERSION_PREFIXES.items():
        version = packages[name]
        prefix = None if version is None else _version_prefix(version, len(expected_prefix))
        ok = prefix == expected_prefix
        expected_text = ".".join(str(part) for part in expected_prefix) + ".x"
        checks.append(
            CheckItem(
                f"package:{name}",
                "PASS" if ok else "FAIL",
                (
                    f"{version} (expected {expected_text})"
                    if version is not None
                    else f"not installed (expected {expected_text})"
                ),
            )
        )

    bitarray_version = packages["bitarray"]
    bitarray_ok = bitarray_version == "3.4.2"
    checks.append(
        CheckItem(
            "package:bitarray",
            "PASS" if bitarray_ok else "FAIL",
            (
                bitarray_version
                if bitarray_version is not None
                else "not installed; install project optional extra 'reference'"
            ),
        )
    )

    if reference_dir is None:
        checks.append(
            CheckItem(
                "reference_checkout",
                "SKIP",
                "no --reference-dir supplied; checkout was not inspected",
            )
        )
    else:
        checkout = reference_dir.expanduser().resolve()
        if not checkout.is_dir():
            checks.append(CheckItem("reference_checkout", "FAIL", f"missing directory: {checkout}"))
        else:
            missing = [name for name in EXPECTED_FILES if not (checkout / name).is_file()]
            if missing:
                checks.append(
                    CheckItem(
                        "reference_files",
                        "FAIL",
                        "missing: " + ", ".join(missing),
                    )
                )
            else:
                checks.append(
                    CheckItem(
                        "reference_files",
                        "PASS",
                        f"all {len(EXPECTED_FILES)} expected files are present",
                    )
                )

            head = _git_head(checkout)
            if head is None:
                checks.append(
                    CheckItem(
                        "reference_git_head",
                        "FAIL",
                        "unable to read Git HEAD (is this a Git checkout and is git available?)",
                    )
                )
            elif head == EXPECTED_COMMIT:
                checks.append(CheckItem("reference_git_head", "PASS", head))
            else:
                checks.append(
                    CheckItem(
                        "reference_git_head",
                        "FAIL",
                        f"expected {EXPECTED_COMMIT}, got {head}",
                    )
                )

    ready = all(item.status != "FAIL" for item in checks)
    return CheckReport(
        ready=ready,
        reference_repository=repository,
        reference_commit=commit,
        python_version=sys.version.split()[0],
        packages=packages,
        checks=tuple(checks),
    )


def format_report(report: CheckReport) -> str:
    lines = ["Stage 3 author-reference environment check"]
    for item in report.checks:
        lines.append(f"[{item.status:4}] {item.name}: {item.detail}")
    lines.append("Stage 3 reference environment: " + ("READY" if report.ready else "NOT READY"))
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check Stage-3 author-reference provenance, local package availability "
            "and optionally a pinned NeuralSteganography checkout."
        )
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        help="Path to a local checkout of harvardnlp/NeuralSteganography.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path for a machine-readable check report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate(args.reference_dir)
    print(format_report(report), end="")
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(report.to_json(), encoding="utf-8")
        print(f"check json: {args.json_output}")
    return 0 if report.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
