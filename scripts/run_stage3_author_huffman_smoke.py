#!/usr/bin/env python3
"""Run a structured smoke test of the pinned Harvard Huffman implementation."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Iterator

from stage3_author_runtime import (
    COMPATIBILITY_PROFILE,
    LegacyCausalLMAdapter,
    force_reference_slow_tokenizer,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "reproducibility" / "author_huffman_gpt2_smoke.json"
DEFAULT_REFERENCE_DIR = REPO_ROOT / "external" / "NeuralSteganography"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "stage3" / "author_smoke" / "huffman_gpt2"
EXPECTED_REPOSITORY = "https://github.com/harvardnlp/NeuralSteganography"
EXPECTED_COMMIT = "14e982564aeaf9a33f7b4de440deda2184d17f12"
EXPECTED_METHOD = "huffman"
EXPECTED_MODE = "author-compatible"
REQUIRED_REFERENCE_FILES = ("utils.py", "huffman.py", "huffman_baseline.py")
PACKAGE_NAMES = (
    "numpy",
    "torch",
    "transformers",
    "tokenizers",
    "huggingface-hub",
    "safetensors",
    "bitarray",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8")), _sha256_bytes(raw)


def _validate_config(config: dict[str, Any]) -> list[int]:
    if config.get("mode") != EXPECTED_MODE:
        raise ValueError(f"expected mode={EXPECTED_MODE!r}")
    if config.get("method") != EXPECTED_METHOD:
        raise ValueError(f"expected method={EXPECTED_METHOD!r}")
    if config.get("reference_repository") != EXPECTED_REPOSITORY:
        raise ValueError("reference_repository does not match the pinned Stage-3 source")
    if config.get("reference_commit") != EXPECTED_COMMIT:
        raise ValueError("reference_commit does not match the pinned Stage-3 source")
    bits_per_word = config.get("bits_per_word")
    if not isinstance(bits_per_word, int) or bits_per_word <= 0:
        raise ValueError("bits_per_word must be a positive integer")
    if bits_per_word > 16:
        raise ValueError("bits_per_word is unexpectedly large for a technical smoke")
    secret = config.get("secret_bits")
    if not isinstance(secret, str) or not secret or set(secret) - {"0", "1"}:
        raise ValueError("secret_bits must be a non-empty binary string")
    if config.get("compatibility_profile") != COMPATIBILITY_PROFILE:
        raise ValueError(f"expected compatibility_profile={COMPATIBILITY_PROFILE!r}")
    return [int(bit) for bit in secret]


def _git(checkout: Path, *args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(checkout), *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def _validate_reference_checkout(reference_dir: Path) -> tuple[str, str]:
    if not reference_dir.is_dir():
        raise FileNotFoundError(f"reference checkout does not exist: {reference_dir}")
    missing = [name for name in REQUIRED_REFERENCE_FILES if not (reference_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("reference checkout is missing: " + ", ".join(missing))
    head = _git(reference_dir, "rev-parse", "HEAD")
    if head != EXPECTED_COMMIT:
        raise RuntimeError(f"expected reference HEAD {EXPECTED_COMMIT}, got {head!r}")
    status = _git(reference_dir, "status", "--porcelain")
    if status is None:
        raise RuntimeError("unable to inspect reference worktree status")
    return head, status


def _package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


@contextmanager
def _reference_import_path(reference_dir: Path) -> Iterator[None]:
    old_dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(reference_dir))
    try:
        yield
    finally:
        try:
            sys.path.remove(str(reference_dir))
        except ValueError:
            pass
        sys.dont_write_bytecode = old_dont_write


def _assert_module_origin(module: Any, reference_dir: Path) -> None:
    module_file = Path(module.__file__).resolve()
    if module_file.parent != reference_dir.resolve():
        raise RuntimeError(f"imported {module.__name__} from unexpected path: {module_file}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _base_result(
    *,
    config: dict[str, Any],
    config_path: Path,
    config_sha256: str,
    reference_dir: Path,
    reference_head: str | None,
    reference_status_before: str | None,
) -> dict[str, Any]:
    bits_per_word = config.get("bits_per_word")
    candidate_count = 2 ** int(bits_per_word) if isinstance(bits_per_word, int) else None
    return {
        "schema_version": "stage3.author_smoke_result.v1",
        "status": "running",
        "mode": config.get("mode"),
        "method": config.get("method"),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "path": str(config_path.resolve()),
            "sha256": config_sha256,
            "model_id": config.get("model_id"),
            "seed": config.get("seed"),
            "bits_per_word_parameter": bits_per_word,
            "candidate_count": candidate_count,
            "finish_sent": config.get("finish_sent"),
            "secret_bits": config.get("secret_bits"),
            "secret_bit_count": len(str(config.get("secret_bits", ""))),
            "context_sha256": _sha256_bytes(str(config.get("context", "")).encode("utf-8")),
        },
        "reference": {
            "repository": EXPECTED_REPOSITORY,
            "expected_commit": EXPECTED_COMMIT,
            "checkout": str(reference_dir.resolve()),
            "actual_commit": reference_head,
            "git_status_before": reference_status_before,
        },
        "environment": {
            "python": sys.version.split()[0],
            "packages": _package_versions(),
        },
        "compatibility": {
            "profile": config.get("compatibility_profile"),
            "tokenizer_use_fast": False,
            "model_call_translation": "past -> past_key_values",
            "cache_translation": (
                "historical stacked [2,batch,heads,seq,head_dim] "
                "<-> modern per-layer (key,value)"
            ),
            "model_return_dict": False,
            "reference_files_modified": False,
        },
    }


def run(config_path: Path, reference_dir: Path, output_dir: Path) -> int:
    config_path = config_path.expanduser().resolve()
    reference_dir = reference_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    result_path = output_dir / "result.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    reference_head: str | None = None
    reference_status_before: str | None = None
    config: dict[str, Any] = {}
    config_sha256 = ""
    result: dict[str, Any]

    try:
        config, config_sha256 = _load_config(config_path)
        secret_bits = _validate_config(config)
        reference_head, reference_status_before = _validate_reference_checkout(reference_dir)
        result = _base_result(
            config=config,
            config_path=config_path,
            config_sha256=config_sha256,
            reference_dir=reference_dir,
            reference_head=reference_head,
            reference_status_before=reference_status_before,
        )
        _write_json(result_path, result)

        with _reference_import_path(reference_dir):
            utils = importlib.import_module("utils")
            huffman_core = importlib.import_module("huffman")
            huffman_baseline = importlib.import_module("huffman_baseline")
            _assert_module_origin(utils, reference_dir)
            _assert_module_origin(huffman_core, reference_dir)
            _assert_module_origin(huffman_baseline, reference_dir)

            import torch

            model_id = str(config["model_id"])
            seed = int(config["seed"])
            bits_per_word = int(config["bits_per_word"])
            finish_sent = bool(config["finish_sent"])
            context = str(config["context"])

            print(f"Loading author-reference model {model_id!r} with slow tokenizer ...", flush=True)
            with force_reference_slow_tokenizer(utils):
                enc, raw_model = utils.get_model(seed=seed, model_name=model_id)
            model = LegacyCausalLMAdapter(raw_model)
            result["compatibility"].update(
                {
                    "tokenizer_class": type(enc).__name__,
                    "model_class": type(raw_model).__name__,
                }
            )
            device = str(next(model.parameters()).device)
            gpu_name = (
                torch.cuda.get_device_name(torch.cuda.current_device())
                if device.startswith("cuda")
                else None
            )

            context_tokens = utils.encode_context(context, enc)
            print(
                f"Running pinned Huffman encode/decode: {len(secret_bits)} secret bits, "
                f"top_candidates=2^{bits_per_word}={2**bits_per_word}, device={device}",
                flush=True,
            )
            output_ids, avg_nll, author_kl, words_per_bit = huffman_baseline.encode_huffman(
                model,
                enc,
                secret_bits,
                context_tokens,
                bits_per_word,
                finish_sent=finish_sent,
                device=device,
            )
            stegotext = enc.decode(output_ids)
            retokenized_ids = enc.encode(stegotext)
            recovered_bits = huffman_baseline.decode_huffman(
                model,
                enc,
                stegotext,
                context_tokens,
                bits_per_word,
                device=device,
            )

            payload_prefix = recovered_bits[: len(secret_bits)]
            exact_payload_recovery = payload_prefix == secret_bits
            text_token_roundtrip_exact = retokenized_ids == output_ids

            if words_per_bit <= 0:
                raise RuntimeError("author words_per_bit must be positive")
            author_bits_consumed_float = len(output_ids) / float(words_per_bit)
            author_bits_consumed = int(round(author_bits_consumed_float))
            if not math.isclose(author_bits_consumed_float, author_bits_consumed, rel_tol=0.0, abs_tol=1e-9):
                raise RuntimeError(
                    "author words_per_bit does not imply an integral consumed-bit count"
                )
            implicit_zero_padding_bits = max(0, author_bits_consumed - len(secret_bits))
            recovered_extra_bits = recovered_bits[len(secret_bits) :]
            trailing_padding_is_zero = all(bit == 0 for bit in recovered_extra_bits)

            (output_dir / "stegotext.txt").write_text(stegotext, encoding="utf-8")
            _write_json(
                output_dir / "token_ids.json",
                {
                    "sender_token_ids": output_ids,
                    "retokenized_token_ids": retokenized_ids,
                    "text_token_roundtrip_exact": text_token_roundtrip_exact,
                },
            )

            result["environment"].update(
                {
                    "torch_cuda_available": bool(torch.cuda.is_available()),
                    "device": device,
                    "gpu_name": gpu_name,
                }
            )
            result["author_metrics"] = {
                "avg_nll_nats_author": float(avg_nll),
                "perplexity_author": float(math.exp(avg_nll)),
                "kl_q_stego_to_p_lm_bits_author": float(author_kl),
                "words_per_bit_author": float(words_per_bit),
                "bits_per_word_author": float(1.0 / words_per_bit),
                "payload_bits_per_token_smoke": float(len(secret_bits) / len(output_ids)),
            }
            result["generation"] = {
                "generated_token_count": len(output_ids),
                "stegotext_utf8_bytes": len(stegotext.encode("utf-8")),
                "sender_token_ids": output_ids,
                "retokenized_token_ids": retokenized_ids,
                "text_token_roundtrip_exact": text_token_roundtrip_exact,
            }
            result["recovery"] = {
                "recovered_bits": "".join(str(bit) for bit in recovered_bits),
                "recovered_bit_count": len(recovered_bits),
                "payload_prefix_bits": "".join(str(bit) for bit in payload_prefix),
                "exact_payload_recovery": exact_payload_recovery,
                "author_bits_consumed": author_bits_consumed,
                "implicit_zero_padding_bits": implicit_zero_padding_bits,
                "recovered_extra_bits": "".join(str(bit) for bit in recovered_extra_bits),
                "trailing_padding_is_zero": trailing_padding_is_zero,
                "decoder_bit_count_matches_author_consumption": (
                    len(recovered_bits) == author_bits_consumed
                ),
            }
            result["status"] = "ok" if exact_payload_recovery else "failed_recovery"

    except Exception as exc:
        if not config:
            result = {
                "schema_version": "stage3.author_smoke_result.v1",
                "status": "error",
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        elif "result" not in locals():
            result = _base_result(
                config=config,
                config_path=config_path,
                config_sha256=config_sha256,
                reference_dir=reference_dir,
                reference_head=reference_head,
                reference_status_before=reference_status_before,
            )
        result["status"] = "error"
        result["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        elapsed = time.perf_counter() - started
        if "result" not in locals():
            result = {"schema_version": "stage3.author_smoke_result.v1", "status": "error"}
        reference_status_after = (
            _git(reference_dir, "status", "--porcelain") if reference_dir.is_dir() else None
        )
        result.setdefault("reference", {})["git_status_after"] = reference_status_after
        result["reference"]["worktree_unchanged"] = (
            reference_status_before is not None
            and reference_status_after == reference_status_before
        )
        result["duration_seconds"] = elapsed
        result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(result_path, result)

    status = result.get("status")
    recovery = result.get("recovery", {})
    print("\nStage 3 author Huffman smoke summary")
    print(f"status: {status}")
    print(f"result: {result_path}")
    print(f"reference worktree unchanged: {result['reference'].get('worktree_unchanged')}")
    if status in {"ok", "failed_recovery"}:
        print(f"exact payload recovery: {recovery.get('exact_payload_recovery')}")
        print(f"author bits consumed: {recovery.get('author_bits_consumed')}")
        print(f"implicit trailing zero padding: {recovery.get('implicit_zero_padding_bits')}")
        print(
            "text/token roundtrip exact: "
            f"{result.get('generation', {}).get('text_token_roundtrip_exact')}"
        )
        print(
            "author KL Q_stego||P_LM [bits]: "
            f"{result.get('author_metrics', {}).get('kl_q_stego_to_p_lm_bits_author')}"
        )
    elif "error" in result:
        print(f"error: {result['error'].get('type')}: {result['error'].get('message')}")

    return 0 if status == "ok" and result["reference"].get("worktree_unchanged") else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run(args.config, args.reference_dir, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
