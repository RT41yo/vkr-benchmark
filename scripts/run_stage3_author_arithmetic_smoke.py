#!/usr/bin/env python3
"""Run a structured smoke test of the pinned Harvard Arithmetic implementation."""

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

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "reproducibility" / "author_arithmetic_gpt2_smoke.json"
DEFAULT_REFERENCE_DIR = REPO_ROOT / "external" / "NeuralSteganography"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "stage3" / "author_smoke" / "arithmetic_gpt2"
EXPECTED_REPOSITORY = "https://github.com/harvardnlp/NeuralSteganography"
EXPECTED_COMMIT = "14e982564aeaf9a33f7b4de440deda2184d17f12"
EXPECTED_METHOD = "arithmetic"
EXPECTED_MODE = "author-compatible"
EXPECTED_COMPATIBILITY_PROFILE = "hf_4_52_arithmetic_native_dynamic_cache"
REQUIRED_REFERENCE_FILES = ("utils.py", "arithmetic.py")
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
    if config.get("compatibility_profile") != EXPECTED_COMPATIBILITY_PROFILE:
        raise ValueError(
            f"expected compatibility_profile={EXPECTED_COMPATIBILITY_PROFILE!r}"
        )

    temperature = config.get("temperature")
    if not isinstance(temperature, (int, float)) or float(temperature) <= 0:
        raise ValueError("temperature must be positive")
    precision = config.get("precision")
    if not isinstance(precision, int) or precision < 2:
        raise ValueError("precision must be an integer >= 2")
    topk = config.get("topk")
    if not isinstance(topk, int) or topk < 2:
        raise ValueError("topk must be an integer >= 2")

    secret = config.get("secret_bits")
    if not isinstance(secret, str) or not secret or set(secret) - {"0", "1"}:
        raise ValueError("secret_bits must be a non-empty binary string")
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
            "temperature": config.get("temperature"),
            "precision": config.get("precision"),
            "topk": config.get("topk"),
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
            "reference_files_modified": False,
            "legacy_model_adapter_used": False,
            "tokenizer_override_used": False,
            "reference_uses_dynamic_cache": True,
        },
        "metric_semantics": {
            "coding_distribution": (
                "temperature-softmax P_tau, probability cutoff, top-k cap, "
                "then finite-precision integer interval Q_stego"
            ),
            "nll_reference_distribution": "untempered masked GPT-2 P_LM",
            "kl_direction": "Q_stego || P_LM_untempered",
            "kl_unit": "bits/token",
            "entropy_return": "mean entropy(P_tau) from reference utils.entropy, in bits",
        },
        "termination_semantics": {
            "encoder_lookahead": "precision-bit window padded with zeros beyond payload",
            "decoder_final_carrier": "emit full precision-bit lower interval bound",
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
            arithmetic = importlib.import_module("arithmetic")
            _assert_module_origin(utils, reference_dir)
            _assert_module_origin(arithmetic, reference_dir)

            import torch

            model_id = str(config["model_id"])
            seed = int(config["seed"])
            temperature = float(config["temperature"])
            precision = int(config["precision"])
            topk = int(config["topk"])
            finish_sent = bool(config["finish_sent"])
            context = str(config["context"])

            print(
                f"Loading author-reference model {model_id!r} with reference tokenizer defaults ...",
                flush=True,
            )
            enc, model = utils.get_model(seed=seed, model_name=model_id)
            result["compatibility"].update(
                {
                    "tokenizer_class": type(enc).__name__,
                    "model_class": type(model).__name__,
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
                f"Running pinned Arithmetic encode/decode: {len(secret_bits)} secret bits, "
                f"temperature={temperature}, precision={precision}, topk={topk}, device={device}",
                flush=True,
            )
            output_ids, avg_nll, author_kl, words_per_bit, avg_hq = arithmetic.encode_arithmetic(
                model,
                enc,
                secret_bits,
                context_tokens,
                temp=temperature,
                finish_sent=finish_sent,
                precision=precision,
                topk=topk,
                device=device,
            )
            stegotext = enc.decode(output_ids)
            retokenized_ids = enc.encode(stegotext)
            recovered_bits = arithmetic.decode_arithmetic(
                model,
                enc,
                stegotext,
                context_tokens,
                temp=temperature,
                precision=precision,
                topk=topk,
                device=device,
            )

            if not output_ids:
                raise RuntimeError("author Arithmetic encoder produced no carrier tokens")
            if words_per_bit <= 0:
                raise RuntimeError("author words_per_bit must be positive")

            author_bits_consumed_float = len(output_ids) / float(words_per_bit)
            author_bits_consumed = int(round(author_bits_consumed_float))
            if not math.isclose(
                author_bits_consumed_float,
                author_bits_consumed,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise RuntimeError(
                    "author words_per_bit does not imply an integral consumed-bit count"
                )
            if author_bits_consumed < len(secret_bits):
                raise RuntimeError(
                    "author encoder stopped before accounting for the full useful payload"
                )

            implicit_zero_lookahead_bits = author_bits_consumed - len(secret_bits)
            payload_prefix = recovered_bits[: len(secret_bits)]
            exact_payload_recovery = payload_prefix == secret_bits

            lookahead_padding = recovered_bits[len(secret_bits) : author_bits_consumed]
            expected_padding = [0] * implicit_zero_lookahead_bits
            lookahead_padding_is_zero = lookahead_padding == expected_padding
            author_consumed_prefix_available = len(recovered_bits) >= author_bits_consumed
            expected_consumed_prefix = secret_bits + expected_padding
            author_consumed_prefix_matches = (
                author_consumed_prefix_available
                and recovered_bits[:author_bits_consumed] == expected_consumed_prefix
            )

            decoder_flush_extra = (
                recovered_bits[author_bits_consumed:]
                if author_consumed_prefix_available
                else []
            )
            decoder_flush_extra_count = len(decoder_flush_extra)
            decoder_flush_within_precision_bound = decoder_flush_extra_count <= precision
            text_token_roundtrip_exact = retokenized_ids == output_ids

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
                "kl_q_stego_to_p_lm_untempered_bits_author": float(author_kl),
                "words_per_bit_author": float(words_per_bit),
                "bits_per_word_author": float(1.0 / words_per_bit),
                "payload_bits_per_token_smoke": float(len(secret_bits) / len(output_ids)),
                "avg_entropy_p_tau_bits_author_helper": float(avg_hq),
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
                "implicit_zero_lookahead_bits": implicit_zero_lookahead_bits,
                "recovered_lookahead_padding_bits": "".join(
                    str(bit) for bit in lookahead_padding
                ),
                "lookahead_padding_is_zero": lookahead_padding_is_zero,
                "author_consumed_prefix_available": author_consumed_prefix_available,
                "author_consumed_prefix_matches_zero_padded_payload": (
                    author_consumed_prefix_matches
                ),
                "decoder_flush_extra_bits": "".join(str(bit) for bit in decoder_flush_extra),
                "decoder_flush_extra_bit_count": decoder_flush_extra_count,
                "decoder_flush_within_precision_bound": decoder_flush_within_precision_bound,
            }
            result["status"] = (
                "ok"
                if exact_payload_recovery
                and author_consumed_prefix_matches
                and decoder_flush_within_precision_bound
                else "failed_recovery"
            )

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
    print("\nStage 3 author Arithmetic smoke summary")
    print(f"status: {status}")
    print(f"result: {result_path}")
    print(f"reference worktree unchanged: {result['reference'].get('worktree_unchanged')}")
    if status in {"ok", "failed_recovery"}:
        print(f"exact payload recovery: {recovery.get('exact_payload_recovery')}")
        print(f"author bits consumed: {recovery.get('author_bits_consumed')}")
        print(
            "implicit zero lookahead bits: "
            f"{recovery.get('implicit_zero_lookahead_bits')}"
        )
        print(
            "decoder final-flush extra bits: "
            f"{recovery.get('decoder_flush_extra_bit_count')}"
        )
        print(
            "text/token roundtrip exact: "
            f"{result.get('generation', {}).get('text_token_roundtrip_exact')}"
        )
        print(
            "author KL Q_stego||P_LM(untempered) [bits]: "
            f"{result.get('author_metrics', {}).get('kl_q_stego_to_p_lm_untempered_bits_author')}"
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
