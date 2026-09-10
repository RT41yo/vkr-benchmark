#!/usr/bin/env python3
"""Run a structured smoke test of the pinned Harvard Bins implementation.

The runner intentionally executes the third-party implementation without editing
it. It imports the pinned checkout from ``external/NeuralSteganography``, loads
GPT-2 through the reference ``utils.get_model`` helper, embeds a fixed binary
payload with ``block_baseline.encode_block``, then decodes from ordinary text
with ``block_baseline.decode_block``.

A machine-readable result is written even when the reference execution fails.
That makes API/model/environment incompatibilities reproducibility evidence
rather than silent local fixes.
"""

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
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "reproducibility" / "author_bins_gpt2_smoke.json"
DEFAULT_REFERENCE_DIR = REPO_ROOT / "external" / "NeuralSteganography"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "stage3" / "author_smoke" / "bins_gpt2"

EXPECTED_REPOSITORY = "https://github.com/harvardnlp/NeuralSteganography"
EXPECTED_COMMIT = "14e982564aeaf9a33f7b4de440deda2184d17f12"
EXPECTED_METHOD = "bins"
EXPECTED_MODE = "author-compatible"
REQUIRED_REFERENCE_FILES = ("utils.py", "block_baseline.py")
PACKAGE_NAMES = (
    "numpy",
    "torch",
    "transformers",
    "tokenizers",
    "huggingface-hub",
    "safetensors",
    "bitarray",
)


COMPATIBILITY_PROFILE = "hf_4_52_legacy_api"


class _LegacyCausalLMAdapter:
    """Bridge the historical GPT-2 cache API to Transformers 4.52.

    The pinned Harvard code was written for an older GPT-2 interface where
    ``past`` is a sequence of per-layer tensors shaped roughly as
    ``[2, batch, heads, seq, head_dim]``.  Transformers 4.52 accepts
    ``past_key_values`` as per-layer ``(key, value)`` pairs instead.

    The adapter converts only that representation at the API boundary:

    * historical stacked cache -> modern ``(key, value)`` pairs before call;
    * modern ``(key, value)`` pairs -> historical stacked cache after call.

    Cache tensor values are not recomputed or modified.  This keeps the pinned
    ``block_baseline.py`` byte-for-byte untouched while satisfying both its
    ``limit_past`` logic and ``past[0].shape[3]`` decoder guard.
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    @staticmethod
    def _legacy_cache_to_modern(past: Any) -> Any:
        if past is None:
            return None

        modern_layers = []
        for layer_index, layer in enumerate(past):
            if isinstance(layer, (tuple, list)):
                if len(layer) < 2:
                    raise RuntimeError(
                        f"cache layer {layer_index} does not contain key/value tensors"
                    )
                modern_layers.append((layer[0], layer[1]))
                continue

            shape = getattr(layer, "shape", None)
            if shape is None or len(shape) != 5 or int(shape[0]) != 2:
                raise RuntimeError(
                    "historical GPT-2 cache layer must have shape "
                    "[2, batch, heads, seq, head_dim]"
                )
            modern_layers.append((layer[0], layer[1]))

        return tuple(modern_layers)

    @staticmethod
    def _modern_cache_to_legacy(past: Any) -> Any:
        if past is None:
            return None

        # Newer cache objects expose an explicit conversion to the tuple form.
        if hasattr(past, "to_legacy_cache"):
            past = past.to_legacy_cache()

        import torch

        legacy_layers = []
        for layer_index, layer in enumerate(past):
            if not isinstance(layer, (tuple, list)) or len(layer) < 2:
                raise RuntimeError(
                    f"modern cache layer {layer_index} is not a key/value pair"
                )
            key, value = layer[0], layer[1]
            if tuple(key.shape) != tuple(value.shape):
                raise RuntimeError(
                    f"cache key/value shapes differ at layer {layer_index}: "
                    f"{tuple(key.shape)} vs {tuple(value.shape)}"
                )
            legacy_layers.append(torch.stack((key, value), dim=0))

        return tuple(legacy_layers)

    def __call__(self, input_ids: Any, past: Any = None, **kwargs: Any) -> tuple[Any, Any]:
        if "past_key_values" in kwargs:
            raise TypeError("pass legacy past= only through this adapter")

        modern_past = self._legacy_cache_to_modern(past)
        outputs = self._model(
            input_ids,
            past_key_values=modern_past,
            use_cache=True,
            return_dict=False,
            **kwargs,
        )
        if not isinstance(outputs, (tuple, list)) or len(outputs) < 2:
            raise RuntimeError("modern model did not return logits and cache")

        legacy_past = self._modern_cache_to_legacy(outputs[1])
        return outputs[0], legacy_past

    def parameters(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.parameters(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


@contextmanager
def _force_reference_slow_tokenizer(utils_module: Any) -> Iterator[None]:
    """Force the slow GPT-2 tokenizer expected by the pinned Harvard code.

    ``GPT2TokenizerFast`` lacks the public ``encoder``/``decoder`` mappings used
    directly by ``run_single.py`` and ``block_baseline.py``. The slow tokenizer
    exposes the historical interface. The patch is scoped to model loading and
    leaves the third-party checkout byte-for-byte untouched.
    """

    original = utils_module.AutoTokenizer.from_pretrained

    def _from_pretrained(*args: Any, **kwargs: Any) -> Any:
        kwargs = dict(kwargs)
        kwargs["use_fast"] = False
        return original(*args, **kwargs)

    with patch.object(utils_module.AutoTokenizer, "from_pretrained", side_effect=_from_pretrained):
        yield


def _preserve_previous_failure(result_path: Path) -> dict[str, Path]:
    """Preserve known failures before a compatibility-layer rerun.

    Stage 3 keeps each observed incompatibility as evidence instead of silently
    overwriting it when the compatibility bridge is extended.
    """

    if not result_path.is_file():
        return {}
    try:
        previous = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if previous.get("status") != "error":
        return {}

    error = previous.get("error") or {}
    message = str(error.get("message", ""))
    compatibility = previous.get("compatibility")

    if not compatibility and "GPT2TokenizerFast has no attribute encoder" in message:
        filename = "raw_reference_failure.json"
        evidence_key = "raw_reference_failure"
    elif compatibility and "'tuple' object has no attribute 'shape'" in message:
        filename = "compat_cache_shape_failure.json"
        evidence_key = "compat_cache_shape_failure"
    else:
        # Unknown failures are intentionally left as result.json so a developer
        # cannot accidentally classify them as a known compatibility finding.
        return {}

    preserved = result_path.with_name(filename)
    if not preserved.exists():
        preserved.write_bytes(result_path.read_bytes())
    return {evidence_key: preserved}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = json.loads(raw.decode("utf-8"))
    return config, _sha256_bytes(raw)


def _validate_config(config: dict[str, Any]) -> list[int]:
    if config.get("mode") != EXPECTED_MODE:
        raise ValueError(f"expected mode={EXPECTED_MODE!r}")
    if config.get("method") != EXPECTED_METHOD:
        raise ValueError(f"expected method={EXPECTED_METHOD!r}")
    if config.get("reference_repository") != EXPECTED_REPOSITORY:
        raise ValueError("reference_repository does not match the pinned Stage-3 source")
    if config.get("reference_commit") != EXPECTED_COMMIT:
        raise ValueError("reference_commit does not match the pinned Stage-3 source")

    block_size = config.get("block_size")
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive integer")

    secret = config.get("secret_bits")
    if not isinstance(secret, str) or not secret or set(secret) - {"0", "1"}:
        raise ValueError("secret_bits must be a non-empty binary string")
    if len(secret) % block_size != 0:
        raise ValueError("secret_bits length must be divisible by block_size for this smoke test")

    profile = config.get("compatibility_profile")
    if profile != COMPATIBILITY_PROFILE:
        raise ValueError(
            f"expected compatibility_profile={COMPATIBILITY_PROFILE!r}, got {profile!r}"
        )

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
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


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
    versions: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


@contextmanager
def _reference_import_path(reference_dir: Path) -> Iterator[None]:
    """Temporarily import top-level modules from the pinned checkout.

    ``dont_write_bytecode`` prevents this diagnostic execution from creating
    ``__pycache__`` files inside the third-party checkout.
    """

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
            "block_size": config.get("block_size"),
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
    preserved_failures = _preserve_previous_failure(result_path)

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
            block = importlib.import_module("block_baseline")
            _assert_module_origin(utils, reference_dir)
            _assert_module_origin(block, reference_dir)

            import torch

            model_id = str(config["model_id"])
            seed = int(config["seed"])
            block_size = int(config["block_size"])
            finish_sent = bool(config["finish_sent"])
            context = str(config["context"])

            print(f"Loading author-reference model {model_id!r} with slow tokenizer ...", flush=True)
            with _force_reference_slow_tokenizer(utils):
                enc, raw_model = utils.get_model(seed=seed, model_name=model_id)
            model = _LegacyCausalLMAdapter(raw_model)
            result["compatibility"].update(
                {
                    "tokenizer_class": type(enc).__name__,
                    "model_class": type(raw_model).__name__,
                }
            )
            device = str(next(model.parameters()).device)
            if device.startswith("cuda"):
                gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
            else:
                gpu_name = None

            context_tokens = utils.encode_context(context, enc)
            bin2words, words2bin = block.get_bins(len(enc.encoder), block_size)

            print(
                f"Running pinned Bins encode/decode: {len(secret_bits)} secret bits, "
                f"block_size={block_size}, device={device}",
                flush=True,
            )
            output_ids, avg_nll, author_kl, words_per_bit = block.encode_block(
                model,
                enc,
                secret_bits,
                context_tokens,
                block_size,
                bin2words,
                words2bin,
                finish_sent=finish_sent,
                device=device,
            )
            stegotext = enc.decode(output_ids)
            retokenized_ids = enc.encode(stegotext)
            recovered_bits = block.decode_block(
                model,
                enc,
                stegotext,
                context_tokens,
                block_size,
                bin2words,
                words2bin,
                device=device,
            )

            payload_recovered = recovered_bits[: len(secret_bits)]
            exact_payload_recovery = payload_recovered == secret_bits
            text_token_roundtrip_exact = retokenized_ids == output_ids

            output_dir.mkdir(parents=True, exist_ok=True)
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
                "payload_prefix_bits": "".join(str(bit) for bit in payload_recovered),
                "exact_payload_recovery": exact_payload_recovery,
            }
            result["status"] = "ok" if exact_payload_recovery else "failed_recovery"

    except Exception as exc:  # preserve the first unmodified author-run failure as evidence
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
            result = {
                "schema_version": "stage3.author_smoke_result.v1",
                "status": "error",
            }
        reference_status_after = (
            _git(reference_dir, "status", "--porcelain") if reference_dir.is_dir() else None
        )
        result.setdefault("reference", {})["git_status_after"] = reference_status_after
        result["reference"]["worktree_unchanged"] = (
            reference_status_before is not None
            and reference_status_after == reference_status_before
        )
        if preserved_failures:
            evidence = result.setdefault("evidence", {})
            for evidence_key, preserved_path in preserved_failures.items():
                evidence[evidence_key] = str(preserved_path.resolve())
            raw_failure = output_dir / "raw_reference_failure.json"
            if raw_failure.is_file():
                evidence.setdefault("raw_reference_failure", str(raw_failure.resolve()))
            cache_failure = output_dir / "compat_cache_shape_failure.json"
            if cache_failure.is_file():
                evidence.setdefault(
                    "compat_cache_shape_failure", str(cache_failure.resolve())
                )
        result["duration_seconds"] = elapsed
        result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(result_path, result)

    status = result.get("status")
    recovery = result.get("recovery", {})
    print("\nStage 3 author Bins smoke summary")
    print(f"status: {status}")
    print(f"result: {result_path}")
    print(f"reference worktree unchanged: {result['reference'].get('worktree_unchanged')}")
    if status in {"ok", "failed_recovery"}:
        print(f"exact payload recovery: {recovery.get('exact_payload_recovery')}")
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
