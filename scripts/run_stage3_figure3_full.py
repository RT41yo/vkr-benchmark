#!/usr/bin/env python3
"""Execute the frozen 23-point Stage-3 Figure-3 author-compatible sweep.

The runner reuses the Step-3.9 first-sentence mirrors, 80 pinned CNN/DailyMail
contexts and the same deterministic long secret-stream construction.  Results
are written atomically in 69 point/replicate shards, so rerunning the command
safely resumes after interruption.  Scientific paper claims are *not* execution
gates; this step only establishes a complete trustworthy dataset for Step 3.11.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import io
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterator

from aggregate_stage3_figure3 import aggregate as aggregate_completed_run
from stage3_author_runtime import LegacyCausalLMAdapter, force_reference_slow_tokenizer
from stage3_arithmetic_cache_compat import (
    ModernGPT2SequenceCacheLimiter,
    UtilsLimitPastProxy,
    patch_arithmetic_limit_past,
)
from stage3_figure3_core import (
    validate_full_config,
    validate_paper_sentence_pilot_result,
    validate_shard_records,
)
from stage3_paper_sentence_core import (
    encode_arithmetic_mirror,
    encode_bins_mirror,
    encode_huffman_mirror,
)
from run_stage3_paper_sentence_pilot import (
    _parity_entry,
    paper_sentence_bits,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "reproducibility" / "figure3_full_run.json"
DEFAULT_REFERENCE_DIR = REPO_ROOT / "external" / "NeuralSteganography"
EXPECTED_REPOSITORY = "https://github.com/harvardnlp/NeuralSteganography"
EXPECTED_COMMIT = "14e982564aeaf9a33f7b4de440deda2184d17f12"
REQUIRED_REFERENCE_FILES = (
    "utils.py", "block_baseline.py", "huffman.py", "huffman_baseline.py", "arithmetic.py"
)
PACKAGE_NAMES = (
    "numpy", "torch", "transformers", "tokenizers", "huggingface-hub", "safetensors", "bitarray"
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bits(bits: list[int]) -> str:
    return hashlib.sha256("".join(str(int(x)) for x in bits).encode("ascii")).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _git_state(path: Path) -> tuple[str, str]:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=path, check=True, capture_output=True, text=True).stdout
    return head, status


def _repo_head() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _validate_reference(reference_dir: Path) -> tuple[str, str]:
    if not reference_dir.is_dir():
        raise FileNotFoundError(reference_dir)
    for relative in REQUIRED_REFERENCE_FILES:
        if not (reference_dir / relative).is_file():
            raise FileNotFoundError(reference_dir / relative)
    head, status = _git_state(reference_dir)
    if head != EXPECTED_COMMIT:
        raise RuntimeError(f"reference commit mismatch: expected {EXPECTED_COMMIT}, got {head}")
    if status:
        raise RuntimeError("reference checkout must be clean before full Figure-3 run")
    return head, status


@contextmanager
def _reference_import_path(reference_dir: Path) -> Iterator[None]:
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(reference_dir))
    try:
        yield
    finally:
        try:
            sys.path.remove(str(reference_dir))
        except ValueError:
            pass
        sys.dont_write_bytecode = old


def _assert_origin(module: Any, reference_dir: Path) -> None:
    if Path(module.__file__).resolve().parent != reference_dir.resolve():
        raise RuntimeError(f"imported {module.__name__} from unexpected path: {module.__file__}")


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _resolved_model_provenance(enc: Any, model: Any) -> dict[str, Any]:
    model_revision = getattr(getattr(model, "config", None), "_commit_hash", None)
    tokenizer_revision = None
    init_kwargs = getattr(enc, "init_kwargs", None)
    if isinstance(init_kwargs, dict):
        tokenizer_revision = init_kwargs.get("_commit_hash")
    if model_revision and tokenizer_revision and str(model_revision) != str(tokenizer_revision):
        raise RuntimeError(
            f"model/tokenizer revision mismatch: model={model_revision}, tokenizer={tokenizer_revision}"
        )
    resolved = model_revision or tokenizer_revision
    if not resolved:
        raise RuntimeError(
            "unable to resolve the Hugging Face GPT-2 Medium commit from loaded model/tokenizer objects"
        )
    return {
        "requested_identifier": "gpt2-medium",
        "resolved_revision": str(resolved),
        "model_config_commit_hash": str(model_revision) if model_revision else None,
        "tokenizer_commit_hash": str(tokenizer_revision) if tokenizer_revision else None,
        "tokenizer_class": type(enc).__name__,
        "model_class": type(model).__name__,
    }


def _load_contexts(config: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    manifest_path = (REPO_ROOT / str(config["context_manifest_path"])).resolve()
    local_path = (REPO_ROOT / str(config["local_context_text_path"])).resolve()
    if _sha256_file(manifest_path) != str(config["context_manifest_sha256"]):
        raise RuntimeError("CNN/DailyMail context manifest SHA-256 mismatch")
    if not local_path.is_file():
        raise FileNotFoundError(
            f"local context text missing: {local_path}; run prepare_stage3_cnndm_contexts.py first"
        )
    manifest = _load_json(manifest_path)
    public = {int(item["selection_rank"]): item for item in manifest.get("contexts", [])}
    local: list[dict[str, Any]] = []
    with local_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                local.append(json.loads(line))
    local = sorted(local, key=lambda item: int(item["selection_rank"]))
    expected_count = int(config["context_count"])
    if len(local) != expected_count:
        raise RuntimeError(f"expected {expected_count} local contexts, got {len(local)}")
    for expected_rank, item in enumerate(local):
        rank = int(item["selection_rank"])
        if rank != expected_rank:
            raise RuntimeError("local contexts must have contiguous selection ranks 0..79")
        pub = public.get(rank)
        if pub is None:
            raise RuntimeError(f"rank {rank} missing from committed manifest")
        context_sha = hashlib.sha256(str(item["context"]).encode("utf-8")).hexdigest()
        if context_sha != str(pub["context_sha256"]):
            raise RuntimeError(f"context SHA mismatch at rank {rank}")
        if str(item["article_id"]) != str(pub["article_id"]):
            raise RuntimeError(f"article id mismatch at rank {rank}")
        item["context_sha256"] = str(pub["context_sha256"])
    return local, _sha256_file(manifest_path)


def _shard_path(output_dir: Path, point_id: str, replicate: int) -> Path:
    return output_dir / "shards" / point_id / f"replicate_{replicate}.jsonl"


def _shard_is_valid(path: Path, point: dict[str, Any], replicate: int, config: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        records = _read_jsonl(path)
        errors = validate_shard_records(records, point=point, replicate=replicate)
        if errors:
            return False
        for record in records:
            rank = int(record["selection_rank"])
            stream = paper_sentence_bits(
                rank,
                seed=int(config["seed_base"]),
                replicate=int(replicate),
                bit_count=int(config["secret_stream"]["bit_count"]),
            )
            if str(record["secret_stream"]["sha256_ascii_bits"]) != _sha256_bits(stream):
                return False
        return True
    except Exception:
        return False


def _decode_with_diagnostics(call: Any) -> tuple[list[int], dict[str, str] | None, str]:
    """Run a public author decoder without making transport recovery a Figure-3 gate."""
    captured = io.StringIO()
    try:
        with redirect_stdout(captured):
            recovered = call()
        return [int(x) for x in recovered], None, captured.getvalue()
    except Exception as exc:  # diagnostic: preserve the sample and record the decoder failure
        return [], {"type": type(exc).__name__, "message": str(exc)}, captured.getvalue()


def _record_from_mirror(
    *, point: dict[str, Any], replicate: int, context: dict[str, Any], stream: list[int],
    mirror: dict[str, Any], enc: Any, utils: Any, recovered_bits: list[int], stegotext: str,
    recovery_error: dict[str, str] | None = None, recovery_stdout: str = "",
    transport_recovery_attempted: bool = True,
) -> dict[str, Any]:
    output_ids = [int(x) for x in mirror["generated_token_ids"]]
    guard = mirror.get("sentence_guard") or {}
    termination_failure = bool(
        point["method"] == "arithmetic"
        and mirror.get("terminal_reason") == "max_generated_tokens"
        and guard.get("final_guard_exhausted") is True
        and guard.get("reached_real_boundary") is False
    )

    if termination_failure:
        # Retokenization/author decoding of an 8192-token non-sentence is not part
        # of the Figure-3 sample and can itself trigger tokenizer/model length
        # warnings. Preserve the generated text but mark transport as N/A.
        retokenized: list[int] | None = None
        positions: list[int] = []
        first_boundary_final = False
    else:
        retokenized = [int(x) for x in enc.encode(stegotext)]
        positions = [idx for idx, token_id in enumerate(output_ids) if bool(utils.is_sent_finish(int(token_id), enc))]
        first_boundary_final = bool(output_ids) and positions == [len(output_ids) - 1]

    payload_bits = int(mirror["payload_bits_confirmed"] )
    source_prefix = stream[:payload_bits]
    no_zeros = not bool(mirror.get("used_implicit_zero_lookahead"))
    payload_valid = payload_bits >= 0 if point["method"] == "arithmetic" else payload_bits > 0

    if termination_failure:
        exact_payload: bool | None = None
        recovery_status = "not_applicable_termination_failure"
        record_status = "sentence_termination_failure"
    else:
        recovered_prefix = [int(x) for x in recovered_bits[:payload_bits]]
        exact_payload = len(recovered_bits) >= payload_bits and recovered_prefix == source_prefix
        recovery_status = (
            "exact_prefix" if exact_payload
            else "decoder_exception" if recovery_error is not None
            else "prefix_mismatch"
        )
        status_ok = bool(
            first_boundary_final and payload_valid and no_zeros
            and mirror.get("terminal_reason") == "sentence_boundary"
        )
        record_status = "ok" if status_ok else "failed"

    metrics: dict[str, Any] = {
        "avg_nll_nats_author": mirror["avg_nll_nats_author"],
        "perplexity_author": math.exp(float(mirror["avg_nll_nats_author"])),
        "kl_q_stego_to_p_lm_bits_author": mirror["kl_q_stego_to_p_lm_bits_author"],
        "bits_per_word_author": mirror["bits_per_word_author"],
        "words_per_bit_author": mirror["words_per_bit_author"],
        "payload_bits_confirmed": payload_bits,
        "carrier_tokens": int(mirror["carrier_tokens"]),
        "metric_eligibility": "excluded_sentence_termination_failure" if termination_failure else "eligible_completed_sentence",
    }
    if "avg_entropy_p_tau_bits_author_helper" in mirror:
        metrics["avg_entropy_p_tau_bits_author_helper"] = mirror["avg_entropy_p_tau_bits_author_helper"]

    cache_compatibility = mirror.get("arithmetic_cache_compatibility")
    record: dict[str, Any] = {
        "run_key": f"{point['id']}|replicate={replicate}|rank={int(context['selection_rank'])}",
        "point_id": str(point["id"]),
        "method": str(point["method"]),
        "point_parameters": {
            k: v for k, v in point.items() if k not in {"id", "method", "compatibility_profile"}
        },
        "replicate": int(replicate),
        "selection_rank": int(context["selection_rank"]),
        "row_index": int(context["row_index"]),
        "article_id": str(context["article_id"]),
        "context_sha256": str(context["context_sha256"]),
        "secret_stream": {
            "bit_count_available": len(stream),
            "sha256_ascii_bits": _sha256_bits(stream),
            "secret_bits_read": int(mirror["secret_bits_read"]),
            "payload_bits_confirmed": payload_bits,
            "remaining_unread_bits_lower_bound": max(0, len(stream) - int(mirror["secret_bits_read"])),
            "used_implicit_zero_lookahead": bool(mirror.get("used_implicit_zero_lookahead")),
        },
        "generation": {
            "generated_token_count": len(output_ids),
            "sender_token_ids": output_ids,
            "retokenized_token_ids": retokenized,
            "text_token_roundtrip_exact": None if retokenized is None else retokenized == output_ids,
            "sentence_finish_token_positions_zero_based": positions,
            "first_sentence_boundary_is_final_token": first_boundary_final,
            "terminal_reason": str(mirror["terminal_reason"]),
            "final_token_text": enc.decode([output_ids[-1]]) if output_ids else "",
            "stegotext_utf8_bytes": len(stegotext.encode("utf-8")),
            "sentence_guard": mirror.get("sentence_guard"),
        },
        "recovery": {
            "recovered_bit_count": None if not transport_recovery_attempted else len(recovered_bits),
            "exact_confirmed_payload_prefix_recovery": exact_payload,
            "decoded_extra_bit_count_after_confirmed_payload": None if not transport_recovery_attempted else max(0, len(recovered_bits) - payload_bits),
            "confirmed_payload_sha256_ascii_bits": _sha256_bits(source_prefix),
            "status": recovery_status,
            "figure3_execution_gate": False,
            "decoder_stdout": recovery_stdout.strip() or None,
            "decoder_exception": recovery_error,
            "attempted": transport_recovery_attempted,
        },
        "author_metrics": metrics,
        "stegotext": stegotext,
        "status": record_status,
    }
    if termination_failure:
        record["termination_failure"] = {
            "kind": "final_adaptive_guard_exhausted_without_sentence_boundary",
            "final_guard_tokens": int(guard.get("final_guard_tokens", 0)),
            "metric_inclusion": "excluded_from_sentence_level_figure3_means",
            "transport_recovery": "not_applicable",
        }
    if point["method"] == "arithmetic":
        record["arithmetic_state"] = {
            "final_interval_width": mirror.get("final_interval_width"),
            "final_effective_precision_bits": mirror.get("final_effective_precision_bits"),
        }
    if cache_compatibility is not None:
        record["arithmetic_cache_compatibility"] = dict(cache_compatibility)
    return record


def _write_shard_or_fail(path: Path, records: list[dict[str, Any]], point: dict[str, Any], replicate: int) -> None:
    errors = validate_shard_records(records, point=point, replicate=replicate)
    if errors:
        failed_path = path.with_name(path.stem + ".failed.jsonl")
        _write_jsonl_atomic(failed_path, records)
        raise RuntimeError(
            f"generated shard failed validation; diagnostic saved at {failed_path}: " + "; ".join(errors[:10])
        )
    _write_jsonl_atomic(path, records)
    failed_path = path.with_name(path.stem + ".failed.jsonl")
    if failed_path.exists():
        failed_path.unlink()


def _fixed_message_parity_legacy(
    *, pilot_config: dict[str, Any], contexts: list[dict[str, Any]], utils: Any,
    block_baseline: Any, huffman_baseline: Any, model: Any, enc: Any, device: str,
) -> list[dict[str, Any]]:
    sentinel_cfg = pilot_config["parity_sentinel"]
    context = contexts[int(sentinel_cfg["selection_rank"])]
    bits = paper_sentence_bits(
        int(context["selection_rank"]), seed=int(pilot_config["seed"]),
        replicate=int(pilot_config["secret_stream"]["replicate"]), bit_count=int(sentinel_cfg["bit_count"]),
    )
    context_tokens = utils.encode_context(str(context["context"]), enc)
    metric_tol = float(sentinel_cfg["require_metric_match_abs_tol"])
    words_tol = float(sentinel_cfg["require_words_per_bit_match_abs_tol"])
    points = {str(p["id"]): p for p in pilot_config["pilot_points"]}

    b = int(points["bins_b3"]["block_size_bits"])
    bin2words, words2bin = block_baseline.get_bins(len(enc.encoder), b)
    ref = block_baseline.encode_block(model, enc, bits, context_tokens, b, bin2words, words2bin, finish_sent=False, device=device)
    mirror = encode_bins_mirror(
        model=model, enc=enc, utils=utils, message=bits, context_tokens=context_tokens,
        block_size=b, bin2words=bin2words, device=device, stop_at_first_sentence=False, max_generated_tokens=None,
    )
    bins_parity = _parity_entry(
        point_id="bins_b3", reference_output=ref[0], reference_nll=ref[1], reference_kl=ref[2],
        reference_words_per_bit=ref[3], mirror=mirror, metric_tol=metric_tol, words_tol=words_tol,
    )

    e = int(points["huffman_e3"]["candidate_pool_exponent"])
    ref = huffman_baseline.encode_huffman(model, enc, bits, context_tokens, e, finish_sent=False, device=device)
    mirror = encode_huffman_mirror(
        model=model, enc=enc, utils=utils, huffman_module=huffman_baseline, message=bits,
        context_tokens=context_tokens, bits_per_word=e, device=device,
        stop_at_first_sentence=False, max_generated_tokens=None,
    )
    huff_parity = _parity_entry(
        point_id="huffman_e3", reference_output=ref[0], reference_nll=ref[1], reference_kl=ref[2],
        reference_words_per_bit=ref[3], mirror=mirror, metric_tol=metric_tol, words_tol=words_tol,
    )
    return [bins_parity, huff_parity]


def _fixed_message_parity_arithmetic(
    *, pilot_config: dict[str, Any], contexts: list[dict[str, Any]], utils: Any,
    arithmetic: Any, model: Any, enc: Any, device: str,
) -> list[dict[str, Any]]:
    sentinel_cfg = pilot_config["parity_sentinel"]
    context = contexts[int(sentinel_cfg["selection_rank"])]
    bits = paper_sentence_bits(
        int(context["selection_rank"]), seed=int(pilot_config["seed"]),
        replicate=int(pilot_config["secret_stream"]["replicate"]), bit_count=int(sentinel_cfg["bit_count"]),
    )
    context_tokens = utils.encode_context(str(context["context"]), enc)
    metric_tol = float(sentinel_cfg["require_metric_match_abs_tol"])
    words_tol = float(sentinel_cfg["require_words_per_bit_match_abs_tol"])
    parity: list[dict[str, Any]] = []
    for point in [p for p in pilot_config["pilot_points"] if p["method"] == "arithmetic"]:
        temp, topk, precision = float(point["temperature"]), int(point["topk"]), int(point["precision"])
        ref = arithmetic.encode_arithmetic(
            model, enc, bits, context_tokens, temp=temp, finish_sent=False,
            precision=precision, topk=topk, device=device,
        )
        mirror = encode_arithmetic_mirror(
            model=model, enc=enc, utils=utils, message=bits, context_tokens=context_tokens,
            precision=precision, topk=topk, temp=temp, device=device,
            stop_at_first_sentence=False, max_generated_tokens=None,
        )
        parity.append(_parity_entry(
            point_id=str(point["id"]), reference_output=ref[0], reference_nll=ref[1], reference_kl=ref[2],
            reference_words_per_bit=ref[3], reference_entropy=ref[4], mirror=mirror,
            metric_tol=metric_tol, words_tol=words_tol,
        ))
    return parity


def _run_legacy_shards(
    *, config: dict[str, Any], pilot_config: dict[str, Any], contexts: list[dict[str, Any]],
    output_dir: Path, utils: Any, block_baseline: Any, huffman_baseline: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    points = [p for p in config["points"] if p["method"] in {"bins", "huffman"}]
    needed = [
        (p, int(r)) for p in points for r in config["replicates"]
        if not _shard_is_valid(_shard_path(output_dir, str(p["id"]), int(r)), p, int(r), config)
    ]
    print(f"Legacy shards pending: {len(needed)}/{len(points) * len(config['replicates'])}", flush=True)
    print("Loading GPT-2 Medium for Bins/Huffman ...", flush=True)
    with force_reference_slow_tokenizer(utils):
        enc, raw_model = utils.get_model(seed=int(config["seed_base"]), model_name=str(config["model_id"]))
    provenance = _resolved_model_provenance(enc, raw_model)
    model = LegacyCausalLMAdapter(raw_model)
    device = str(next(raw_model.parameters()).device)
    print(
        f"Legacy-compatible model ready on {device}; tokenizer={type(enc).__name__}; "
        f"revision={provenance['resolved_revision']}",
        flush=True,
    )

    parity = _fixed_message_parity_legacy(
        pilot_config=pilot_config, contexts=contexts, utils=utils, block_baseline=block_baseline,
        huffman_baseline=huffman_baseline, model=model, enc=enc, device=device,
    )
    if not all(bool(item["passed"]) for item in parity):
        raise RuntimeError("Bins/Huffman fixed-message mirror parity failed")
    print("Bins/Huffman fixed-message mirror parity: PASS", flush=True)

    bit_count = int(config["secret_stream"]["bit_count"])
    max_tokens = int(config["sentence_stop"]["max_generated_tokens"])
    for point, replicate in needed:
        point_id = str(point["id"])
        path = _shard_path(output_dir, point_id, replicate)
        print(f"[{point_id}] replicate {replicate}: generating 80 contexts", flush=True)
        records: list[dict[str, Any]] = []
        if point["method"] == "bins":
            b = int(point["block_size_bits"])
            bin2words, words2bin = block_baseline.get_bins(len(enc.encoder), b)
        else:
            e = int(point["candidate_pool_exponent"])

        for idx, context in enumerate(contexts):
            rank = int(context["selection_rank"])
            stream = paper_sentence_bits(rank, seed=int(config["seed_base"]), replicate=replicate, bit_count=bit_count)
            context_tokens = utils.encode_context(str(context["context"]), enc)
            if point["method"] == "bins":
                mirror = encode_bins_mirror(
                    model=model, enc=enc, utils=utils, message=stream, context_tokens=context_tokens,
                    block_size=b, bin2words=bin2words, device=device,
                    stop_at_first_sentence=True, max_generated_tokens=max_tokens,
                )
                stegotext = enc.decode(mirror["generated_token_ids"])
                recovered, recovery_error, recovery_stdout = _decode_with_diagnostics(
                    lambda: block_baseline.decode_block(
                        model, enc, stegotext, context_tokens, b, bin2words, words2bin, device=device,
                    )
                )
            else:
                mirror = encode_huffman_mirror(
                    model=model, enc=enc, utils=utils, huffman_module=huffman_baseline, message=stream,
                    context_tokens=context_tokens, bits_per_word=e, device=device,
                    stop_at_first_sentence=True, max_generated_tokens=max_tokens,
                )
                stegotext = enc.decode(mirror["generated_token_ids"])
                recovered, recovery_error, recovery_stdout = _decode_with_diagnostics(
                    lambda: huffman_baseline.decode_huffman(
                        model, enc, stegotext, context_tokens, e, device=device,
                    )
                )
            records.append(_record_from_mirror(
                point=point, replicate=replicate, context=context, stream=stream, mirror=mirror,
                enc=enc, utils=utils, recovered_bits=recovered, stegotext=stegotext,
                recovery_error=recovery_error, recovery_stdout=recovery_stdout,
            ))
            if (idx + 1) % 20 == 0 or idx + 1 == len(contexts):
                print(f"  {idx + 1}/80", flush=True)
        recovery_failures = sum(
            not bool(record["recovery"]["exact_confirmed_payload_prefix_recovery"]) for record in records
        )
        if recovery_failures:
            print(
                f"  transport recovery diagnostic: {recovery_failures}/80 prefix failures; "
                "samples remain valid for Figure-3 generation metrics",
                flush=True,
            )
        _write_shard_or_fail(path, records, point, replicate)
        print(f"  shard saved: {path.relative_to(REPO_ROOT)}", flush=True)

    del model, raw_model, enc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return parity, provenance


def _arithmetic_guard_caps(config: dict[str, Any]) -> list[int]:
    guard = config["sentence_stop"]["adaptive_arithmetic_guard"]
    caps = [int(x) for x in guard["cap_sequence_tokens"]]
    if not caps or caps[0] != int(config["sentence_stop"]["max_generated_tokens"]):
        raise RuntimeError("invalid Arithmetic adaptive sentence-guard configuration")
    if caps != sorted(set(caps)):
        raise RuntimeError("Arithmetic adaptive sentence guards must be strictly increasing")
    return caps


def _encode_arithmetic_with_adaptive_sentence_guard(
    *, config: dict[str, Any], model: Any, enc: Any, utils: Any, stream: list[int],
    context_tokens: list[int], precision: int, topk: int, temp: float, device: str,
) -> tuple[dict[str, Any], ModernGPT2SequenceCacheLimiter]:
    """Rerun cap-hit deterministic Arithmetic samples with larger engineering guards.

    A real first author boundary is the only completed-sentence outcome. If the
    final 8192-token guard is exhausted, the caller records a documented
    sentence_termination_failure instead of extending the guard or inventing a
    boundary.
    """

    caps = _arithmetic_guard_caps(config)
    attempts: list[dict[str, Any]] = []
    final_mirror: dict[str, Any] | None = None
    final_limiter: ModernGPT2SequenceCacheLimiter | None = None
    for cap in caps:
        limiter = ModernGPT2SequenceCacheLimiter(max_cache_tokens=1022)
        arithmetic_utils = UtilsLimitPastProxy(utils, limiter)
        mirror = encode_arithmetic_mirror(
            model=model, enc=enc, utils=arithmetic_utils, message=stream, context_tokens=context_tokens,
            precision=precision, topk=topk, temp=temp, device=device,
            stop_at_first_sentence=True, max_generated_tokens=cap,
        )
        attempts.append({
            "guard_tokens": cap,
            "terminal_reason": str(mirror.get("terminal_reason")),
            "carrier_tokens": int(mirror.get("carrier_tokens", 0)),
        })
        final_mirror, final_limiter = mirror, limiter
        if mirror.get("terminal_reason") != "max_generated_tokens":
            break

    assert final_mirror is not None and final_limiter is not None
    final_guard_exhausted = bool(
        final_mirror.get("terminal_reason") == "max_generated_tokens"
        and attempts[-1]["guard_tokens"] == caps[-1]
    )
    final_mirror["sentence_guard"] = {
        "initial_guard_tokens": caps[0],
        "final_guard_tokens": attempts[-1]["guard_tokens"],
        "escalation_count": len(attempts) - 1,
        "attempts": attempts,
        "reached_real_boundary": final_mirror.get("terminal_reason") == "sentence_boundary",
        "final_guard_exhausted": final_guard_exhausted,
        "classification": "sentence_termination_failure" if final_guard_exhausted else "completed_or_other_terminal",
    }
    return final_mirror, final_limiter


def _run_arithmetic_shards(
    *, config: dict[str, Any], pilot_config: dict[str, Any], contexts: list[dict[str, Any]],
    output_dir: Path, utils: Any, arithmetic: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    points = [p for p in config["points"] if p["method"] == "arithmetic"]
    needed = [
        (p, int(r)) for p in points for r in config["replicates"]
        if not _shard_is_valid(_shard_path(output_dir, str(p["id"]), int(r)), p, int(r), config)
    ]
    print(f"Arithmetic shards pending: {len(needed)}/{len(points) * len(config['replicates'])}", flush=True)
    print("Loading raw GPT-2 Medium for Arithmetic ...", flush=True)
    enc, model = utils.get_model(seed=int(config["seed_base"]), model_name=str(config["model_id"]))
    provenance = _resolved_model_provenance(enc, model)
    device = str(next(model.parameters()).device)
    print(
        f"Arithmetic model ready on {device}; tokenizer={type(enc).__name__}; "
        f"revision={provenance['resolved_revision']}",
        flush=True,
    )

    parity = _fixed_message_parity_arithmetic(
        pilot_config=pilot_config, contexts=contexts, utils=utils, arithmetic=arithmetic,
        model=model, enc=enc, device=device,
    )
    if not all(bool(item["passed"]) for item in parity):
        raise RuntimeError("Arithmetic fixed-message mirror parity failed")
    print("Arithmetic fixed-message mirror parity: PASS", flush=True)

    bit_count = int(config["secret_stream"]["bit_count"])
    for point, replicate in needed:
        point_id = str(point["id"])
        path = _shard_path(output_dir, point_id, replicate)
        temp, topk, precision = float(point["temperature"]), int(point["topk"]), int(point["precision"])
        print(
            f"[{point_id}] replicate {replicate}: 80 contexts, temp={temp}, topk={topk}, precision={precision}",
            flush=True,
        )
        records: list[dict[str, Any]] = []
        for idx, context in enumerate(contexts):
            rank = int(context["selection_rank"])
            stream = paper_sentence_bits(rank, seed=int(config["seed_base"]), replicate=replicate, bit_count=bit_count)
            context_tokens = utils.encode_context(str(context["context"]), enc)
            mirror, encode_cache_limiter = _encode_arithmetic_with_adaptive_sentence_guard(
                config=config, model=model, enc=enc, utils=utils, stream=stream,
                context_tokens=context_tokens, precision=precision, topk=topk, temp=temp, device=device,
            )
            stegotext = enc.decode(mirror["generated_token_ids"])
            guard = mirror.get("sentence_guard") or {}
            termination_failure = bool(
                mirror.get("terminal_reason") == "max_generated_tokens"
                and guard.get("final_guard_exhausted") is True
                and guard.get("reached_real_boundary") is False
            )
            if termination_failure:
                recovered, recovery_error, recovery_stdout = [], None, ""
                decode_cache_limiter = ModernGPT2SequenceCacheLimiter(max_cache_tokens=1022)
                transport_recovery_attempted = False
            else:
                decode_cache_limiter = ModernGPT2SequenceCacheLimiter(max_cache_tokens=1022)
                with patch_arithmetic_limit_past(arithmetic, decode_cache_limiter):
                    recovered, recovery_error, recovery_stdout = _decode_with_diagnostics(
                        lambda: arithmetic.decode_arithmetic(
                            model, enc, stegotext, context_tokens, temp=temp, precision=precision, topk=topk, device=device,
                        )
                    )
                transport_recovery_attempted = True
            mirror["arithmetic_cache_compatibility"] = {
                "max_cache_tokens": 1022,
                "encode_trim_events": int(encode_cache_limiter.trim_events),
                "encode_max_seen_sequence_tokens": int(encode_cache_limiter.max_seen_sequence_tokens),
                "decode_trim_events": int(decode_cache_limiter.trim_events),
                "decode_max_seen_sequence_tokens": int(decode_cache_limiter.max_seen_sequence_tokens),
            }
            records.append(_record_from_mirror(
                point=point, replicate=replicate, context=context, stream=stream, mirror=mirror,
                enc=enc, utils=utils, recovered_bits=recovered, stegotext=stegotext,
                recovery_error=recovery_error, recovery_stdout=recovery_stdout,
                transport_recovery_attempted=transport_recovery_attempted,
            ))
            if (idx + 1) % 20 == 0 or idx + 1 == len(contexts):
                print(f"  {idx + 1}/80", flush=True)
        recovery_failures = sum(
            int(record["author_metrics"]["payload_bits_confirmed"]) > 0
            and not bool(record["recovery"]["exact_confirmed_payload_prefix_recovery"])
            for record in records
        )
        zero_payload = sum(int(record["author_metrics"]["payload_bits_confirmed"]) == 0 for record in records)
        if recovery_failures:
            print(
                f"  transport recovery diagnostic: {recovery_failures}/80 positive-payload prefix failures; "
                "samples remain valid for Figure-3 generation metrics",
                flush=True,
            )
        if zero_payload:
            print(
                f"  zero-payload diagnostic: {zero_payload}/80 valid Arithmetic sentences confirmed 0 bits; "
                "included with bits/word=0",
                flush=True,
            )
        encode_trim_runs = sum(
            int((record.get("arithmetic_cache_compatibility") or {}).get("encode_trim_events", 0)) > 0
            for record in records
        )
        encode_trim_events = sum(
            int((record.get("arithmetic_cache_compatibility") or {}).get("encode_trim_events", 0))
            for record in records
        )
        decode_trim_runs = sum(
            int((record.get("arithmetic_cache_compatibility") or {}).get("decode_trim_events", 0)) > 0
            for record in records
        )
        if encode_trim_runs or decode_trim_runs:
            print(
                "  Arithmetic cache compatibility diagnostic: "
                f"encode-trim runs={encode_trim_runs}/80, encode-trim events={encode_trim_events}, "
                f"decode-trim runs={decode_trim_runs}/80",
                flush=True,
            )
        escalated = [
            r for r in records
            if int(((r.get("generation") or {}).get("sentence_guard") or {}).get("escalation_count", 0)) > 0
        ]
        if escalated:
            max_guard_used = max(
                int(r["generation"]["sentence_guard"]["final_guard_tokens"]) for r in escalated
            )
            print(
                "  sentence-guard diagnostic: "
                f"adaptive reruns={len(escalated)}/80, max guard used={max_guard_used}",
                flush=True,
            )
        termination_failures = [r for r in records if r.get("status") == "sentence_termination_failure"]
        if termination_failures:
            ranks = [int(r["selection_rank"]) for r in termination_failures]
            print(
                "  sentence-termination diagnostic: "
                f"failures={len(termination_failures)}/80 at final 8192-token guard; "
                f"ranks={ranks}; excluded from sentence-level Figure-3 means",
                flush=True,
            )
        _write_shard_or_fail(path, records, point, replicate)
        print(f"  shard saved: {path.relative_to(REPO_ROOT)}", flush=True)

    del model, enc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return parity, provenance


def run(config_path: Path, reference_dir: Path, output_dir: Path | None = None) -> int:
    started = time.perf_counter()
    config_path = config_path.expanduser().resolve()
    reference_dir = reference_dir.expanduser().resolve()
    config = _load_json(config_path)
    matrix_path = (REPO_ROOT / str(config["matrix_path"])).resolve()
    pilot_config_path = (REPO_ROOT / str(config["paper_sentence_pilot_config_path"])).resolve()
    pilot_result_path = (REPO_ROOT / str(config["paper_sentence_pilot_result_path"])).resolve()
    core_path = (REPO_ROOT / str(config["paper_sentence_core_path"])).resolve()
    if _sha256_file(matrix_path) != str(config["matrix_sha256"]):
        raise RuntimeError("frozen paper matrix SHA-256 mismatch")
    if _sha256_file(pilot_config_path) != str(config["paper_sentence_pilot_config_sha256"]):
        raise RuntimeError("Step-3.9 pilot config SHA-256 mismatch")
    if _sha256_file(core_path) != str(config["paper_sentence_core_sha256"]):
        raise RuntimeError("validated Step-3.9 paper-sentence core has changed")
    matrix = _load_json(matrix_path)
    plan = validate_full_config(config, matrix)
    if not pilot_result_path.is_file():
        raise FileNotFoundError(f"Step-3.9 committed result missing: {pilot_result_path}")
    pilot_result = _load_json(pilot_result_path)
    validate_paper_sentence_pilot_result(pilot_result)
    if str(pilot_result.get("config", {}).get("sha256")) != str(config["paper_sentence_pilot_config_sha256"]):
        raise RuntimeError("Step-3.9 result was produced by a different pilot config")
    pilot_config = _load_json(pilot_config_path)
    contexts, manifest_sha = _load_contexts(config)
    reference_head, reference_status_before = _validate_reference(reference_dir)

    if output_dir is None:
        output_dir = (REPO_ROOT / str(config["storage"]["output_directory"])).resolve()
    else:
        output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / str(config["storage"]["run_state_path"])
    state = {
        "schema_version": "stage3.figure3_full_run_state.v1",
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_commit_at_invocation": _repo_head(),
        "config": {"path": str(config_path), "sha256": _sha256_file(config_path)},
        "matrix": {"path": str(matrix_path), "sha256": _sha256_file(matrix_path)},
        "context_manifest_sha256": manifest_sha,
        "model_id": str(config["model_id"]),
        "environment": {"python": sys.version.split()[0], "packages": _package_versions()},
        "reference": {
            "repository": EXPECTED_REPOSITORY,
            "expected_commit": EXPECTED_COMMIT,
            "actual_commit_before": reference_head,
            "git_status_before": reference_status_before,
            "checkout": str(reference_dir),
        },
        "expected": plan,
        "arithmetic_long_context_compatibility": config["arithmetic_long_context_compatibility"],
    }
    _write_json(state_path, state)

    try:
        with _reference_import_path(reference_dir):
            utils = importlib.import_module("utils")
            block_baseline = importlib.import_module("block_baseline")
            huffman_baseline = importlib.import_module("huffman_baseline")
            arithmetic = importlib.import_module("arithmetic")
            for module in (utils, block_baseline, huffman_baseline, arithmetic):
                _assert_origin(module, reference_dir)

            import torch
            state["environment"].update({
                "torch_cuda_available": bool(torch.cuda.is_available()),
                "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else None,
            })
            _write_json(state_path, state)

            legacy_parity, legacy_model_provenance = _run_legacy_shards(
                config=config, pilot_config=pilot_config, contexts=contexts, output_dir=output_dir,
                utils=utils, block_baseline=block_baseline, huffman_baseline=huffman_baseline,
            )
            state["legacy_fixed_message_parity"] = legacy_parity
            state["model_provenance_legacy"] = legacy_model_provenance
            _write_json(state_path, state)

            arithmetic_parity, arithmetic_model_provenance = _run_arithmetic_shards(
                config=config, pilot_config=pilot_config, contexts=contexts, output_dir=output_dir,
                utils=utils, arithmetic=arithmetic,
            )
            state["arithmetic_fixed_message_parity"] = arithmetic_parity
            state["model_provenance_arithmetic"] = arithmetic_model_provenance
            same_revision = (
                legacy_model_provenance["resolved_revision"]
                == arithmetic_model_provenance["resolved_revision"]
            )
            state["model_revision_cross_phase_equal"] = same_revision
            if not same_revision:
                raise RuntimeError(
                    "legacy and Arithmetic phases resolved different GPT-2 Medium revisions"
                )
            state["resolved_model_revision"] = legacy_model_provenance["resolved_revision"]
            _write_json(state_path, state)

        reference_head_after, reference_status_after = _git_state(reference_dir)
        unchanged = (
            reference_head_after == reference_head
            and reference_status_after == reference_status_before
            and reference_status_after == ""
        )
        state["reference"].update({
            "actual_commit_after": reference_head_after,
            "git_status_after": reference_status_after,
            "worktree_unchanged": unchanged,
        })
        state["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        state["duration_seconds"] = time.perf_counter() - started
        state["status"] = "completed" if unchanged else "failed"
        _write_json(state_path, state)
        if not unchanged:
            raise RuntimeError("pinned reference worktree changed during full Figure-3 run")

        summary = aggregate_completed_run(config_path, output_dir=output_dir, write=True)
        ready = bool(summary["execution_gate_ready_for_step_3_11_interpretation"])
        print("\nStage 3 full Figure-3 execution summary")
        print(f"runs: {summary['run_count']}/{summary['expected_run_count']}")
        print(f"points: {summary['point_count']}/{summary['expected_point_count']}")
        print(f"Step-3.9 overlap continuity: {summary['pilot_continuity']['passed']}")
        print(f"reference worktree unchanged: {summary['reference_worktree_unchanged']}")
        cache_diag = summary["arithmetic_cache_compatibility_diagnostic"]
        print(
            "Arithmetic cache compatibility: "
            f"encode-trim-runs={cache_diag['runs_with_encode_trim']}; "
            f"encode-trim-events={cache_diag['total_encode_trim_events']}; "
            f"decode-trim-runs={cache_diag['runs_with_decode_trim']}"
        )
        print("scientific paper claims evaluated in this step: False")
        print(
            "Stage 3 full Figure-3 run: "
            + ("READY FOR STEP 3.11 INTERPRETATION" if ready else "NOT READY")
        )
        return 0 if ready else 1
    except Exception as exc:
        state["status"] = "error"
        state["error"] = {"type": type(exc).__name__, "message": str(exc)}
        state["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        state["duration_seconds"] = time.perf_counter() - started
        try:
            head_after, status_after = _git_state(reference_dir)
            state["reference"].update({
                "actual_commit_after": head_after,
                "git_status_after": status_after,
                "worktree_unchanged": head_after == reference_head and status_after == reference_status_before == "",
            })
        except Exception:
            pass
        _write_json(state_path, state)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    return run(args.config, args.reference_dir, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
