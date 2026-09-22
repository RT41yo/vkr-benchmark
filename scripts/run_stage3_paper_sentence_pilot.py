#!/usr/bin/env python3
"""Run the Stage-3 paper-sentence author-compatible pilot on GPT-2 Medium.

Unlike the earlier fixed-24-bit pilot, this driver uses a long deterministic
uniform-looking bitstream and stops embedding immediately after the *first*
output token satisfying the pinned author's ``utils.is_sent_finish`` predicate.
No greedy completion tail is added and Arithmetic is forbidden from reading
implicit zero look-ahead.  The original Figure-3 batch driver is unavailable,
so this stopping policy is explicitly a frozen Stage-3 operationalization of the
paper phrase "generate an entire sentence given a uniform random message".
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterator

from stage3_author_runtime import LegacyCausalLMAdapter, force_reference_slow_tokenizer
from stage3_paper_sentence_core import (
    encode_arithmetic_mirror,
    encode_bins_mirror,
    encode_huffman_mirror,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "reproducibility" / "paper_sentence_pilot.json"
DEFAULT_REFERENCE_DIR = REPO_ROOT / "external" / "NeuralSteganography"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "stage3" / "paper_reproduction" / "paper_sentence_pilot"
EXPECTED_REPOSITORY = "https://github.com/harvardnlp/NeuralSteganography"
EXPECTED_COMMIT = "14e982564aeaf9a33f7b4de440deda2184d17f12"
EXPECTED_MODEL = "gpt2-medium"
EXPECTED_SCHEMA = "stage3.paper_sentence_pilot.v1"
EXPECTED_POINTS = {
    "bins_b3",
    "huffman_e3",
    "arithmetic_t0.9_k300",
    "arithmetic_t1.0_k50256",
}
REQUIRED_REFERENCE_FILES = (
    "utils.py",
    "block_baseline.py",
    "huffman.py",
    "huffman_baseline.py",
    "arithmetic.py",
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


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bits(bits: list[int]) -> str:
    return _sha256_bytes("".join(str(int(bit)) for bit in bits).encode("ascii"))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _git_reference_state(reference_dir: Path) -> tuple[str, str]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=reference_dir, check=True, capture_output=True, text=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=reference_dir, check=True, capture_output=True, text=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("unable to inspect reference worktree status") from exc
    return head, status


def _validate_reference_checkout(reference_dir: Path) -> tuple[str, str]:
    if not reference_dir.is_dir():
        raise FileNotFoundError(reference_dir)
    for relative in REQUIRED_REFERENCE_FILES:
        if not (reference_dir / relative).is_file():
            raise FileNotFoundError(reference_dir / relative)
    head, status = _git_reference_state(reference_dir)
    if head != EXPECTED_COMMIT:
        raise RuntimeError(f"reference commit mismatch: expected {EXPECTED_COMMIT}, got {head}")
    if status:
        raise RuntimeError("reference checkout must be clean before the paper-sentence pilot")
    return head, status


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


def _package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def paper_sentence_bits(selection_rank: int, *, seed: int, replicate: int, bit_count: int) -> list[int]:
    """Deterministic uniform-looking stream, paired across methods for one context."""

    bits: list[int] = []
    counter = 0
    while len(bits) < bit_count:
        material = (
            f"stage3-paper-sentence|seed={seed}|replicate={replicate}|"
            f"selection_rank={selection_rank}|counter={counter}"
        ).encode("ascii")
        for byte in hashlib.sha256(material).digest():
            for shift in range(7, -1, -1):
                bits.append((byte >> shift) & 1)
                if len(bits) == bit_count:
                    return bits
        counter += 1
    return bits


def _validate_config(config: dict[str, Any], config_path: Path) -> None:
    if config.get("schema_version") != EXPECTED_SCHEMA:
        raise ValueError("unexpected paper-sentence pilot schema")
    if config.get("reference_repository") != EXPECTED_REPOSITORY:
        raise ValueError("unexpected reference repository")
    if config.get("reference_commit") != EXPECTED_COMMIT:
        raise ValueError("unexpected reference commit")
    if config.get("model_id") != EXPECTED_MODEL:
        raise ValueError("paper-sentence pilot must use gpt2-medium")
    if int(config.get("pilot_context_count", 0)) != 8:
        raise ValueError("paper-sentence pilot must use eight frozen contexts")
    stream = config.get("secret_stream") or {}
    if int(stream.get("bit_count", 0)) < 4096:
        raise ValueError("paper-sentence bitstream must be deliberately long")
    stop = config.get("sentence_stop") or {}
    if stop.get("stop_at_first_boundary") is not True:
        raise ValueError("paper-sentence pilot must stop at the first boundary")
    if stop.get("predicate") != "pinned_reference_utils.is_sent_finish":
        raise ValueError("paper-sentence pilot must use the pinned reference sentence predicate")
    if int(stop.get("max_generated_tokens", 0)) != 256:
        raise ValueError("paper-sentence pilot safety cap must remain 256 tokens")
    points = config.get("pilot_points") or []
    if {str(point.get("id")) for point in points} != EXPECTED_POINTS:
        raise ValueError("paper-sentence pilot points differ from the frozen representative set")
    if int((config.get("execution") or {}).get("expected_run_count", 0)) != 32:
        raise ValueError("paper-sentence pilot expected_run_count must remain 32")
    if not config_path.is_file():
        raise FileNotFoundError(config_path)


def _load_and_validate_contexts(config: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    manifest_path = (REPO_ROOT / str(config["context_manifest_path"])).resolve()
    text_path = (REPO_ROOT / str(config["local_context_text_path"])).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    actual_manifest_sha = _sha256_file(manifest_path)
    expected_manifest_sha = str(config["context_manifest_sha256"])
    if actual_manifest_sha != expected_manifest_sha:
        raise RuntimeError(
            f"context manifest SHA-256 mismatch: expected {expected_manifest_sha}, got {actual_manifest_sha}"
        )
    if not text_path.is_file():
        raise FileNotFoundError(
            f"local context text is missing: {text_path}; run prepare_stage3_cnndm_contexts.py first"
        )

    manifest = _load_json(manifest_path)
    public_by_rank = {int(item["selection_rank"]): item for item in manifest.get("contexts", [])}
    local_records: list[dict[str, Any]] = []
    with text_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                local_records.append(json.loads(line))

    pilot_count = int(config["pilot_context_count"])
    pilot_records = sorted(
        (item for item in local_records if bool(item.get("pilot"))),
        key=lambda item: int(item["selection_rank"]),
    )
    if len(pilot_records) != pilot_count:
        raise RuntimeError(f"expected {pilot_count} local pilot contexts, got {len(pilot_records)}")
    for expected_rank, item in enumerate(pilot_records):
        rank = int(item["selection_rank"])
        if rank != expected_rank:
            raise RuntimeError(f"pilot selection ranks must be 0..{pilot_count - 1}")
        public = public_by_rank.get(rank)
        if public is None or not bool(public.get("pilot")):
            raise RuntimeError(f"pilot rank {rank} missing from committed manifest")
        context = str(item["context"])
        if _sha256_bytes(context.encode("utf-8")) != str(public["context_sha256"]):
            raise RuntimeError(f"context hash mismatch at selection_rank={rank}")
        if str(item["article_id"]) != str(public["article_id"]):
            raise RuntimeError(f"article id mismatch at selection_rank={rank}")
    return pilot_records, actual_manifest_sha


def _metric_close(a: float, b: float, tol: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)


def _parity_entry(
    *, point_id: str, reference_output: list[int], reference_nll: float, reference_kl: float,
    reference_words_per_bit: float, mirror: dict[str, Any], metric_tol: float, words_tol: float,
    reference_entropy: float | None = None,
) -> dict[str, Any]:
    exact_ids = [int(x) for x in reference_output] == [int(x) for x in mirror["generated_token_ids"]]
    nll_ok = _metric_close(reference_nll, mirror["avg_nll_nats_author"], metric_tol)
    kl_ok = _metric_close(reference_kl, mirror["kl_q_stego_to_p_lm_bits_author"], metric_tol)
    wpb_ok = _metric_close(reference_words_per_bit, mirror["words_per_bit_author"], words_tol)
    entropy_ok = True
    entropy_delta = None
    if reference_entropy is not None:
        entropy_delta = abs(float(reference_entropy) - float(mirror["avg_entropy_p_tau_bits_author_helper"]))
        entropy_ok = entropy_delta <= metric_tol
    return {
        "point_id": point_id,
        "exact_generated_token_ids": exact_ids,
        "absolute_nll_delta": abs(float(reference_nll) - float(mirror["avg_nll_nats_author"])),
        "absolute_kl_delta": abs(float(reference_kl) - float(mirror["kl_q_stego_to_p_lm_bits_author"])),
        "absolute_words_per_bit_delta": abs(float(reference_words_per_bit) - float(mirror["words_per_bit_author"])),
        "absolute_entropy_delta": entropy_delta,
        "passed": bool(exact_ids and nll_ok and kl_ok and wpb_ok and entropy_ok),
    }


def _sentence_positions(output_ids: list[int], enc: Any, utils: Any) -> list[int]:
    return [
        idx for idx, token_id in enumerate(output_ids)
        if bool(utils.is_sent_finish(int(token_id), enc))
    ]


def _record_from_mirror(
    *, point: dict[str, Any], context: dict[str, Any], stream: list[int], mirror: dict[str, Any],
    enc: Any, utils: Any, recovered_bits: list[int], stegotext: str,
) -> dict[str, Any]:
    output_ids = [int(x) for x in mirror["generated_token_ids"]]
    retokenized = [int(x) for x in enc.encode(stegotext)]
    payload_bits = int(mirror["payload_bits_confirmed"])
    source_prefix = stream[:payload_bits]
    recovered_prefix = [int(x) for x in recovered_bits[:payload_bits]]
    exact_payload = recovered_prefix == source_prefix and len(recovered_bits) >= payload_bits
    positions = _sentence_positions(output_ids, enc, utils)
    first_boundary_is_final = bool(output_ids) and positions == [len(output_ids) - 1]
    positive_payload = payload_bits > 0
    no_implicit_zeros = not bool(mirror.get("used_implicit_zero_lookahead"))
    stopped_at_boundary = mirror.get("terminal_reason") == "sentence_boundary"
    status_ok = exact_payload and first_boundary_is_final and positive_payload and no_implicit_zeros and stopped_at_boundary

    final_token_text = enc.decode([output_ids[-1]]) if output_ids else ""
    return {
        "point_id": str(point["id"]),
        "method": str(point["method"]),
        "point_parameters": {k: v for k, v in point.items() if k not in {"id", "method", "compatibility_profile"}},
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
            "text_token_roundtrip_exact": retokenized == output_ids,
            "sentence_finish_token_positions_zero_based": positions,
            "first_sentence_boundary_is_final_token": first_boundary_is_final,
            "terminal_reason": str(mirror["terminal_reason"]),
            "final_token_text": final_token_text,
            "stegotext_utf8_bytes": len(stegotext.encode("utf-8")),
        },
        "recovery": {
            "recovered_bit_count": len(recovered_bits),
            "exact_confirmed_payload_prefix_recovery": exact_payload,
            "decoded_extra_bit_count_after_confirmed_payload": max(0, len(recovered_bits) - payload_bits),
            "confirmed_payload_sha256_ascii_bits": _sha256_bits(source_prefix),
        },
        "author_metrics": {
            "avg_nll_nats_author": mirror["avg_nll_nats_author"],
            "perplexity_author": math.exp(float(mirror["avg_nll_nats_author"])) if mirror["avg_nll_nats_author"] is not None else None,
            "kl_q_stego_to_p_lm_bits_author": mirror["kl_q_stego_to_p_lm_bits_author"],
            "bits_per_word_author": mirror["bits_per_word_author"],
            "words_per_bit_author": mirror["words_per_bit_author"],
            "payload_bits_confirmed": payload_bits,
            "carrier_tokens": int(mirror["carrier_tokens"]),
        },
        "status": "ok" if status_ok else "failed",
    }


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for point_id in sorted({str(record["point_id"]) for record in records}):
        subset = [record for record in records if record["point_id"] == point_id]
        ok = [record for record in subset if record.get("status") == "ok"]
        metrics = [record["author_metrics"] for record in ok]
        summary[point_id] = {
            "run_count": len(subset),
            "ok_count": len(ok),
            "mean_carrier_tokens": (sum(float(m["carrier_tokens"]) for m in metrics) / len(metrics)) if metrics else None,
            "mean_payload_bits_confirmed": (sum(float(m["payload_bits_confirmed"]) for m in metrics) / len(metrics)) if metrics else None,
            "mean_bits_per_word_author": (sum(float(m["bits_per_word_author"]) for m in metrics) / len(metrics)) if metrics else None,
            "mean_kl_q_stego_to_p_lm_bits_author": (sum(float(m["kl_q_stego_to_p_lm_bits_author"]) for m in metrics) / len(metrics)) if metrics else None,
            "exact_payload_recovery_count": sum(bool(r["recovery"]["exact_confirmed_payload_prefix_recovery"]) for r in subset),
            "first_boundary_final_count": sum(bool(r["generation"]["first_sentence_boundary_is_final_token"]) for r in subset),
            "text_token_roundtrip_exact_count": sum(bool(r["generation"]["text_token_roundtrip_exact"]) for r in subset),
        }
    return summary


def _legacy_phase(
    *, config: dict[str, Any], contexts: list[dict[str, Any]], utils: Any,
    block_baseline: Any, huffman_baseline: Any, text_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import torch

    print("Loading GPT-2 Medium for paper-sentence Bins/Huffman ...", flush=True)
    with force_reference_slow_tokenizer(utils):
        enc, raw_model = utils.get_model(seed=int(config["seed"]), model_name=str(config["model_id"]))
    model = LegacyCausalLMAdapter(raw_model)
    device = str(next(raw_model.parameters()).device)
    print(f"Legacy-compatible model ready on {device}; tokenizer={type(enc).__name__}", flush=True)

    points = {str(point["id"]): point for point in config["pilot_points"]}
    bins_point = points["bins_b3"]
    huff_point = points["huffman_e3"]
    block_size = int(bins_point["block_size_bits"])
    exponent = int(huff_point["candidate_pool_exponent"])
    bin2words, words2bin = block_baseline.get_bins(len(enc.encoder), block_size)
    max_tokens = int(config["sentence_stop"]["max_generated_tokens"])

    sentinel_cfg = config["parity_sentinel"]
    sentinel_context = contexts[int(sentinel_cfg["selection_rank"])]
    sentinel_bits = paper_sentence_bits(
        int(sentinel_context["selection_rank"]), seed=int(config["seed"]), replicate=int(config["secret_stream"]["replicate"]),
        bit_count=int(sentinel_cfg["bit_count"]),
    )
    sentinel_context_tokens = utils.encode_context(str(sentinel_context["context"]), enc)
    metric_tol = float(sentinel_cfg["require_metric_match_abs_tol"])
    words_tol = float(sentinel_cfg["require_words_per_bit_match_abs_tol"])

    ref_bins = block_baseline.encode_block(
        model, enc, sentinel_bits, sentinel_context_tokens, block_size, bin2words, words2bin,
        finish_sent=False, device=device,
    )
    mirror_bins = encode_bins_mirror(
        model=model, enc=enc, utils=utils, message=sentinel_bits, context_tokens=sentinel_context_tokens,
        block_size=block_size, bin2words=bin2words, device=device,
        stop_at_first_sentence=False, max_generated_tokens=None,
    )
    parity_bins = _parity_entry(
        point_id="bins_b3", reference_output=ref_bins[0], reference_nll=ref_bins[1], reference_kl=ref_bins[2],
        reference_words_per_bit=ref_bins[3], mirror=mirror_bins, metric_tol=metric_tol, words_tol=words_tol,
    )

    ref_huff = huffman_baseline.encode_huffman(
        model, enc, sentinel_bits, sentinel_context_tokens, exponent, finish_sent=False, device=device,
    )
    mirror_huff = encode_huffman_mirror(
        model=model, enc=enc, utils=utils, huffman_module=huffman_baseline, message=sentinel_bits,
        context_tokens=sentinel_context_tokens, bits_per_word=exponent, device=device,
        stop_at_first_sentence=False, max_generated_tokens=None,
    )
    parity_huff = _parity_entry(
        point_id="huffman_e3", reference_output=ref_huff[0], reference_nll=ref_huff[1], reference_kl=ref_huff[2],
        reference_words_per_bit=ref_huff[3], mirror=mirror_huff, metric_tol=metric_tol, words_tol=words_tol,
    )
    parity = [parity_bins, parity_huff]
    if not all(item["passed"] for item in parity):
        raise RuntimeError("paper-sentence Bins/Huffman mirrors failed pinned-reference parity")
    print("Bins/Huffman fixed-message mirror parity: PASS", flush=True)

    records: list[dict[str, Any]] = []
    bit_count = int(config["secret_stream"]["bit_count"])
    replicate = int(config["secret_stream"]["replicate"])
    for context in contexts:
        rank = int(context["selection_rank"])
        stream = paper_sentence_bits(rank, seed=int(config["seed"]), replicate=replicate, bit_count=bit_count)
        context_tokens = utils.encode_context(str(context["context"]), enc)
        for point in (bins_point, huff_point):
            print(f"[{point['id']}] context {rank}: stop at first sentence boundary", flush=True)
            if point["method"] == "bins":
                mirror = encode_bins_mirror(
                    model=model, enc=enc, utils=utils, message=stream, context_tokens=context_tokens,
                    block_size=block_size, bin2words=bin2words, device=device,
                    stop_at_first_sentence=True, max_generated_tokens=max_tokens,
                )
                stegotext = enc.decode(mirror["generated_token_ids"])
                recovered = block_baseline.decode_block(
                    model, enc, stegotext, context_tokens, block_size, bin2words, words2bin, device=device,
                )
            else:
                mirror = encode_huffman_mirror(
                    model=model, enc=enc, utils=utils, huffman_module=huffman_baseline, message=stream,
                    context_tokens=context_tokens, bits_per_word=exponent, device=device,
                    stop_at_first_sentence=True, max_generated_tokens=max_tokens,
                )
                stegotext = enc.decode(mirror["generated_token_ids"])
                recovered = huffman_baseline.decode_huffman(
                    model, enc, stegotext, context_tokens, exponent, device=device,
                )
            record = _record_from_mirror(
                point=point, context=context, stream=stream, mirror=mirror, enc=enc, utils=utils,
                recovered_bits=[int(x) for x in recovered], stegotext=stegotext,
            )
            records.append(record)
            text_records.append({
                "point_id": str(point["id"]), "selection_rank": rank,
                "context_sha256": str(context["context_sha256"]), "stegotext": stegotext,
            })

    del model
    del raw_model
    del enc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records, parity


def _arithmetic_phase(
    *, config: dict[str, Any], contexts: list[dict[str, Any]], utils: Any,
    arithmetic: Any, text_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import torch

    print("Loading raw GPT-2 Medium for paper-sentence Arithmetic ...", flush=True)
    enc, model = utils.get_model(seed=int(config["seed"]), model_name=str(config["model_id"]))
    device = str(next(model.parameters()).device)
    print(f"Arithmetic model ready on {device}; tokenizer={type(enc).__name__}", flush=True)

    points = [point for point in config["pilot_points"] if point["method"] == "arithmetic"]
    max_tokens = int(config["sentence_stop"]["max_generated_tokens"])
    sentinel_cfg = config["parity_sentinel"]
    sentinel_context = contexts[int(sentinel_cfg["selection_rank"])]
    sentinel_bits = paper_sentence_bits(
        int(sentinel_context["selection_rank"]), seed=int(config["seed"]), replicate=int(config["secret_stream"]["replicate"]),
        bit_count=int(sentinel_cfg["bit_count"]),
    )
    sentinel_context_tokens = utils.encode_context(str(sentinel_context["context"]), enc)
    metric_tol = float(sentinel_cfg["require_metric_match_abs_tol"])
    words_tol = float(sentinel_cfg["require_words_per_bit_match_abs_tol"])
    parity: list[dict[str, Any]] = []
    for point in points:
        temp = float(point["temperature"])
        topk = int(point["topk"])
        precision = int(point["precision"])
        ref = arithmetic.encode_arithmetic(
            model, enc, sentinel_bits, sentinel_context_tokens, temp=temp, finish_sent=False,
            precision=precision, topk=topk, device=device,
        )
        mirror = encode_arithmetic_mirror(
            model=model, enc=enc, utils=utils, message=sentinel_bits, context_tokens=sentinel_context_tokens,
            precision=precision, topk=topk, temp=temp, device=device,
            stop_at_first_sentence=False, max_generated_tokens=None,
        )
        parity.append(_parity_entry(
            point_id=str(point["id"]), reference_output=ref[0], reference_nll=ref[1], reference_kl=ref[2],
            reference_words_per_bit=ref[3], reference_entropy=ref[4], mirror=mirror,
            metric_tol=metric_tol, words_tol=words_tol,
        ))
    if not all(item["passed"] for item in parity):
        raise RuntimeError("paper-sentence Arithmetic mirror failed pinned-reference parity")
    print("Arithmetic fixed-message mirror parity: PASS", flush=True)

    records: list[dict[str, Any]] = []
    bit_count = int(config["secret_stream"]["bit_count"])
    replicate = int(config["secret_stream"]["replicate"])
    for context in contexts:
        rank = int(context["selection_rank"])
        stream = paper_sentence_bits(rank, seed=int(config["seed"]), replicate=replicate, bit_count=bit_count)
        context_tokens = utils.encode_context(str(context["context"]), enc)
        for point in points:
            temp = float(point["temperature"])
            topk = int(point["topk"])
            precision = int(point["precision"])
            print(
                f"[{point['id']}] context {rank}: temp={temp}, topk={topk}, precision={precision}, first-boundary stop",
                flush=True,
            )
            mirror = encode_arithmetic_mirror(
                model=model, enc=enc, utils=utils, message=stream, context_tokens=context_tokens,
                precision=precision, topk=topk, temp=temp, device=device,
                stop_at_first_sentence=True, max_generated_tokens=max_tokens,
            )
            stegotext = enc.decode(mirror["generated_token_ids"])
            recovered = arithmetic.decode_arithmetic(
                model, enc, stegotext, context_tokens, temp=temp, precision=precision, topk=topk, device=device,
            )
            record = _record_from_mirror(
                point=point, context=context, stream=stream, mirror=mirror, enc=enc, utils=utils,
                recovered_bits=[int(x) for x in recovered], stegotext=stegotext,
            )
            if "avg_entropy_p_tau_bits_author_helper" in mirror:
                record["author_metrics"]["avg_entropy_p_tau_bits_author_helper"] = mirror[
                    "avg_entropy_p_tau_bits_author_helper"
                ]
            record["arithmetic_state"] = {
                "final_interval_width": mirror.get("final_interval_width"),
                "final_effective_precision_bits": mirror.get("final_effective_precision_bits"),
            }
            records.append(record)
            text_records.append({
                "point_id": str(point["id"]), "selection_rank": rank,
                "context_sha256": str(context["context_sha256"]), "stegotext": stegotext,
            })

    del model
    del enc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records, parity


def run(config_path: Path, reference_dir: Path, output_dir: Path) -> int:
    config_path = config_path.expanduser().resolve()
    reference_dir = reference_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    text_path = output_dir / "stegotexts.jsonl"
    started_perf = time.perf_counter()
    result: dict[str, Any] = {
        "schema_version": "stage3.paper_sentence_pilot_result.v1",
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(result_path, result)

    try:
        config = _load_json(config_path)
        _validate_config(config, config_path)
        contexts, manifest_sha = _load_and_validate_contexts(config)
        reference_head, reference_status_before = _validate_reference_checkout(reference_dir)
        result.update({
            "mode": "author-compatible paper-sentence operationalization",
            "model_id": str(config["model_id"]),
            "config": {
                "path": str(config_path),
                "sha256": _sha256_file(config_path),
                "context_manifest_sha256": manifest_sha,
                "pilot_context_count": len(contexts),
                "secret_stream_bit_count": int(config["secret_stream"]["bit_count"]),
                "sentence_stop_predicate": str(config["sentence_stop"]["predicate"]),
                "max_generated_tokens": int(config["sentence_stop"]["max_generated_tokens"]),
            },
            "reference": {
                "repository": EXPECTED_REPOSITORY,
                "expected_commit": EXPECTED_COMMIT,
                "actual_commit": reference_head,
                "checkout": str(reference_dir),
                "git_status_before": reference_status_before,
            },
            "environment": {"python": sys.version.split()[0], "packages": _package_versions()},
            "metric_semantics": {
                "paper_capacity": "confirmed author payload bits at first sentence boundary / generated carrier tokens",
                "author_kl": "D_KL(Q_stego || P_LM), bits/token, averaged over all tokens of the generated first-boundary sentence",
                "sentence_rule": "stop immediately after the first token satisfying pinned utils.is_sent_finish; no greedy completion tail",
                "arithmetic_lookahead": "full precision-bit real secret look-ahead required on every measured step; implicit zero padding is forbidden",
                "historical_claim": "Stage-3 operationalization because the original Figure-3 batch driver is not present in the public pinned repository",
            },
        })
        _write_json(result_path, result)

        with _reference_import_path(reference_dir):
            utils = importlib.import_module("utils")
            block_baseline = importlib.import_module("block_baseline")
            huffman_baseline = importlib.import_module("huffman_baseline")
            arithmetic = importlib.import_module("arithmetic")
            for module in (utils, block_baseline, huffman_baseline, arithmetic):
                _assert_module_origin(module, reference_dir)

            import torch
            result["environment"].update({
                "torch_cuda_available": bool(torch.cuda.is_available()),
                "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else None,
            })

            text_records: list[dict[str, Any]] = []
            legacy_records, legacy_parity = _legacy_phase(
                config=config, contexts=contexts, utils=utils, block_baseline=block_baseline,
                huffman_baseline=huffman_baseline, text_records=text_records,
            )
            _write_jsonl(text_path, text_records)
            arithmetic_records, arithmetic_parity = _arithmetic_phase(
                config=config, contexts=contexts, utils=utils, arithmetic=arithmetic, text_records=text_records,
            )
            records = legacy_records + arithmetic_records
            parity = legacy_parity + arithmetic_parity
            _write_jsonl(text_path, text_records)

        reference_head_after, reference_status_after = _git_reference_state(reference_dir)
        worktree_unchanged = (
            reference_head_after == reference_head
            and reference_status_after == reference_status_before
            and reference_status_after == ""
        )
        expected_count = int(config["execution"]["expected_run_count"])
        parity_ok = len(parity) == 4 and all(bool(item["passed"]) for item in parity)
        run_count_ok = len(records) == expected_count
        record_status_ok = all(record.get("status") == "ok" for record in records)
        first_boundary_ok = all(bool(record["generation"]["first_sentence_boundary_is_final_token"]) for record in records)
        terminal_reason_ok = all(record["generation"]["terminal_reason"] == "sentence_boundary" for record in records)
        payload_positive = all(int(record["author_metrics"]["payload_bits_confirmed"]) > 0 for record in records)
        recovery_ok = all(bool(record["recovery"]["exact_confirmed_payload_prefix_recovery"]) for record in records)
        no_implicit_zero = all(not bool(record["secret_stream"]["used_implicit_zero_lookahead"]) for record in records)
        no_stream_exhaustion = all(record["generation"]["terminal_reason"] != "bitstream_exhausted" for record in records)
        no_cap = all(record["generation"]["terminal_reason"] != "max_generated_tokens" for record in records)

        result.update({
            "parity_checks": parity,
            "records": records,
            "summary_by_point": _summary(records),
            "reference": {
                **result["reference"],
                "actual_commit_after": reference_head_after,
                "git_status_after": reference_status_after,
                "worktree_unchanged": worktree_unchanged,
            },
            "gate": {
                "expected_run_count": expected_count,
                "actual_run_count": len(records),
                "run_count_ok": run_count_ok,
                "all_four_fixed_message_parity_checks_passed": parity_ok,
                "all_runs_stop_at_first_sentence_boundary": first_boundary_ok and terminal_reason_ok,
                "all_payload_counts_positive": payload_positive,
                "all_confirmed_payload_prefixes_recovered": recovery_ok,
                "all_measured_steps_free_of_implicit_zero_lookahead": no_implicit_zero,
                "no_secret_stream_exhaustion": no_stream_exhaustion,
                "no_safety_cap_hits": no_cap,
                "all_record_status_ok": record_status_ok,
                "reference_worktree_unchanged": worktree_unchanged,
                "ready_for_full_figure3_runner_implementation": bool(
                    parity_ok and run_count_ok and record_status_ok and first_boundary_ok and terminal_reason_ok
                    and payload_positive and recovery_ok and no_implicit_zero and no_stream_exhaustion and no_cap
                    and worktree_unchanged
                ),
            },
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": time.perf_counter() - started_perf,
        })
        result["status"] = "ok" if result["gate"]["ready_for_full_figure3_runner_implementation"] else "failed"
        _write_json(result_path, result)

        print("\nStage 3 paper-sentence pilot summary")
        print(f"status: {result['status']}")
        print(f"records: {len(records)}/{expected_count}")
        print(f"fixed-message mirror parity: {parity_ok}")
        print(f"all stop at first boundary: {first_boundary_ok and terminal_reason_ok}")
        print(f"no implicit zero look-ahead: {no_implicit_zero}")
        print(f"exact confirmed payload prefix recovery: {recovery_ok}")
        print(f"reference worktree unchanged: {worktree_unchanged}")
        for point_id, summary in sorted(result["summary_by_point"].items()):
            print(
                f"{point_id}: ok={summary['ok_count']}/8, "
                f"mean tokens={summary['mean_carrier_tokens']}, "
                f"mean payload bits={summary['mean_payload_bits_confirmed']}, "
                f"mean author bpw={summary['mean_bits_per_word_author']}, "
                f"mean author KL bits={summary['mean_kl_q_stego_to_p_lm_bits_author']}"
            )
        return 0 if result["status"] == "ok" else 1
    except Exception as exc:
        result["status"] = "error"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        result["duration_seconds"] = time.perf_counter() - started_perf
        _write_json(result_path, result)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    return run(args.config, args.reference_dir, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
