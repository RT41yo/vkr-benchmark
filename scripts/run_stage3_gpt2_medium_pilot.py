#!/usr/bin/env python3
"""Run the Stage-3 author-compatible GPT-2 Medium paper pilot.

The pilot deliberately stays outside the normalized benchmark implementation. It
executes the pinned Harvard Bins, Huffman, and Arithmetic algorithms on the eight
frozen CNN/DailyMail contexts. The public reference ``finish_sent`` behavior is
preserved: payload bits are embedded first and greedy top-1 tokens are appended
until the reference sentence-finish predicate fires; author statistics therefore
cover only the payload-carrying prefix, exactly as documented in run_single.py.

This is a technical pre-sweep gate, not a Figure-3 measurement. The original
batch driver and its exact random-message length are not present in the pinned
repository, so the 24-bit pilot payload is explicitly marked as a Stage-3
operationalization used only to validate GPT-2 Medium execution and accounting.
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
import traceback
from typing import Any, Iterator

from stage3_author_runtime import LegacyCausalLMAdapter, force_reference_slow_tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "reproducibility" / "gpt2_medium_pilot.json"
DEFAULT_REFERENCE_DIR = REPO_ROOT / "external" / "NeuralSteganography"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "stage3" / "paper_reproduction" / "gpt2_medium_pilot"
EXPECTED_REPOSITORY = "https://github.com/harvardnlp/NeuralSteganography"
EXPECTED_COMMIT = "14e982564aeaf9a33f7b4de440deda2184d17f12"
EXPECTED_MODEL = "gpt2-medium"
EXPECTED_SCHEMA = "stage3.gpt2_medium_pilot.v1"
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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _git_reference_state(reference_dir: Path) -> tuple[str, str]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=reference_dir,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=reference_dir,
            check=True,
            capture_output=True,
            text=True,
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
        raise RuntimeError("reference checkout must be clean before the pilot")
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


def pilot_secret_bits(selection_rank: int, *, seed: int, bit_count: int) -> list[int]:
    """Return a deterministic uniform-looking bitstream for one pilot context."""

    bits: list[int] = []
    counter = 0
    while len(bits) < bit_count:
        material = (
            f"stage3-gpt2-medium-pilot|seed={seed}|"
            f"selection_rank={selection_rank}|counter={counter}"
        ).encode("ascii")
        digest = hashlib.sha256(material).digest()
        for byte in digest:
            for shift in range(7, -1, -1):
                bits.append((byte >> shift) & 1)
                if len(bits) == bit_count:
                    return bits
        counter += 1
    return bits


def _validate_config(config: dict[str, Any], config_path: Path) -> None:
    if config.get("schema_version") != EXPECTED_SCHEMA:
        raise ValueError("unexpected GPT-2 Medium pilot schema")
    if config.get("reference_repository") != EXPECTED_REPOSITORY:
        raise ValueError("unexpected reference repository")
    if config.get("reference_commit") != EXPECTED_COMMIT:
        raise ValueError("unexpected reference commit")
    if config.get("model_id") != EXPECTED_MODEL:
        raise ValueError("pilot must use gpt2-medium")
    if int(config.get("pilot_context_count", 0)) != 8:
        raise ValueError("pilot must use eight frozen contexts")
    payload = config.get("payload") or {}
    if int(payload.get("bit_count", 0)) != 24:
        raise ValueError("technical pilot payload must remain 24 bits")
    if config.get("finish_sent") is not True:
        raise ValueError("paper pilot must use reference finish_sent=True")
    points = config.get("pilot_points") or []
    expected_ids = {
        "bins_b3",
        "huffman_e3",
        "arithmetic_t0.9_k300",
        "arithmetic_t1.0_k50256",
    }
    if {str(point.get("id")) for point in points} != expected_ids:
        raise ValueError("pilot points differ from the frozen four-point gate")
    if int((config.get("execution") or {}).get("expected_run_count", 0)) != 32:
        raise ValueError("pilot expected_run_count must remain 32")
    if not config_path.is_file():
        raise FileNotFoundError(config_path)


def _load_and_validate_contexts(config: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    manifest_path = (REPO_ROOT / str(config["context_manifest_path"])).resolve()
    text_path = (REPO_ROOT / str(config["local_context_text_path"])).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    expected_manifest_sha = str(config["context_manifest_sha256"])
    actual_manifest_sha = _sha256_file(manifest_path)
    if actual_manifest_sha != expected_manifest_sha:
        raise RuntimeError(
            f"context manifest SHA-256 mismatch: expected {expected_manifest_sha}, got {actual_manifest_sha}"
        )
    if not text_path.is_file():
        raise FileNotFoundError(
            f"local context text is missing: {text_path}; run prepare_stage3_cnndm_contexts.py first"
        )

    manifest = _load_json(manifest_path)
    public_by_rank = {
        int(item["selection_rank"]): item for item in manifest.get("contexts", [])
    }
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
        context_sha = _sha256_bytes(context.encode("utf-8"))
        if context_sha != str(public["context_sha256"]):
            raise RuntimeError(f"context hash mismatch at selection_rank={rank}")
        if str(item["article_id"]) != str(public["article_id"]):
            raise RuntimeError(f"article id mismatch at selection_rank={rank}")
    return pilot_records, actual_manifest_sha


def _sentence_diagnostics(output_ids: list[int], enc: Any, utils: Any) -> dict[str, Any]:
    positions = [
        index for index, token_id in enumerate(output_ids)
        if bool(utils.is_sent_finish(int(token_id), enc))
    ]
    return {
        "sentence_finish_token_positions_zero_based": positions,
        "sentence_finish_token_count": len(positions),
        "final_token_finishes_sentence": bool(output_ids) and bool(positions) and positions[-1] == len(output_ids) - 1,
        "has_early_sentence_finish": any(pos < len(output_ids) - 1 for pos in positions),
    }


def _base_record(
    *,
    point: dict[str, Any],
    context: dict[str, Any],
    secret_bits: list[int],
) -> dict[str, Any]:
    return {
        "point_id": str(point["id"]),
        "method": str(point["method"]),
        "selection_rank": int(context["selection_rank"]),
        "row_index": int(context["row_index"]),
        "article_id": str(context["article_id"]),
        "context_sha256": str(context["context_sha256"]),
        "payload_bit_count": len(secret_bits),
        "payload_bits": "".join(str(bit) for bit in secret_bits),
        "finish_sent": True,
        "status": "running",
    }


def _finalize_common_record(
    record: dict[str, Any],
    *,
    output_ids: list[int],
    stegotext: str,
    retokenized_ids: list[int],
    recovered_bits: list[int],
    avg_nll: float,
    author_kl: float,
    words_per_bit: float,
    enc: Any,
    utils: Any,
    entropy_p_tau: float | None = None,
) -> None:
    payload_bits = [int(ch) for ch in str(record["payload_bits"])]
    payload_len = len(payload_bits)
    payload_prefix = recovered_bits[:payload_len]
    exact_payload_prefix = payload_prefix == payload_bits
    if words_per_bit <= 0:
        raise RuntimeError("author words_per_bit must be positive")

    record.update(
        {
            "status": "ok" if exact_payload_prefix else "payload_mismatch",
            "generation": {
                "generated_token_count_total": len(output_ids),
                "stegotext_utf8_bytes": len(stegotext.encode("utf-8")),
                "sender_token_ids": output_ids,
                "retokenized_token_ids": retokenized_ids,
                "text_token_roundtrip_exact": retokenized_ids == output_ids,
                **_sentence_diagnostics(output_ids, enc, utils),
            },
            "recovery": {
                "recovered_bit_count": len(recovered_bits),
                "payload_prefix_bits": "".join(str(bit) for bit in payload_prefix),
                "exact_payload_prefix_recovery": exact_payload_prefix,
                "decoded_extra_bit_count_after_payload_prefix": max(0, len(recovered_bits) - payload_len),
            },
            "author_metrics": {
                "avg_nll_nats_author": float(avg_nll),
                "perplexity_author": math.exp(float(avg_nll)),
                "kl_q_stego_to_p_lm_bits_author": float(author_kl),
                "words_per_bit_author_stats_prefix": float(words_per_bit),
                "bits_per_word_author_stats_prefix": 1.0 / float(words_per_bit),
                "useful_payload_bits_per_total_generated_token": payload_len / len(output_ids),
            },
        }
    )
    if entropy_p_tau is not None:
        record["author_metrics"]["avg_entropy_p_tau_bits_author_helper"] = float(entropy_p_tau)


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    point_ids = sorted({str(record["point_id"]) for record in records})
    summary: dict[str, Any] = {}
    for point_id in point_ids:
        subset = [record for record in records if record["point_id"] == point_id]
        ok = [record for record in subset if record.get("status") == "ok"]
        metrics = [record["author_metrics"] for record in ok if "author_metrics" in record]
        generation = [record["generation"] for record in ok if "generation" in record]
        summary[point_id] = {
            "run_count": len(subset),
            "ok_count": len(ok),
            "exact_payload_recovery_count": sum(
                bool(record.get("recovery", {}).get("exact_payload_prefix_recovery")) for record in subset
            ),
            "text_token_roundtrip_exact_count": sum(
                bool(record.get("generation", {}).get("text_token_roundtrip_exact")) for record in subset
            ),
            "early_sentence_finish_count": sum(
                bool(record.get("generation", {}).get("has_early_sentence_finish")) for record in subset
            ),
            "mean_generated_token_count_total": (
                sum(item["generated_token_count_total"] for item in generation) / len(generation)
                if generation else None
            ),
            "mean_bits_per_word_author_stats_prefix": (
                sum(item["bits_per_word_author_stats_prefix"] for item in metrics) / len(metrics)
                if metrics else None
            ),
            "mean_kl_q_stego_to_p_lm_bits_author": (
                sum(item["kl_q_stego_to_p_lm_bits_author"] for item in metrics) / len(metrics)
                if metrics else None
            ),
        }
    return summary


def _run_legacy_points(
    *,
    config: dict[str, Any],
    contexts: list[dict[str, Any]],
    utils: Any,
    block_baseline: Any,
    huffman_baseline: Any,
    output_text_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    import torch

    print("Loading GPT-2 Medium for Bins/Huffman with the scoped slow-tokenizer compatibility bridge ...", flush=True)
    with force_reference_slow_tokenizer(utils):
        enc, raw_model = utils.get_model(seed=int(config["seed"]), model_name=str(config["model_id"]))
    model = LegacyCausalLMAdapter(raw_model)
    device = str(next(raw_model.parameters()).device)
    print(f"Legacy-compatible model ready on {device}; tokenizer={type(enc).__name__}", flush=True)

    points = {str(point["id"]): point for point in config["pilot_points"]}
    bins_point = points["bins_b3"]
    huffman_point = points["huffman_e3"]
    block_size = int(bins_point["block_size_bits"])
    bin2words, words2bin = block_baseline.get_bins(len(enc.encoder), block_size)
    records: list[dict[str, Any]] = []

    for context in contexts:
        rank = int(context["selection_rank"])
        secret_bits = pilot_secret_bits(rank, seed=int(config["seed"]), bit_count=int(config["payload"]["bit_count"]))
        context_tokens = utils.encode_context(str(context["context"]), enc)

        for point in (bins_point, huffman_point):
            record = _base_record(point=point, context=context, secret_bits=secret_bits)
            print(f"[{point['id']}] context {rank}: author encode/decode", flush=True)
            if point["method"] == "bins":
                output_ids, avg_nll, author_kl, words_per_bit = block_baseline.encode_block(
                    model,
                    enc,
                    secret_bits,
                    context_tokens,
                    block_size,
                    bin2words,
                    words2bin,
                    finish_sent=True,
                    device=device,
                )
                stegotext = enc.decode(output_ids)
                retokenized_ids = enc.encode(stegotext)
                recovered_bits = block_baseline.decode_block(
                    model,
                    enc,
                    stegotext,
                    context_tokens,
                    block_size,
                    bin2words,
                    words2bin,
                    device=device,
                )
            else:
                exponent = int(huffman_point["candidate_pool_exponent"])
                output_ids, avg_nll, author_kl, words_per_bit = huffman_baseline.encode_huffman(
                    model,
                    enc,
                    secret_bits,
                    context_tokens,
                    exponent,
                    finish_sent=True,
                    device=device,
                )
                stegotext = enc.decode(output_ids)
                retokenized_ids = enc.encode(stegotext)
                recovered_bits = huffman_baseline.decode_huffman(
                    model,
                    enc,
                    stegotext,
                    context_tokens,
                    exponent,
                    device=device,
                )

            if not output_ids:
                raise RuntimeError(f"{point['id']} produced no generated tokens at context {rank}")
            _finalize_common_record(
                record,
                output_ids=[int(x) for x in output_ids],
                stegotext=stegotext,
                retokenized_ids=[int(x) for x in retokenized_ids],
                recovered_bits=[int(x) for x in recovered_bits],
                avg_nll=float(avg_nll),
                author_kl=float(author_kl),
                words_per_bit=float(words_per_bit),
                enc=enc,
                utils=utils,
            )
            records.append(record)
            output_text_records.append(
                {
                    "point_id": str(point["id"]),
                    "selection_rank": rank,
                    "context_sha256": str(context["context_sha256"]),
                    "stegotext": stegotext,
                }
            )

    del model
    del raw_model
    del enc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records


def _run_arithmetic_points(
    *,
    config: dict[str, Any],
    contexts: list[dict[str, Any]],
    utils: Any,
    arithmetic: Any,
    output_text_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    print("Loading raw reference GPT-2 Medium for Arithmetic (native DynamicCache path) ...", flush=True)
    enc, model = utils.get_model(seed=int(config["seed"]), model_name=str(config["model_id"]))
    device = str(next(model.parameters()).device)
    print(f"Arithmetic model ready on {device}; tokenizer={type(enc).__name__}", flush=True)

    arithmetic_points = [point for point in config["pilot_points"] if point["method"] == "arithmetic"]
    records: list[dict[str, Any]] = []
    for context in contexts:
        rank = int(context["selection_rank"])
        secret_bits = pilot_secret_bits(rank, seed=int(config["seed"]), bit_count=int(config["payload"]["bit_count"]))
        context_tokens = utils.encode_context(str(context["context"]), enc)
        for point in arithmetic_points:
            record = _base_record(point=point, context=context, secret_bits=secret_bits)
            temperature = float(point["temperature"])
            topk = int(point["topk"])
            precision = int(point["precision"])
            print(
                f"[{point['id']}] context {rank}: temp={temperature}, topk={topk}, precision={precision}",
                flush=True,
            )
            output_ids, avg_nll, author_kl, words_per_bit, avg_hq = arithmetic.encode_arithmetic(
                model,
                enc,
                secret_bits,
                context_tokens,
                temp=temperature,
                finish_sent=True,
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
                raise RuntimeError(f"{point['id']} produced no generated tokens at context {rank}")
            _finalize_common_record(
                record,
                output_ids=[int(x) for x in output_ids],
                stegotext=stegotext,
                retokenized_ids=[int(x) for x in retokenized_ids],
                recovered_bits=[int(x) for x in recovered_bits],
                avg_nll=float(avg_nll),
                author_kl=float(author_kl),
                words_per_bit=float(words_per_bit),
                enc=enc,
                utils=utils,
                entropy_p_tau=float(avg_hq),
            )
            records.append(record)
            output_text_records.append(
                {
                    "point_id": str(point["id"]),
                    "selection_rank": rank,
                    "context_sha256": str(context["context_sha256"]),
                    "stegotext": stegotext,
                }
            )
    return records


def run(config_path: Path, reference_dir: Path, output_dir: Path) -> int:
    config_path = config_path.expanduser().resolve()
    reference_dir = reference_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    text_path = output_dir / "stegotexts.jsonl"

    started_perf = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    result: dict[str, Any] = {
        "schema_version": "stage3.gpt2_medium_pilot_result.v1",
        "status": "running",
        "started_at_utc": started_utc,
    }
    _write_json(result_path, result)

    try:
        config = _load_json(config_path)
        _validate_config(config, config_path)
        config_sha = _sha256_file(config_path)
        contexts, manifest_sha = _load_and_validate_contexts(config)
        reference_head, reference_status_before = _validate_reference_checkout(reference_dir)
        result.update(
            {
                "mode": "author-compatible",
                "model_id": str(config["model_id"]),
                "config": {
                    "path": str(config_path),
                    "sha256": config_sha,
                    "payload_bit_count": int(config["payload"]["bit_count"]),
                    "finish_sent": bool(config["finish_sent"]),
                    "pilot_context_count": len(contexts),
                    "context_manifest_sha256": manifest_sha,
                },
                "reference": {
                    "repository": EXPECTED_REPOSITORY,
                    "expected_commit": EXPECTED_COMMIT,
                    "actual_commit": reference_head,
                    "checkout": str(reference_dir),
                    "git_status_before": reference_status_before,
                },
                "environment": {
                    "python": sys.version.split()[0],
                    "packages": _package_versions(),
                },
                "metric_semantics": {
                    "author_kl": "D_KL(Q_stego || P_LM) returned by pinned Harvard helpers, bits/token over payload-carrying statistics prefix",
                    "author_bpw": "1 / words_per_bit returned by pinned encoder; reference run_single notes these statistics exclude greedy sentence-completion tokens",
                    "pilot_surface_payload_rate": "24 useful pilot payload bits divided by all generated tokens, stored only as a diagnostic",
                    "figure3_status": "not a final Figure-3 reproduction measurement",
                },
            }
        )
        _write_json(result_path, result)

        with _reference_import_path(reference_dir):
            utils = importlib.import_module("utils")
            block_baseline = importlib.import_module("block_baseline")
            huffman_baseline = importlib.import_module("huffman_baseline")
            arithmetic = importlib.import_module("arithmetic")
            for module in (utils, block_baseline, huffman_baseline, arithmetic):
                _assert_module_origin(module, reference_dir)

            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
            else:
                gpu_name = None
            result["environment"].update(
                {
                    "torch_cuda_available": bool(torch.cuda.is_available()),
                    "gpu_name": gpu_name,
                }
            )

            text_records: list[dict[str, Any]] = []
            records = _run_legacy_points(
                config=config,
                contexts=contexts,
                utils=utils,
                block_baseline=block_baseline,
                huffman_baseline=huffman_baseline,
                output_text_records=text_records,
            )
            _write_jsonl(text_path, text_records)
            result["records"] = records
            result["summary_by_point"] = _summarize(records)
            _write_json(result_path, result)

            arithmetic_records = _run_arithmetic_points(
                config=config,
                contexts=contexts,
                utils=utils,
                arithmetic=arithmetic,
                output_text_records=text_records,
            )
            records.extend(arithmetic_records)
            _write_jsonl(text_path, text_records)
            result["records"] = records
            result["summary_by_point"] = _summarize(records)

        reference_head_after, reference_status_after = _git_reference_state(reference_dir)
        worktree_unchanged = (
            reference_head_after == reference_head
            and reference_status_after == reference_status_before
            and reference_status_after == ""
        )
        expected_run_count = int(config["execution"]["expected_run_count"])
        run_count_ok = len(records) == expected_run_count
        payload_ok = all(
            bool(record.get("recovery", {}).get("exact_payload_prefix_recovery"))
            for record in records
        )
        sentence_end_ok = all(
            bool(record.get("generation", {}).get("final_token_finishes_sentence"))
            for record in records
        )
        record_status_ok = all(record.get("status") == "ok" for record in records)
        early_sentence_count = sum(
            bool(record.get("generation", {}).get("has_early_sentence_finish"))
            for record in records
        )

        result["reference"].update(
            {
                "actual_commit_after": reference_head_after,
                "git_status_after": reference_status_after,
                "worktree_unchanged": worktree_unchanged,
            }
        )
        result["gate"] = {
            "expected_run_count": expected_run_count,
            "actual_run_count": len(records),
            "run_count_ok": run_count_ok,
            "all_exact_payload_prefix_recovery": payload_ok,
            "all_final_tokens_finish_sentence": sentence_end_ok,
            "all_record_status_ok": record_status_ok,
            "early_sentence_finish_run_count": early_sentence_count,
            "paper_sentence_shape_review_required": early_sentence_count > 0,
            "reference_worktree_unchanged": worktree_unchanged,
        }
        result["status"] = (
            "ok"
            if run_count_ok and payload_ok and sentence_end_ok and record_status_ok and worktree_unchanged
            else "failed"
        )
        result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        result["duration_seconds"] = time.perf_counter() - started_perf
        _write_json(result_path, result)

        print("\nStage 3 GPT-2 Medium author-compatible pilot summary")
        print(f"status: {result['status']}")
        print(f"records: {len(records)}/{expected_run_count}")
        print(f"exact payload prefix recovery: {payload_ok}")
        print(f"all final tokens finish sentence: {sentence_end_ok}")
        print(f"early sentence-finish runs (diagnostic): {early_sentence_count}")
        print(f"reference worktree unchanged: {worktree_unchanged}")
        for point_id, point_summary in result["summary_by_point"].items():
            print(
                f"{point_id}: ok={point_summary['ok_count']}/8, "
                f"mean author bpw={point_summary['mean_bits_per_word_author_stats_prefix']}, "
                f"mean author KL bits={point_summary['mean_kl_q_stego_to_p_lm_bits_author']}"
            )
        return 0 if result["status"] == "ok" else 1

    except Exception as exc:  # preserve failure evidence for Stage-3 debugging
        result["status"] = "error"
        result["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        result["duration_seconds"] = time.perf_counter() - started_perf
        try:
            if reference_dir.exists():
                head_after, status_after = _git_reference_state(reference_dir)
                result.setdefault("reference", {}).update(
                    {
                        "actual_commit_after": head_after,
                        "git_status_after": status_after,
                    }
                )
        except Exception:
            pass
        _write_json(result_path, result)
        print(f"Stage 3 GPT-2 Medium pilot ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


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
