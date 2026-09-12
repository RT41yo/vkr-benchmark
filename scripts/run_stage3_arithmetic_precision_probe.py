#!/usr/bin/env python3
"""Diagnose finite-precision behavior of the author Arithmetic implementation.

This script intentionally does not modify the pinned Harvard checkout.  It mirrors
its executable arithmetic-coding loop while recording the integer interval state
and three KL decompositions at every payload-carrying step:

* author KL: exact distribution used by the pinned encoder;
* truncation-only KL: LM probabilities renormalized on the retained prefix;
* terminal-fill counterfactual: same rounded integer widths, but any shortfall is
  assigned to the last candidate instead of the first candidate induced by the
  author's cumulative-vector shift.

The mirror must first reproduce the pinned encoder on a sentinel run.  The probe
is diagnostic only and is not part of the frozen Figure-3 curve.
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
import statistics
import subprocess
import sys
import time
import traceback
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "reproducibility" / "arithmetic_precision_probe.json"
DEFAULT_REFERENCE_DIR = REPO_ROOT / "external" / "NeuralSteganography"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "stage3" / "paper_reproduction" / "arithmetic_precision_probe"
EXPECTED_SCHEMA = "stage3.arithmetic_precision_probe.v1"
EXPECTED_RESULT_SCHEMA = "stage3.arithmetic_precision_probe_result.v1"
EXPECTED_COMMIT = "14e982564aeaf9a33f7b4de440deda2184d17f12"
EXPECTED_MODEL = "gpt2-medium"
PACKAGE_NAMES = ("numpy", "torch", "transformers", "tokenizers", "huggingface-hub", "safetensors", "bitarray")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def _git_state(reference_dir: Path) -> tuple[str, str]:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=reference_dir, check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=reference_dir, check=True, capture_output=True, text=True).stdout
    return head, status


def _validate_reference(reference_dir: Path) -> tuple[str, str]:
    for name in ("utils.py", "arithmetic.py"):
        if not (reference_dir / name).is_file():
            raise FileNotFoundError(reference_dir / name)
    head, status = _git_state(reference_dir)
    if head != EXPECTED_COMMIT:
        raise RuntimeError(f"reference commit mismatch: expected {EXPECTED_COMMIT}, got {head}")
    if status:
        raise RuntimeError("reference checkout must be clean before precision probe")
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


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != EXPECTED_SCHEMA:
        raise ValueError("unexpected arithmetic precision probe schema")
    if config.get("reference_commit") != EXPECTED_COMMIT:
        raise ValueError("unexpected reference commit")
    if config.get("model_id") != EXPECTED_MODEL:
        raise ValueError("precision probe must use gpt2-medium")
    if int(config.get("pilot_context_count", 0)) != 8:
        raise ValueError("precision probe must use the eight frozen pilot contexts")
    point = config.get("arithmetic_point") or {}
    if float(point.get("temperature", -1)) != 1.0 or int(point.get("topk", 0)) != 50256:
        raise ValueError("precision probe must target tau=1, topk=50256")
    if [int(x) for x in point.get("precision_values", [])] != [26, 32, 40, 48]:
        raise ValueError("precision grid must remain [26, 32, 40, 48]")
    if bool(point.get("finish_sent", True)):
        raise ValueError("precision probe must isolate payload coding with finish_sent=false")
    if int((config.get("payload") or {}).get("bit_count", 0)) != 256:
        raise ValueError("precision probe payload must remain 256 bits")
    if int((config.get("execution") or {}).get("expected_run_count", 0)) != 32:
        raise ValueError("precision probe must contain 32 runs")


def _load_contexts(config: dict[str, Any]) -> list[dict[str, Any]]:
    manifest_path = REPO_ROOT / str(config["context_manifest_path"])
    local_path = REPO_ROOT / str(config["local_context_text_path"])
    pilot_path = REPO_ROOT / str(config["pilot_result_path"])
    if _sha256_file(manifest_path) != str(config["context_manifest_sha256"]):
        raise RuntimeError("context manifest SHA-256 mismatch")
    if _sha256_file(pilot_path) != str(config["pilot_result_sha256"]):
        raise RuntimeError("committed GPT-2 Medium pilot result SHA-256 mismatch")
    if not local_path.is_file():
        raise FileNotFoundError(f"missing local frozen context text: {local_path}")
    manifest = _load_json(manifest_path)
    public = {int(x["selection_rank"]): x for x in manifest["contexts"]}
    local: list[dict[str, Any]] = []
    for line in local_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if bool(item.get("pilot")):
                local.append(item)
    local.sort(key=lambda x: int(x["selection_rank"]))
    if [int(x["selection_rank"]) for x in local] != list(range(8)):
        raise RuntimeError("local pilot contexts must have selection_rank 0..7")
    for item in local:
        rank = int(item["selection_rank"])
        if rank not in public or not bool(public[rank]["pilot"]):
            raise RuntimeError(f"pilot rank {rank} missing from committed manifest")
        if _sha256_text(str(item["context"])) != str(public[rank]["context_sha256"]):
            raise RuntimeError(f"context SHA mismatch at selection_rank={rank}")
        item = item
    return local


def probe_bits(selection_rank: int, *, seed: int, bit_count: int) -> list[int]:
    bits: list[int] = []
    counter = 0
    while len(bits) < bit_count:
        material = f"stage3-arithmetic-precision-probe|seed={seed}|selection_rank={selection_rank}|counter={counter}".encode("ascii")
        for byte in hashlib.sha256(material).digest():
            for shift in range(7, -1, -1):
                bits.append((byte >> shift) & 1)
                if len(bits) == bit_count:
                    return bits
        counter += 1
    return bits


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    denom = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    if denom == 0:
        return None
    return sum(x * y for x, y in zip(dx, dy)) / denom


def _instrumented_encode(
    *, model: Any, enc: Any, utils: Any, message: list[int], context_tokens: list[int],
    precision: int, topk: int, temp: float, device: str,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from transformers import DynamicCache

    context = torch.tensor(context_tokens[-1022:], device=device, dtype=torch.long)
    max_val = 2 ** precision
    cur_interval = [0, max_val]
    prev = context
    past = None
    output = context
    total_log_probs = 0.0
    total_kl = 0.0
    total_num_for_stats = 0
    i = 0
    steps: list[dict[str, Any]] = []
    max_steps = 4096

    with torch.no_grad():
        while i < len(message):
            if len(steps) >= max_steps:
                raise RuntimeError("precision probe exceeded max_steps")
            i_before = i
            width_before = int(cur_interval[1] - cur_interval[0])
            out = model(prev.unsqueeze(0), past_key_values=DynamicCache.from_legacy_cache(past), use_cache=True)
            logits = out.logits
            past = utils.limit_past(out.past_key_values)
            logits[0, -1, -1] = -1e20
            logits[0, -1, 628] = -1e20
            logits, indices = logits[0, -1, :].sort(descending=True)
            logits = logits.double()
            logits_temp = logits / temp
            probs_temp = F.softmax(logits_temp, dim=0)
            log_probs = F.log_softmax(logits, dim=0)

            threshold = 1.0 / width_before
            first_small = (probs_temp < threshold).nonzero()
            if len(first_small) == 0:
                raise RuntimeError("author code assumption failed: no probability below current threshold")
            k_before = min(max(2, int(first_small[0].item())), topk)
            scaled = probs_temp[:k_before] / probs_temp[:k_before].sum() * width_before
            rounded = scaled.round().long()
            cum_pre = rounded.cumsum(0)
            overfill = (cum_pre > width_before).nonzero()
            if len(overfill) > 0:
                cum_pre = cum_pre[: overfill[0]]
            if len(cum_pre) < 2:
                raise RuntimeError("author candidate prefix collapsed below two tokens")
            k_after = int(len(cum_pre))
            residual = int(width_before - int(cum_pre[-1].item()))
            if residual < 0:
                raise RuntimeError("negative rounding residual after author overfill trimming")

            # Exact author semantics: shift the complete cumulative vector by the shortfall.
            cum_author = cum_pre + residual
            widths_author = cum_author.clone()
            widths_author[1:] = cum_author[1:] - cum_author[:-1]
            q_author = widths_author.double() / widths_author.sum()
            kl_author = float(utils.kl(q_author, q_author.log(), log_probs[:k_after]))

            # Diagnostic 1: pure support truncation before integer rounding.
            retained_mass = float(probs_temp[:k_after].sum().item())
            q_trunc = probs_temp[:k_after].double() / probs_temp[:k_after].double().sum()
            kl_trunc = float(utils.kl(q_trunc, q_trunc.log(), log_probs[:k_after]))

            # Diagnostic 2: same rounded widths but put shortfall in the final bin.
            cum_terminal = cum_pre.clone()
            cum_terminal[-1] += residual
            widths_terminal = cum_terminal.clone()
            widths_terminal[1:] = cum_terminal[1:] - cum_terminal[:-1]
            q_terminal = widths_terminal.double() / widths_terminal.sum()
            kl_terminal = float(utils.kl(q_terminal, q_terminal.log(), log_probs[:k_after]))

            message_bits = message[i:i + precision]
            implicit_zeros = max(0, i + precision - len(message))
            if implicit_zeros:
                message_bits = message_bits + [0] * implicit_zeros
            message_idx = utils.bits2int(reversed(message_bits))
            cum_abs = cum_author + cur_interval[0]
            selection = int((cum_abs > message_idx).nonzero()[0].item())
            new_bottom = cum_abs[selection - 1] if selection > 0 else cur_interval[0]
            new_top = cum_abs[selection]
            lower_bits = list(reversed(utils.int2bits(new_bottom, precision)))
            upper_bits = list(reversed(utils.int2bits(new_top - 1, precision)))
            consumed = int(utils.num_same_from_beg(lower_bits, upper_bits))
            i += consumed
            lower_rest = lower_bits[consumed:] + [0] * consumed
            upper_rest = upper_bits[consumed:] + [1] * consumed
            cur_interval[0] = utils.bits2int(reversed(lower_rest))
            cur_interval[1] = utils.bits2int(reversed(upper_rest)) + 1
            width_after = int(cur_interval[1] - cur_interval[0])

            total_log_probs += float(log_probs[selection].item())
            total_kl += kl_author
            total_num_for_stats += 1
            selected_token_id = int(indices[selection].item())
            prev = indices[selection].view(1)
            output = torch.cat((output, prev))

            steps.append({
                "step_index": len(steps),
                "secret_bit_offset_before": i_before,
                "secret_bit_offset_after": i,
                "bits_consumed": consumed,
                "implicit_zero_lookahead_bits": implicit_zeros,
                "interval_width_before": width_before,
                "effective_precision_bits_before": math.log2(width_before),
                "threshold": threshold,
                "candidate_count_before_overfill": k_before,
                "candidate_count_after_overfill": k_after,
                "retained_lm_mass": retained_mass,
                "rounding_residual_integer_mass": residual,
                "rounding_residual_fraction": residual / width_before,
                "author_kl_bits": kl_author,
                "truncation_only_kl_bits": kl_trunc,
                "terminal_fill_counterfactual_kl_bits": kl_terminal,
                "selected_rank": selection,
                "selected_token_id": selected_token_id,
                "interval_width_after": width_after,
                "effective_precision_bits_after": math.log2(width_after),
                "no_prefix_bits_consumed": consumed == 0,
            })

    generated = output[len(context):].tolist()
    words_per_bit = total_num_for_stats / i
    return {
        "generated_token_ids": [int(x) for x in generated],
        "avg_nll_nats_author": -total_log_probs / total_num_for_stats,
        "avg_kl_bits_author": total_kl / total_num_for_stats,
        "words_per_bit_author": words_per_bit,
        "bits_per_word_author": 1.0 / words_per_bit,
        "payload_bits_consumed": i,
        "step_count": total_num_for_stats,
        "steps": steps,
    }


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for precision in (26, 32, 40, 48):
        subset = [r for r in records if int(r["precision"]) == precision]
        all_steps = [s for r in subset for s in r["steps"]]
        clean = [s for s in all_steps if int(s["implicit_zero_lookahead_bits"]) == 0]
        widths = [float(s["effective_precision_bits_before"]) for s in clean]
        kls = [float(s["author_kl_bits"]) for s in clean]
        residuals = [float(s["rounding_residual_fraction"]) for s in clean]
        out[str(precision)] = {
            "run_count": len(subset),
            "total_step_count": len(all_steps),
            "zero_padding_free_step_count": len(clean),
            "mean_run_author_kl_bits": statistics.mean(float(r["avg_kl_bits_author"]) for r in subset),
            "mean_zero_padding_free_author_kl_bits": statistics.mean(kls) if kls else None,
            "mean_zero_padding_free_truncation_only_kl_bits": statistics.mean(float(s["truncation_only_kl_bits"]) for s in clean) if clean else None,
            "mean_zero_padding_free_terminal_fill_counterfactual_kl_bits": statistics.mean(float(s["terminal_fill_counterfactual_kl_bits"]) for s in clean) if clean else None,
            "mean_effective_precision_bits_before": statistics.mean(widths) if widths else None,
            "minimum_effective_precision_bits_before": min(widths) if widths else None,
            "mean_rounding_residual_fraction": statistics.mean(residuals) if residuals else None,
            "max_rounding_residual_fraction": max(residuals) if residuals else None,
            "fraction_zero_prefix_consumption_steps": (sum(bool(s["no_prefix_bits_consumed"]) for s in clean) / len(clean)) if clean else None,
            "pearson_effective_precision_vs_author_kl": _pearson(widths, kls),
            "pearson_rounding_residual_fraction_vs_author_kl": _pearson(residuals, kls),
        }
    return out


def run(config_path: Path, reference_dir: Path, output_dir: Path) -> int:
    started = time.perf_counter()
    result_path = output_dir / "result.json"
    result: dict[str, Any] = {"schema_version": EXPECTED_RESULT_SCHEMA, "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()}
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(result_path, result)
    try:
        config = _load_json(config_path)
        _validate_config(config)
        contexts = _load_contexts(config)
        head_before, status_before = _validate_reference(reference_dir)
        result.update({
            "mode": "author-compatible diagnostic mirror",
            "model_id": EXPECTED_MODEL,
            "config": {"path": str(config_path.resolve()), "sha256": _sha256_file(config_path)},
            "reference": {"expected_commit": EXPECTED_COMMIT, "actual_commit": head_before, "git_status_before": status_before},
            "environment": {"python": sys.version.split()[0], "packages": _package_versions()},
            "records": [],
        })
        with _reference_import_path(reference_dir):
            utils = importlib.import_module("utils")
            arithmetic = importlib.import_module("arithmetic")
            if Path(utils.__file__).resolve().parent != reference_dir.resolve() or Path(arithmetic.__file__).resolve().parent != reference_dir.resolve():
                raise RuntimeError("reference modules imported from unexpected path")
            import torch
            enc, model = utils.get_model(seed=int(config["seed"]), model_name=EXPECTED_MODEL)
            device = str(next(model.parameters()).device)
            result["environment"].update({"torch_cuda_available": bool(torch.cuda.is_available()), "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else None})
            print(f"Arithmetic precision probe model ready on {device}; tokenizer={type(enc).__name__}", flush=True)

            # Sentinel mirror parity against the pinned executable reference.
            sentinel_rank = int(config["parity_check"]["selection_rank"])
            sentinel_precision = int(config["parity_check"]["precision"])
            sentinel_context = contexts[sentinel_rank]
            sentinel_bits = probe_bits(sentinel_rank, seed=int(config["seed"]), bit_count=int(config["payload"]["bit_count"]))
            sentinel_ctx_tokens = utils.encode_context(str(sentinel_context["context"]), enc)
            mirror = _instrumented_encode(model=model, enc=enc, utils=utils, message=sentinel_bits, context_tokens=sentinel_ctx_tokens, precision=sentinel_precision, topk=50256, temp=1.0, device=device)
            ref_out, ref_nll, ref_kl, ref_wpb, _ = arithmetic.encode_arithmetic(model, enc, sentinel_bits, sentinel_ctx_tokens, finish_sent=False, device=device, temp=1.0, precision=sentinel_precision, topk=50256)
            token_match = [int(x) for x in ref_out] == mirror["generated_token_ids"]
            kl_delta = abs(float(ref_kl) - float(mirror["avg_kl_bits_author"]))
            wpb_delta = abs(float(ref_wpb) - float(mirror["words_per_bit_author"]))
            parity_ok = token_match and kl_delta <= float(config["parity_check"]["require_author_kl_match_abs_tol"]) and wpb_delta <= float(config["parity_check"]["require_words_per_bit_match_abs_tol"])
            result["parity_check"] = {
                "selection_rank": sentinel_rank,
                "precision": sentinel_precision,
                "exact_generated_token_ids": token_match,
                "reference_avg_kl_bits": float(ref_kl),
                "mirror_avg_kl_bits": float(mirror["avg_kl_bits_author"]),
                "absolute_kl_delta": kl_delta,
                "reference_words_per_bit": float(ref_wpb),
                "mirror_words_per_bit": float(mirror["words_per_bit_author"]),
                "absolute_words_per_bit_delta": wpb_delta,
                "passed": parity_ok,
            }
            if not parity_ok:
                raise RuntimeError("instrumented arithmetic mirror failed pinned-reference parity check")
            print("Sentinel parity against pinned arithmetic.py: PASS", flush=True)

            records: list[dict[str, Any]] = []
            for context in contexts:
                rank = int(context["selection_rank"])
                bits = probe_bits(rank, seed=int(config["seed"]), bit_count=int(config["payload"]["bit_count"]))
                ctx_tokens = utils.encode_context(str(context["context"]), enc)
                for precision in [int(x) for x in config["arithmetic_point"]["precision_values"]]:
                    print(f"[precision={precision}] context {rank}", flush=True)
                    diag = _instrumented_encode(model=model, enc=enc, utils=utils, message=bits, context_tokens=ctx_tokens, precision=precision, topk=50256, temp=1.0, device=device)
                    records.append({
                        "selection_rank": rank,
                        "row_index": int(context["row_index"]),
                        "article_id": str(context["article_id"]),
                        "context_sha256": str(context["context_sha256"]),
                        "precision": precision,
                        "temperature": 1.0,
                        "topk": 50256,
                        "payload_bit_count": len(bits),
                        **diag,
                    })
            result["records"] = records
            result["summary_by_precision"] = _summary(records)

        head_after, status_after = _git_state(reference_dir)
        unchanged = head_after == head_before and status_after == status_before == ""
        result["reference"].update({"actual_commit_after": head_after, "git_status_after": status_after, "worktree_unchanged": unchanged})
        run_count_ok = len(result["records"]) == int(config["execution"]["expected_run_count"])
        result["gate"] = {
            "expected_run_count": int(config["execution"]["expected_run_count"]),
            "actual_run_count": len(result["records"]),
            "run_count_ok": run_count_ok,
            "mirror_parity_passed": bool(result["parity_check"]["passed"]),
            "reference_worktree_unchanged": unchanged,
            "full_sweep_remains_blocked_until_review": True,
        }
        result["status"] = "ok" if run_count_ok and result["parity_check"]["passed"] and unchanged else "failed"
        result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        result["duration_seconds"] = time.perf_counter() - started
        _write_json(result_path, result)

        print("\nStage 3 Arithmetic precision probe summary")
        print(f"status: {result['status']}")
        print(f"runs: {len(result['records'])}/32")
        print(f"mirror parity: {result['parity_check']['passed']}")
        print(f"reference worktree unchanged: {unchanged}")
        for precision, item in result["summary_by_precision"].items():
            print(
                f"precision={precision}: full-run mean author KL={item['mean_run_author_kl_bits']}; "
                f"zero-padding-free author KL={item['mean_zero_padding_free_author_kl_bits']}; "
                f"zero-padding-free trunc-only KL={item['mean_zero_padding_free_truncation_only_kl_bits']}; "
                f"zero-padding-free terminal-fill counterfactual KL={item['mean_zero_padding_free_terminal_fill_counterfactual_kl_bits']}; "
                f"min clean effective bits={item['minimum_effective_precision_bits_before']}"
            )
        return 0 if result["status"] == "ok" else 1
    except Exception as exc:
        result["status"] = "error"
        result["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        result["duration_seconds"] = time.perf_counter() - started
        _write_json(result_path, result)
        print(f"Stage 3 Arithmetic precision probe ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    return run(args.config.expanduser().resolve(), args.reference_dir.expanduser().resolve(), args.output_dir.expanduser().resolve())


if __name__ == "__main__":
    raise SystemExit(main())
