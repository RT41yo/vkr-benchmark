#!/usr/bin/env python3
"""Run Stage-3 Step 3.12 matched author-compatible vs normalized pairs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from stage3_matched_comparison_core import (  # noqa: E402
    EXPECTED_CONTEXT_RANKS,
    EXPECTED_MODEL_REVISION,
    aggregate_records,
    load_author_records,
    load_json,
    load_jsonl,
    paper_sentence_bits,
    sha256_bits,
    sha256_file,
    token_agreement,
    validate_config,
    validate_context_records,
)
from stage3_matched_gpt2 import Stage3GPT2NormalizedAdapter  # noqa: E402
from vkr_benchmark.distributions import GenerationPolicy, ReferenceDistributionBuilder  # noqa: E402
from vkr_benchmark.metrics import (  # noqa: E402
    compute_capacity_entropy_metrics,
    compute_distribution_distortion_metrics,
    compute_raw_lm_quality_metrics,
)
from vkr_benchmark.methods import ArithmeticMethod, BinsMethod, HuffmanMethod  # noqa: E402
from vkr_benchmark.randomness import MethodRandomSource, SecretSource  # noqa: E402
from vkr_benchmark.runner.streaming import (  # noqa: E402
    decode_streaming_tokens,
    encode_fixed_carrier_tokens,
    method_environment_from_builder,
)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "reproducibility" / "matched_author_normalized.json"


class FixedBitSecretSource(SecretSource):
    """Finite exact bit stream used only by the frozen matched comparison."""

    def __init__(self, bits: list[int]) -> None:
        self._bits = tuple(int(x) for x in bits)
        if any(x not in (0, 1) for x in self._bits):
            raise ValueError("secret stream must be binary")
        self._position = 0

    @property
    def position(self) -> int:
        return self._position

    def read_bits(self, count: int) -> tuple[int, ...]:
        if count < 0:
            raise ValueError("count must be non-negative")
        stop = self._position + count
        if stop > len(self._bits):
            raise RuntimeError("Step-3.12 secret stream exhausted")
        out = self._bits[self._position:stop]
        self._position = stop
        return out


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "+inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def _method(point: dict[str, Any]):
    method_id = point["method_id"]
    if method_id == "bins":
        return BinsMethod()
    if method_id == "huffman":
        return HuffmanMethod()
    if method_id == "arithmetic_coding":
        return ArithmeticMethod()
    raise ValueError(f"unsupported matched method_id: {method_id}")


def _rng(point: dict[str, Any]):
    seed = point.get("method_random_seed")
    return None if seed is None else MethodRandomSource(int(seed))


def _resolved_revision(model: Any, tokenizer: Any, requested: str) -> str:
    observed = getattr(getattr(model, "config", None), "_commit_hash", None)
    tok = getattr(tokenizer, "init_kwargs", {}) or {}
    observed_tok = tok.get("_commit_hash")
    for value in (observed, observed_tok):
        if value is not None and str(value) != requested:
            raise RuntimeError(
                f"GPT-2 Medium cache resolved to {value}, expected frozen {requested}"
            )
    return requested


def _load_model(config: dict[str, Any]):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("torch + transformers are required for Step 3.12") from exc

    revision = str(config["model_revision"])
    model_id = str(config["model_id"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        use_fast=False,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=True,
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    resolved = _resolved_revision(model, tokenizer, revision)
    adapter = Stage3GPT2NormalizedAdapter(
        model=model,
        tokenizer=tokenizer,
        resolved_revision=resolved,
        device=device,
    )
    return adapter, model, tokenizer


def run(config_path: Path = DEFAULT_CONFIG) -> int:
    config_path = config_path.resolve()
    config = load_json(config_path)
    validate_config(config)

    manifest_path = REPO_ROOT / config["context_manifest_path"]
    if sha256_file(manifest_path) != config["context_manifest_sha256"]:
        raise RuntimeError("frozen CNN/DailyMail context manifest SHA-256 mismatch")
    manifest = load_json(manifest_path)
    local_context_path = REPO_ROOT / config["local_context_text_path"]
    if not local_context_path.is_file():
        raise FileNotFoundError(
            f"local frozen context text is missing: {local_context_path}; "
            "regenerate it with prepare_stage3_cnndm_contexts.py"
        )
    contexts = validate_context_records(load_jsonl(local_context_path), manifest=manifest)
    context_by_rank = {int(x["selection_rank"]): x for x in contexts}
    author = load_author_records(repo_root=REPO_ROOT, config=config)

    figure3 = load_json(REPO_ROOT / config["figure3_summary_path"])
    if figure3.get("execution_gate_ready_for_step_3_11_interpretation") is not True:
        raise RuntimeError("committed Figure-3 author sweep is not execution-complete")
    if figure3.get("model_id") != config["model_id"]:
        raise RuntimeError("Figure-3 model id differs from matched-comparison model")

    print("Loading frozen GPT-2 Medium for normalized matched comparison ...", flush=True)
    adapter, _model_obj, _tokenizer_obj = _load_model(config)
    print(
        f"Normalized model ready on {adapter.device}; revision={adapter.revision}",
        flush=True,
    )

    records: list[dict[str, Any]] = []
    stream_cfg = config["secret_stream"]
    for point in config["points"]:
        point_id = str(point["id"])
        policy = GenerationPolicy(**point["generation_policy"])
        builder = ReferenceDistributionBuilder(token_space=adapter.token_space, policy=policy)
        environment = method_environment_from_builder(builder)
        print(f"[{point_id}] 8 matched contexts", flush=True)

        for rank in EXPECTED_CONTEXT_RANKS:
            author_record = author[(point_id, rank)]
            context_record = context_by_rank[rank]
            context = str(context_record["context"])
            author_tokens = tuple(int(x) for x in author_record["generation"]["sender_token_ids"])
            carrier_tokens = len(author_tokens)
            if carrier_tokens != int(author_record["author_metrics"]["carrier_tokens"]):
                raise RuntimeError(f"author carrier-token accounting mismatch: {point_id} rank {rank}")

            bits = paper_sentence_bits(
                rank,
                seed=int(stream_cfg["seed"]),
                replicate=int(stream_cfg["replicate"]),
                bit_count=int(stream_cfg["bit_count"]),
            )
            stream_sha = sha256_bits(bits)
            if stream_sha != author_record["secret_stream"]["sha256_ascii_bits"]:
                raise RuntimeError(f"secret-stream mismatch: {point_id} rank {rank}")

            method = _method(point)
            encode = encode_fixed_carrier_tokens(
                lm_adapter=adapter,
                reference_builder=builder,
                method=method,
                method_config=point["method_config"],
                environment=environment,
                prompt_text=context,
                carrier_tokens=carrier_tokens,
                secret_source=FixedBitSecretSource(bits),
                method_random_source=_rng(point),
            )
            if encode.carrier_tokens != carrier_tokens:
                raise RuntimeError(
                    f"normalized method stopped before matched carrier length: {point_id} rank {rank}"
                )

            decode = decode_streaming_tokens(
                lm_adapter=adapter,
                reference_builder=builder,
                method=method,
                method_config=point["method_config"],
                environment=environment,
                prompt_text=context,
                observed_token_ids=encode.carrier_token_ids,
                method_random_source=_rng(point),
                expected_payload_bits=encode.payload_bits,
            )
            recovered = tuple(decode.recovered_bits)
            payload = tuple(encode.payload_secret_bits)
            exact_decode = bool(decode.finalization.complete and recovered == payload)
            if not exact_decode:
                raise RuntimeError(f"normalized token-ID decode mismatch: {point_id} rank {rank}")

            capacity = compute_capacity_entropy_metrics(
                payload_bits=encode.payload_bits,
                reference_entropies_bits=encode.step_reference_entropy_bits,
            )
            distortion = compute_distribution_distortion_metrics(
                encode.step_distribution_distortion
            )
            quality = compute_raw_lm_quality_metrics(encode.step_raw_lm_nll_nats)
            agreement = token_agreement(author_tokens, encode.carrier_token_ids)

            record = {
                "schema_version": "stage3.matched_pair.v1",
                "pair_key": f"{point_id}|rank={rank}",
                "point_id": point_id,
                "method_id": point["method_id"],
                "method_config": point["method_config"],
                "generation_policy": point["generation_policy"],
                "selection_rank": rank,
                "article_id": context_record["article_id"],
                "context_sha256": context_record["context_sha256"],
                "secret_stream_sha256_ascii_bits": stream_sha,
                "carrier_alignment": {
                    "author_carrier_tokens": carrier_tokens,
                    "normalized_carrier_tokens": encode.carrier_tokens,
                    "equal": encode.carrier_tokens == carrier_tokens,
                },
                "author": {
                    "run_key": author_record.get("run_key"),
                    "payload_bits_confirmed": int(author_record["author_metrics"]["payload_bits_confirmed"]),
                    "bits_per_token_author": float(author_record["author_metrics"]["bits_per_word_author"]),
                    "kl_q_stego_to_p_lm_bits_author": float(author_record["author_metrics"]["kl_q_stego_to_p_lm_bits_author"]),
                    "avg_nll_nats_author": float(author_record["author_metrics"]["avg_nll_nats_author"]),
                    "sender_token_ids": list(author_tokens),
                },
                "normalized": {
                    "payload_bits_confirmed": encode.payload_bits,
                    "secret_bits_read": encode.secret_bits_read,
                    "bits_per_token": capacity.bits_per_token,
                    "kl_ref_to_stego_mean_bits": distortion.kl_ref_to_stego_mean_bits,
                    "kl_stego_to_ref_mean_bits": distortion.kl_stego_to_ref_mean_bits,
                    "kl_ref_to_stego_infinite_steps": distortion.kl_ref_to_stego_infinite_steps,
                    "kl_stego_to_ref_infinite_steps": distortion.kl_stego_to_ref_infinite_steps,
                    "tvd_mean": distortion.tvd_mean,
                    "nll_raw_lm_nats_per_token": quality.nll_raw_lm_nats_per_token,
                    "carrier_token_ids": list(encode.carrier_token_ids),
                    "exact_token_id_decode": exact_decode,
                    "recovered_payload_bits": len(recovered),
                },
                "token_sequence_agreement": agreement,
                "comparison_scope": {
                    "metric_closeness_evaluated": False,
                    "note": "Step 3.12 records paired differentials only; Step 3.13 interprets conformance/discrepancy.",
                },
            }
            records.append(record)
            print(
                f"  rank {rank}: tokens={carrier_tokens}; "
                f"payload author/norm={record['author']['payload_bits_confirmed']}/{encode.payload_bits}; "
                f"token agreement={agreement['agreement_rate']:.3f}",
                flush=True,
            )

    summary = aggregate_records(records, config=config)
    summary["config_path"] = str(config_path.relative_to(REPO_ROOT))
    summary["config_sha256"] = sha256_file(config_path)
    summary["context_manifest_sha256"] = config["context_manifest_sha256"]
    summary["model_provenance"] = {
        "model_id": adapter.model_id,
        "requested_revision": config["model_revision"],
        "resolved_revision": adapter.revision,
        "device": adapter.device,
    }
    out_dir = REPO_ROOT / config["execution"]["output_directory"]
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_records = [_json_safe(record) for record in records]
    _atomic_text(
        out_dir / "records.jsonl",
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in safe_records),
    )
    _atomic_text(
        out_dir / "summary.json",
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
    )

    csv_path = out_dir / "paired_comparison.csv"
    temp_csv = csv_path.with_suffix(".csv.tmp")
    with temp_csv.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "point_id", "selection_rank", "carrier_tokens",
            "author_payload_bits", "normalized_payload_bits",
            "author_bits_per_token", "normalized_bits_per_token",
            "author_kl_q_to_p_bits", "normalized_kl_stego_to_ref_bits",
            "normalized_kl_ref_to_stego_bits", "normalized_tvd",
            "token_agreement_rate", "exact_token_sequence_parity",
            "normalized_exact_token_id_decode",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({
                "point_id": record["point_id"],
                "selection_rank": record["selection_rank"],
                "carrier_tokens": record["carrier_alignment"]["author_carrier_tokens"],
                "author_payload_bits": record["author"]["payload_bits_confirmed"],
                "normalized_payload_bits": record["normalized"]["payload_bits_confirmed"],
                "author_bits_per_token": record["author"]["bits_per_token_author"],
                "normalized_bits_per_token": record["normalized"]["bits_per_token"],
                "author_kl_q_to_p_bits": record["author"]["kl_q_stego_to_p_lm_bits_author"],
                "normalized_kl_stego_to_ref_bits": record["normalized"]["kl_stego_to_ref_mean_bits"],
                "normalized_kl_ref_to_stego_bits": record["normalized"]["kl_ref_to_stego_mean_bits"],
                "normalized_tvd": record["normalized"]["tvd_mean"],
                "token_agreement_rate": record["token_sequence_agreement"]["agreement_rate"],
                "exact_token_sequence_parity": record["token_sequence_agreement"]["exact"],
                "normalized_exact_token_id_decode": record["normalized"]["exact_token_id_decode"],
            })
    temp_csv.replace(csv_path)

    print("\nStage 3 Step 3.12 matched comparison execution summary")
    print(f"pairs: {summary['pair_count']}/{summary['expected_pair_count']}")
    print(f"normalized exact token-ID decode: {summary['normalized_exact_decode_count']}/{summary['pair_count']}")
    print("metric closeness evaluated in this step: False")
    print("Stage 3 Step 3.12: READY FOR STEP 3.13 CONFORMANCE ANALYSIS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    return run(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
