#!/usr/bin/env python3
"""Series reliability check for normalized Stage-4 RRC.

The check deliberately separates two questions:

1. Algorithmic reliability: can the RRC receiver recover the secret when it is
   given the exact token IDs emitted by the sender?
2. Ordinary-text reliability: after ``token IDs -> text -> retokenization``, do
   the recovered token IDs and secret still match?

Only the first question is a conformance gate.  Text-channel failures caused by
retokenization are an observed transport property and are reported rather than
silently repaired or converted into an RRC algorithm failure.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import statistics
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vkr_benchmark.config import LocalModelConfig
from vkr_benchmark.distributions import GenerationPolicy, ReferenceDistributionBuilder
from vkr_benchmark.errors import BenchmarkError
from vkr_benchmark.inputs import PromptRegistry
from vkr_benchmark.lm import HFCausalLMAdapter
from vkr_benchmark.methods import RRCMethod
from vkr_benchmark.randomness import MethodRandomSource, Shake256SecretSource
from vkr_benchmark.runner import (
    decode_streaming_tokens,
    encode_fixed_payload_bits,
    method_environment_from_builder,
)
from vkr_benchmark.transport import TextChannel

REFERENCE_COMMIT = "dae326259e4fca8bc4fcf460dafdbc0e88a0a71a"


@dataclass(frozen=True, slots=True)
class SampleResult:
    sample_index: int
    secret_id: str
    carrier_tokens: int | None
    bits_per_token: float | None
    termination_retries: int | None
    direct_decoder_complete: bool
    direct_secret_exact: bool
    token_sequence_roundtrip_exact: bool
    first_token_mismatch: int | None
    token_count_delta: int | None
    text_decoder_complete: bool
    text_secret_exact: bool
    encode_error: str | None = None
    direct_decode_error: str | None = None
    text_decode_error: str | None = None


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a JSON object: {path}")
    return payload


def _rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def summarize(samples: list[SampleResult]) -> dict[str, Any]:
    total = len(samples)
    encode_failures = sum(s.encode_error is not None for s in samples)
    direct_exact = sum(s.direct_secret_exact for s in samples)
    token_exact = sum(s.token_sequence_roundtrip_exact for s in samples)
    text_complete = sum(s.text_decoder_complete for s in samples)
    text_exact = sum(s.text_secret_exact for s in samples)
    token_exact_but_secret_wrong = sum(
        s.token_sequence_roundtrip_exact and not s.text_secret_exact for s in samples
    )
    token_changed_but_secret_recovered = sum(
        (not s.token_sequence_roundtrip_exact) and s.text_secret_exact for s in samples
    )

    carrier = [s.carrier_tokens for s in samples if s.carrier_tokens is not None]
    bpt = [s.bits_per_token for s in samples if s.bits_per_token is not None]
    retries = [
        s.termination_retries
        for s in samples
        if s.termination_retries is not None
    ]
    deltas = [s.token_count_delta for s in samples if s.token_count_delta is not None]

    direct_gate = (
        encode_failures == 0
        and direct_exact == total
        and token_exact_but_secret_wrong == 0
    )

    return {
        "sample_count": total,
        "encode_failures": encode_failures,
        "direct_token_id_recovery": {
            "exact": direct_exact,
            "rate": _rate(direct_exact, total),
        },
        "text_transport": {
            "token_sequence_exact": token_exact,
            "token_sequence_exact_rate": _rate(token_exact, total),
            "decoder_complete": text_complete,
            "decoder_complete_rate": _rate(text_complete, total),
            "secret_exact": text_exact,
            "secret_exact_rate": _rate(text_exact, total),
            "token_exact_but_secret_wrong": token_exact_but_secret_wrong,
            "token_changed_but_secret_recovered": token_changed_but_secret_recovered,
        },
        "carrier_tokens": {
            "min": min(carrier) if carrier else None,
            "max": max(carrier) if carrier else None,
            "mean": statistics.fmean(carrier) if carrier else None,
        },
        "bits_per_token": {
            "min": min(bpt) if bpt else None,
            "max": max(bpt) if bpt else None,
            "mean": statistics.fmean(bpt) if bpt else None,
        },
        "termination_retries": {
            "sum": sum(retries) if retries else 0,
            "max": max(retries) if retries else 0,
        },
        "token_count_delta": {
            "min": min(deltas) if deltas else None,
            "max": max(deltas) if deltas else None,
            "mean": statistics.fmean(deltas) if deltas else None,
        },
        "status": "pass" if direct_gate else "fail",
        "status_semantics": (
            "PASS gates exact direct token-ID recovery only; ordinary-text "
            "reliability is measured and may be below 100% because retokenization "
            "can change the carrier token sequence."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/reproducibility/stage4_rrc_text_channel.json",
    )
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    cfg_path = args.config.expanduser().resolve()
    cfg = _load_json(cfg_path)

    if cfg.get("reference_commit") != REFERENCE_COMMIT:
        raise ValueError("RRC text-channel config does not pin the expected reference commit")

    sample_count = int(args.samples if args.samples is not None else cfg["samples"])
    if sample_count <= 0:
        raise ValueError("samples must be positive")
    progress_every = int(args.progress_every)
    if progress_every <= 0:
        raise ValueError("progress-every must be positive")

    model_path = REPO_ROOT / str(cfg["model_config"])
    model_config = LocalModelConfig.from_json(model_path)
    prompt = PromptRegistry.from_jsonl(REPO_ROOT / "data/prompts.jsonl").get(
        str(cfg["prompt_id"])
    )
    generation = cfg["generation"]
    policy = GenerationPolicy(
        temperature=float(generation.get("temperature", 1.0)),
        top_k=generation.get("top_k"),
        top_p=float(generation.get("top_p", 1.0)),
        exclude_special_tokens=bool(generation.get("exclude_special_tokens", True)),
    )

    payload_bits = int(cfg["payload_bits"])
    max_carrier_tokens = int(cfg["max_carrier_tokens"])
    key = cfg["key"]
    precision_margin = int(cfg["precision_margin"])
    secret_prefix = str(cfg["secret_id_prefix"])

    print("Stage 4 RRC text-channel reliability")
    print("reference commit:", REFERENCE_COMMIT)
    print("model:", model_config.model_id)
    print("revision:", model_config.revision)
    print("prompt_id:", prompt.prompt_id)
    print("payload bits:", payload_bits)
    print("max carrier tokens:", max_carrier_tokens)
    print("samples:", sample_count)
    print("key:", key)
    print("precision margin:", precision_margin)
    print("Loading local model...")

    lm = HFCausalLMAdapter.from_local_config(model_config)
    builder = ReferenceDistributionBuilder(token_space=lm.token_space, policy=policy)
    environment = method_environment_from_builder(builder)
    method = RRCMethod()
    method_config = {
        "precision_margin": precision_margin,
        "verify_termination": True,
    }
    channel = TextChannel(lm)

    results: list[SampleResult] = []
    for sample_index in range(sample_count):
        secret_id = f"{secret_prefix}-{sample_index + 1:06d}"
        try:
            encoded = encode_fixed_payload_bits(
                lm_adapter=lm,
                reference_builder=builder,
                method=method,
                method_config=method_config,
                environment=environment,
                prompt_text=prompt.text,
                target_payload_bits=payload_bits,
                max_carrier_tokens=max_carrier_tokens,
                secret_source=Shake256SecretSource(secret_id),
                method_random_source=MethodRandomSource(key),
                key=key,
            )
        except BenchmarkError as exc:
            results.append(
                SampleResult(
                    sample_index=sample_index,
                    secret_id=secret_id,
                    carrier_tokens=None,
                    bits_per_token=None,
                    termination_retries=None,
                    direct_decoder_complete=False,
                    direct_secret_exact=False,
                    token_sequence_roundtrip_exact=False,
                    first_token_mismatch=None,
                    token_count_delta=None,
                    text_decoder_complete=False,
                    text_secret_exact=False,
                    encode_error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        direct_complete = False
        direct_exact = False
        direct_error = None
        try:
            direct = decode_streaming_tokens(
                lm_adapter=lm,
                reference_builder=builder,
                method=method,
                method_config=method_config,
                environment=environment,
                prompt_text=prompt.text,
                observed_token_ids=encoded.carrier_token_ids,
                method_random_source=MethodRandomSource(key),
                expected_payload_bits=payload_bits,
                key=key,
            )
            direct_complete = bool(direct.finalization.complete)
            direct_exact = (
                direct_complete
                and direct.recovered_bits == encoded.payload_secret_bits
            )
        except BenchmarkError as exc:
            direct_error = f"{type(exc).__name__}: {exc}"

        transport = channel.transmit(encoded.carrier_token_ids)
        text_complete = False
        text_exact = False
        text_error = None
        try:
            decoded_text = decode_streaming_tokens(
                lm_adapter=lm,
                reference_builder=builder,
                method=method,
                method_config=method_config,
                environment=environment,
                prompt_text=prompt.text,
                observed_token_ids=transport.receiver_token_ids,
                method_random_source=MethodRandomSource(key),
                expected_payload_bits=payload_bits,
                key=key,
            )
            text_complete = bool(decoded_text.finalization.complete)
            text_exact = (
                text_complete
                and decoded_text.recovered_bits == encoded.payload_secret_bits
            )
        except BenchmarkError as exc:
            text_error = f"{type(exc).__name__}: {exc}"

        retries_raw = encoded.finalization.metadata.get("termination_retries")
        retries = int(retries_raw) if retries_raw is not None else None
        results.append(
            SampleResult(
                sample_index=sample_index,
                secret_id=secret_id,
                carrier_tokens=encoded.carrier_tokens,
                bits_per_token=payload_bits / encoded.carrier_tokens,
                termination_retries=retries,
                direct_decoder_complete=direct_complete,
                direct_secret_exact=direct_exact,
                token_sequence_roundtrip_exact=transport.token_sequence_roundtrip_exact,
                first_token_mismatch=transport.first_token_mismatch,
                token_count_delta=transport.token_count_delta,
                text_decoder_complete=text_complete,
                text_secret_exact=text_exact,
                direct_decode_error=direct_error,
                text_decode_error=text_error,
            )
        )

        done = sample_index + 1
        if done % progress_every == 0 or done == sample_count:
            partial = summarize(results)
            print(
                f"  {done}/{sample_count}: "
                f"direct={partial['direct_token_id_recovery']['exact']}/{len(results)} "
                f"token-text={partial['text_transport']['token_sequence_exact']}/{len(results)} "
                f"secret-text={partial['text_transport']['secret_exact']}/{len(results)}",
                flush=True,
            )

    summary = summarize(results)
    failures = [
        asdict(sample)
        for sample in results
        if (
            sample.encode_error is not None
            or not sample.direct_secret_exact
            or not sample.token_sequence_roundtrip_exact
            or not sample.text_secret_exact
        )
    ]
    payload = {
        "reference_commit": REFERENCE_COMMIT,
        "config": str(cfg_path.relative_to(REPO_ROOT)),
        "model_id": model_config.model_id,
        "model_revision": model_config.revision,
        "prompt_id": prompt.prompt_id,
        "payload_bits": payload_bits,
        "samples_requested": sample_count,
        "summary": summary,
        "failure_samples": failures,
    }

    print()
    print("direct token-ID recovery:", f"{summary['direct_token_id_recovery']['exact']}/{sample_count}", f"({100 * summary['direct_token_id_recovery']['rate']:.1f}%)")
    print("text token sequence exact:", f"{summary['text_transport']['token_sequence_exact']}/{sample_count}", f"({100 * summary['text_transport']['token_sequence_exact_rate']:.1f}%)")
    print("text secret recovery exact:", f"{summary['text_transport']['secret_exact']}/{sample_count}", f"({100 * summary['text_transport']['secret_exact_rate']:.1f}%)")
    print("token-exact but secret wrong:", summary["text_transport"]["token_exact_but_secret_wrong"])
    print("token-changed but secret recovered:", summary["text_transport"]["token_changed_but_secret_recovered"])
    print("carrier tokens min/mean/max:", f"{summary['carrier_tokens']['min']}/{summary['carrier_tokens']['mean']:.3f}/{summary['carrier_tokens']['max']}")
    print("BPT mean:", f"{summary['bits_per_token']['mean']:.6f}")
    print("termination retries sum/max:", f"{summary['termination_retries']['sum']}/{summary['termination_retries']['max']}")
    print("status:", summary["status"].upper())
    print("note: text-channel recovery is measured separately and is not required to be 100% for PASS")

    output = args.json_output
    if output is not None:
        output = output.expanduser()
        if not output.is_absolute():
            output = REPO_ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print("json:", output.relative_to(REPO_ROOT))

    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
