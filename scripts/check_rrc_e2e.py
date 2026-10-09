#!/usr/bin/env python3
"""Short real-LM smoke test for normalized RRC.

The script reports algorithmic recovery from the sender token IDs separately
from reliability after ordinary-text transport.  This distinction is important
for RRC because retokenization ambiguity can break extraction even when the
range-coding algorithm itself is correct.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vkr_benchmark.config import LocalModelConfig
from vkr_benchmark.distributions import GenerationPolicy, ReferenceDistributionBuilder
from vkr_benchmark.lm import HFCausalLMAdapter
from vkr_benchmark.metrics import compute_distribution_distortion_metrics
from vkr_benchmark.methods import RRCMethod
from vkr_benchmark.randomness import MethodRandomSource, Shake256SecretSource
from vkr_benchmark.runner import (
    decode_streaming_tokens,
    encode_fixed_payload_bits,
    method_environment_from_builder,
)
from vkr_benchmark.transport import TextChannel


def _preview(bits: tuple[int, ...], limit: int = 128) -> str:
    text = "".join(str(bit) for bit in bits[:limit])
    return text + ("..." if len(bits) > limit else "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_config", type=Path)
    parser.add_argument("--payload-bits", type=int, default=128)
    parser.add_argument("--max-carrier-tokens", type=int, default=256)
    parser.add_argument("--secret-id", default="000001")
    parser.add_argument("--key", type=int, default=42)
    parser.add_argument("--precision-margin", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--prompt",
        default="The history of artificial intelligence began",
    )
    args = parser.parse_args()

    config = LocalModelConfig.from_json(args.model_config)
    print("RRC end-to-end smoke test")
    print("model:", config.model_id)
    print("revision:", config.revision)
    print("target payload bits:", args.payload_bits)
    print("max carrier tokens:", args.max_carrier_tokens)
    print("secret_id:", args.secret_id)
    print("shared PRNG key:", args.key)
    print("precision margin:", args.precision_margin)
    print("common generation top-k:", args.top_k)
    print("common generation top-p:", args.top_p)
    print("Loading local model...")

    lm = HFCausalLMAdapter.from_local_config(config)
    builder = ReferenceDistributionBuilder(
        token_space=lm.token_space,
        policy=GenerationPolicy(top_k=args.top_k, top_p=args.top_p),
    )
    environment = method_environment_from_builder(builder)
    method = RRCMethod()
    method_config = {
        "precision_margin": args.precision_margin,
        "verify_termination": True,
    }

    encoded = encode_fixed_payload_bits(
        lm_adapter=lm,
        reference_builder=builder,
        method=method,
        method_config=method_config,
        environment=environment,
        prompt_text=args.prompt,
        target_payload_bits=args.payload_bits,
        max_carrier_tokens=args.max_carrier_tokens,
        secret_source=Shake256SecretSource(args.secret_id),
        method_random_source=MethodRandomSource(args.key),
        key=args.key,
    )

    direct = decode_streaming_tokens(
        lm_adapter=lm,
        reference_builder=builder,
        method=method,
        method_config=method_config,
        environment=environment,
        prompt_text=args.prompt,
        observed_token_ids=encoded.carrier_token_ids,
        method_random_source=MethodRandomSource(args.key),
        expected_payload_bits=encoded.payload_bits,
        key=args.key,
    )
    direct_exact = (
        direct.finalization.complete
        and direct.recovered_bits == encoded.payload_secret_bits
    )

    transport = TextChannel(lm).transmit(encoded.carrier_token_ids)
    text_decode = decode_streaming_tokens(
        lm_adapter=lm,
        reference_builder=builder,
        method=method,
        method_config=method_config,
        environment=environment,
        prompt_text=args.prompt,
        observed_token_ids=transport.receiver_token_ids,
        method_random_source=MethodRandomSource(args.key),
        expected_payload_bits=encoded.payload_bits,
        key=args.key,
    )
    text_exact = (
        transport.token_sequence_roundtrip_exact
        and text_decode.finalization.complete
        and text_decode.recovered_bits == encoded.payload_secret_bits
    )

    distortion = compute_distribution_distortion_metrics(
        encoded.step_distribution_distortion
    )

    print()
    print("generated carrier tokens:", encoded.carrier_tokens)
    print("payload bits:", encoded.payload_bits)
    print("bits per token:", f"{encoded.payload_bits / encoded.carrier_tokens:.6f}")
    print("secret bits read:", encoded.secret_bits_read)
    print("step bits consumed:", list(encoded.step_bits_consumed))
    print("precision digits:", encoded.finalization.metadata.get("precision_digits"))
    print("termination retries:", encoded.finalization.metadata.get("termination_retries"))
    print("q_mode:", distortion.q_mode.value)
    print("q_source:", encoded.step_distribution_distortion[0].q_source.value)
    print("KL P_ref->Q max:", distortion.kl_ref_to_stego_max_bits)
    print("KL Q->P_ref max:", distortion.kl_stego_to_ref_max_bits)
    print("TVD max:", distortion.tvd_max)
    print()
    print("direct token-ID recovery exact:", direct_exact)
    print("direct decoder complete:", direct.finalization.complete)
    print("expected secret bits:", _preview(encoded.payload_secret_bits))
    print("direct recovered bits:", _preview(direct.recovered_bits))
    print()
    print("token_sequence_roundtrip_exact:", transport.token_sequence_roundtrip_exact)
    print("first_token_mismatch:", transport.first_token_mismatch)
    print("token_count_delta:", transport.token_count_delta)
    print("text-channel decoder complete:", text_decode.finalization.complete)
    print("text-channel secret recovery exact:", text_exact)
    print("text-channel recovered bits:", _preview(text_decode.recovered_bits))
    print()
    print("stegotext:")
    print(transport.text)
    print()
    print("RRC smoke test: COMPLETED")


if __name__ == "__main__":
    main()
