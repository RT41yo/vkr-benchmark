#!/usr/bin/env python3
"""Short real-LM end-to-end smoke test for normalized Bins.

This script is intentionally not a benchmark experiment runner yet. It checks
that the first adapted stegomethod can traverse the complete infrastructure:
LM -> P_reference -> Bins -> ordinary text -> retokenization -> Bins decoder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vkr_benchmark.config import LocalModelConfig
from vkr_benchmark.distributions import GenerationPolicy, ReferenceDistributionBuilder
from vkr_benchmark.lm import HFCausalLMAdapter
from vkr_benchmark.methods import BinsMethod
from vkr_benchmark.randomness import MethodRandomSource, Shake256SecretSource
from vkr_benchmark.runner import method_environment_from_builder, run_streaming_text_roundtrip


def bits_preview(bits: tuple[int, ...], limit: int = 96) -> str:
    text = "".join(str(bit) for bit in bits[:limit])
    return text + ("..." if len(bits) > limit else "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_config", type=Path)
    parser.add_argument("--block-size", type=int, default=2)
    parser.add_argument("--carrier-tokens", type=int, default=16)
    parser.add_argument("--secret-id", default="000001")
    parser.add_argument("--method-seed", type=int, default=12345)
    parser.add_argument(
        "--prompt",
        default="The history of artificial intelligence began",
    )
    args = parser.parse_args()

    config = LocalModelConfig.from_json(args.model_config)
    print("Bins end-to-end smoke test")
    print("model:", config.model_id)
    print("revision:", config.revision)
    print("block_size:", args.block_size)
    print("target carrier tokens:", args.carrier_tokens)
    print("secret_id:", args.secret_id)
    print("method seed (smoke-test only):", args.method_seed)
    print("Loading local model...")

    lm = HFCausalLMAdapter.from_local_config(config)
    builder = ReferenceDistributionBuilder(
        token_space=lm.token_space,
        policy=GenerationPolicy(),
    )
    environment = method_environment_from_builder(builder)

    # Encoder and decoder must reconstruct the same fixed keyed partition, so
    # they receive two independent RNG objects initialized from the same seed.
    result = run_streaming_text_roundtrip(
        lm_adapter=lm,
        reference_builder=builder,
        method=BinsMethod(),
        method_config={"block_size": args.block_size},
        environment=environment,
        prompt_text=args.prompt,
        carrier_tokens=args.carrier_tokens,
        secret_source=Shake256SecretSource(args.secret_id),
        encoder_random_source=MethodRandomSource(args.method_seed),
        decoder_random_source=MethodRandomSource(args.method_seed),
    )

    print()
    print("generated carrier tokens:", result.encode.carrier_tokens)
    print("payload bits:", result.encode.payload_bits)
    print("bits per token (diagnostic):", f"{result.encode.payload_bits / result.encode.carrier_tokens:.6f}")
    print("sender token ids:", list(result.encode.carrier_token_ids))
    print("receiver token ids:", list(result.transport.receiver_token_ids))
    print("token_sequence_roundtrip_exact:", result.transport.token_sequence_roundtrip_exact)
    print("first_token_mismatch:", result.transport.first_token_mismatch)
    print("token_count_delta:", result.transport.token_count_delta)
    print("expected secret bits:", bits_preview(result.encode.consumed_secret_bits))
    print("recovered secret bits:", bits_preview(result.decode.recovered_bits))
    print("expected_length_bits:", result.expected_length_bits)
    print("recovered_length_bits:", result.recovered_length_bits)
    print("recovered_extra_bits:", result.recovered_extra_bits)
    print("length_delta_bits:", result.length_delta_bits)
    print("first_bit_mismatch:", result.first_bit_mismatch)
    print("roundtrip_exact:", result.roundtrip_exact)
    print("decoder complete:", result.decode.finalization.complete)
    print()
    print("stegotext:")
    print(result.transport.text)
    print()
    print("Bins end-to-end pipeline: COMPLETED")


if __name__ == "__main__":
    main()
