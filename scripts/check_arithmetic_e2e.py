#!/usr/bin/env python3
"""Short real-LM end-to-end smoke test for normalized Arithmetic Coding.

This is still a Stage-2 integration check, not the final experiment runner.
It exercises the shared path:
LM -> P_reference -> Arithmetic Coding -> ordinary text -> retokenization -> decoder.

Arithmetic Coding differs from Bins/Huffman because it maintains a precision-bit
secret look-ahead window. Therefore ``secret bits read`` can exceed useful
``payload bits``; only the confirmed prefix counts as embedded payload.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vkr_benchmark.config import LocalModelConfig
from vkr_benchmark.distributions import GenerationPolicy, ReferenceDistributionBuilder
from vkr_benchmark.lm import HFCausalLMAdapter
from vkr_benchmark.methods import ArithmeticMethod
from vkr_benchmark.randomness import Shake256SecretSource
from vkr_benchmark.runner import method_environment_from_builder, run_streaming_text_roundtrip


def bits_preview(bits: tuple[int, ...], limit: int = 128) -> str:
    text = "".join(str(bit) for bit in bits[:limit])
    return text + ("..." if len(bits) > limit else "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_config", type=Path)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=50_000)
    parser.add_argument("--carrier-tokens", type=int, default=16)
    parser.add_argument("--secret-id", default="000001")
    parser.add_argument(
        "--prompt",
        default="The history of artificial intelligence began",
    )
    args = parser.parse_args()

    config = LocalModelConfig.from_json(args.model_config)
    print("Arithmetic Coding end-to-end smoke test")
    print("model:", config.model_id)
    print("revision:", config.revision)
    print("precision:", args.precision)
    print("method-internal top_k:", args.top_k)
    print("target carrier tokens:", args.carrier_tokens)
    print("secret_id:", args.secret_id)
    print("Loading local model...")

    lm = HFCausalLMAdapter.from_local_config(config)
    builder = ReferenceDistributionBuilder(
        token_space=lm.token_space,
        policy=GenerationPolicy(),
    )
    environment = method_environment_from_builder(builder)

    result = run_streaming_text_roundtrip(
        lm_adapter=lm,
        reference_builder=builder,
        method=ArithmeticMethod(),
        method_config={"precision": args.precision, "top_k": args.top_k},
        environment=environment,
        prompt_text=args.prompt,
        carrier_tokens=args.carrier_tokens,
        secret_source=Shake256SecretSource(args.secret_id),
        encoder_random_source=None,
        decoder_random_source=None,
    )

    print()
    print("generated carrier tokens:", result.encode.carrier_tokens)
    print("payload bits:", result.encode.payload_bits)
    print("secret bits read (includes look-ahead):", result.encode.secret_bits_read)
    print("look-ahead overhead bits:", result.encode.secret_bits_read - result.encode.payload_bits)
    print(
        "bits per token (diagnostic):",
        f"{result.encode.payload_bits / result.encode.carrier_tokens:.6f}",
    )
    print("step bits consumed:", list(result.encode.step_bits_consumed))
    print("sender token ids:", list(result.encode.carrier_token_ids))
    print("receiver token ids:", list(result.transport.receiver_token_ids))
    print(
        "token_sequence_roundtrip_exact:",
        result.transport.token_sequence_roundtrip_exact,
    )
    print("first_token_mismatch:", result.transport.first_token_mismatch)
    print("token_count_delta:", result.transport.token_count_delta)
    print("payload secret bits:", bits_preview(result.encode.payload_secret_bits))
    print("read secret bits:", bits_preview(result.encode.read_secret_bits))
    print("recovered secret bits:", bits_preview(result.decode.recovered_bits))
    print("expected_length_bits:", result.expected_length_bits)
    print("recovered_length_bits:", result.recovered_length_bits)
    print("recovered_extra_bits:", result.recovered_extra_bits)
    print("length_delta_bits:", result.length_delta_bits)
    print("first_bit_mismatch:", result.first_bit_mismatch)
    print("roundtrip_exact:", result.roundtrip_exact)
    print("decoder complete:", result.decode.finalization.complete)
    print("encoder final interval:", result.encode.finalization.metadata.get("final_interval"))
    print("decoder final interval:", result.decode.finalization.metadata.get("final_interval"))
    print()
    print("stegotext:")
    print(result.transport.text)
    print()
    print("Arithmetic Coding end-to-end pipeline: COMPLETED")


if __name__ == "__main__":
    main()
