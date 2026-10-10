#!/usr/bin/env python3
"""Short real-LM end-to-end smoke test for normalized ADG."""

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
from vkr_benchmark.methods import ADGMethod
from vkr_benchmark.randomness import MethodRandomSource, Shake256SecretSource
from vkr_benchmark.runner import method_environment_from_builder, run_streaming_text_roundtrip


def _preview(bits: tuple[int, ...], limit: int = 128) -> str:
    text = "".join(str(bit) for bit in bits[:limit])
    return text + ("..." if len(bits) > limit else "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_config", type=Path)
    parser.add_argument("--carrier-tokens", type=int, default=16)
    parser.add_argument("--secret-id", default="000001")
    parser.add_argument("--method-seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--prompt",
        default="The history of artificial intelligence began",
    )
    args = parser.parse_args()

    config = LocalModelConfig.from_json(args.model_config)
    print("ADG end-to-end smoke test")
    print("reference commit: b4a7e802a97bc3a3b84d07e63f24627b16c51a10")
    print("model:", config.model_id)
    print("revision:", config.revision)
    print("target carrier tokens:", args.carrier_tokens)
    print("secret_id:", args.secret_id)
    print("method random seed:", args.method_seed)
    print("common generation top-k:", args.top_k)
    print("common generation top-p:", args.top_p)
    print("Loading local model...")

    lm = HFCausalLMAdapter.from_local_config(config)
    builder = ReferenceDistributionBuilder(
        token_space=lm.token_space,
        policy=GenerationPolicy(top_k=args.top_k, top_p=args.top_p),
    )
    environment = method_environment_from_builder(builder)

    result = run_streaming_text_roundtrip(
        lm_adapter=lm,
        reference_builder=builder,
        method=ADGMethod(),
        method_config={},
        environment=environment,
        prompt_text=args.prompt,
        carrier_tokens=args.carrier_tokens,
        secret_source=Shake256SecretSource(args.secret_id),
        encoder_random_source=MethodRandomSource(args.method_seed),
        decoder_random_source=None,
    )

    distortion = result.encode.step_distribution_distortion
    print()
    print("generated carrier tokens:", result.encode.carrier_tokens)
    print("payload bits:", result.encode.payload_bits)
    print("bits per token:", f"{result.encode.payload_bits / result.encode.carrier_tokens:.6f}")
    print("secret bits read:", result.encode.secret_bits_read)
    print("step bits consumed:", list(result.encode.step_bits_consumed))
    if distortion:
        print("q_mode:", distortion[0].q_mode.value)
        print("q_source:", distortion[0].q_source.value)
        print("KL P_ref->Q max:", max(x.kl_ref_to_stego_bits for x in distortion))
        print("KL Q->P_ref max:", max(x.kl_stego_to_ref_bits for x in distortion))
        print("TVD max:", max(x.tvd for x in distortion))
    print()
    print("sender token ids:", list(result.encode.carrier_token_ids))
    print("receiver token ids:", list(result.transport.receiver_token_ids))
    print("token_sequence_roundtrip_exact:", result.transport.token_sequence_roundtrip_exact)
    print("first_token_mismatch:", result.transport.first_token_mismatch)
    print("expected secret bits:", _preview(result.encode.payload_secret_bits))
    print("recovered secret bits:", _preview(result.decode.recovered_bits))
    print("roundtrip_exact:", result.roundtrip_exact)
    print("decoder complete:", result.decode.finalization.complete)
    print()
    print("stegotext:")
    print(result.transport.text)
    print()
    print("ADG smoke test: COMPLETED")


if __name__ == "__main__":
    main()
