#!/usr/bin/env python3
"""Short real-LM end-to-end smoke test for normalized DAIRstega."""

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
from vkr_benchmark.methods import DAIRMethod
from vkr_benchmark.randomness import Shake256SecretSource
from vkr_benchmark.runner import method_environment_from_builder, run_streaming_text_roundtrip


def _preview(bits: tuple[int, ...], limit: int = 128) -> str:
    text = "".join(str(bit) for bit in bits[:limit])
    return text + ("..." if len(bits) > limit else "")


def _finite_max(values: list[float]) -> str:
    finite = [value for value in values if value != float("inf")]
    return "none" if not finite else str(max(finite))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_config", type=Path)
    parser.add_argument("--carrier-tokens", type=int, default=16)
    parser.add_argument("--secret-id", default="000001")
    parser.add_argument("--alpha", type=int, default=8)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--method-top-k", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--prompt",
        default="The history of artificial intelligence began",
    )
    args = parser.parse_args()

    config = LocalModelConfig.from_json(args.model_config)
    print("DAIRstega end-to-end smoke test")
    print("reference commit: 8d85edf98d48c3efa827a125b6d4e90f88141ea2")
    print("model:", config.model_id)
    print("revision:", config.revision)
    print("target carrier tokens:", args.carrier_tokens)
    print("secret_id:", args.secret_id)
    print("alpha:", args.alpha)
    print("beta:", args.beta)
    print("method-local top-k:", args.method_top_k)
    print("common generation top-k:", args.top_k)
    print("common generation top-p:", args.top_p)
    print("Loading local model...")

    lm = HFCausalLMAdapter.from_local_config(config)
    builder = ReferenceDistributionBuilder(
        token_space=lm.token_space,
        policy=GenerationPolicy(top_k=args.top_k, top_p=args.top_p),
    )
    environment = method_environment_from_builder(builder)
    method_config = {
        "alpha": args.alpha,
        "beta": args.beta,
        "top_k": args.method_top_k,
    }

    result = run_streaming_text_roundtrip(
        lm_adapter=lm,
        reference_builder=builder,
        method=DAIRMethod(),
        method_config=method_config,
        environment=environment,
        prompt_text=args.prompt,
        carrier_tokens=args.carrier_tokens,
        secret_source=Shake256SecretSource(args.secret_id),
        encoder_random_source=None,
        decoder_random_source=None,
    )

    distortion = result.encode.step_distribution_distortion
    print()
    print("generated carrier tokens:", result.encode.carrier_tokens)
    print("payload bits:", result.encode.payload_bits)
    print("bits per token:", f"{result.encode.payload_bits / result.encode.carrier_tokens:.6f}")
    print("secret bits read (includes alpha-bit look-ahead):", result.encode.secret_bits_read)
    print("step bits consumed:", list(result.encode.step_bits_consumed))
    if distortion:
        print("q_mode:", distortion[0].q_mode.value)
        print("q_source:", distortion[0].q_source.value)
        forward = [x.kl_ref_to_stego_bits for x in distortion]
        reverse = [x.kl_stego_to_ref_bits for x in distortion]
        print("KL P_ref->Q infinite steps:", sum(x == float("inf") for x in forward))
        print("KL P_ref->Q finite max:", _finite_max(forward))
        print("KL Q->P_ref max:", max(reverse))
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
    print("DAIRstega smoke test: COMPLETED")


if __name__ == "__main__":
    main()
