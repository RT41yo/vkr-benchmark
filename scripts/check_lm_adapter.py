#!/usr/bin/env python3
"""GPU smoke test for LMAdapter + canonical P_reference.

Run explicitly on the Manjaro benchmark workstation; this script does not
participate in ordinary unit tests and never downloads model files.

Besides the ordinary adapter smoke test, this version reproduces the Stage-1
``model_probe.py`` DynamicCache call pattern independently on the same loaded
model and compares raw logits before and after one cached token step.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import DynamicCache

from vkr_benchmark.config import LocalModelConfig
from vkr_benchmark.distributions import GenerationPolicy, ReferenceDistributionBuilder
from vkr_benchmark.lm import HFCausalLMAdapter


def _probe_style_prefill(model, prompt_ids: tuple[int, ...], device: str):
    """Independent copy of the Stage-1 ``model_probe.py`` prefill path."""

    ids = torch.tensor([list(prompt_ids)], device=device, dtype=torch.long)
    mask = torch.ones_like(ids, dtype=torch.long)
    cache = DynamicCache()
    pos = torch.arange(ids.shape[1], device=ids.device, dtype=torch.long)
    with torch.inference_mode():
        out = model(
            input_ids=ids,
            attention_mask=mask,
            past_key_values=cache,
            cache_position=pos,
            use_cache=True,
            logits_to_keep=1,
        )
    return out.logits[0, -1, :], out.past_key_values, mask, pos[-1:] + 1


def _probe_style_step(model, token_id: int, cache, mask, pos, device: str):
    """Independent copy of one Stage-1 cached autoregressive step."""

    token = torch.tensor([[token_id]], device=device, dtype=torch.long)
    mask = torch.cat([mask, mask.new_ones((1, 1))], dim=-1)
    with torch.inference_mode():
        out = model(
            input_ids=token,
            attention_mask=mask,
            past_key_values=cache,
            cache_position=pos,
            use_cache=True,
            logits_to_keep=1,
        )
    return out.logits[0, -1, :], out.past_key_values, mask, pos[-1:] + 1


def _compare_logits(label: str, adapter_logits: torch.Tensor, probe_logits: torch.Tensor) -> None:
    exact = bool(torch.equal(adapter_logits, probe_logits))
    max_diff = float((adapter_logits.float() - probe_logits.float()).abs().max().item())
    print(f"{label} exact: {exact}")
    print(f"{label} max abs diff: {max_diff:.9g}")
    if not exact:
        raise RuntimeError(
            f"{label} differs from the Stage-1 model_probe-style computation path"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_config", type=Path)
    parser.add_argument(
        "--prompt",
        default="The history of artificial intelligence began",
    )
    args = parser.parse_args()

    config = LocalModelConfig.from_json(args.model_config)
    print(f"Loading: {config.model_id}")
    print(f"Revision: {config.revision}")
    print(f"Local path: {config.local_path}")
    print(f"dtype: {config.dtype}; device: {config.device}")
    print(f"prompt_add_special_tokens: {config.prompt_add_special_tokens}")

    adapter = HFCausalLMAdapter.from_local_config(config)
    space = adapter.token_space
    print(f"output vocab: {space.output_vocab_size}")
    print(f"tokenizer vocab: {space.tokenizer_vocab_size}")
    print(f"tokenizer ids in output: {space.shared_vocab_size}")
    print(f"output-only ids: {space.output_only_count}")
    print(f"special ids: {len(space.special_token_ids)}")

    prompt_ids = adapter.encode_prompt(args.prompt)
    print(f"prompt tokens: {len(prompt_ids)}")
    print(f"prompt token ids: {list(prompt_ids)}")

    # Adapter path.
    state = adapter.prefill(prompt_ids)
    raw_logits, state = adapter.next_logits(state)
    print(f"raw logits dtype: {raw_logits.dtype}")
    print(f"raw logits device: {raw_logits.device}")
    if raw_logits.device.type != torch.device(config.device).type:
        raise RuntimeError(
            f"raw logits are on {raw_logits.device}, expected device type {config.device}"
        )

    # Independent Stage-1 model_probe-style path on the same loaded model. This
    # intentionally accesses the backend only inside this diagnostic script;
    # benchmark methods never receive the model object.
    probe_logits, probe_cache, probe_mask, probe_pos = _probe_style_prefill(
        adapter._model,  # noqa: SLF001 - intentional diagnostics-only access
        prompt_ids,
        config.device,
    )
    _compare_logits("prefill logits", raw_logits, probe_logits)

    builder = ReferenceDistributionBuilder(
        token_space=space,
        policy=GenerationPolicy(
            temperature=1.0,
            top_k=None,
            top_p=1.0,
            exclude_special_tokens=True,
        ),
    )
    reference = builder.build(raw_logits)

    total = float(np.sum(reference.probabilities, dtype=np.float64))
    forbidden_mass = float(
        np.sum(reference.probabilities[~builder.allowed_mask], dtype=np.float64)
    )
    first_token = int(reference.token_order[0])
    print(f"P_reference sum: {total:.9f}")
    print(f"forbidden mass: {forbidden_mass:.9f}")
    print(f"support size: {reference.support_size}")
    print(f"top token id: {first_token}")

    # Advance exactly one step through the adapter KV-cache path.
    raw_logits_2, state_2 = adapter.next_logits(state, first_token)
    reference_2 = builder.build(raw_logits_2)

    # Advance the independently reproduced Stage-1 path by the same token.
    probe_logits_2, _, _, _ = _probe_style_step(
        adapter._model,  # noqa: SLF001 - intentional diagnostics-only access
        first_token,
        probe_cache,
        probe_mask,
        probe_pos,
        config.device,
    )
    _compare_logits("cached-step logits", raw_logits_2, probe_logits_2)

    print(f"state length after one generated token: {state_2.sequence_length}")
    print(
        "next P_reference sum: "
        f"{float(np.sum(reference_2.probabilities, dtype=np.float64)):.9f}"
    )
    print("LM adapter smoke test: OK")


if __name__ == "__main__":
    main()
