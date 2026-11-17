"""Canonical ordinary generation from P_reference (no stegomethod, no secret).

The FP64 CDF is only a numerically stable *selection* mechanism. It does not
change the shared FP32 P_reference or the LM's cached inference path.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vkr_benchmark.distributions import ReferenceDistributionBuilder
from vkr_benchmark.distributions.types import ReferenceDistribution
from vkr_benchmark.errors import ConfigurationError, DistributionError
from vkr_benchmark.lm import LMAdapter
from vkr_benchmark.randomness import RandomSource
from vkr_benchmark.transport import TextChannel


def sample_reference_token(reference: ReferenceDistribution, *, random_source: RandomSource) -> int:
    """Inverse CDF over ascending token IDs; right-open intervals [a,b).

    A floating point FP32 probability array may sum to approximately (rather
    than exactly) 1. Scaling u by its actual FP64 sum avoids both an undefined
    right-tail event and a biased 'clip to last token' workaround.
    """
    probabilities = reference.probabilities
    total = float(np.sum(probabilities, dtype=np.float64))
    if not (np.isfinite(total) and total > 0):
        raise DistributionError("invalid P_reference total for control sampling")
    u = float(random_source.random())
    if not np.isfinite(u) or not 0.0 <= u < 1.0:
        raise DistributionError("control random source must return u in [0,1)")
    cdf = np.cumsum(probabilities, dtype=np.float64)
    idx = int(np.searchsorted(cdf, u * total, side="right"))
    # Guard only against a rounding error in floating-point search near 1.0;
    # never silently choose a zero-probability token.
    if idx >= len(probabilities):
        idx = int(np.flatnonzero(probabilities > 0)[-1])
    if probabilities[idx] <= 0:
        raise DistributionError("control sampler selected a zero-probability token")
    return idx


@dataclass(frozen=True, slots=True)
class CoverGeneration:
    prompt_token_ids: tuple[int, ...]
    sender_token_ids: tuple[int, ...]
    text: str
    receiver_token_ids: tuple[int, ...]
    token_sequence_roundtrip_exact: bool
    first_token_mismatch: int | None


def generate_cover(
    *,
    lm_adapter: LMAdapter,
    reference_builder: ReferenceDistributionBuilder,
    prompt_text: str,
    carrier_tokens: int,
    random_source: RandomSource,
) -> CoverGeneration:
    """Sample exactly N carrier tokens through the existing cached LM adapter."""
    if carrier_tokens <= 0:
        raise ConfigurationError("carrier_tokens must be positive")
    if not prompt_text:
        raise ConfigurationError("cover prompt must be nonempty")
    if reference_builder.token_space != lm_adapter.token_space:
        raise ConfigurationError("cover builder token space disagrees with LM")
    prompt_ids = lm_adapter.encode_prompt(prompt_text)
    state = lm_adapter.prefill(prompt_ids)
    selected: list[int] = []
    previous: int | None = None
    for _ in range(carrier_tokens):
        logits, state = lm_adapter.next_logits(state, previous_token_id=previous)
        reference = reference_builder.build(logits)
        previous = sample_reference_token(reference, random_source=random_source)
        selected.append(previous)
    transport = TextChannel(lm_adapter).transmit(selected)
    return CoverGeneration(
        prompt_token_ids=tuple(prompt_ids),
        sender_token_ids=tuple(selected),
        text=transport.text,
        receiver_token_ids=transport.receiver_token_ids,
        token_sequence_roundtrip_exact=transport.token_sequence_roundtrip_exact,
        first_token_mismatch=transport.first_token_mismatch,
    )
