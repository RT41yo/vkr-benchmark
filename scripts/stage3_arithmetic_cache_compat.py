"""Long-context cache compatibility for Stage-3 author Arithmetic runs.

The pinned Harvard repository's 2025 tuple-cache branch in ``utils.limit_past``
keeps the historical slice ``[:, :, :, -1022:]``.  That slice was correct for
the historical stacked cache layout ``[2, batch, heads, seq, head_dim]`` but,
for modern GPT-2 tuple key/value tensors ``[batch, heads, seq, head_dim]``, it
trims ``head_dim`` instead of ``seq``.  Short runs are unaffected; sufficiently
long runs can therefore exceed GPT-2's learned 1024-position embedding table.

This module is a Stage-3 compatibility shim only.  It does not alter token
probabilities, Arithmetic interval math, payload accounting, or the normalized
benchmark implementation.  It restores the intended 1022-token sliding-cache
behavior on modern tuple/DynamicCache tensors.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


class ModernGPT2SequenceCacheLimiter:
    """Trim modern GPT-2 KV caches along the sequence axis.

    ``DynamicCache.from_legacy_cache`` accepts the returned tuple-of-tuples, so
    callers can preserve the exact public Arithmetic forward path while keeping
    the cache bounded to the intended historical 1022 tokens.
    """

    def __init__(self, max_cache_tokens: int = 1022) -> None:
        self.max_cache_tokens = int(max_cache_tokens)
        self.trim_events = 0
        self.max_seen_sequence_tokens = 0

    def __call__(self, past: Any) -> Any:
        if past is None:
            return None
        layers = past.to_legacy_cache() if hasattr(past, "to_legacy_cache") else tuple(past)
        trimmed_layers: list[tuple[Any, Any]] = []
        did_trim = False
        call_max = 0

        for layer_index, layer in enumerate(layers):
            if not isinstance(layer, (tuple, list)) or len(layer) < 2:
                raise RuntimeError(
                    f"modern Arithmetic cache layer {layer_index} is not a key/value pair"
                )
            key, value = layer[0], layer[1]
            if getattr(key, "ndim", None) != 4 or getattr(value, "ndim", None) != 4:
                raise RuntimeError(
                    "modern GPT-2 key/value cache tensors must have layout "
                    "[batch, heads, seq, head_dim]"
                )
            if tuple(key.shape) != tuple(value.shape):
                raise RuntimeError(
                    f"cache key/value shapes differ at layer {layer_index}: "
                    f"{tuple(key.shape)} vs {tuple(value.shape)}"
                )
            seq_len = int(key.shape[-2])
            call_max = max(call_max, seq_len)
            if seq_len > self.max_cache_tokens:
                key = key[:, :, -self.max_cache_tokens :, :]
                value = value[:, :, -self.max_cache_tokens :, :]
                did_trim = True
            trimmed_layers.append((key, value))

        self.max_seen_sequence_tokens = max(self.max_seen_sequence_tokens, call_max)
        if did_trim:
            self.trim_events += 1
        return tuple(trimmed_layers)


class UtilsLimitPastProxy:
    """Delegate all author ``utils`` attributes except ``limit_past``."""

    def __init__(self, base_utils: Any, limiter: ModernGPT2SequenceCacheLimiter) -> None:
        self._base_utils = base_utils
        self.limit_past = limiter

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_utils, name)


@contextmanager
def patch_arithmetic_limit_past(
    arithmetic_module: Any, limiter: ModernGPT2SequenceCacheLimiter
) -> Iterator[None]:
    """Temporarily repair only the imported ``limit_past`` used by decoder."""

    original = arithmetic_module.limit_past
    arithmetic_module.limit_past = limiter
    try:
        yield
    finally:
        arithmetic_module.limit_past = original
