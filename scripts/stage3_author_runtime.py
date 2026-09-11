"""Shared runtime compatibility helpers for Stage-3 author-reference smokes.

This module is deliberately outside the normalized benchmark implementation.
It bridges historical GPT-2 API conventions used by the pinned Harvard NLP
reference to Transformers 4.52 without changing reference algorithm files,
logits, token probabilities, or cache tensor values.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator
from unittest.mock import patch

COMPATIBILITY_PROFILE = "hf_4_52_legacy_api"


class LegacyCausalLMAdapter:
    """Translate between the historical Harvard GPT-2 cache API and HF 4.52."""

    def __init__(self, model: Any) -> None:
        self._model = model

    @staticmethod
    def legacy_cache_to_modern(past: Any) -> Any:
        if past is None:
            return None

        modern_layers = []
        for layer_index, layer in enumerate(past):
            if isinstance(layer, (tuple, list)):
                if len(layer) < 2:
                    raise RuntimeError(
                        f"cache layer {layer_index} does not contain key/value tensors"
                    )
                modern_layers.append((layer[0], layer[1]))
                continue

            shape = getattr(layer, "shape", None)
            if shape is None or len(shape) != 5 or int(shape[0]) != 2:
                raise RuntimeError(
                    "historical GPT-2 cache layer must have shape "
                    "[2, batch, heads, seq, head_dim]"
                )
            modern_layers.append((layer[0], layer[1]))

        return tuple(modern_layers)

    @staticmethod
    def modern_cache_to_legacy(past: Any) -> Any:
        if past is None:
            return None
        if hasattr(past, "to_legacy_cache"):
            past = past.to_legacy_cache()

        import torch

        legacy_layers = []
        for layer_index, layer in enumerate(past):
            if not isinstance(layer, (tuple, list)) or len(layer) < 2:
                raise RuntimeError(
                    f"modern cache layer {layer_index} is not a key/value pair"
                )
            key, value = layer[0], layer[1]
            if tuple(key.shape) != tuple(value.shape):
                raise RuntimeError(
                    f"cache key/value shapes differ at layer {layer_index}: "
                    f"{tuple(key.shape)} vs {tuple(value.shape)}"
                )
            legacy_layers.append(torch.stack((key, value), dim=0))

        return tuple(legacy_layers)

    def __call__(self, input_ids: Any, past: Any = None, **kwargs: Any) -> tuple[Any, Any]:
        if "past_key_values" in kwargs:
            raise TypeError("pass legacy past= only through this adapter")

        modern_past = self.legacy_cache_to_modern(past)
        outputs = self._model(
            input_ids,
            past_key_values=modern_past,
            use_cache=True,
            return_dict=False,
            **kwargs,
        )
        if not isinstance(outputs, (tuple, list)) or len(outputs) < 2:
            raise RuntimeError("modern model did not return logits and cache")

        return outputs[0], self.modern_cache_to_legacy(outputs[1])

    def parameters(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.parameters(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


@contextmanager
def force_reference_slow_tokenizer(utils_module: Any) -> Iterator[None]:
    """Force the slow tokenizer exposing ``encoder``/``decoder`` mappings."""

    original = utils_module.AutoTokenizer.from_pretrained

    def _from_pretrained(*args: Any, **kwargs: Any) -> Any:
        kwargs = dict(kwargs)
        kwargs["use_fast"] = False
        return original(*args, **kwargs)

    with patch.object(utils_module.AutoTokenizer, "from_pretrained", side_effect=_from_pretrained):
        yield
