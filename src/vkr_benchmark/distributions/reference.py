"""Canonical construction of P_reference from raw LM logits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vkr_benchmark.distributions.ordering import deterministic_token_order
from vkr_benchmark.distributions.types import ReferenceDistribution
from vkr_benchmark.errors import ConfigurationError, DistributionError, NumericalError
from vkr_benchmark.lm import TokenSpace


@dataclass(frozen=True, slots=True)
class GenerationPolicy:
    """Common generation transformations applied before a stegomethod.

    Baseline v0.1 is temperature=1, no top-k, top-p=1, special-token exclusion.
    """

    temperature: float = 1.0
    top_k: int | None = None
    top_p: float = 1.0
    exclude_special_tokens: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ConfigurationError("temperature must be finite and > 0")
        if self.top_k is not None and self.top_k <= 0:
            raise ConfigurationError("top_k must be positive or None")
        if not np.isfinite(self.top_p) or not (0.0 < self.top_p <= 1.0):
            raise ConfigurationError("top_p must be in (0, 1]")


class ReferenceDistributionBuilder:
    """Apply the specification-v0.1 canonical logits-to-probabilities pipeline."""

    def __init__(self, *, token_space: TokenSpace, policy: GenerationPolicy) -> None:
        self._token_space = token_space
        self._policy = policy
        self._allowed_mask = token_space.allowed_mask(
            exclude_special_tokens=policy.exclude_special_tokens
        )
        if not bool(np.any(self._allowed_mask)):
            raise DistributionError("allowed token set is empty")

    @property
    def token_space(self) -> TokenSpace:
        return self._token_space

    @property
    def policy(self) -> GenerationPolicy:
        return self._policy

    @property
    def allowed_mask(self) -> np.ndarray:
        return self._allowed_mask

    @property
    def allowed_token_count(self) -> int:
        return int(np.count_nonzero(self._allowed_mask))

    def build(self, raw_logits: Any) -> ReferenceDistribution:
        """Build immutable FP32 P_reference without mutating raw logits.

        The main LM path uses torch FP32 softmax on the same device as the
        logits. A NumPy FP32 fallback exists for synthetic/unit tests.
        Deterministic top-k/top-p support selection is then performed using the
        benchmark-wide (-probability, token_id) order.
        """

        probs = self._softmax_after_common_mask_and_temperature(raw_logits)
        probs = self._apply_top_k(probs)
        probs = self._apply_top_p(probs)
        probs = self._renormalize_fp32(probs)

        # Preserve exact zeros for all forbidden ids after every transformation.
        probs = np.array(probs, dtype=np.float32, copy=True)
        probs[~self._allowed_mask] = np.float32(0.0)
        probs = self._renormalize_fp32(probs)
        return ReferenceDistribution(probs)

    def _softmax_after_common_mask_and_temperature(self, raw_logits: Any) -> np.ndarray:
        # Keep torch optional at import time. The concrete LM environment has it.
        try:
            import torch
        except ImportError:  # pragma: no cover - core-only environments
            torch = None  # type: ignore[assignment]

        if torch is not None and isinstance(raw_logits, torch.Tensor):
            return self._torch_softmax(raw_logits, torch)
        return self._numpy_softmax(raw_logits)

    def _torch_softmax(self, raw_logits: Any, torch: Any) -> np.ndarray:
        if raw_logits.ndim != 1:
            raise DistributionError("raw logits must be one-dimensional")
        if int(raw_logits.numel()) != self._token_space.output_vocab_size:
            raise DistributionError(
                f"raw logits length {int(raw_logits.numel())} does not match "
                f"output vocab {self._token_space.output_vocab_size}"
            )
        if not bool(torch.isfinite(raw_logits).all().item()):
            raise NumericalError("raw LM logits contain NaN or Inf")

        logits = raw_logits.detach().to(dtype=torch.float32).clone()
        allowed = torch.tensor(
            np.array(self._allowed_mask, copy=True),
            dtype=torch.bool,
            device=logits.device,
        )
        logits.masked_fill_(~allowed, float("-inf"))
        logits.div_(float(self._policy.temperature))
        probs = torch.softmax(logits, dim=-1, dtype=torch.float32)

        if not bool(torch.isfinite(probs).all().item()):
            raise NumericalError("softmax produced NaN or Inf")
        return np.array(probs.detach().cpu().numpy(), dtype=np.float32, copy=True)

    def _numpy_softmax(self, raw_logits: Any) -> np.ndarray:
        logits = np.array(raw_logits, dtype=np.float32, copy=True)
        if logits.ndim != 1:
            raise DistributionError("raw logits must be one-dimensional")
        if logits.size != self._token_space.output_vocab_size:
            raise DistributionError(
                f"raw logits length {logits.size} does not match "
                f"output vocab {self._token_space.output_vocab_size}"
            )
        if not np.all(np.isfinite(logits)):
            raise NumericalError("raw LM logits contain NaN or Inf")

        logits[~self._allowed_mask] = np.float32(-np.inf)
        logits = np.asarray(logits / np.float32(self._policy.temperature), dtype=np.float32)

        finite_logits = logits[self._allowed_mask]
        if finite_logits.size == 0:
            raise DistributionError("allowed token set is empty")
        max_logit = np.max(finite_logits)
        shifted = np.asarray(logits - max_logit, dtype=np.float32)
        exp_values = np.zeros_like(shifted, dtype=np.float32)
        exp_values[self._allowed_mask] = np.exp(
            shifted[self._allowed_mask], dtype=np.float32
        )
        denominator = np.sum(exp_values, dtype=np.float32)
        if not np.isfinite(denominator) or denominator <= 0:
            raise NumericalError("softmax normalization is invalid")
        return np.asarray(exp_values / denominator, dtype=np.float32)

    def _apply_top_k(self, probs: np.ndarray) -> np.ndarray:
        top_k = self._policy.top_k
        if top_k is None:
            return probs

        order = deterministic_token_order(probs)
        if top_k >= order.size:
            return probs

        filtered = np.zeros_like(probs, dtype=np.float32)
        keep = order[:top_k]
        filtered[keep] = probs[keep]
        return self._renormalize_fp32(filtered)

    def _apply_top_p(self, probs: np.ndarray) -> np.ndarray:
        top_p = self._policy.top_p
        if top_p >= 1.0:
            return probs

        order = deterministic_token_order(probs)
        if order.size == 0:
            raise DistributionError("cannot apply top-p to empty support")

        cumulative = np.cumsum(probs[order], dtype=np.float64)
        last_index = int(np.searchsorted(cumulative, top_p, side="left"))
        last_index = min(last_index, order.size - 1)
        keep = order[: last_index + 1]

        filtered = np.zeros_like(probs, dtype=np.float32)
        filtered[keep] = probs[keep]
        return self._renormalize_fp32(filtered)

    @staticmethod
    def _renormalize_fp32(probs: np.ndarray) -> np.ndarray:
        values = np.array(probs, dtype=np.float32, copy=True)
        total = np.sum(values, dtype=np.float32)
        if not np.isfinite(total) or total <= 0:
            raise NumericalError("probability normalization total is invalid")
        values = np.asarray(values / total, dtype=np.float32)
        return values
