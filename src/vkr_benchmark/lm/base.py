"""Common interface for causal language-model adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from vkr_benchmark.lm.types import TokenSpace


@dataclass(frozen=True, slots=True)
class LMState:
    """Opaque state of one autoregressive generation path.

    ``backend_state`` is intentionally backend-specific (for Hugging Face it is
    the KV cache). ``pending_logits`` are the logits for the next token after
    the already processed prefix.
    """

    backend_state: Any
    pending_logits: Any
    sequence_length: int

    def __post_init__(self) -> None:
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")


class LMAdapter(ABC):
    """Model-independent interface fixed by benchmark specification v0.1."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Logical model id recorded in benchmark configurations."""

    @property
    @abstractmethod
    def revision(self) -> str:
        """Pinned model revision recorded for reproducibility."""

    @property
    @abstractmethod
    def token_space(self) -> TokenSpace:
        """Output/tokenizer vocabulary relation and special-token metadata."""

    @abstractmethod
    def encode_prompt(self, text: str) -> tuple[int, ...]:
        """Tokenize an exact prompt using the adapter's explicit prompt policy."""

    @abstractmethod
    def encode_text(
        self,
        text: str,
        *,
        add_special_tokens: bool,
    ) -> tuple[int, ...]:
        """Tokenize arbitrary text with an explicit special-token policy.

        This is intentionally separate from ``encode_prompt``: prompt
        tokenization may add model-specific BOS tokens, while text received
        from the transport channel must be retokenized with
        ``add_special_tokens=False``.
        """

    @abstractmethod
    def decode_tokens(self, token_ids: tuple[int, ...] | list[int]) -> str:
        """Decode carrier ids without skipping special tokens or cleanup."""

    @abstractmethod
    def prefill(self, prompt_token_ids: tuple[int, ...] | list[int]) -> LMState:
        """Process the prompt once and create the initial KV-cache state."""

    @abstractmethod
    def next_logits(
        self,
        state: LMState,
        previous_token_id: int | None = None,
    ) -> tuple[Any, LMState]:
        """Return next-token logits, optionally advancing state by one token.

        ``previous_token_id=None`` returns the logits already produced by
        ``prefill`` (or the latest advance) without an additional LM forward.
        Supplying a token performs exactly one cached autoregressive step.
        """
