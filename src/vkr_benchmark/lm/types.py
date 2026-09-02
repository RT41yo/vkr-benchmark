"""Model-independent LM adapter types."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vkr_benchmark.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class TokenSpace:
    """Relationship between LM output positions and actual tokenizer ids.

    ``tokenizer_vocab_size`` is kept for reporting (``len(tokenizer)``), while
    ``tokenizer_token_ids`` is the authoritative set obtained from
    ``tokenizer.get_vocab().values()``.  The distinction matters because a
    tokenizer is not required to use one contiguous id interval.

    Normalized runs can select only ids that both exist in the model output
    head and have an actual tokenizer mapping. Special ids are additionally
    excluded by the baseline generation policy.
    """

    output_vocab_size: int
    tokenizer_vocab_size: int
    special_token_ids: frozenset[int]
    tokenizer_token_ids: frozenset[int] | None = None

    def __post_init__(self) -> None:
        if self.output_vocab_size <= 0:
            raise ConfigurationError("output_vocab_size must be positive")
        if self.tokenizer_vocab_size <= 0:
            raise ConfigurationError("tokenizer_vocab_size must be positive")
        if self.tokenizer_token_ids is not None and any(
            token_id < 0 for token_id in self.tokenizer_token_ids
        ):
            raise ConfigurationError("tokenizer token ids must be non-negative")
        if any(token_id < 0 for token_id in self.special_token_ids):
            raise ConfigurationError("special token ids must be non-negative")

    @property
    def actual_tokenizer_ids(self) -> frozenset[int]:
        """Authoritative tokenizer-id set, with a contiguous fallback for tests."""

        if self.tokenizer_token_ids is not None:
            return self.tokenizer_token_ids
        return frozenset(range(self.tokenizer_vocab_size))

    @property
    def valid_tokenizer_ids(self) -> frozenset[int]:
        """Tokenizer ids that are also valid positions in the LM output head."""

        return frozenset(
            token_id
            for token_id in self.actual_tokenizer_ids
            if token_id < self.output_vocab_size
        )

    @property
    def shared_vocab_size(self) -> int:
        """Number of LM output ids with an actual tokenizer mapping."""

        return len(self.valid_tokenizer_ids)

    @property
    def output_only_count(self) -> int:
        return self.output_vocab_size - self.shared_vocab_size

    @property
    def output_only_ids(self) -> tuple[int, ...]:
        valid = self.valid_tokenizer_ids
        return tuple(
            token_id
            for token_id in range(self.output_vocab_size)
            if token_id not in valid
        )

    def allowed_mask(self, *, exclude_special_tokens: bool = True) -> np.ndarray:
        """Return an immutable boolean mask over the LM output vocabulary."""

        mask = np.zeros(self.output_vocab_size, dtype=np.bool_)
        valid = self.valid_tokenizer_ids
        if valid:
            mask[np.fromiter(valid, dtype=np.int64, count=len(valid))] = True

        if exclude_special_tokens:
            for token_id in self.special_token_ids:
                if 0 <= token_id < self.output_vocab_size:
                    mask[token_id] = False

        mask.setflags(write=False)
        return mask

    def is_tokenizer_id(self, token_id: int) -> bool:
        return (
            0 <= token_id < self.output_vocab_size
            and token_id in self.actual_tokenizer_ids
        )
