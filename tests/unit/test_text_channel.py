from __future__ import annotations

import numpy as np

from vkr_benchmark.lm import LMAdapter, LMState, TokenSpace
from vkr_benchmark.transport import TextChannel


class _TransportLM(LMAdapter):
    def __init__(self, *, perturb: bool = False) -> None:
        self._perturb = perturb
        self._space = TokenSpace(
            output_vocab_size=8,
            tokenizer_vocab_size=8,
            tokenizer_token_ids=frozenset(range(8)),
            special_token_ids=frozenset({7}),
        )

    @property
    def model_id(self) -> str:
        return "fake"

    @property
    def revision(self) -> str:
        return "test"

    @property
    def token_space(self) -> TokenSpace:
        return self._space

    def encode_prompt(self, text: str) -> tuple[int, ...]:
        del text
        return (0,)

    def encode_text(self, text: str, *, add_special_tokens: bool) -> tuple[int, ...]:
        assert add_special_tokens is False
        ids = tuple(int(part) for part in text.split()) if text else ()
        if self._perturb and len(ids) >= 2:
            return (ids[1], ids[0], *ids[2:])
        return ids

    def decode_tokens(self, token_ids: tuple[int, ...] | list[int]) -> str:
        return " ".join(str(int(x)) for x in token_ids)

    def prefill(self, prompt_token_ids: tuple[int, ...] | list[int]) -> LMState:
        del prompt_token_ids
        return LMState(None, np.zeros(8, dtype=np.float32), 1)

    def next_logits(self, state: LMState, previous_token_id: int | None = None):
        del previous_token_id
        return state.pending_logits, state


def test_text_channel_exact_roundtrip() -> None:
    result = TextChannel(_TransportLM()).transmit((1, 4, 2, 6))

    assert result.text == "1 4 2 6"
    assert result.receiver_token_ids == (1, 4, 2, 6)
    assert result.token_sequence_roundtrip_exact is True
    assert result.first_token_mismatch is None
    assert result.token_count_delta == 0


def test_text_channel_reports_first_token_mismatch() -> None:
    result = TextChannel(_TransportLM(perturb=True)).transmit((1, 4, 2, 6))

    assert result.receiver_token_ids == (4, 1, 2, 6)
    assert result.token_sequence_roundtrip_exact is False
    assert result.first_token_mismatch == 0
