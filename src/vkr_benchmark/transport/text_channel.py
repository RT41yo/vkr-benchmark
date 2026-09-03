"""Ordinary-text transport required by benchmark reliability checks."""

from __future__ import annotations

from dataclasses import dataclass

from vkr_benchmark.lm import LMAdapter


def _first_mismatch(a: tuple[int, ...], b: tuple[int, ...]) -> int | None:
    for index, (left, right) in enumerate(zip(a, b, strict=False)):
        if left != right:
            return index
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


@dataclass(frozen=True, slots=True)
class TextTransportResult:
    """Result of sending carrier token ids through ordinary text only."""

    sender_token_ids: tuple[int, ...]
    text: str
    receiver_token_ids: tuple[int, ...]
    token_sequence_roundtrip_exact: bool
    first_token_mismatch: int | None

    @property
    def token_count_delta(self) -> int:
        return len(self.receiver_token_ids) - len(self.sender_token_ids)


class TextChannel:
    """Tokenizer-backed ordinary-text channel.

    The sender decodes carrier token ids to text. Only that text is considered
    transmitted. The receiver retokenizes it without injecting prompt special
    tokens. The channel deliberately performs no BPE repair or method-specific
    correction.
    """

    def __init__(self, lm_adapter: LMAdapter) -> None:
        self._lm_adapter = lm_adapter

    def transmit(self, token_ids: tuple[int, ...] | list[int]) -> TextTransportResult:
        sender = tuple(int(token_id) for token_id in token_ids)
        text = self._lm_adapter.decode_tokens(sender)
        receiver = self._lm_adapter.encode_text(
            text,
            add_special_tokens=False,
        )
        mismatch = _first_mismatch(sender, receiver)
        return TextTransportResult(
            sender_token_ids=sender,
            text=text,
            receiver_token_ids=receiver,
            token_sequence_roundtrip_exact=mismatch is None,
            first_token_mismatch=mismatch,
        )
