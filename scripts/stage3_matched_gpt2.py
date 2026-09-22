"""GPT-2 Medium LM adapter used only by Stage-3 matched normalized runs.

The adapter feeds the *normalized* benchmark pipeline while matching the author
experiment's model and context prefix.  It does not import author method code
and does not change any normalized method semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vkr_benchmark.lm import LMAdapter, LMState, TokenSpace


@dataclass(frozen=True, slots=True)
class _GPT2State:
    past_key_values: Any


class Stage3GPT2NormalizedAdapter(LMAdapter):
    """Short-horizon cached GPT-2 adapter with author-matched EOT prompt prefix."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        resolved_revision: str,
        device: str,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._revision = str(resolved_revision)
        self._device = str(device)

        output = model.get_output_embeddings()
        if output is None or not hasattr(output, "weight"):
            raise RuntimeError("GPT-2 model does not expose output embeddings")
        output_vocab = int(output.weight.shape[0])
        vocab = tokenizer.get_vocab()
        tokenizer_ids = frozenset(int(v) for v in vocab.values() if int(v) >= 0)
        # The author helper clears eos/bos/unk attributes after loading.  The
        # normalized benchmark must still apply its canonical special-token
        # exclusion, so the GPT-2 EOT id is pinned explicitly here.
        eot_id = int(vocab.get("<|endoftext|>", 50256))
        self._eot_id = eot_id
        self._token_space = TokenSpace(
            output_vocab_size=output_vocab,
            tokenizer_vocab_size=int(len(tokenizer)),
            tokenizer_token_ids=tokenizer_ids,
            special_token_ids=frozenset({eot_id}),
        )

    @property
    def model_id(self) -> str:
        return "gpt2-medium"

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def token_space(self) -> TokenSpace:
        return self._token_space

    @property
    def device(self) -> str:
        return self._device

    def encode_prompt(self, text: str) -> tuple[int, ...]:
        # Exact public author context rule: encode('<|endoftext|>') + encode(raw).
        body = tuple(int(x) for x in self._tokenizer.encode(text, add_special_tokens=False))
        return (self._eot_id, *body)

    def encode_text(self, text: str, *, add_special_tokens: bool) -> tuple[int, ...]:
        return tuple(
            int(x)
            for x in self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        )

    def decode_tokens(self, token_ids: tuple[int, ...] | list[int]) -> str:
        return str(
            self._tokenizer.decode(
                list(token_ids),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )

    def prefill(self, prompt_token_ids: tuple[int, ...] | list[int]) -> LMState:
        if not prompt_token_ids:
            raise ValueError("GPT-2 matched prefill requires a non-empty prompt")
        import torch

        ids = list(int(x) for x in prompt_token_ids)
        # Author mirrors retain the last 1022 context tokens.  Frozen pilot
        # contexts are much shorter, but apply the same rule explicitly.
        ids = ids[-1022:]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self._device)
        with torch.inference_mode():
            out = self._model(input_ids=input_ids, use_cache=True, return_dict=True)
        logits = out.logits[0, -1, :]
        return LMState(
            backend_state=_GPT2State(out.past_key_values),
            pending_logits=logits,
            sequence_length=len(ids),
        )

    def next_logits(
        self,
        state: LMState,
        previous_token_id: int | None = None,
    ) -> tuple[Any, LMState]:
        if previous_token_id is None:
            return state.pending_logits, state
        if state.sequence_length >= 1024:
            raise RuntimeError(
                "Step-3.12 matched run exceeded GPT-2's 1024-token context; "
                "the frozen eight-context protocol is expected to remain below this bound"
            )

        import torch

        backend = state.backend_state
        if not isinstance(backend, _GPT2State):
            raise RuntimeError("unexpected Stage-3 GPT-2 adapter state")
        input_ids = torch.tensor(
            [[int(previous_token_id)]], dtype=torch.long, device=self._device
        )
        with torch.inference_mode():
            out = self._model(
                input_ids=input_ids,
                past_key_values=backend.past_key_values,
                use_cache=True,
                return_dict=True,
            )
        logits = out.logits[0, -1, :]
        return logits, LMState(
            backend_state=_GPT2State(out.past_key_values),
            pending_logits=logits,
            sequence_length=state.sequence_length + 1,
        )
