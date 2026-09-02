"""Hugging Face causal-LM adapter for local Llama/Qwen checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vkr_benchmark.config import LocalModelConfig
from vkr_benchmark.errors import LMError, ModelLoadError
from vkr_benchmark.lm.base import LMAdapter, LMState
from vkr_benchmark.lm.types import TokenSpace


@dataclass(frozen=True, slots=True)
class _HFBackendState:
    """Hugging Face state required to reproduce the Stage-1 cache path."""

    cache: Any
    attention_mask: Any
    next_cache_position: Any


class HFCausalLMAdapter(LMAdapter):
    """One cached inference path shared by encoder and decoder.

    The concrete Hugging Face calls intentionally mirror ``lm_probe`` v0.2:
    an explicit ``DynamicCache``, attention mask, cache position,
    ``use_cache=True`` and ``logits_to_keep=1`` are used for both prefill and
    every autoregressive step.
    """

    def __init__(self, *, config: LocalModelConfig, model: Any, tokenizer: Any) -> None:
        self._config = config
        self._model = model
        self._tokenizer = tokenizer

        output_embeddings = model.get_output_embeddings()
        if output_embeddings is None or not hasattr(output_embeddings, "weight"):
            raise ModelLoadError("model does not expose output embedding weights")

        output_vocab_size = int(output_embeddings.weight.shape[0])
        tokenizer_vocab_size = int(len(tokenizer))

        try:
            tokenizer_vocab = tokenizer.get_vocab()
            tokenizer_token_ids = frozenset(
                int(token_id)
                for token_id in tokenizer_vocab.values()
                if int(token_id) >= 0
            )
        except Exception as exc:  # noqa: BLE001 - normalize backend metadata errors
            raise ModelLoadError(f"cannot inspect tokenizer vocabulary: {exc}") from exc

        special_ids = frozenset(
            int(token_id)
            for token_id in getattr(tokenizer, "all_special_ids", [])
            if token_id is not None
        )
        self._token_space = TokenSpace(
            output_vocab_size=output_vocab_size,
            tokenizer_vocab_size=tokenizer_vocab_size,
            tokenizer_token_ids=tokenizer_token_ids,
            special_token_ids=special_ids,
        )

    @classmethod
    def from_local_config(cls, config: LocalModelConfig) -> "HFCausalLMAdapter":
        """Load a pinned local model without any network access."""

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised in LM environment
            raise ModelLoadError(
                "LM dependencies are missing; install the benchmark LM environment"
            ) from exc

        if not config.local_path.exists():
            raise ModelLoadError(f"local model path does not exist: {config.local_path}")

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map[config.dtype]

        if config.device.startswith("cuda") and not torch.cuda.is_available():
            raise ModelLoadError(
                f"model config requests {config.device}, but CUDA is unavailable"
            )

        tokenizer_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": False,
        }
        model_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": False,
            "torch_dtype": torch_dtype,
        }
        if config.attn_implementation is not None:
            model_kwargs["attn_implementation"] = config.attn_implementation

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(config.local_path), **tokenizer_kwargs
            )
            model = AutoModelForCausalLM.from_pretrained(
                str(config.local_path), **model_kwargs
            )
            model.to(config.device)
            model.eval()
        except Exception as exc:  # noqa: BLE001 - normalize backend load errors
            raise ModelLoadError(
                f"failed to load local model {config.model_id} from {config.local_path}: {exc}"
            ) from exc

        return cls(config=config, model=model, tokenizer=tokenizer)

    @property
    def model_id(self) -> str:
        return self._config.model_id

    @property
    def revision(self) -> str:
        return self._config.revision

    @property
    def token_space(self) -> TokenSpace:
        return self._token_space

    @property
    def device(self) -> str:
        return self._config.device

    @property
    def prompt_add_special_tokens(self) -> bool:
        return self._config.prompt_add_special_tokens

    def encode_prompt(self, text: str) -> tuple[int, ...]:
        token_ids = self._tokenizer.encode(
            text,
            add_special_tokens=self._config.prompt_add_special_tokens,
        )
        return tuple(int(token_id) for token_id in token_ids)

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
            raise LMError("prefill requires at least one prompt token")
        self._validate_tokenizer_ids(prompt_token_ids)

        import torch
        from transformers import DynamicCache

        input_ids = torch.tensor(
            [list(prompt_token_ids)],
            dtype=torch.long,
            device=self._config.device,
        )
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        cache = DynamicCache()
        cache_position = torch.arange(
            input_ids.shape[1], device=input_ids.device, dtype=torch.long
        )

        with torch.inference_mode():
            output = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=cache,
                cache_position=cache_position,
                use_cache=True,
                logits_to_keep=1,
            )

        if output.past_key_values is None:
            raise LMError("model did not return a KV cache during prefill")

        logits = output.logits[0, -1, :]
        self._validate_logits_shape(logits)
        backend_state = _HFBackendState(
            cache=output.past_key_values,
            attention_mask=attention_mask,
            next_cache_position=cache_position[-1:] + 1,
        )
        return LMState(
            backend_state=backend_state,
            pending_logits=logits,
            sequence_length=len(prompt_token_ids),
        )

    def next_logits(
        self,
        state: LMState,
        previous_token_id: int | None = None,
    ) -> tuple[Any, LMState]:
        if previous_token_id is None:
            self._validate_logits_shape(state.pending_logits)
            return state.pending_logits, state

        self._validate_tokenizer_ids((previous_token_id,))
        backend = state.backend_state
        if not isinstance(backend, _HFBackendState):
            raise LMError("unexpected Hugging Face backend state")

        import torch

        input_ids = torch.tensor(
            [[previous_token_id]],
            dtype=torch.long,
            device=self._config.device,
        )
        attention_mask = torch.cat(
            [backend.attention_mask, backend.attention_mask.new_ones((1, 1))],
            dim=-1,
        )

        with torch.inference_mode():
            output = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=backend.cache,
                cache_position=backend.next_cache_position,
                use_cache=True,
                logits_to_keep=1,
            )

        if output.past_key_values is None:
            raise LMError("model did not return a KV cache during cached step")

        logits = output.logits[0, -1, :]
        self._validate_logits_shape(logits)
        updated_backend = _HFBackendState(
            cache=output.past_key_values,
            attention_mask=attention_mask,
            next_cache_position=backend.next_cache_position[-1:] + 1,
        )
        updated = LMState(
            backend_state=updated_backend,
            pending_logits=logits,
            sequence_length=state.sequence_length + 1,
        )
        return logits, updated

    def _validate_logits_shape(self, logits: Any) -> None:
        shape = tuple(int(value) for value in logits.shape)
        if shape != (self._token_space.output_vocab_size,):
            raise LMError(
                "LM returned unexpected logits shape: "
                f"{shape}; expected ({self._token_space.output_vocab_size},)"
            )

    def _validate_tokenizer_ids(self, token_ids: tuple[int, ...] | list[int]) -> None:
        invalid = [
            int(token_id)
            for token_id in token_ids
            if not self._token_space.is_tokenizer_id(int(token_id))
        ]
        if invalid:
            preview = invalid[:5]
            raise LMError(f"invalid tokenizer token ids: {preview}")
