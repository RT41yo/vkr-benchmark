"""Generation-only Stage-7 stegotexts via the unchanged normalized encoders.

No KL/TVD instrumentation or receiver decoding is performed. This module never
writes Stage-6 results. The target distribution, encoder and termination
semantics are identical to the Stage-6 encode path.
"""
from __future__ import annotations

from dataclasses import dataclass

from vkr_benchmark.config import ExperimentConfig
from vkr_benchmark.distributions import ReferenceDistributionBuilder, StepContext
from vkr_benchmark.errors import ConfigurationError, ContractError
from vkr_benchmark.inputs import create_secret_source
from vkr_benchmark.lm import LMAdapter
from vkr_benchmark.randomness import RecordingSecretSource
from vkr_benchmark.runner.experiment import create_method_runtime
from vkr_benchmark.runner.streaming import method_environment_from_builder
from vkr_benchmark.transport import TextChannel


class Stage7FixedPayloadNonTermination(ContractError):
    """RRC did not finish its fixed payload before the frozen token ceiling.

    The partial carrier is *not* a valid Stage-7 stegotext. Keep only aggregate
    diagnostic counters: no secret or partial text is exposed as a dataset item.
    """

    def __init__(
        self, *, target_payload_bits: int, max_carrier_tokens: int,
        generated_carrier_tokens: int, termination_candidates: int,
        verification_rejections: int,
    ) -> None:
        self.target_payload_bits = target_payload_bits
        self.max_carrier_tokens = max_carrier_tokens
        self.generated_carrier_tokens = generated_carrier_tokens
        self.termination_candidates = termination_candidates
        self.verification_rejections = verification_rejections
        super().__init__(
            "Stage-7 fixed-payload encoder did not terminate "
            f"after {generated_carrier_tokens}/{max_carrier_tokens} carriers "
            f"for {target_payload_bits} payload bits; "
            f"termination_candidates={termination_candidates}, "
            f"verification_rejections={verification_rejections}"
        )


@dataclass(frozen=True, slots=True)
class Stage7StegoGeneration:
    prompt_token_ids: tuple[int, ...]
    sender_token_ids: tuple[int, ...]
    receiver_token_ids: tuple[int, ...]
    text: str
    payload_bits: int
    secret_bits_read: int
    termination_reason: str
    token_sequence_roundtrip_exact: bool
    first_token_mismatch: int | None


def generate_stage7_stego(
    *, config: ExperimentConfig, lm_adapter: LMAdapter, prompt_text: str,
    reference_builder: ReferenceDistributionBuilder,
) -> Stage7StegoGeneration:
    """Encode-only pass with no detector metadata exposed to the generated text.

    Does not invoke metric-only `distribution_info`, entropy/NLL or decode. It
    retains the same `encoder.step`/`encoder.finalize` and cached LM path.
    """
    if not prompt_text:
        raise ConfigurationError("Stage-7 stego prompt must be nonempty")
    if reference_builder.token_space != lm_adapter.token_space:
        raise ConfigurationError("Stage-7 builder token space disagrees with LM")
    if reference_builder.policy != config.generation.to_policy():
        raise ConfigurationError("Stage-7 reference policy disagrees with run config")
    if config.method.method_id == "rrc":
        if config.termination.mode != "fixed_payload_bits":
            raise ConfigurationError("RRC requires fixed_payload_bits in Stage 7")
    elif config.termination.mode != "fixed_carrier_tokens":
        raise ConfigurationError("streaming methods require fixed_carrier_tokens")

    environment = method_environment_from_builder(reference_builder)
    runtime = create_method_runtime(config)
    source = RecordingSecretSource(create_secret_source(config.secret_id))
    target = (config.termination.target_payload_bits
              if config.termination.mode == "fixed_payload_bits" else None)
    limit = (config.termination.max_carrier_tokens
             if target is not None else config.termination.target_carrier_tokens)
    assert limit is not None and limit > 0
    prompt_ids = lm_adapter.encode_prompt(prompt_text)
    state = lm_adapter.prefill(prompt_ids)
    encoder = runtime.method.create_encoder(
        config=config.method.params, environment=environment,
        secret_source=source, random_source=runtime.encoder_random_source,
        key=config.method.key, target_payload_bits=target,
    )
    generated: list[int] = []
    step_bits: list[int | None] = []
    termination_candidates = 0
    verification_rejections = 0
    for step_index in range(limit):
        raw_logits, state = lm_adapter.next_logits(state)
        reference = reference_builder.build(raw_logits)
        decision = encoder.step(StepContext(step_index=step_index, reference=reference))
        token_id = int(decision.token_id)
        if not environment.is_allowed(token_id):
            raise ContractError(f"Stage-7 method emitted forbidden token {token_id}")
        generated.append(token_id)
        step_bits.append(decision.bits_consumed)
        if target is not None:
            # Read diagnostics already emitted by the encoder. This does not
            # invoke distribution metrics, receiver decoding or extra LM work.
            trace = decision.method_trace
            if trace.get("termination_test") is True:
                termination_candidates += 1
                if trace.get("termination_verified") is False:
                    verification_rejections += 1
        if encoder.done or step_index + 1 == limit:
            break
        _, state = lm_adapter.next_logits(state, token_id)
    if target is not None and not encoder.done:
        raise Stage7FixedPayloadNonTermination(
            target_payload_bits=target,
            max_carrier_tokens=limit,
            generated_carrier_tokens=len(generated),
            termination_candidates=termination_candidates,
            verification_rejections=verification_rejections,
        )
    final = encoder.finalize()
    if target is not None and final.payload_bits != target:
        raise ContractError("Stage-7 RRC payload length disagrees with target")
    if final.payload_bits > source.position:
        raise ContractError("Stage-7 encoder claims more payload than source bits read")
    if all(bits is not None for bits in step_bits):
        if sum(int(bits) for bits in step_bits) != final.payload_bits:
            raise ContractError("Stage-7 per-step payload accounting mismatch")
    if target is None and len(generated) != limit:
        raise ContractError("Stage-7 streaming method terminated before 256 carriers")
    transport = TextChannel(lm_adapter).transmit(generated)
    if not transport.text:
        raise ContractError("Stage-7 stegotext transport text is empty")
    return Stage7StegoGeneration(
        prompt_token_ids=tuple(prompt_ids), sender_token_ids=tuple(generated),
        receiver_token_ids=transport.receiver_token_ids, text=transport.text,
        payload_bits=final.payload_bits, secret_bits_read=source.position,
        termination_reason=final.termination_reason,
        token_sequence_roundtrip_exact=transport.token_sequence_roundtrip_exact,
        first_token_mismatch=transport.first_token_mismatch,
    )
