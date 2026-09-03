from typing import Any, Mapping

import numpy as np

from vkr_benchmark.distributions import ReferenceDistribution, StepContext
from vkr_benchmark.methods import (
    DecodeProgress,
    DecoderFinalization,
    DecoderSession,
    EncodeDecision,
    EncoderFinalization,
    EncoderSession,
    MethodEnvironment,
    StegoMethod,
)
from vkr_benchmark.randomness import RandomSource, SecretSource, Shake256SecretSource


class FixedMessageEncoder(EncoderSession):
    """Synthetic RRC-like session: no meaningful bits-per-step accounting."""

    def __init__(self) -> None:
        self.steps = 0

    @property
    def done(self) -> bool:
        return self.steps >= 2

    def step(self, context: StepContext) -> EncodeDecision:
        token = int(context.reference.token_order[0])
        self.steps += 1
        return EncodeDecision(token_id=token, bits_consumed=None)

    def finalize(self) -> EncoderFinalization:
        return EncoderFinalization(payload_bits=8, termination_reason="method_defined")


class FixedMessageDecoder(DecoderSession):
    def __init__(self) -> None:
        self.steps = 0

    @property
    def done(self) -> bool:
        return self.steps >= 2

    def observe(self, context: StepContext, observed_token_id: int) -> DecodeProgress:
        self.steps += 1
        return DecodeProgress()

    def finalize(self) -> DecoderFinalization:
        return DecoderFinalization(recovered_bits=(1, 0, 1, 0, 1, 0, 1, 0))


class FixedMessageMethod(StegoMethod):
    method_id = "synthetic_fixed_message"

    def create_encoder(
        self,
        *,
        config: Mapping[str, Any],
        environment: MethodEnvironment,
        secret_source: SecretSource,
        random_source: RandomSource | None = None,
        key: bytes | str | int | None = None,
    ) -> EncoderSession:
        del environment
        return FixedMessageEncoder()

    def create_decoder(
        self,
        *,
        config: Mapping[str, Any],
        environment: MethodEnvironment,
        random_source: RandomSource | None = None,
        key: bytes | str | int | None = None,
        expected_payload_bits: int | None = None,
    ) -> DecoderSession:
        del environment
        return FixedMessageDecoder()


def test_session_contract_supports_fixed_message_method() -> None:
    method = FixedMessageMethod()
    secret = Shake256SecretSource("contract-test")
    environment = MethodEnvironment(output_vocab_size=2, allowed_token_ids=(0, 1))
    encoder = method.create_encoder(
        config={}, environment=environment, secret_source=secret
    )
    reference = ReferenceDistribution(np.array([0.6, 0.4], dtype=np.float32))

    decisions = []
    while not encoder.done:
        decisions.append(encoder.step(StepContext(len(decisions), reference)))

    final_encode = encoder.finalize()
    assert [d.bits_consumed for d in decisions] == [None, None]
    assert final_encode.payload_bits == 8

    decoder = method.create_decoder(
        config={}, environment=environment, expected_payload_bits=8
    )
    for i, decision in enumerate(decisions):
        progress = decoder.observe(
            StepContext(i, reference),
            observed_token_id=decision.token_id,
        )
        assert progress.recovered_bits == ()

    final_decode = decoder.finalize()
    assert final_decode.recovered_bits == (1, 0, 1, 0, 1, 0, 1, 0)
