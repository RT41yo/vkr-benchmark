"""Stateful method-adapter lifecycle used by all steganographic methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from vkr_benchmark.distributions import StepContext
from vkr_benchmark.methods.types import (
    DecodeProgress,
    DecoderFinalization,
    EncodeDecision,
    EncoderFinalization,
    MethodEnvironment,
)
from vkr_benchmark.randomness import RandomSource, SecretSource

MethodConfig = Mapping[str, Any]
KeyMaterial = bytes | str | int | None


class EncoderSession(ABC):
    """Stateful encoder for one benchmark run."""

    @property
    @abstractmethod
    def done(self) -> bool:
        """Whether method-defined encoding has terminated."""

    @abstractmethod
    def step(self, context: StepContext) -> EncodeDecision:
        """Choose the next carrier token for the current P_reference."""

    @abstractmethod
    def finalize(self) -> EncoderFinalization:
        """Return authoritative payload accounting and final method metadata."""


class DecoderSession(ABC):
    """Stateful decoder for one benchmark run."""

    @property
    @abstractmethod
    def done(self) -> bool:
        """Whether the method has enough information to finalize decoding."""

    @abstractmethod
    def observe(self, context: StepContext, observed_token_id: int) -> DecodeProgress:
        """Consume one token reconstructed from the transmitted text."""

    @abstractmethod
    def finalize(self) -> DecoderFinalization:
        """Recover the final payload after all required observations."""


class StegoMethod(ABC):
    """Factory for matching encoder and decoder sessions."""

    method_id: str

    @abstractmethod
    def create_encoder(
        self,
        *,
        config: MethodConfig,
        environment: MethodEnvironment,
        secret_source: SecretSource,
        random_source: RandomSource | None = None,
        key: KeyMaterial = None,
    ) -> EncoderSession:
        """Create an encoder session isolated from LM/tokenizer ownership."""

    @abstractmethod
    def create_decoder(
        self,
        *,
        config: MethodConfig,
        environment: MethodEnvironment,
        random_source: RandomSource | None = None,
        key: KeyMaterial = None,
        expected_payload_bits: int | None = None,
    ) -> DecoderSession:
        """Create a decoder session matching the encoder configuration."""
