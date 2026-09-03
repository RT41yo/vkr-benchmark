"""Deterministic secret-bit sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from hashlib import shake_256

_SECRET_DOMAIN_PREFIX = "VKR-BENCHMARK-v0.1|secret|secret_id="


class SecretSource(ABC):
    """Sequential source of binary secret payload bits."""

    @property
    @abstractmethod
    def position(self) -> int:
        """Number of bits already consumed from the source."""

    @abstractmethod
    def read_bits(self, count: int) -> tuple[int, ...]:
        """Consume and return exactly ``count`` bits."""


class Shake256SecretSource(SecretSource):
    """Specification-v0.1 SHAKE256 secret stream, bytes read MSB-first."""

    def __init__(self, secret_id: str) -> None:
        if not secret_id:
            raise ValueError("secret_id must be non-empty")
        self._secret_id = secret_id
        self._seed = f"{_SECRET_DOMAIN_PREFIX}{secret_id}".encode("utf-8")
        self._position = 0

    @property
    def secret_id(self) -> str:
        return self._secret_id

    @property
    def position(self) -> int:
        return self._position

    def read_bits(self, count: int) -> tuple[int, ...]:
        if count < 0:
            raise ValueError("count must be non-negative")
        if count == 0:
            return ()

        start = self._position
        stop = start + count
        required_bytes = (stop + 7) // 8
        stream_prefix = shake_256(self._seed).digest(required_bytes)

        bits = tuple(
            (stream_prefix[bit_index // 8] >> (7 - (bit_index % 8))) & 1
            for bit_index in range(start, stop)
        )
        self._position = stop
        return bits


class RecordingSecretSource(SecretSource):
    """Transparent wrapper that records exactly the bits consumed in a run."""

    def __init__(self, source: SecretSource) -> None:
        self._source = source
        self._consumed: list[int] = []

    @property
    def position(self) -> int:
        return self._source.position

    @property
    def consumed_bits(self) -> tuple[int, ...]:
        return tuple(self._consumed)

    def read_bits(self, count: int) -> tuple[int, ...]:
        bits = self._source.read_bits(count)
        self._consumed.extend(bits)
        return bits
