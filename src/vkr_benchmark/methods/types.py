"""Shared method-session input and result types."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from vkr_benchmark.distributions import DistributionInfo


def _metadata(values: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(values or {}))


@dataclass(frozen=True, slots=True)
class MethodEnvironment:
    """Run-invariant information exposed to a steganographic method.

    The environment contains only benchmark-normalized token-space information.
    It deliberately does not expose the LM, tokenizer, logits, device, or common
    generation-policy implementation to the method.

    ``allowed_token_ids`` is canonicalized to increasing token ID order so that
    a keyed/randomized method partition does not depend on incidental caller
    ordering.
    """

    output_vocab_size: int
    allowed_token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.output_vocab_size <= 0:
            raise ValueError("output_vocab_size must be positive")

        token_ids = tuple(sorted(int(token_id) for token_id in self.allowed_token_ids))
        if not token_ids:
            raise ValueError("allowed_token_ids must be non-empty")
        if len(set(token_ids)) != len(token_ids):
            raise ValueError("allowed_token_ids must not contain duplicates")
        if token_ids[0] < 0 or token_ids[-1] >= self.output_vocab_size:
            raise ValueError("allowed token ID lies outside output vocabulary")

        object.__setattr__(self, "allowed_token_ids", token_ids)

    def is_allowed(self, token_id: int) -> bool:
        """Check V_allowed membership in O(log |V|) without another large set."""

        value = int(token_id)
        index = bisect_left(self.allowed_token_ids, value)
        return index < len(self.allowed_token_ids) and self.allowed_token_ids[index] == value


@dataclass(frozen=True, slots=True)
class EncodeDecision:
    """One carrier-token decision produced by an encoder session.

    ``bits_consumed`` is optional by design. Streaming methods can report it
    directly; fixed-message/stateful methods such as RRC may report ``None``
    and provide authoritative payload accounting only at finalization.
    """

    token_id: int
    bits_consumed: int | None = None
    distribution_info: DistributionInfo = field(
        default_factory=DistributionInfo.unavailable
    )
    method_trace: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.token_id < 0:
            raise ValueError("token_id must be non-negative")
        if self.bits_consumed is not None and self.bits_consumed < 0:
            raise ValueError("bits_consumed must be non-negative or None")
        object.__setattr__(self, "method_trace", _metadata(self.method_trace))


@dataclass(frozen=True, slots=True)
class DecodeProgress:
    """Incremental decoder output for one observed carrier token."""

    recovered_bits: tuple[int, ...] = ()
    method_trace: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(bit not in (0, 1) for bit in self.recovered_bits):
            raise ValueError("recovered_bits must contain only 0/1")
        object.__setattr__(self, "method_trace", _metadata(self.method_trace))


@dataclass(frozen=True, slots=True)
class EncoderFinalization:
    """Authoritative payload accounting after generation terminates."""

    payload_bits: int
    termination_reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.payload_bits < 0:
            raise ValueError("payload_bits must be non-negative")
        if not self.termination_reason:
            raise ValueError("termination_reason must be non-empty")
        object.__setattr__(self, "metadata", _metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class DecoderFinalization:
    """Recovered payload returned after the full carrier has been observed."""

    recovered_bits: tuple[int, ...]
    complete: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(bit not in (0, 1) for bit in self.recovered_bits):
            raise ValueError("recovered_bits must contain only 0/1")
        object.__setattr__(self, "metadata", _metadata(self.metadata))
