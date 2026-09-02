"""Core probability-distribution and per-step context types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from vkr_benchmark.distributions.ordering import deterministic_token_order
from vkr_benchmark.errors import DistributionError

_REFERENCE_SUM_ATOL = 1e-5
_Q_SUM_ATOL = 1e-8


class QMode(StrEnum):
    """How the induced stego distribution is obtained."""

    ANALYTIC_EXACT = "analytic_exact"
    EXACT_ENUMERATION = "exact_enumeration"
    MONTE_CARLO_ESTIMATE = "monte_carlo_estimate"
    UNAVAILABLE = "unavailable"


class QRepresentation(StrEnum):
    """Concrete representation made available to the metric layer."""

    EXPLICIT_PROBABILITIES = "explicit_probabilities"
    REFERENCE_EQUALITY_CERTIFICATE = "reference_equality_certificate"
    NONE = "none"


class QSource(StrEnum):
    """Provenance of information about Q_stego.

    This is intentionally more explicit than specification v0.1 so that later
    analyses do not confuse a theoretical equality claim with an independently
    constructed or estimated distribution.
    """

    ADAPTER_EXACT = "adapter_exact"
    ANALYTIC_THEORY = "analytic_theory"
    INDEPENDENT_ENUMERATION = "independent_enumeration"
    MONTE_CARLO_VALIDATION = "monte_carlo_validation"
    UNAVAILABLE = "unavailable"


def _readonly_float_array(values: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _readonly_metadata(metadata: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(metadata or {}))


@dataclass(frozen=True, slots=True)
class ReferenceDistribution:
    """Canonical per-step P_reference supplied by common infrastructure.

    Probabilities are stored as an immutable FP32 copy. Methods are free to
    derive FP64/Decimal/integer representations internally, but must not mutate
    or replace this canonical distribution.
    """

    probabilities: np.ndarray
    token_order: np.ndarray | None = None

    def __post_init__(self) -> None:
        probs = _readonly_float_array(self.probabilities, dtype=np.float32)
        if probs.ndim != 1 or probs.size == 0:
            raise DistributionError("P_reference must be a non-empty 1D array")
        if not np.all(np.isfinite(probs)):
            raise DistributionError("P_reference must contain only finite values")
        if np.any(probs < 0):
            raise DistributionError("P_reference probabilities must be non-negative")

        total = float(np.sum(probs, dtype=np.float64))
        if not np.isclose(total, 1.0, rtol=0.0, atol=_REFERENCE_SUM_ATOL):
            raise DistributionError(
                f"P_reference must sum to 1 within atol={_REFERENCE_SUM_ATOL}; got {total}"
            )

        if self.token_order is None:
            order = deterministic_token_order(probs)
        else:
            order = np.array(self.token_order, dtype=np.int64, copy=True)
            if order.ndim != 1:
                raise DistributionError("token_order must be one-dimensional")
            expected = deterministic_token_order(probs)
            if not np.array_equal(order, expected):
                raise DistributionError(
                    "token_order must exactly follow (-probability, token_id) "
                    "over positive-probability support"
                )
            order.setflags(write=False)

        object.__setattr__(self, "probabilities", probs)
        object.__setattr__(self, "token_order", order)

    @property
    def vocab_size(self) -> int:
        return int(self.probabilities.size)

    @property
    def support_size(self) -> int:
        assert self.token_order is not None
        return int(self.token_order.size)


@dataclass(frozen=True, slots=True)
class StepContext:
    """Immutable method-facing context for one generated carrier token."""

    step_index: int
    reference: ReferenceDistribution

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")


@dataclass(frozen=True, slots=True)
class DistributionInfo:
    """Tagged report describing the method-induced Q_stego for one step."""

    mode: QMode
    representation: QRepresentation
    source: QSource
    probabilities: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        metadata = _readonly_metadata(self.metadata)
        object.__setattr__(self, "metadata", metadata)

        if self.representation == QRepresentation.EXPLICIT_PROBABILITIES:
            if self.probabilities is None:
                raise DistributionError(
                    "explicit Q representation requires probabilities"
                )
            q = _readonly_float_array(self.probabilities, dtype=np.float64)
            if q.ndim != 1 or q.size == 0:
                raise DistributionError("Q_stego must be a non-empty 1D array")
            if not np.all(np.isfinite(q)):
                raise DistributionError("Q_stego must contain only finite values")
            if np.any(q < 0):
                raise DistributionError("Q_stego probabilities must be non-negative")
            total = float(np.sum(q, dtype=np.float64))
            if not np.isclose(total, 1.0, rtol=0.0, atol=_Q_SUM_ATOL):
                raise DistributionError(
                    f"Q_stego must sum to 1 within atol={_Q_SUM_ATOL}; got {total}"
                )
            object.__setattr__(self, "probabilities", q)
        elif self.probabilities is not None:
            raise DistributionError(
                "probabilities are allowed only for explicit Q representation"
            )

        if self.mode == QMode.UNAVAILABLE:
            if self.representation != QRepresentation.NONE:
                raise DistributionError(
                    "q_mode=unavailable must use representation=none"
                )
            if self.source != QSource.UNAVAILABLE:
                raise DistributionError(
                    "q_mode=unavailable must use source=unavailable"
                )

    @classmethod
    def explicit(
        cls,
        probabilities: np.ndarray,
        *,
        mode: QMode,
        source: QSource,
        metadata: Mapping[str, Any] | None = None,
    ) -> "DistributionInfo":
        if mode == QMode.UNAVAILABLE:
            raise DistributionError("explicit Q cannot use q_mode=unavailable")
        return cls(
            mode=mode,
            representation=QRepresentation.EXPLICIT_PROBABILITIES,
            source=source,
            probabilities=probabilities,
            metadata=metadata or {},
        )

    @classmethod
    def reference_equality(
        cls,
        *,
        source: QSource = QSource.ANALYTIC_THEORY,
        metadata: Mapping[str, Any] | None = None,
    ) -> "DistributionInfo":
        return cls(
            mode=QMode.ANALYTIC_EXACT,
            representation=QRepresentation.REFERENCE_EQUALITY_CERTIFICATE,
            source=source,
            metadata=metadata or {},
        )

    @classmethod
    def unavailable(
        cls, *, metadata: Mapping[str, Any] | None = None
    ) -> "DistributionInfo":
        return cls(
            mode=QMode.UNAVAILABLE,
            representation=QRepresentation.NONE,
            source=QSource.UNAVAILABLE,
            metadata=metadata or {},
        )
