"""Shared exception hierarchy for benchmark-controlled failures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RoundtripFailureDiagnostics:
    """Non-normative sender/transport state captured before a hard failure.

    The diagnostics exist only to preserve evidence needed to attribute failed
    end-to-end runs.  They do not change run status, reliability semantics, or
    method behavior.
    """

    failure_phase: str
    sender_token_ids: tuple[int, ...]
    payload_bits: int
    secret_bits_read: int
    transport_text: str | None = None
    receiver_token_ids: tuple[int, ...] | None = None
    token_sequence_roundtrip_exact: bool | None = None
    first_token_mismatch: int | None = None
    token_count_delta: int | None = None


_ROUNDTRIP_DIAGNOSTICS_ATTR = "_vkr_roundtrip_failure_diagnostics"


def attach_roundtrip_failure_diagnostics(
    error: Exception,
    diagnostics: RoundtripFailureDiagnostics,
) -> None:
    """Attach best-effort diagnostic state without changing the exception type."""

    try:
        setattr(error, _ROUNDTRIP_DIAGNOSTICS_ATTR, diagnostics)
    except Exception:
        # Diagnostic preservation must never replace or hide the real failure.
        return


def get_roundtrip_failure_diagnostics(
    error: Exception,
) -> RoundtripFailureDiagnostics | None:
    """Return attached hard-failure diagnostics when available."""

    value = getattr(error, _ROUNDTRIP_DIAGNOSTICS_ATTR, None)
    return value if isinstance(value, RoundtripFailureDiagnostics) else None



class BenchmarkError(Exception):
    """Base exception for failures that should be represented in run status."""


class ContractError(BenchmarkError):
    """Raised when a component violates a benchmark interface contract."""


class ConfigurationError(BenchmarkError):
    """Raised for an invalid or unsupported benchmark configuration."""


class UnsupportedConfigurationError(ConfigurationError):
    """Raised when individually valid components form an unsupported run."""


class DistributionError(ContractError):
    """Raised when a probability distribution is malformed."""


class MetricError(BenchmarkError):
    """Raised when a benchmark metric cannot be computed validly."""


class NumericalError(BenchmarkError):
    """Raised when LM/distribution computation contains invalid numerics."""


class LMError(BenchmarkError):
    """Raised for failures in the common language-model path."""


class ModelLoadError(LMError):
    """Raised when a configured local model cannot be loaded faithfully."""


class MethodError(BenchmarkError):
    """Raised when a steganographic method cannot complete a valid operation."""


class SessionStateError(ContractError):
    """Raised when an encoder/decoder session is used in an invalid state."""


class StorageError(BenchmarkError):
    """Raised when canonical run persistence cannot be completed safely."""
