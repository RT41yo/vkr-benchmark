"""Shared exception hierarchy for benchmark-controlled failures."""


class BenchmarkError(Exception):
    """Base exception for failures that should be represented in run status."""


class ContractError(BenchmarkError):
    """Raised when a component violates a benchmark interface contract."""


class ConfigurationError(BenchmarkError):
    """Raised for an invalid or unsupported benchmark configuration."""


class DistributionError(ContractError):
    """Raised when a probability distribution is malformed."""


class NumericalError(BenchmarkError):
    """Raised when LM/distribution computation contains invalid numerics."""


class LMError(BenchmarkError):
    """Raised for failures in the common language-model path."""


class ModelLoadError(LMError):
    """Raised when a configured local model cannot be loaded faithfully."""


class SessionStateError(ContractError):
    """Raised when an encoder/decoder session is used in an invalid state."""
