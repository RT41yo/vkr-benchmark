"""Shared exception hierarchy for benchmark-controlled failures."""


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
