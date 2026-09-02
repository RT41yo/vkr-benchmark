import numpy as np
import pytest

from vkr_benchmark.distributions import (
    DistributionInfo,
    QMode,
    QRepresentation,
    QSource,
)
from vkr_benchmark.errors import DistributionError


def test_explicit_q_is_read_only() -> None:
    info = DistributionInfo.explicit(
        np.array([0.75, 0.25]),
        mode=QMode.ANALYTIC_EXACT,
        source=QSource.ADAPTER_EXACT,
    )
    assert info.representation == QRepresentation.EXPLICIT_PROBABILITIES
    assert info.probabilities is not None
    assert not info.probabilities.flags.writeable


def test_reference_equality_keeps_theory_distinct_from_explicit_q() -> None:
    info = DistributionInfo.reference_equality()
    assert info.representation == QRepresentation.REFERENCE_EQUALITY_CERTIFICATE
    assert info.source == QSource.ANALYTIC_THEORY
    assert info.probabilities is None


def test_unavailable_q_cannot_carry_probabilities() -> None:
    with pytest.raises(DistributionError):
        DistributionInfo(
            mode=QMode.UNAVAILABLE,
            representation=QRepresentation.NONE,
            source=QSource.UNAVAILABLE,
            probabilities=np.array([1.0]),
        )
