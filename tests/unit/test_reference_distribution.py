import numpy as np
import pytest

from vkr_benchmark.distributions import ReferenceDistribution
from vkr_benchmark.errors import DistributionError


def test_reference_distribution_is_fp32_read_only_and_ordered() -> None:
    reference = ReferenceDistribution(np.array([0.2, 0.5, 0.3]))

    assert reference.probabilities.dtype == np.float32
    assert reference.token_order.tolist() == [1, 2, 0]
    assert not reference.probabilities.flags.writeable
    assert not reference.token_order.flags.writeable


def test_reference_distribution_rejects_non_normalized_probabilities() -> None:
    with pytest.raises(DistributionError):
        ReferenceDistribution(np.array([0.2, 0.2], dtype=np.float32))


def test_reference_distribution_rejects_noncanonical_custom_order() -> None:
    with pytest.raises(DistributionError):
        ReferenceDistribution(
            np.array([0.5, 0.5], dtype=np.float32),
            token_order=np.array([1, 0]),
        )
