import numpy as np

from vkr_benchmark.distributions import deterministic_token_order


def test_probability_order_uses_token_id_for_exact_ties() -> None:
    probs = np.array([0.25, 0.25, 0.40, 0.10], dtype=np.float32)
    assert deterministic_token_order(probs).tolist() == [2, 0, 1, 3]


def test_zero_probability_tokens_are_excluded_by_default() -> None:
    probs = np.array([0.5, 0.0, 0.5, 0.0], dtype=np.float32)
    assert deterministic_token_order(probs).tolist() == [0, 2]
