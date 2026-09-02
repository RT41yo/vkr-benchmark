import numpy as np
import pytest

from vkr_benchmark.distributions import (
    GenerationPolicy,
    ReferenceDistributionBuilder,
)
from vkr_benchmark.errors import NumericalError
from vkr_benchmark.lm import TokenSpace


def _space() -> TokenSpace:
    return TokenSpace(
        output_vocab_size=6,
        tokenizer_vocab_size=5,
        special_token_ids=frozenset({0}),
    )


def test_baseline_builder_masks_special_and_output_only_and_normalizes() -> None:
    builder = ReferenceDistributionBuilder(
        token_space=_space(),
        policy=GenerationPolicy(),
    )
    raw = np.array([10.0, 2.0, 1.0, 0.0, -1.0, 20.0], dtype=np.float32)
    raw_before = raw.copy()

    reference = builder.build(raw)

    assert np.array_equal(raw, raw_before)
    assert reference.probabilities.dtype == np.float32
    assert reference.probabilities[0] == 0.0  # special
    assert reference.probabilities[5] == 0.0  # output-only
    assert np.isclose(
        np.sum(reference.probabilities, dtype=np.float64),
        1.0,
        atol=1e-6,
    )
    assert reference.token_order[0] == 1


def test_top_k_uses_token_id_tie_breaking() -> None:
    builder = ReferenceDistributionBuilder(
        token_space=TokenSpace(4, 4, frozenset()),
        policy=GenerationPolicy(top_k=2),
    )
    reference = builder.build(np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32))
    assert reference.token_order.tolist() == [0, 1]
    assert reference.probabilities[2] == 0.0
    assert reference.probabilities[3] == 0.0


def test_top_p_keeps_minimal_deterministic_prefix() -> None:
    builder = ReferenceDistributionBuilder(
        token_space=TokenSpace(3, 3, frozenset()),
        policy=GenerationPolicy(top_p=0.70),
    )
    # softmax ~= [0.665, 0.245, 0.090], therefore 0.70 needs first two tokens.
    reference = builder.build(np.array([2.0, 1.0, 0.0], dtype=np.float32))
    assert reference.token_order.tolist() == [0, 1]
    assert reference.probabilities[2] == 0.0


def test_temperature_changes_distribution_but_not_order() -> None:
    logits = np.array([2.0, 1.0], dtype=np.float32)
    cold = ReferenceDistributionBuilder(
        token_space=TokenSpace(2, 2, frozenset()),
        policy=GenerationPolicy(temperature=0.5),
    ).build(logits)
    warm = ReferenceDistributionBuilder(
        token_space=TokenSpace(2, 2, frozenset()),
        policy=GenerationPolicy(temperature=2.0),
    ).build(logits)

    assert cold.token_order.tolist() == warm.token_order.tolist() == [0, 1]
    assert cold.probabilities[0] > warm.probabilities[0]


def test_nonfinite_raw_logits_are_rejected() -> None:
    builder = ReferenceDistributionBuilder(
        token_space=TokenSpace(2, 2, frozenset()),
        policy=GenerationPolicy(),
    )
    with pytest.raises(NumericalError):
        builder.build(np.array([0.0, np.nan], dtype=np.float32))


def test_torch_logits_follow_fp32_path_when_torch_is_available() -> None:
    torch = pytest.importorskip("torch")
    builder = ReferenceDistributionBuilder(
        token_space=TokenSpace(3, 3, frozenset({2})),
        policy=GenerationPolicy(),
    )
    logits = torch.tensor([2.0, 1.0, 100.0], dtype=torch.float32)
    reference = builder.build(logits)
    assert reference.probabilities[2] == 0.0
    assert reference.probabilities[0] > reference.probabilities[1]
