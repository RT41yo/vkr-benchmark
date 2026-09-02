import pytest

from vkr_benchmark.methods import MethodEnvironment


def test_method_environment_canonicalizes_allowed_token_order() -> None:
    env = MethodEnvironment(output_vocab_size=8, allowed_token_ids=(7, 1, 4, 2))
    assert env.allowed_token_ids == (1, 2, 4, 7)


def test_method_environment_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        MethodEnvironment(output_vocab_size=4, allowed_token_ids=(0, 1, 1))


def test_method_environment_rejects_out_of_range_id() -> None:
    with pytest.raises(ValueError, match="outside"):
        MethodEnvironment(output_vocab_size=4, allowed_token_ids=(0, 4))
