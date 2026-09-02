import numpy as np

from vkr_benchmark.lm import TokenSpace


def test_llama_like_token_space_has_no_output_only_ids() -> None:
    space = TokenSpace(
        output_vocab_size=128256,
        tokenizer_vocab_size=128256,
        special_token_ids=frozenset({128000, 128001}),
    )
    assert space.output_only_count == 0
    mask = space.allowed_mask()
    assert mask.shape == (128256,)
    assert not mask[128000]
    assert not mask[128001]
    assert int(np.count_nonzero(mask)) == 128254


def test_qwen_exact_output_only_count_is_267() -> None:
    space = TokenSpace(
        output_vocab_size=151936,
        tokenizer_vocab_size=151669,
        special_token_ids=frozenset(),
    )
    assert space.output_only_count == 267
    assert len(space.output_only_ids) == 267
    assert space.output_only_ids[0] == 151669
    assert space.output_only_ids[-1] == 151935

    mask = space.allowed_mask()
    assert bool(np.all(mask[:151669]))
    assert not bool(np.any(mask[151669:]))


def test_special_exclusion_can_be_disabled_without_enabling_output_only_ids() -> None:
    space = TokenSpace(
        output_vocab_size=6,
        tokenizer_vocab_size=4,
        special_token_ids=frozenset({1}),
    )
    mask = space.allowed_mask(exclude_special_tokens=False)
    assert mask.tolist() == [True, True, True, True, False, False]


def test_non_contiguous_tokenizer_ids_create_internal_output_only_hole() -> None:
    space = TokenSpace(
        output_vocab_size=6,
        tokenizer_vocab_size=5,
        special_token_ids=frozenset(),
        tokenizer_token_ids=frozenset({0, 1, 3, 4, 5}),
    )
    assert space.output_only_ids == (2,)
    assert space.output_only_count == 1
    assert space.allowed_mask().tolist() == [True, True, False, True, True, True]
    assert not space.is_tokenizer_id(2)
    assert space.is_tokenizer_id(5)
