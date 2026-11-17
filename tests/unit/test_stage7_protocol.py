from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
import pytest

from vkr_benchmark.distributions import GenerationPolicy, ReferenceDistributionBuilder
from vkr_benchmark.distributions.types import ReferenceDistribution
from vkr_benchmark.errors import ConfigurationError, DistributionError
from vkr_benchmark.lm import TokenSpace
from vkr_benchmark.lm.base import LMAdapter, LMState
from vkr_benchmark.randomness import ControlRandomSource, RandomSource
from vkr_benchmark.steganalysis.cover import generate_cover, sample_reference_token
from vkr_benchmark.steganalysis.protocol import cover_seed, grouped_split, secret_id_for_prompt


class _FixedRandom(RandomSource):
    def __init__(self, u: float) -> None:
        self.u = u
    def random(self) -> float:
        return self.u
    def randbelow(self, upper: int) -> int:
        return 0


def test_inverse_cdf_samples_probability_intervals_and_skips_zero_mass() -> None:
    p = ReferenceDistribution(np.array([0.5, 0, 0.3, 0.2], dtype=np.float32))
    assert sample_reference_token(p, random_source=_FixedRandom(0.0)) == 0
    assert sample_reference_token(p, random_source=_FixedRandom(0.49)) == 0
    assert sample_reference_token(p, random_source=_FixedRandom(0.5)) == 2
    assert sample_reference_token(p, random_source=_FixedRandom(0.79)) == 2
    assert sample_reference_token(p, random_source=_FixedRandom(0.99)) == 3
    with pytest.raises(DistributionError, match=r"\[0,1\)"):
        sample_reference_token(p, random_source=_FixedRandom(1.0))


def test_inverse_cdf_empirical_probabilities_match_reference() -> None:
    p = ReferenceDistribution(np.array([0.7, 0.2, 0.1], dtype=np.float32))
    rng = ControlRandomSource(12345)
    outputs = np.bincount([sample_reference_token(p, random_source=rng) for _ in range(12000)], minlength=3) / 12000
    assert np.allclose(outputs, [0.7, 0.2, 0.1], atol=0.02)


def test_grouped_split_seed_reproducible_and_fails_duplicates() -> None:
    pairs = [(f"s7p{i:04d}", f"article-{i}") for i in range(1, 129)]
    assigned = grouped_split(pairs)
    assert assigned == grouped_split(reversed(pairs))
    assert set(assigned.values()) == {"train", "validation"}
    assert list(assigned.values()).count("train") == 96
    assert list(assigned.values()).count("validation") == 32
    with pytest.raises(ConfigurationError, match="duplicate"):
        grouped_split(pairs[:-1] + [("s7p0128", "article-1")])
    assert secret_id_for_prompt("s7p0001") == "stage7-det-s7p0001-r01"
    assert cover_seed(model_id="m", article_id="a", method_point_id="bins/b1", replicate_id="r01") == cover_seed(
        model_id="m", article_id="a", method_point_id="bins/b1", replicate_id="r01")
    assert cover_seed(model_id="m", article_id="a", method_point_id="bins/b1", replicate_id="r01") != cover_seed(
        model_id="m", article_id="a", method_point_id="bins/b4", replicate_id="r01")


class _FakeLM(LMAdapter):
    def __init__(self) -> None:
        self._space = TokenSpace(
            output_vocab_size=4,
            tokenizer_vocab_size=4,
            tokenizer_token_ids=frozenset({0, 1, 2, 3}),
            special_token_ids=frozenset({0}),
        )
    @property
    def model_id(self) -> str:
        return "fake"
    @property
    def revision(self) -> str:
        return "rev"
    @property
    def token_space(self) -> TokenSpace:
        return self._space
    def encode_prompt(self, text: str) -> tuple[int, ...]:
        return (1,)
    def encode_text(self, text: str, *, add_special_tokens: bool) -> tuple[int, ...]:
        return tuple(int(x) for x in text.split())
    def decode_tokens(self, token_ids: tuple[int, ...] | list[int]) -> str:
        return " ".join(str(t) for t in token_ids)
    def prefill(self, prompt_token_ids: tuple[int, ...] | list[int]) -> LMState:
        return LMState(backend_state=None, pending_logits=np.array([-2.0, 0.0, 1.0, -0.2]), sequence_length=len(prompt_token_ids))
    def next_logits(self, state: LMState, previous_token_id: int | None = None) -> tuple[np.ndarray, LMState]:
        if previous_token_id is None:
            return state.pending_logits, state
        return state.pending_logits, LMState(backend_state=None, pending_logits=state.pending_logits,
                                             sequence_length=state.sequence_length + 1)


def test_cover_generation_is_deterministic_exact_length_and_special_ids_excluded() -> None:
    lm = _FakeLM()
    builder = ReferenceDistributionBuilder(token_space=lm.token_space,
                                           policy=GenerationPolicy(temperature=1.0, top_k=None,
                                                                   top_p=1.0, exclude_special_tokens=True))
    r1 = generate_cover(lm_adapter=lm, reference_builder=builder, prompt_text="hello", carrier_tokens=8,
                        random_source=ControlRandomSource(55))
    r2 = generate_cover(lm_adapter=lm, reference_builder=builder, prompt_text="hello", carrier_tokens=8,
                        random_source=ControlRandomSource(55))
    assert r1 == r2
    assert len(r1.sender_token_ids) == 8
    assert 0 not in r1.sender_token_ids
    assert r1.token_sequence_roundtrip_exact
    assert r1.text == " ".join(str(i) for i in r1.sender_token_ids)
    with pytest.raises(ConfigurationError):
        generate_cover(lm_adapter=lm, reference_builder=builder, prompt_text="hello", carrier_tokens=0,
                       random_source=ControlRandomSource(55))


def _prompt_record(rank: int) -> dict:
    text = f"Article {rank}. Sentence two. Sentence three."
    return {"selection_rank": rank, "row_index": rank, "article_id": f"article-{rank}",
            "context": text, "context_sha256": hashlib.sha256(text.encode()).hexdigest()}


def test_stage7_prompt_split_no_source_text_in_manifest() -> None:
    root = Path(__file__).resolve().parents[2]
    namespace = runpy.run_path(str(root / "scripts/prepare_stage7_prompts.py"), run_name="test")
    stage6 = [{"prompt_id": f"s6p{index+1:04d}", "article_id": f"article-{index+8}",
               "row_index": index+8, "stage3_selection_rank": index+8,
               "context_sha256": _prompt_record(index+8)["context_sha256"]} for index in range(24)]
    prompts, manifest = namespace["build_stage7_records"](
        [_prompt_record(rank) for rank in range(160)], frozen_stage6_records=stage6)
    assert len(prompts) == 128
    assert prompts[0]["prompt_id"] == "s7p0001"
    assert manifest["split"]["train"] == 96
    assert sum(x["partition"] == "test" for x in manifest["rows"]) == 24
    assert "Sentence three" not in json.dumps(manifest)
    with pytest.raises(ConfigurationError, match="duplicate"):
        duplicates = [_prompt_record(rank) for rank in range(160)]
        duplicates[70]["article_id"] = duplicates[50]["article_id"]
        namespace["build_stage7_records"](duplicates, frozen_stage6_records=stage6)
