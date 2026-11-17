"""Stage-7 trainval identity and generation-only regression tests (CPU only)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from vkr_benchmark.config import (
    ExperimentConfig, GenerationRunConfig, MethodRunConfig, TerminationConfig,
)
from vkr_benchmark.distributions import GenerationPolicy, ReferenceDistributionBuilder
from vkr_benchmark.errors import ConfigurationError
from vkr_benchmark.inputs import create_secret_source
from vkr_benchmark.lm import TokenSpace
from vkr_benchmark.lm.base import LMAdapter, LMState
from vkr_benchmark.runner.experiment import create_method_runtime
from vkr_benchmark.runner.streaming import encode_fixed_carrier_tokens, method_environment_from_builder
from vkr_benchmark.steganalysis.plan import build_trainval_stego_plan, validate_stego_plan_row
from vkr_benchmark.steganalysis.protocol import method_random_material
from vkr_benchmark.steganalysis.stego import generate_stage7_stego

ROOT = Path(__file__).resolve().parents[2]


class TinyLM(LMAdapter):
    def __init__(self):
        self._space = TokenSpace(output_vocab_size=5, tokenizer_vocab_size=5,
                                 tokenizer_token_ids=frozenset(range(5)),
                                 special_token_ids=frozenset({0}))
    @property
    def model_id(self): return 'toy'
    @property
    def revision(self): return 'toy-rev'
    @property
    def token_space(self): return self._space
    def encode_prompt(self, text): return (1,)
    def encode_text(self, text, *, add_special_tokens): return tuple(map(int, text.split()))
    def decode_tokens(self, token_ids): return ' '.join(map(str, token_ids))
    def prefill(self, prompt_token_ids):
        return LMState(backend_state=None, pending_logits=np.array([-2, 2, 1, 0, -1], dtype=np.float32),
                       sequence_length=len(prompt_token_ids))
    def next_logits(self, state, previous_token_id=None):
        if previous_token_id is None: return state.pending_logits, state
        return state.pending_logits, LMState(backend_state=None, pending_logits=state.pending_logits,
                                              sequence_length=state.sequence_length+1)


def _toy_config(method_id, params, seed=None, key=None, termination=None):
    return ExperimentConfig(
        benchmark_version='1.0', run_kind='normalized', model_config_path=ROOT/'configs/models/llama-3.2-3b.local.json',
        prompt_id='s7p0001', method=MethodRunConfig(method_id=method_id, params=params,
                                                  random_seed=seed, key=key),
        generation=GenerationRunConfig(), secret_id='stage7-det-s7p0001-r01',
        termination=termination or TerminationConfig(mode='fixed_carrier_tokens', target_carrier_tokens=8))


@pytest.mark.parametrize('method_id,params,seed,key', [
    ('bins', {'block_size':1}, 'bins-key', None),
    ('huffman', {'bits_per_word':2}, None, None),
    ('adg', {}, 'adg-seed', None),
    ('discop', {}, None, 'discop-key'),
])
def test_generation_only_matches_instrumented_encoder(method_id, params, seed, key):
    """A changed sender sequence would violate Stage-6 encoder semantics."""
    lm = TinyLM()
    config = _toy_config(method_id, params, seed, key)
    builder = ReferenceDistributionBuilder(token_space=lm.token_space, policy=GenerationPolicy())
    actual = generate_stage7_stego(config=config, lm_adapter=lm,
                                   reference_builder=builder, prompt_text='prompt')
    runtime = create_method_runtime(config)
    expected = encode_fixed_carrier_tokens(
        lm_adapter=lm, reference_builder=builder, method=runtime.method,
        method_config=params, environment=method_environment_from_builder(builder),
        prompt_text='prompt', carrier_tokens=8,
        secret_source=create_secret_source(config.secret_id),
        method_random_source=runtime.encoder_random_source, key=key)
    assert actual.sender_token_ids == expected.carrier_token_ids
    assert actual.payload_bits == expected.payload_bits
    assert actual.secret_bits_read == expected.secret_bits_read
    assert actual.token_sequence_roundtrip_exact


def test_fixed_payload_rrc_preserved_on_toy_lm():
    lm = TinyLM()
    cfg = _toy_config('rrc', {'precision_margin':40, 'verify_termination':True},
                      key='toy-rrc-key', termination=TerminationConfig(
                          mode='fixed_payload_bits', target_payload_bits=8, max_carrier_tokens=100))
    b = ReferenceDistributionBuilder(token_space=lm.token_space, policy=GenerationPolicy())
    output = generate_stage7_stego(config=cfg, lm_adapter=lm,
                                   reference_builder=b, prompt_text='prompt')
    assert 1 <= len(output.sender_token_ids) <= 100
    assert output.payload_bits == 8
    assert output.termination_reason == 'fixed_payload_bits'


def test_method_randomness_shared_across_points_but_not_articles():
    assert method_random_material(article_id='a', method_id='bins', replicate_id='r01') == \
           method_random_material(article_id='a', method_id='bins', replicate_id='r01')
    assert method_random_material(article_id='a', method_id='bins', replicate_id='r01') != \
           method_random_material(article_id='b', method_id='bins', replicate_id='r01')
    assert method_random_material(article_id='a', method_id='discop', replicate_id='r01')[1] is not None
    assert method_random_material(article_id='a', method_id='dairstega', replicate_id='r01') == (None,None)


def test_exact_3840_trainval_run_plan_and_tamper_detection():
    frozen = json.loads((ROOT/'results/stage6/dataset_v1.0_manifest.json').read_text(encoding='utf-8'))
    rows = []
    for i in range(1,153):
        rows.append({'prompt_id': f's7p{i:04d}' if i <=128 else f's6p{i-128:04d}',
                     'article_id': f'article-{i}', 'context_sha256': f'hash-{i}',
                     'partition': 'train' if i <=96 else 'validation' if i <=128 else 'test'})
    plan = build_trainval_stego_plan(
        split_manifest={'schema_version':'stage7.prompt_split.v1', 'rows':rows},
        frozen_manifest=frozen, project_root=ROOT)
    assert len(plan) == 3840
    assert len({r['sample_id'] for r in plan}) == 3840
    assert len({r['stego_run_id'] for r in plan}) == 3840
    assert sum(r['method_id'] == 'rrc' for r in plan) == 256
    defaults = [r for r in plan if r['prompt_id'] == 's7p0001' and r['model_id'].startswith('meta')
                and r['method_point_id']=='default']
    assert {r['method_id'] for r in defaults} == {'adg','discop'}
    from vkr_benchmark.config import LocalModelConfig
    row = next(r for r in plan if r['model_id'].startswith('meta'))
    model = LocalModelConfig.from_json(ROOT/row['model_config'], project_root=ROOT)
    assert validate_stego_plan_row(row, model).method.method_id == row['method_id']
    modified = copy.deepcopy(row)
    modified['cover_seed'] += 1
    with pytest.raises(ConfigurationError, match='cover seed'):
        validate_stego_plan_row(modified, model)
    modified = copy.deepcopy(row)
    modified['canonical_run_config']['method']['params'] = {'arbitrary': True}
    with pytest.raises(ConfigurationError):
        validate_stego_plan_row(modified, model)


def test_stego_refuses_wrong_generation_policy_and_stopping_mode():
    lm = TinyLM()
    wrong = ReferenceDistributionBuilder(token_space=lm.token_space,
                                         policy=GenerationPolicy(top_k=2))
    with pytest.raises(ConfigurationError, match='reference policy'):
        generate_stage7_stego(config=_toy_config('bins', {'block_size':1}, seed=1),
                              lm_adapter=lm, reference_builder=wrong, prompt_text='prompt')
    with pytest.raises(ConfigurationError, match='fixed_payload_bits'):
        generate_stage7_stego(config=_toy_config('rrc', {}, key='key'),
                              lm_adapter=lm,
                              reference_builder=ReferenceDistributionBuilder(
                                  token_space=lm.token_space, policy=GenerationPolicy()),
                              prompt_text='prompt')
