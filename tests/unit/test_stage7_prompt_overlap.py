from __future__ import annotations
import hashlib
import pytest
from vkr_benchmark.errors import ConfigurationError
from vkr_benchmark.steganalysis.leakage import audit_prompt_overlap


def _fixture(make_duplicate: bool):
    groups=[]; stage6=[]; stage7=[]
    for i in range(152):
        pid=f'p{i}'
        text=f'This article {i} explores several specific topics about weather climate forecasts and extensive research findings.'
        if make_duplicate and i==150:
            text='THIS ARTICLE 3 explores several specific topics about weather climate forecasts and extensive research findings.'
        groups.append({'prompt_id':pid, 'context_sha256':hashlib.sha256(text.encode()).hexdigest(),
                       'partition':'test' if i>=128 else 'train' if i<96 else 'validation'})
        (stage7 if i<128 else stage6).append({'prompt_id':pid, 'text':text})
    return {'rows':groups}, stage6, stage7


def test_overlap_audit_clean_prompts():
    split, s6, s7 = _fixture(False)
    result = audit_prompt_overlap(split_manifest=split, stage6_prompts=s6, stage7_prompts=s7)
    assert result['status']=='ok'
    assert result['prompt_count']==152


def test_overlap_audit_detects_case_insensitive_near_duplicate_cross_split():
    split, s6, s7 = _fixture(True)
    result = audit_prompt_overlap(split_manifest=split, stage6_prompts=s6, stage7_prompts=s7)
    assert result['status']=='review_required'
    assert any(c['left_prompt_id']=='p3' and c['right_prompt_id']=='p150' or
               c['left_prompt_id']=='p150' and c['right_prompt_id']=='p3' for c in result['candidates'])


def test_overlap_audit_fails_on_changed_prompt_hash():
    split, s6, s7 = _fixture(False)
    s7[0]['text']='wrong text'
    with pytest.raises(ConfigurationError, match='hash mismatch'):
        audit_prompt_overlap(split_manifest=split, stage6_prompts=s6, stage7_prompts=s7)
