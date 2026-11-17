"""Read-only Stage-7 prompt overlap audit, including near-duplicate candidates."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from vkr_benchmark.errors import ConfigurationError
from vkr_benchmark.steganalysis.dataset import read_json, read_jsonl


def normalized_tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


def shingles(text: str, *, width: int = 5) -> frozenset[tuple[str, ...]]:
    tokens = normalized_tokens(text)
    if not tokens:
        return frozenset()
    if len(tokens) < width:
        return frozenset({tokens})
    return frozenset(tuple(tokens[i:i + width]) for i in range(len(tokens) - width + 1))


def audit_prompt_overlap(
    *, split_manifest: dict[str, Any], stage6_prompts: list[dict[str, Any]],
    stage7_prompts: list[dict[str, Any]], threshold: float = 0.85,
) -> dict[str, Any]:
    if not 0 < threshold <= 1:
        raise ConfigurationError('near-duplicate threshold must lie in (0,1]')
    groups = split_manifest['rows']
    if len(groups) != 152:
        raise ConfigurationError('overlap audit requires all 152 prompt groups')
    by_id = {r['prompt_id']:r for r in groups}
    if len(by_id) != 152:
        raise ConfigurationError('split contains duplicate prompt ids')
    prompts = stage6_prompts + stage7_prompts
    if len(prompts) != 152 or len({r['prompt_id'] for r in prompts}) != 152:
        raise ConfigurationError('overlap audit requires 24+128 distinct prompt texts')
    if {r['prompt_id'] for r in prompts} != set(by_id):
        raise ConfigurationError('prompt text inventory differs from split manifest')
    rows = []
    for prompt in prompts:
        group = by_id[prompt['prompt_id']]
        text = prompt['text']
        if hashlib.sha256(text.encode('utf-8')).hexdigest() != group['context_sha256']:
            raise ConfigurationError(f'prompt content hash mismatch: {prompt["prompt_id"]}')
        rows.append((prompt['prompt_id'], group['partition'], shingles(text)))
    found = []
    for i, (left_id, left_partition, left_shingles) in enumerate(rows):
        for right_id, right_partition, right_shingles in rows[i+1:]:
            if left_partition == right_partition:
                continue
            denom = len(left_shingles | right_shingles)
            similarity = len(left_shingles & right_shingles)/denom if denom else 1.0
            if similarity >= threshold:
                found.append({'left_prompt_id':left_id, 'right_prompt_id':right_id,
                              'left_partition':left_partition, 'right_partition':right_partition,
                              'shingle_jaccard':round(similarity,6)})
    return {'schema_version':'stage7.prompt_overlap_audit.v1',
            'status':'review_required' if found else 'ok', 'prompt_count':len(rows),
            'cross_split_candidate_count':len(found), 'threshold':threshold,
            'token_shingle_width':5, 'candidates':sorted(found, key=lambda r:(r['left_prompt_id'],r['right_prompt_id']))}
