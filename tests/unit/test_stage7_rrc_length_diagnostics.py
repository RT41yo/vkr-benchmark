from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import stage7_rrc_length_diagnostics as diag


def inputs():
    groups, prompts, matched, scores = [], [], [], []
    for i in range(24):
        pid, aid, text = f'p{i}', f'a{i}', f'Context {i}'
        groups.append(dict(prompt_id=pid, article_id=aid, partition='test',
                           context_sha256=sha256(text.encode()).hexdigest()))
        prompts.append(dict(prompt_id=pid, text=text))
        for model, (_, count) in diag.MODELS.items():
            for k in range(count):
                rid = f'r{k}'
                sid = f'{model}-{i}-{k}'
                row = dict(sample_id=sid, partition='test', model_id=model, article_id=aid,
                           prompt_id=pid, method_id='rrc', method_point_id=diag.POINT,
                           replicate_id=rid, carrier_tokens=300+i, stego_run_id=sid,
                           cover_seed=diag.cover_seed(model_id=model, article_id=aid,
                           method_point_id=f'rrc/{diag.POINT}', replicate_id=rid))
                matched.append(row)
                for d in diag.DETECTORS:
                    for y in (0,1):
                        scores.append(dict(row, pair_id=sid, detector=d, label=y, score_stego=.5))
    return matched, scores, prompts, {'rows':groups}


def test_plan_preserves_seeds_and_changes_only_stopping_length():
    matched, scores, prompts, split = inputs()
    plan = diag.build_plan(matched,scores,prompts,split)
    original = {r['sample_id']:r for r in matched}
    assert len(plan)==120
    for r in plan:
        assert r['carrier_tokens']==256
        assert r['matched_carrier_tokens']==original[r['sample_id']]['carrier_tokens']
        assert r['cover_seed']==original[r['sample_id']]['cover_seed']
    assert all(r['carrier_tokens']>=300 for r in matched)


def test_plan_rejects_missing_prediction_or_wrong_prompt():
    matched,scores,prompts,split=inputs()
    with pytest.raises(ValueError,match='score support'):
        diag.build_plan(matched,scores[:-1],prompts,split)
    prompts[0]['text']='changed'
    with pytest.raises(ValueError,match='prompt text changed'):
        diag.build_plan(matched,scores,prompts,split)


def test_cover_validation_preserves_cr_and_rejects_wrong_length(tmp_path):
    row=diag.build_plan(*inputs())[0]
    # Synthetic fixture uses slash-containing IDs; pathlib handles them locally.
    folder=tmp_path/row['sample_id'];folder.mkdir(parents=True)
    text='a\r\nb\u2028c'
    (folder/'cover.txt').write_bytes(text.encode())
    marker=dict(row,model_revision=diag.REVISIONS[row['model_id']],text_sha256=sha256(text.encode()).hexdigest())
    (folder/'cover_result.json').write_text(json.dumps(marker))
    ids=folder/'token_ids.json';ids.write_text(json.dumps({'sender_token_ids':[1]*256}))
    assert diag.read_cover(row,tmp_path)==text
    ids.write_text(json.dumps({'sender_token_ids':[1]*255}))
    with pytest.raises(ValueError,match='invalid unmatched cover'):
        diag.read_cover(row,tmp_path)


def test_paired_delta_matches_literal_article_resampling():
    old=[dict(article_id=a,label=y,score_stego=s) for a,y,s in [('a',0,.2),('a',1,.8),('b',0,.6),('b',1,.4)]]
    new=[dict(r,score_stego=.1 if r['label']==0 else r['score_stego']) for r in old]
    draws=np.array([[2,0],[1,1],[0,2]])
    delta=diag.paired_delta(old,new,['a','b'],draws)
    expected=[]
    for w in draws:
        auc=[]
        for rows in (old,new):
            weights=np.repeat(w,2)
            auc.append(roc_auc_score([r['label'] for r in rows],[r['score_stego'] for r in rows],sample_weight=weights))
        expected.append(auc[1]-auc[0])
    assert delta['ci95_lower']==pytest.approx(np.quantile(expected,.025))
    assert delta['ci95_upper']==pytest.approx(np.quantile(expected,.975))


def test_refuses_overwrite(tmp_path):
    target=tmp_path/'artifact'
    diag.write_new(target,b'original')
    with pytest.raises(FileExistsError):
        diag.write_new(target,b'new')
    assert target.read_bytes()==b'original'


def test_common_prefix_checks_separately_generated_sender_ids(tmp_path):
    row=diag.build_plan(*inputs())[0]
    old_root,new_root=tmp_path/'old',tmp_path/'new'
    for root,n in [(old_root,row['matched_carrier_tokens']),(new_root,256)]:
        folder=root/row['sample_id'];folder.mkdir(parents=True)
        (folder/'cover.txt').write_bytes(b'text')
        marker=dict(row,carrier_tokens=n,model_revision=diag.REVISIONS[row['model_id']],
                    text_sha256=sha256(b'text').hexdigest())
        (folder/'cover_result.json').write_text(json.dumps(marker))
        (folder/'token_ids.json').write_text(json.dumps({'sender_token_ids':list(range(n))}))
    assert diag.common_prefix_exact(row,old_root,new_root)
    (new_root/row['sample_id']/'token_ids.json').write_text(json.dumps({'sender_token_ids':[999]*256}))
    assert not diag.common_prefix_exact(row,old_root,new_root)
