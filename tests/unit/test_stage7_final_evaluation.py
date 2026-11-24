from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))

import vkr_benchmark.steganalysis.final_evaluation as fe
from vkr_benchmark.errors import ConfigurationError


def test_article_bootstrap_repeats_whole_cluster_with_ties():
    rows = [{'article_id': article, 'label': label}
            for article in ('a', 'b') for label in (0, 1)]
    scores = [0.1, 0.9, 0.8, 0.2]
    draws = np.array([[1, 1], [2, 0], [0, 2], [1, 1]])
    out = fe.article_bootstrap_auc(rows=rows, scores=scores,
                                   article_ids=['a', 'b'], draw_counts=draws)
    assert out['roc_auc'] == 0.75
    assert out['bootstrap_n'] == 4
    assert out['ci95_lower'] == pytest.approx(0.05625)
    assert out['ci95_upper'] == pytest.approx(0.98125)
    tied = fe.article_bootstrap_auc(rows=rows, scores=[0.5,0.5,0.5,0.5],
                                    article_ids=['a','b'],draw_counts=draws)
    assert (tied['roc_auc'], tied['ci95_lower'], tied['ci95_upper']) == (0.5, 0.5, 0.5)


def test_bootstrap_weighted_pairwise_exactly_matches_literal_article_resampling():
    from collections import Counter
    rows = []
    scores = []
    for article, score_pairs in [('a',[(0,.2),(1,.4),(0,.7),(1,.5)]),
                                 ('b',[(0,.6),(1,.9)]),
                                 ('c',[(0,.8),(1,.3)])]:
        for label, score in score_pairs:
            rows.append({'article_id':article,'label':label})
            scores.append(score)
    rng = np.random.default_rng(15)
    draws = np.stack([rng.multinomial(3, [1/3]*3) for _ in range(100)])
    assert fe.bootstrap_article_weights(['a','b','c'],iterations=11,seed=731).shape==(11,3)
    value = fe.article_bootstrap_auc(rows=rows,scores=scores,
                                     article_ids=['a','b','c'],draw_counts=draws)
    literal = []
    for multiplicity in draws:
        y, s = [], []
        for g, article in enumerate(['a','b','c']):
            for _ in range(int(multiplicity[g])):
                for row, score in zip(rows, scores):
                    if row['article_id'] == article:
                        y.append(row['label']);s.append(score)
        literal.append(roc_auc_score(y,s))
    assert value['ci95_lower'] == pytest.approx(float(np.quantile(literal,.025)))
    assert value['ci95_upper'] == pytest.approx(float(np.quantile(literal,.975)))


def test_metrics_use_frozen_threshold_and_correct_confusion():
    rows = [{'article_id': a,'label': y} for a in ('a','b') for y in (0,1)]
    result = fe.evaluate_scored_rows(rows,[.2,.9,.6,.4],threshold=.5,
        article_ids=['a','b'],draw_counts=np.array([[1,1],[2,0],[0,2]]))
    assert result['roc_auc'] == .75
    assert result['confusion'] == {'tn':1,'fp':1,'fn':1,'tp':1}
    assert result['balanced_accuracy'] == 0.5
    assert result['record_count'] == 4 and result['article_group_count']==2
    assert 0<=result['pr_auc_trapezoid']<=1
    with pytest.raises(ConfigurationError,match='threshold'):
        fe.evaluate_scored_rows(rows,[.2,.9,.6,.4],threshold=1.1,
            article_ids=['a','b'],draw_counts=np.array([[1,1]]))


def _write(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(fe.canonical_bytes(data))


def _fake_lock_sources(tmp_path, monkeypatch):
    config = tmp_path/'detector_training.json'
    real = Path(__file__).resolve().parents[2]/'configs/stage7/detector_training.json'
    cfg = json.loads(real.read_text())
    cfg['roberta']['base_revision'] = fe.PINNED_REVISION
    _write(config,cfg)
    monkeypatch.setattr(fe,'PINNED_PRIMARY_CONFIG',fe.file_sha(config))
    manifest = tmp_path/'dataset_manifest.json'
    _write(manifest,{'dataset_sha256':fe.FROZEN_DATASET_SHA256,
                     'sample_count':11278,'pair_count':5639,
                     'status':'complete_with_documented_nontermination',
                     'stage6_test_sources':1800, 'group_count':152,
                     'class_counts':{'cover':5639,'stego':5639},
                     'partition_counts':{'train':5760,'validation':1918,'test':3600},
                     'excluded_nontermination_run_ids':['0b0e22af25613edb']})
    audit = tmp_path/'roberta_primary_audit.json'
    model_root=tmp_path/'models'
    baseline=tmp_path/'baseline'
    roberta=tmp_path/'roberta'
    all_runs=[]
    for model_id, slug in fe.MODEL_SLUGS.items():
        _write(baseline/slug/'summary.json',{
            'status':'complete','model_id':model_id,'dataset_sha256':fe.FROZEN_DATASET_SHA256,
            'test_access':'none','baseline':cfg['baseline'],
            'validation_roc_auc':1.0, 'validation_threshold':0.9,
            'validation_balanced_accuracy':1.0,'config_sha256':'a'*64,
            'counts':{'train':2,'validation':2}})
        (baseline/slug/'validation_predictions.jsonl').write_text(
            ''.join(json.dumps({'record_id':f'{slug}-{i}', 'partition':'validation',
                                'label':i,'score_stego': .1 if i==0 else .9})+'\n'
                    for i in (0,1)),encoding='utf-8')
        joblib=model_root/slug/'tfidf_logreg.joblib'
        joblib.parent.mkdir(parents=True,exist_ok=True)
        joblib.write_bytes(b'pretend model bytes')
        for seed in (711,712,713):
            summary_path=roberta/slug/f'seed-{seed}'/'summary.json'
            _write(summary_path, {'status':'complete','test_access':'none',
                'model_id':model_id,'seed':seed,'dataset_sha256':fe.FROZEN_DATASET_SHA256,
                'validation_threshold':.51,'validation_roc_auc': .6 + (seed-711)*.1})
            weights=model_root/slug/'roberta'/f'seed-{seed}'
            weights.mkdir(parents=True)
            hashes={}
            for name in fe.RO_BERT_ASSETS:
                path=weights/name
                path.write_bytes((f'{seed}-{name}').encode())
                hashes[name]=fe.file_sha(path)
            all_runs.append({
                'model_id':model_id,'seed':seed,'best_epoch':4,
                'validation_roc_auc':.6 + (seed-711)*.1,
                'validation_threshold':.51,
                'summary_sha256':fe.file_sha(summary_path),
                'checkpoint_asset_sha256':hashes,
            })
    _write(audit, {'status':'verified','test_access':'none','detector_count':6,
                   'dataset_sha256':fe.FROZEN_DATASET_SHA256,
                   'primary_configuration_sha256':fe.PINNED_PRIMARY_CONFIG,
                   'base_revision':fe.PINNED_REVISION,'runs':all_runs})
    monkeypatch.setattr(fe,'PINNED_PRIMARY_AUDIT',fe.file_sha(audit))
    proto = json.loads((Path(__file__).resolve().parents[2]/'configs/stage7/final_evaluation.json').read_text())
    proto['primary_configuration_sha256']=fe.PINNED_PRIMARY_CONFIG
    proto['primary_audit_sha256']=fe.PINNED_PRIMARY_AUDIT
    protocol=tmp_path/'final_evaluation.json'
    _write(protocol,proto)
    return dict(protocol=protocol, detector_config=config, primary_audit=audit,
                dataset_manifest=manifest, baseline_results=baseline,
                roberta_results=roberta, models_root=model_root)


def test_preflight_seals_selected_validation_models_without_reading_test(tmp_path,monkeypatch):
    paths=_fake_lock_sources(tmp_path,monkeypatch)
    sealed=fe.make_lock(**paths)
    assert sealed['status']=='sealed' and sealed['test_access']=='none'
    assert len(sealed['selected'])==2
    assert all(x['roberta_e4']['seed']==713 for x in sealed['selected'])
    assert all(x['baseline']['threshold']==.9 for x in sealed['selected'])
    assert b'pretend model bytes' not in fe.canonical_bytes(sealed)


def test_preflight_refuses_modified_checkpoint_or_validation_scores(tmp_path,monkeypatch):
    paths=_fake_lock_sources(tmp_path,monkeypatch)
    fe.make_lock(**paths)
    weight=paths['models_root']/'llama-3.2-3b/roberta/seed-713/model.safetensors'
    weight.write_bytes(b'changed')
    with pytest.raises(ConfigurationError,match='checkpoint weights'):
        fe.make_lock(**paths)
    weight.write_bytes(b'713-model.safetensors')
    pred=paths['baseline_results']/'llama-3.2-3b/validation_predictions.jsonl'
    pred.write_text(pred.read_text().replace('0.9', '0.4'))
    with pytest.raises(ConfigurationError,match='baseline validation'):
        fe.make_lock(**paths)


def test_protocol_rejects_posthoc_seed_changes():
    path=Path(__file__).resolve().parents[2]/'configs/stage7/final_evaluation.json'
    cfg=json.loads(path.read_text())
    fe.validate_protocol(cfg)
    cfg['selected_roberta_seeds'][fe.QWEN_ID]=711
    with pytest.raises(ConfigurationError,match='protocol differs'):
        fe.validate_protocol(cfg)


def test_test_runner_requires_explicit_unlock_but_dry_run_avoids_test(tmp_path,monkeypatch,capsys):
    import evaluate_stage7_final as runner
    protocol=Path(__file__).resolve().parents[2]/'configs/stage7/final_evaluation.json'
    lock={'schema_version':'stage7.final_evaluation_lock.v1','status':'sealed',
          'dataset_sha256':fe.FROZEN_DATASET_SHA256}
    lock_path=tmp_path/'lock.json'
    lock_path.write_bytes(fe.canonical_bytes(lock))
    monkeypatch.setattr(runner,'_committed',lambda path:None)
    monkeypatch.setattr(runner,'make_lock',lambda **kwargs:lock)
    def sealed(**kwargs):
        raise AssertionError('frozen test must not be read in preflight')
    monkeypatch.setattr(runner,'load_frozen_test',sealed)
    monkeypatch.setattr(sys,'argv',['evaluate_stage7_final.py','--protocol',str(protocol),
                     '--lock',str(lock_path),'--dry-run'])
    assert runner.main()==0
    assert 'committed_lock_verified' in capsys.readouterr().out
    monkeypatch.setattr(sys,'argv',['evaluate_stage7_final.py','--protocol',str(protocol),
                     '--lock',str(lock_path),'--open-frozen-test','--lock-sha256','0'*64])
    assert runner.main()==1
    assert 'test remains sealed' in capsys.readouterr().err


def test_final_aggregates_all_15_points_two_detectors_and_four_pooled(tmp_path,monkeypatch):
    import evaluate_stage7_final as runner
    protocol=json.loads((Path(__file__).resolve().parents[2]/'configs/stage7/final_evaluation.json').read_text())
    protocol['bootstrap']['resamples']=9
    rows_by_model={}
    reference=[]
    for model_id, slug in fe.MODEL_SLUGS.items():
        rs=[]
        for article in range(24):
            for method, point in protocol['operating_points']:
                for repeat in range(3 if model_id==fe.LLAMA_ID else 2):
                    for label in (0,1):
                        rs.append({'record_id':f'{slug}:{article}:{method}:{point}:{repeat}:{label}',
                                   'article_id':f'article-{article}',
                                   'pair_id':f'{slug}:{article}:{method}:{point}:{repeat}',
                                   'method_id':method,'method_point_id':point,
                                   'model_id':model_id,'label':label,'text':f'some {label} text'})
        rows_by_model[model_id]=rs
        joblib=tmp_path/slug/'tfidf_logreg.joblib'
        joblib.parent.mkdir(parents=True, exist_ok=True)
        joblib.write_bytes(b'fake-but-frozen')
        reference.append({'model_id':model_id,
                          'baseline':{'joblib_sha256':fe.file_sha(joblib),'threshold':.5},
                          'roberta_e4':{'threshold':.5}})
    monkeypatch.setattr(runner,'_baseline_scores',
        lambda rows,path,lock_meta:[.2 if r['label']==0 else .8 for r in rows])
    monkeypatch.setattr(runner,'_roberta_scores',
        lambda rows,model_dir,**kwargs:([.3 if r['label']==0 else .7 for r in rows],
                                        [False]*len(rows)))
    report,pred=runner._evaluate(rows_by_model,lock={'selected':reference,
                            'dataset_sha256':fe.FROZEN_DATASET_SHA256,
                            'protocol_sha256':'test'}, protocol=protocol,model_root=tmp_path)
    assert report['point_rows']==60 and report['pooled_rows']==4
    assert len(report['results'])==64 and len(pred)==7200
    assert all(r['roc_auc']==1.0 for r in report['results'])
    assert all(r['ci95_lower']==1.0 for r in report['results'])
    assert all(r['confusion']['fp']==0 for r in report['results'])
    assert {r['detector'] for r in report['results']}=={'tfidf_logreg','roberta_e4'}
