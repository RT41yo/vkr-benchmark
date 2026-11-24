from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))

import audit_stage7_roberta_primary as audit
import train_stage7_roberta_epochs8 as e8
import compare_stage7_roberta_epochs8 as compare
from vkr_benchmark.errors import ConfigurationError
from vkr_benchmark.steganalysis.detectors import LLAMA_ID, QWEN_ID


def _write(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True) + '\n', encoding='utf-8')


def _records(tmp_path: Path):
    rows=[]
    for split in ('train','validation','test'):
        for model, slug in ((LLAMA_ID,'llama'),(QWEN_ID,'qwen')):
            pair=f'{split}-{slug}'
            for label, kind in ((0,'cover'),(1,'stego')):
                text=f'{split} {kind} text\r\n'
                rows.append({
                    'record_id':f'{pair}-{kind}', 'pair_id':pair,
                    'article_id':f'{split}-article', 'prompt_id':f'{split}-prompt',
                    'model_id':model, 'method_id':'bins', 'method_point_id':'b1',
                    'replicate_id':'r01', 'partition':split, 'label':label,
                    'text':text, 'text_sha256':sha256(text.encode()).hexdigest(),
                })
    content = (''.join(json.dumps(row, sort_keys=True)+'\n' for row in rows)).encode('utf-8')
    dataset=tmp_path/'detector_records.jsonl';dataset.write_bytes(content)
    manifest=tmp_path/'dataset_manifest.json'
    counts={k:sum(r['partition']==k for r in rows) for k in ('train','validation','test')}
    _write(manifest,{
        'schema_version':'stage7.detector_dataset.v1','status':'complete_with_documented_nontermination',
        'excluded_nontermination_run_ids':['0b0e22af25613edb'],
        'sample_count':len(rows),'pair_count':len(rows)//2,
        'partition_counts':counts,
        'per_model_counts':{LLAMA_ID:6,QWEN_ID:6},
        'class_counts':{'cover':6,'stego':6},
        'dataset_sha256':sha256(content).hexdigest(),
    })
    return dataset,manifest,sha256(content).hexdigest()


def _primary_fixture(tmp_path: Path):
    dataset, manifest, digest = _records(tmp_path)
    revision='a'*40
    config=tmp_path/'primary_config.json'
    _write(config,{'schema_version':'stage7.detector_training.v1','dataset_sha256':digest,
                   'roberta': {'base_revision':revision,'max_epochs':4,'training_seeds':[711,712,713]}})
    base=tmp_path/'base.json'
    _write(base,{'revision':revision,'status':'pinned'})
    results=tmp_path/'primary_runs'
    models=tmp_path/'primary_models'
    for model_id, slug in ((LLAMA_ID,'llama-3.2-3b'),(QWEN_ID,'qwen3-4b-base')):
        for seed in (711,712,713):
            result=results/slug/f'seed-{seed}'
            _write(result/'summary.json', {
                'status':'complete','model_id':model_id,'seed':seed,
                'dataset_sha256':digest, 'configuration_sha256':audit.file_hash(config),
                'base_report_sha256':audit.file_hash(base),'base_revision':revision,
                'test_access':'none','max_epochs':4,
                'counts':{'train':2,'validation':2},
                'microbatch':4,'gradient_accumulation':8,'effective_batch':32,
                'best_epoch':1,'history':[{'epoch':1,'validation_roc_auc':1.0}],
                'validation_roc_auc':1.0,'validation_threshold':0.9,
                'validation_balanced_accuracy':1.0,
            })
            preds=[]
            for kind,score in (('cover',0.1),('stego',0.9)):
                preds.append({'record_id':f'validation-{('llama' if model_id==LLAMA_ID else 'qwen')}-{kind}',
                              'pair_id':f'validation-{('llama' if model_id==LLAMA_ID else 'qwen')}',
                              'label':int(kind=='stego'),'partition':'validation','score_stego':score})
            (result/'validation_predictions.jsonl').write_text(
                ''.join(json.dumps(x)+'\n' for x in preds),encoding='utf-8')
            mroot=models/slug/'roberta'/f'seed-{seed}'
            mroot.mkdir(parents=True)
            for filename in audit.REQUIRED_FILES:
                (mroot/filename).write_bytes(b'asset-'+filename.encode())
    report=audit.audit_primary(dataset=dataset,manifest=manifest,config=config,
        base_report=base,results_root=results,model_root=models,
        frozen_sha256=digest,frozen_structure=False)
    report_path=tmp_path/'primary_audit.json'; _write(report_path,report)
    return dataset,manifest,digest,config,base,results,models,report_path,report


def test_primary_audit_full_six_runs_recomputes_validation_and_hashes(tmp_path):
    *_, report = _primary_fixture(tmp_path)
    assert report['detector_count']==6
    assert report['test_access']=='none'
    assert {r['model_id'] for r in report['runs']}=={LLAMA_ID,QWEN_ID}
    assert all(r['validation_roc_auc']==1.0 for r in report['runs'])
    assert all(len(r['checkpoint_asset_sha256'])==6 for r in report['runs'])


def test_primary_audit_rejects_modified_validation_prediction(tmp_path):
    dataset,manifest,digest,config,base,results,models,*_ = _primary_fixture(tmp_path)
    path=results/'llama-3.2-3b'/'seed-711'/'validation_predictions.jsonl'
    corrupted=path.read_text().replace('0.9','0.2')
    path.write_text(corrupted)
    with pytest.raises(ConfigurationError,match='validation ROC-AUC|validation threshold|balanced accuracy'):
        audit.audit_primary(dataset=dataset,manifest=manifest,config=config,
            base_report=base,results_root=results,model_root=models,
            frozen_sha256=digest,frozen_structure=False)


def test_primary_audit_rejects_missing_weight_file(tmp_path):
    dataset,manifest,digest,config,base,results,models,*_ = _primary_fixture(tmp_path)
    (models/'qwen3-4b-base'/'roberta'/'seed-712'/'model.safetensors').unlink()
    with pytest.raises(ConfigurationError,match='missing or empty checkpoint file'):
        audit.audit_primary(dataset=dataset,manifest=manifest,config=config,
            base_report=base,results_root=results,model_root=models,
            frozen_sha256=digest,frozen_structure=False)


def _ablation_fixture(tmp_path, monkeypatch):
    dataset,manifest,digest,config,base,primary_results,models,audit_path,primary_report = _primary_fixture(tmp_path)
    experiment=dict(e8.EXPECTED_ABLATION)
    experiment.update(dataset_sha256=digest,primary_configuration_sha256=audit.file_hash(config),
                      base_report_sha256=audit.file_hash(base),base_revision='a'*40)
    exp_path=tmp_path/'e8_config.json';_write(exp_path,experiment)
    monkeypatch.setattr(e8,'PINNED_PRIMARY_CONFIG',audit.file_hash(config))
    monkeypatch.setattr(e8,'PINNED_BASE_REPORT',audit.file_hash(base))
    monkeypatch.setattr(e8,'EXPECTED_ABLATION',experiment)
    monkeypatch.setattr(e8,'FROZEN_PRIMARY_ROBERTA',json.loads(config.read_text())['roberta'])
    monkeypatch.setattr(e8,'FROZEN_DATASET_SHA256',digest)
    monkeypatch.setattr(e8,'PINNED_REVISION','a'*40)
    monkeypatch.setattr(e8,'verify_snapshot',lambda snapshot,report: None)
    outputs=tmp_path/'e8_results'
    model_root=tmp_path/'e8_models'
    return (dataset,manifest,digest,config,base,primary_results,models,audit_path,
            primary_report,exp_path,experiment,outputs,model_root)


def test_e8_preflight_checks_protocol_and_never_opens_test(tmp_path,monkeypatch):
    ds,manifest,digest,config,base,_,_,audit_path,_,exp_path,_,outputs,models = _ablation_fixture(tmp_path,monkeypatch)
    info, data=e8.validate_ablation_preflight(cfg_path=config,ablation_path=exp_path,
        report_path=base,snapshot=tmp_path/'snapshot',audit_path=audit_path,
        results_root=outputs,model_root=models,dataset=ds,manifest=manifest,
        model_id=LLAMA_ID,seed=711,frozen_structure=False)
    assert info['max_epochs']==8
    assert info['test_access']=='none'
    assert len(data.train)==2 and len(data.validation)==2
    assert all(r['partition']!='test' for r in data.train+data.validation)
    assert not outputs.exists()


def test_e8_preflight_rejects_changed_primary_or_experiment_config(tmp_path,monkeypatch):
    ds,manifest,_,config,base,_,_,audit_path,_,exp_path,experiment,outputs,models = _ablation_fixture(tmp_path,monkeypatch)
    kwargs=dict(cfg_path=config,ablation_path=exp_path,report_path=base,
        snapshot=tmp_path/'snapshot',audit_path=audit_path,
        results_root=outputs,model_root=models,dataset=ds,manifest=manifest,
        model_id=QWEN_ID,seed=713,frozen_structure=False)
    changed=dict(experiment,max_epochs=9)
    _write(exp_path,changed)
    with pytest.raises(ConfigurationError,match='ablation configuration changed'):
        e8.validate_ablation_preflight(**kwargs)
    _write(exp_path,experiment)
    _write(config,{'corrupt':'changed'})
    with pytest.raises(ConfigurationError,match='primary Stage-7 detector configuration changed'):
        e8.validate_ablation_preflight(**kwargs)


def test_e8_preflight_rejects_primary_output_overlap(tmp_path,monkeypatch):
    ds,manifest,_,config,base,_,_,audit_path,_,exp_path,_,outputs,models = _ablation_fixture(tmp_path,monkeypatch)
    with pytest.raises(ConfigurationError,match='overlap primary checkpoints'):
        e8.validate_ablation_preflight(cfg_path=config,ablation_path=exp_path,
            report_path=base,snapshot=tmp_path/'snapshot',audit_path=audit_path,
            results_root=e8.ROOT/'results/stage7/checkpoints/roberta',model_root=models,
            dataset=ds,manifest=manifest,model_id=LLAMA_ID,seed=711,frozen_structure=False)


def test_e8_comparison_preserves_primary_without_test(tmp_path,monkeypatch):
    ds,manifest,digest,config,base,primary_results,_,audit_path,primary_report,exp_path,experiment,outputs,_ = _ablation_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(compare,'FROZEN_DATASET_SHA256',digest)
    monkeypatch.setattr(compare,'PINNED_PRIMARY_CONFIG',audit.file_hash(config))
    monkeypatch.setattr(compare,'PINNED_BASE_REPORT',audit.file_hash(base))
    monkeypatch.setattr(compare,'EXPECTED_ABLATION',experiment)
    for model,slug in ((LLAMA_ID,'llama-3.2-3b'),(QWEN_ID,'qwen3-4b-base')):
        for seed in (711,712,713):
            summary={
                'status':'complete','study_id':e8.E8_STUDY,'model_id':model,
                'seed':seed,'test_access':'none','device':'cuda','max_epochs':8,
                'dataset_sha256':digest,
                'primary_configuration_sha256':audit.file_hash(config),
                'base_report_sha256':audit.file_hash(base),
                'ablation_configuration_sha256':audit.file_hash(exp_path),
                'primary_audit_sha256':audit.file_hash(audit_path),
                'counts':{'train':2,'validation':2},'best_epoch':2,
                'history':[{'epoch':1,'validation_roc_auc':0.9},
                           {'epoch':2,'validation_roc_auc':0.93}],
                'validation_roc_auc':0.93,
            }
            _write(outputs/slug/f'seed-{seed}'/'summary.json',summary)
    result=compare.compare(primary_audit=audit_path,primary_results=primary_results,
                            e8_results=outputs,ablation_config=exp_path)
    assert result['status']=='validation_only_exploratory'
    assert result['primary_checkpoint_selection_changed'] is False
    assert len(result['models'])==2
    assert all(abs(row['ablation_minus_primary_mean_roc_auc']+.07)<1e-12 for row in result['models'])
    assert result['test_access']=='none'
    path=outputs/'qwen3-4b-base'/'seed-712'/'summary.json'
    bad=json.loads(path.read_text());bad['test_access']='yes';_write(path,bad)
    with pytest.raises(ConfigurationError,match='inconsistent primary/E8 pair'):
        compare.compare(primary_audit=audit_path,primary_results=primary_results,
                        e8_results=outputs,ablation_config=exp_path)
