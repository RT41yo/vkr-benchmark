from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))

from vkr_benchmark.errors import ConfigurationError
from vkr_benchmark.steganalysis.detectors import (
    LLAMA_ID,QWEN_ID,choose_validation_threshold,load_detector_training_data,
)


def fixture_data(tmp_path:Path):
    rows=[]
    for partition,article in (('train','a1'),('validation','a2'),('test','a3')):
        for label,role in ((0,'cover'),(1,'stego')):
            text=f'{role}\r\n{article} text'
            rows.append({'record_id':f'{article}-{role}','pair_id':article,
                         'article_id':article,'prompt_id':'p'+article,
                         'model_id':LLAMA_ID,'method_id':'bins',
                         'method_point_id':'b1','replicate_id':'r01',
                         'partition':partition,'label':label,'text':text,
                         'text_sha256':sha256(text.encode()).hexdigest()})
    blob=(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows)).encode()
    dataset=tmp_path/'records.jsonl'; dataset.write_bytes(blob)
    meta={'schema_version':'stage7.detector_dataset.v1','status':'complete_with_documented_nontermination',
          'excluded_nontermination_run_ids':['0b0e22af25613edb'],
          'sample_count':len(rows),'pair_count':3,
          'partition_counts':{'train':2,'validation':2,'test':2},
          'per_model_counts':{LLAMA_ID:6},'class_counts':{'cover':3,'stego':3},
          'dataset_sha256':sha256(blob).hexdigest()}
    manifest=tmp_path/'manifest.json'; manifest.write_text(json.dumps(meta))
    return dataset,manifest,meta


def load(tmp_path):
    dataset,manifest,meta=fixture_data(tmp_path)
    value=load_detector_training_data(dataset=dataset,manifest=manifest,model_id=LLAMA_ID,
                expected_sha256=meta['dataset_sha256'],require_frozen_structure=False)
    return value,dataset,manifest,meta


def test_detector_loader_keeps_test_sealed_and_preserves_carriage_returns(tmp_path):
    data,_,_,_=load(tmp_path)
    assert len(data.train)==2 and len(data.validation)==2
    assert all(r['partition']!='test' for r in data.train+data.validation)
    assert all('\r\n' in r['text'] for r in data.train+data.validation)
    assert data.counts=={'train':2,'validation':2}


def test_detector_loader_detects_changed_dataset_bytes(tmp_path):
    _,dataset,manifest,meta=load(tmp_path)
    dataset.write_bytes(dataset.read_bytes()+b' ')
    with pytest.raises(ConfigurationError,match='frozen SHA-256'):
        load_detector_training_data(dataset=dataset,manifest=manifest,model_id=LLAMA_ID,
            expected_sha256=meta['dataset_sha256'],require_frozen_structure=False)


def test_detector_loader_detects_broken_pair(tmp_path):
    _,dataset,manifest,_=load(tmp_path)
    rows=[json.loads(r) for r in dataset.read_text().splitlines()]
    rows[0]['label']=1
    raw=(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows)).encode()
    dataset.write_bytes(raw)
    meta=json.loads(manifest.read_text());meta['dataset_sha256']=sha256(raw).hexdigest()
    meta['class_counts']={'cover':2,'stego':4}
    manifest.write_text(json.dumps(meta))
    with pytest.raises(ConfigurationError,match='pairing'):
        load_detector_training_data(dataset=dataset,manifest=manifest,model_id=LLAMA_ID,
            expected_sha256=sha256(raw).hexdigest(),require_frozen_structure=False)


def test_detector_loader_detects_cross_split_article(tmp_path):
    _,dataset,manifest,_=load(tmp_path)
    rows=[json.loads(r) for r in dataset.read_text().splitlines()]
    rows[4]['article_id']='a1';rows[5]['article_id']='a1'
    raw=(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows)).encode()
    dataset.write_bytes(raw)
    meta=json.loads(manifest.read_text());meta['dataset_sha256']=sha256(raw).hexdigest()
    manifest.write_text(json.dumps(meta))
    with pytest.raises(ConfigurationError,match='article crosses'):
        load_detector_training_data(dataset=dataset,manifest=manifest,model_id=LLAMA_ID,
            expected_sha256=sha256(raw).hexdigest(),require_frozen_structure=False)


def test_validation_threshold_only_uses_validation_classes():
    t,score=choose_validation_threshold([0,0,1,1],[0.1,0.2,0.8,0.9])
    assert t==0.8 and score==1.0
    with pytest.raises(ConfigurationError,match='both validation classes'):
        choose_validation_threshold([1,1],[0.3,0.4])


def test_roberta_pinning_and_missing_weights(tmp_path):
    from prepare_stage7_roberta_base import manifest_for_snapshot,verify_snapshot
    for file in ('config.json','tokenizer.json','vocab.json','merges.txt','model.safetensors'):
        (tmp_path/file).write_text(file)
    sha='a'*40
    manifest=manifest_for_snapshot(tmp_path,sha)
    verify_snapshot(tmp_path,manifest)
    (tmp_path/'tokenizer.json').write_text('tampered')
    with pytest.raises(ConfigurationError,match='changed'):
        verify_snapshot(tmp_path,manifest)
    with pytest.raises(ConfigurationError,match='commit SHA'):
        manifest_for_snapshot(tmp_path,'main')


def test_roberta_selection_uses_auc_then_earlier_epoch_then_seed():
    from select_stage7_roberta_checkpoint import select_summary
    from vkr_benchmark.steganalysis.detectors import FROZEN_DATASET_SHA256
    base={'status':'complete','dataset_sha256':FROZEN_DATASET_SHA256,
          'test_access':'none','base_revision':'a'*40,'base_report_sha256':'b'*64,
          'configuration_sha256':'c'*64,'model_id':QWEN_ID,
          'validation_threshold':0.5,'validation_balanced_accuracy':0.6}
    rows=[{**base,'seed':711,'best_epoch':4,'validation_roc_auc':0.85},
          {**base,'seed':712,'best_epoch':2,'validation_roc_auc':0.85},
          {**base,'seed':713,'best_epoch':2,'validation_roc_auc':0.85}]
    result=select_summary(rows)
    assert result['selected_seed']==712 and result['selected_epoch']==2
    assert result['test_used_for_selection'] is False
    with pytest.raises(ConfigurationError,match='three frozen'):
        select_summary(rows[:2])


def test_baseline_trains_text_only_and_emits_validation_not_test(tmp_path, monkeypatch):
    import train_stage7_baseline as runner
    from vkr_benchmark.steganalysis.detectors import DetectorData
    cfg_path=Path(__file__).resolve().parents[2]/'configs/stage7/detector_training.json'
    records=[]
    for partition in ('train','validation'):
        for role,label,text in (
            ('cover',0,'ordinary plain natural ordinary ordinary story'),
            ('stego',1,'encoded hidden secret hidden hidden story')):
            for i in range(4):
                records.append({'record_id':f'{partition}-{role}-{i}',
                                'pair_id':f'{partition}-{i}', 'label':label,'text':text,
                                'partition':partition})
    fake=DetectorData(model_id=LLAMA_ID,
        train=tuple(r for r in records if r['partition']=='train'),
        validation=tuple(r for r in records if r['partition']=='validation'),
        dataset_sha256=runner.FROZEN_DATASET_SHA256,
        counts={'train':8,'validation':8})
    monkeypatch.setattr(runner,'load_detector_training_data',lambda **kwargs:fake)
    monkeypatch.setattr(sys,'argv',['train_stage7_baseline.py','--model-id',LLAMA_ID,
                      '--config',str(cfg_path),'--output-root',str(tmp_path/'runs'),
                      '--model-root',str(tmp_path/'models')])
    assert runner.main()==0
    summary=json.loads((tmp_path/'runs/llama-3.2-3b/summary.json').read_text())
    assert summary['validation_roc_auc']==1.0
    assert summary['test_access']=='none'
    assert summary['tfidf_feature_count']>0
    text=(tmp_path/'runs/llama-3.2-3b/validation_predictions.jsonl').read_text()
    assert len(text.splitlines())==8
    assert 'test' not in text
    assert (tmp_path/'models/llama-3.2-3b/tfidf_logreg.joblib').exists()
    # Re-running must never silently overwrite an already-fitted detector.
    assert runner.main()==1


def test_roberta_dry_run_requires_pinned_revision_and_no_gpu(tmp_path,monkeypatch):
    import train_stage7_roberta as runner
    from prepare_stage7_roberta_base import manifest_for_snapshot
    from vkr_benchmark.steganalysis.detectors import DetectorData
    snapshot=tmp_path/'snapshot';snapshot.mkdir()
    for name in ('config.json','tokenizer.json','vocab.json','merges.txt','model.safetensors'):
        (snapshot/name).write_text(name)
    rev='f'*40
    report=tmp_path/'snapshot_report.json'
    report.write_text(json.dumps(manifest_for_snapshot(snapshot,rev)))
    source_cfg=json.loads((Path(__file__).resolve().parents[2]/'configs/stage7/detector_training.json').read_text())
    config=tmp_path/'training.json';config.write_text(json.dumps(source_cfg))
    monkeypatch.setattr(sys,'argv',['train_stage7_roberta.py','--model-id',QWEN_ID,
                     '--seed','711','--snapshot',str(snapshot),'--base-report',str(report),
                     '--config',str(config),'--dry-run'])
    assert runner.main()==1  # null floating revision must be refused
    source_cfg['roberta']['base_revision']=rev
    config.write_text(json.dumps(source_cfg))
    data=DetectorData(model_id=QWEN_ID,train=({'text':'a','label':0},),
                      validation=({'text':'b','label':1},),
                      dataset_sha256=runner.FROZEN_DATASET_SHA256,
                      counts={'train':1,'validation':1})
    monkeypatch.setattr(runner,'load_detector_training_data',lambda **kwargs:data)
    assert runner.main()==0
