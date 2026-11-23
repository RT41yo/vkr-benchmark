#!/usr/bin/env python3
"""Train one pinned RoBERTa-base seed on Stage-7 frozen train; select epoch on validation.

The external test split is never supplied to this process. Run the three frozen
seeds per LM separately; combine validation summaries before opening test.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vkr_benchmark.errors import ConfigurationError  # noqa: E402
from vkr_benchmark.steganalysis.detectors import (  # noqa: E402
    FROZEN_DATASET_SHA256, LLAMA_ID, QWEN_ID,
    choose_validation_threshold, load_detector_training_data,
)
from prepare_stage7_roberta_base import verify_snapshot  # noqa: E402


def _slug(model_id: str) -> str:
    return 'llama-3.2-3b' if model_id == LLAMA_ID else 'qwen3-4b-base'


def _write(path: Path, data: object) -> None:
    path.write_text(json.dumps(data,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')


def _validation_scores(model, loader, device, torch):
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in loader:
            label = batch.pop('labels')
            del label
            batch = {k:v.to(device) for k,v in batch.items()}
            logits = model(**batch).logits
            scores.extend(torch.softmax(logits.float(),dim=-1)[:,1].cpu().tolist())
    return scores


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-id',required=True,choices=[LLAMA_ID,QWEN_ID])
    p.add_argument('--seed',required=True,type=int,choices=[711,712,713])
    p.add_argument('--snapshot',required=True,type=Path,help='Pinned local RoBERTa snapshot')
    p.add_argument('--base-report',type=Path,default=ROOT/'results/stage7/checkpoints/roberta_base_snapshot.json')
    p.add_argument('--dataset',type=Path,default=ROOT/'data/stage7/dataset/detector_records.jsonl')
    p.add_argument('--manifest',type=Path,default=ROOT/'results/stage7/dataset_manifest.json')
    p.add_argument('--config',type=Path,default=ROOT/'configs/stage7/detector_training.json')
    p.add_argument('--output-root',type=Path,default=ROOT/'results/stage7/checkpoints/roberta')
    p.add_argument('--model-root',type=Path,default=ROOT/'models/detectors/stage7')
    p.add_argument('--dry-run',action='store_true',help='Validate frozen data, config, snapshot; no GPU or training')
    a = p.parse_args()
    try:
        cfg_raw = a.config.read_bytes()
        cfg = json.loads(cfg_raw)
        frozen = {'base_model':'FacebookAI/roberta-base', 'max_length':512,
            'truncation':'right', 'dynamic_padding':True, 'learning_rate':0.00002,
            'weight_decay':0.01, 'max_epochs':4, 'warmup_ratio':0.1,
            'microbatch':4, 'gradient_accumulation':8,
            'training_seeds':[711,712,713], 'early_stopping_patience':2}
        params = cfg.get('roberta',{})
        revision = params.get('base_revision')
        if (cfg.get('schema_version') != 'stage7.detector_training.v1' or
                cfg.get('dataset_sha256') != FROZEN_DATASET_SHA256 or
                {k:v for k,v in params.items() if k != 'base_revision'} != frozen or
                not isinstance(revision,str) or re.fullmatch(r'[0-9a-f]{40}',revision) is None):
            raise ConfigurationError('frozen RoBERTa configuration requires a pinned 40-hex revision')
        report = json.loads(a.base_report.read_text(encoding='utf-8'))
        if report.get('revision') != revision:
            raise ConfigurationError('RoBERTa revision differs from preflight manifest')
        verify_snapshot(a.snapshot,report)
        data = load_detector_training_data(dataset=a.dataset,manifest=a.manifest,
                    model_id=a.model_id,expected_sha256=FROZEN_DATASET_SHA256)
        slug = _slug(a.model_id)
        result_dir = a.output_root/slug/f'seed-{a.seed}'
        model_dir = a.model_root/slug/'roberta'/f'seed-{a.seed}'
        if result_dir.exists() or model_dir.exists():
            raise ConfigurationError('detector output already exists; refuse overwrite')
        info = {'status':'preflight_ok','model_id':a.model_id,'seed':a.seed,
                'base_revision':revision,'dataset_sha256':data.dataset_sha256,
                'base_report_sha256':sha256(a.base_report.read_bytes()).hexdigest(),
                'configuration_sha256':sha256(cfg_raw).hexdigest(),
                'counts':data.counts,'test_access':'none'}
        if a.dry_run:
            print(json.dumps(info,indent=2,sort_keys=True))
            return 0
        try:
            import numpy as np
            import torch
            from sklearn.metrics import roc_auc_score
            from torch.utils.data import DataLoader
            from transformers import (AutoTokenizer,AutoModelForSequenceClassification,
                                      DataCollatorWithPadding,get_linear_schedule_with_warmup)
        except ImportError as exc:
            raise ConfigurationError('Stage-7 RoBERTa requires numpy, torch, scikit-learn, transformers') from exc
        random.seed(a.seed)
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(a.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends.cuda.matmul,'allow_tf32'):
            torch.backends.cuda.matmul.allow_tf32 = False
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = AutoTokenizer.from_pretrained(str(a.snapshot),local_files_only=True,use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(str(a.snapshot),
            local_files_only=True,num_labels=2)
        model.to(device)
        train = list(data.train)
        validation = list(data.validation)
        # Tokenize carrier text only, never structured metadata, prompt or token IDs.
        def encode(rows):
            return [dict(tokenizer(row['text'],max_length=512,truncation=True,
                                   add_special_tokens=True),labels=row['label']) for row in rows]
        collator = DataCollatorWithPadding(tokenizer=tokenizer,padding=True,return_tensors='pt')
        train_encoded = encode(train)
        val_encoded = encode(validation)
        gen = torch.Generator()
        gen.manual_seed(a.seed)
        train_loader = DataLoader(train_encoded,batch_size=4,shuffle=True,collate_fn=collator,
                                  generator=gen)
        val_loader = DataLoader(val_encoded,batch_size=4,shuffle=False,collate_fn=collator)
        optimizer = torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=0.01)
        steps_per_epoch = math.ceil(len(train_loader)/8)
        total_steps = steps_per_epoch*4
        warmup_steps = math.ceil(total_steps*0.1)
        scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)
        val_y = [row['label'] for row in validation]
        best_auc = -1.0
        best_epoch = None
        best_scores = None
        patience = 0
        epochs = []
        for epoch in range(1,5):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            batches = len(train_loader)
            for step,batch in enumerate(train_loader):
                batch = {k:v.to(device) for k,v in batch.items()}
                logits = model(**batch)
                group_size = min(8, batches - (step//8)*8)
                (logits.loss/group_size).backward()
                total_loss += float(logits.loss.detach().cpu())
                if (step+1)%8 == 0 or step+1 == batches:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            scores = _validation_scores(model,val_loader,device,torch)
            auc = float(roc_auc_score(val_y,scores))
            epochs.append({'epoch':epoch,'validation_roc_auc':auc,
                           'mean_batch_loss':total_loss/batches})
            print(f'Stage-7 RoBERTa {slug} seed={a.seed} epoch={epoch}: validation ROC-AUC={auc:.6f}',
                  flush=True)
            if auc > best_auc + 1e-12:
                best_auc,best_epoch,best_scores = auc,epoch,scores
                patience = 0
                model_dir.mkdir(parents=True,exist_ok=True)
                model.save_pretrained(model_dir,safe_serialization=True)
                tokenizer.save_pretrained(model_dir)
            else:
                patience += 1
                if patience >= 2:
                    break
        if best_scores is None:
            raise ConfigurationError('RoBERTa validation never produced scores')
        threshold,bacc = choose_validation_threshold(val_y,best_scores)
        summary = {**info,'status':'complete','best_epoch':best_epoch,
                   'validation_roc_auc':best_auc,'validation_threshold':threshold,
                   'validation_balanced_accuracy':bacc,'history':epochs,
                   'optimizer':'AdamW','microbatch':4,'gradient_accumulation':8,
                   'effective_batch':32,'max_epochs':4,'warmup_steps':warmup_steps,
                   'total_scheduled_steps':total_steps,'device':str(device),
                   'torch_version':torch.__version__,'transformers_base_revision':revision}
        result_dir.mkdir(parents=True,exist_ok=False)
        _write(result_dir/'summary.json',summary)
        predictions = [{'record_id':row['record_id'],'pair_id':row['pair_id'],
                         'label':row['label'],'partition':'validation','score_stego':score}
                       for row,score in zip(validation,best_scores)]
        (result_dir/'validation_predictions.jsonl').write_text(
            ''.join(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n'
                    for row in predictions),encoding='utf-8')
        print(json.dumps(summary,ensure_ascii=False,indent=2,sort_keys=True))
        return 0
    except (OSError,ValueError,ConfigurationError) as exc:
        print(f'Stage-7 RoBERTa training: ERROR ({type(exc).__name__}: {exc})',file=sys.stderr)
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
