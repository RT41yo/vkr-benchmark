#!/usr/bin/env python3
"""Read-only audit of frozen test scores; optional derived outputs, no inference."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
from html import escape
import io
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
    balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score)

REPORT_SHA = '9249515e6a4d2e49a71d0d7a50acbbfc0065a98f57533990f0c86d157e6987b9'
PRED_SHA = 'fc321b9f0d1dd2d7ad2c15376e19e6f872764beb20c426b58e39a83ed622f521'
STAGE6_SHA = '9b0324dcedd4c03cc0f13c6103f8825411b1a0086678f8eac4651f1c5e4f10a4'
LOCK_SHA = 'ba88ee49f92d7c12b64c7baad7329e10f6af463ab8b0857d2f27e900ed2654ce'
DATASET_SHA = '43ea3b2ded135a8004a2ada7f3c3897ee3873baf0cd9283408418f99298587d9'
PROTOCOL_SHA = '56bc75b01c208c85ea6b389d4410a654b186277be27176f467f3359dbfad9a4e'
MODELS = {'meta-llama/Llama-3.2-3B': ('Llama-3.2-3B', 3), 'Qwen/Qwen3-4B-Base': ('Qwen3-4B-Base', 2)}
DETECTORS = ('tfidf_logreg', 'roberta_e4')
POINTS = (('bins','b1'),('bins','b4'),('bins','b6'),('huffman','e2'),('huffman','e4'),('huffman','e8'),
          ('arithmetic_coding','p32-k128'),('arithmetic_coding','p32-k2048'),('arithmetic_coding','p32-k50000'),
          ('adg','default'),('discop','default'),('dairstega','a8-b1-k100'),('dairstega','a16-b075-k100'),
          ('dairstega','a32-b05-k100'),('rrc','payload1024-margin40'))
THRESHOLDS = {('meta-llama/Llama-3.2-3B','tfidf_logreg'): 0.5120935467373745,
              ('meta-llama/Llama-3.2-3B','roberta_e4'): 0.4882860481739044,
              ('Qwen/Qwen3-4B-Base','tfidf_logreg'): 0.5332555668581335,
              ('Qwen/Qwen3-4B-Base','roberta_e4'): 0.5310582518577576}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_bytes(path, expected):
    blob = path.read_bytes()
    require(sha256(blob).hexdigest() == expected, f'SHA-256 mismatch: {path}')
    return blob


def weighted_rank_auc(y, scores, groups, weights):
    """Weighted Mann–Whitney statistic via ascending score groups, half credit for ties.

    This is independent of the final evaluator's pairwise comparison matrix.
    Batch the draws to limit memory use.
    """
    y, scores, groups, weights = map(np.asarray, (y, scores, groups, weights))
    order = np.argsort(scores, kind='stable')
    y, scores, groups = y[order], scores[order], groups[order]
    starts = np.r_[0, np.flatnonzero(scores[1:] != scores[:-1]) + 1]
    result = []
    for i in range(0, len(weights), 128):
        w = weights[i:i+128, groups].astype(float)
        pos = np.add.reduceat(w * y, starts, axis=1)
        neg = np.add.reduceat(w * (1-y), starts, axis=1)
        denom = pos.sum(axis=1) * neg.sum(axis=1)
        require(np.all(denom > 0), 'bootstrap draw lacks one class')
        result.extend((pos * (np.cumsum(neg, axis=1) - .5*neg)).sum(axis=1) / denom)
    return np.asarray(result)


def metrics(rows, threshold, article_index, draws):
    y = np.array([r['label'] for r in rows]); s = np.array([r['score_stego'] for r in rows])
    g = np.array([article_index[r['article_id']] for r in rows]); pred = s >= threshold
    samples = weighted_rank_auc(y, s, g, draws)
    lo, hi = np.quantile(samples, [.025,.975]); p, recall, _ = precision_recall_curve(y,s)
    tn, fp, fn, tp = confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(roc_auc=float(roc_auc_score(y,s)), ci95_lower=float(lo), ci95_upper=float(hi),
        bootstrap_n=len(samples), record_count=len(rows), pair_count=len(rows)//2,
        article_group_count=len(set(g)), threshold_from_pooled_validation=threshold,
        accuracy=float(accuracy_score(y,pred)), balanced_accuracy=float(balanced_accuracy_score(y,pred)),
        precision=float(precision_score(y,pred,zero_division=0)), recall=float(recall_score(y,pred,zero_division=0)),
        f1=float(f1_score(y,pred,zero_division=0)), pr_auc_trapezoid=float(auc(recall,p)),
        average_precision=float(average_precision_score(y,s)),
        confusion=dict(tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp)))


def audit(report, predictions):
    require(report['status']=='complete' and report['dataset_sha256']==DATASET_SHA and
        report['lock_sha256']==LOCK_SHA and report['protocol_sha256']==PROTOCOL_SHA and
        report['predictions_sha256']==PRED_SHA and report['primary_model']=='E4_seed_713' and
        report['ablation_e8_used_for_test'] is False and report['threshold_selection']=='pooled_validation_only' and
        report['bootstrap']==dict(confidence=.95,method='percentile',resamples=2000,seed=731,shared_draws=True,unit='article_id') and
        (report['test_pair_count'],report['test_article_groups'],report['point_rows'],report['pooled_rows'],report['row_count'])==(1800,24,60,4,64),
        'unexpected report metadata')
    ids = {}; pairs = defaultdict(list); strata = defaultdict(list); support = Counter(); records = {}
    fields = ('article_id','model_id','method_id','method_point_id','pair_id','label','partition')
    require(len(predictions)==7200,'unexpected prediction count')
    for r in predictions:
        m,d,a,p,y = r['model_id'],r['detector'],r['article_id'],(r['method_id'],r['method_point_id']),r['label']
        score = r['score_stego']; rid = r['record_id']
        require(m in MODELS and d in DETECTORS and p in POINTS and type(y) is int and y in (0,1) and
            r['partition']=='test' and type(score) in (float,int) and math.isfinite(score) and 0<=score<=1 and
            all(isinstance(r[k],str) and r[k] for k in ('article_id','pair_id','record_id')), 'invalid prediction')
        require((d,rid) not in ids, 'duplicate detector-record')
        ids[d,rid]=True; meta=tuple(r[k] for k in fields)
        require(rid not in records or records[rid]==meta, 'metadata differs across detectors')
        records[rid]=meta; pairs[d,r['pair_id']].append(r)
        strata[m,d,*p].append(r); strata[m,d,None,None].append(r)
        support[m,d,*p,a,y]+=1
    articles=sorted({r['article_id'] for r in predictions}); require(len(articles)==24 and len(records)==3600 and len(pairs)==3600,'unexpected article/record/pair count')
    require(len({r['pair_id'] for r in predictions})==1800,'pair count mismatch')
    for rows in pairs.values():
        require(len(rows)==2 and {r['label'] for r in rows}=={0,1} and len({tuple(r[k] for k in ('article_id','model_id','method_id','method_point_id')) for r in rows})==1,'invalid cover/stego pair')
    for m,(_,n) in MODELS.items():
        for d in DETECTORS:
            for p in POINTS:
                for a in articles:
                    for y in (0,1):
                        require(support[m,d,*p,a,y]==n,'article/point support mismatch')
    rng=np.random.default_rng(731); positions=rng.integers(0,24,size=(2000,24))
    draws=np.stack([np.bincount(v,minlength=24) for v in positions]); ai={a:i for i,a in enumerate(articles)}
    seen=set(); max_error=0.; checked=0
    for row in report['results']:
        m,d,p,q=(row[k] for k in ('model_id','detector','method_id','method_point_id')); key=(m,d,p,q)
        require(key in strata and key not in seen and row['scope']==('pooled' if p is None else 'operating_point'),'unexpected report stratum')
        seen.add(key); calc=metrics(strata[key],THRESHOLDS[m,d],ai,draws)
        for k,v in calc.items():
            if isinstance(v,float):
                delta=abs(row[k]-v); require(math.isfinite(delta) and delta<=1e-10,f'{key}: {k} mismatch: {delta}'); max_error=max(max_error,delta)
            else:
                require(row[k]==v,f'{key}: {k} mismatch')
            checked+=1
    require(seen==set(strata) and len(seen)==64,'missing report strata')
    return dict(status='verified', report_sha256=REPORT_SHA,predictions_sha256=PRED_SHA,lock_sha256=LOCK_SHA,
        dataset_sha256=DATASET_SHA, protocol_sha256=PROTOCOL_SHA, prediction_count=len(predictions), record_count=len(records),
        pair_count=1800,article_groups=len(articles),strata_verified=len(seen),metric_fields_verified=checked,
        bootstrap_intervals_verified=len(seen),max_absolute_numeric_difference=max_error,absolute_tolerance=1e-10,
        bootstrap_algorithm='weighted_tie_aware_score_groups',test_access='stored_predictions_only',
        length_diagnostics='copied_from_original_report_not_independently_recomputed',inference_repeated=False)


def join_stage6(report, stage6):
    by={(r['model_id'],r['method_id'],r['method_point_id'],r['detector']):r for r in report['results'] if r['scope']=='operating_point'}
    models={v[0]:k for k,v in MODELS.items()}; seen=set(); joined=[]
    for r in stage6:
        m=models[r['model']]; p=(r['method_id'],r['method_point_id']); key=(m,*p)
        require(p in POINTS and key not in seen and int(r['n_total'])==MODELS[m][1]*24,'Stage 6 key/support mismatch'); seen.add(key)
        out=dict(r); out['model_id']=m
        require(by[m,*p,DETECTORS[0]]['length_diagnostics']==by[m,*p,DETECTORS[1]]['length_diagnostics'], 'length diagnostics differ across detectors')
        for d in DETECTORS:
            a=by[m,*p,d]
            for k in ('roc_auc','ci95_lower','ci95_upper','balanced_accuracy','precision','recall','f1','average_precision','pr_auc_trapezoid'):
                out[f'{d}_{k}']=a[k]
            for c in ('cover','stego'):
                for k,v in a['length_diagnostics'][c].items():
                    out[f'{c}_{k}']=v
        joined.append(out)
    require(seen=={(m,*p) for m in MODELS for p in POINTS},'missing Stage 6 point')
    return joined


def correlations(joined):
    result=[]
    cols=('kl_stego_to_ref_mean_bits_median','tvd_mean_median','ppl_raw_lm_median','bits_per_token_median',
          'kl_ref_to_stego_infinite_rate_ok','kl_ref_to_stego_finite_median_bits')
    for m in MODELS:
        for d in DETECTORS:
            for col in cols:
                xy=[(float(r[col]),float(r[f'{d}_roc_auc'])) for r in joined if r['model_id']==m and r[col]!='' and math.isfinite(float(r[col]))]
                x,y=zip(*xy) if xy else ([],[]); rho=None
                if len(x)>=3 and len(set(x))>1 and len(set(y))>1:
                    rho=float(spearmanr(x,y).statistic)
                result.append(dict(model_id=m,detector=d,stage6_metric=col,n=len(x),spearman_rho=rho,
                    status='finite_subset_only' if col=='kl_ref_to_stego_finite_median_bits' else 'descriptive'))
    return result


def forest_svg(report, model):
    rows={(r['method_id'],r['method_point_id'],r['detector']):r for r in report['results'] if r['model_id']==model and r['scope']=='operating_point'}
    w,h=1000,700; left,right=250,905; x=lambda v:left+(right-left)*v
    s=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
       '<rect width="100%" height="100%" fill="white"/><g font-family="DejaVu Sans, sans-serif" fill="#172536">',
       f'<text x="25" y="32" font-size="21" font-weight="bold">{escape(MODELS[model][0])} — frozen test ROC-AUC</text>',
       '<text x="25" y="56" font-size="13">95% article-cluster bootstrap CI; 24 articles; raw score orientation</text>']
    for t in (0,.2,.4,.5,.6,.8,1):
        s.append(f'<line x1="{x(t)}" x2="{x(t)}" y1="85" y2="628" stroke="{ "#5c6873" if t==.5 else "#e0e5e9"}" stroke-dasharray="{ "5 4" if t==.5 else "none"}"/><text x="{x(t)}" y="648" font-size="12" text-anchor="middle">{t:g}</text>')
    for i,p in enumerate(POINTS):
        y=100+i*35; s.append(f'<text x="25" y="{y+4}" font-size="13">{escape(" / ".join(p))}</text>')
        for d,c,dy in [('tfidf_logreg','#bc7330',-6),('roberta_e4','#246c9b',6)]:
            a=rows[*p,d]; yy=y+dy; l,v,u=(x(a[k]) for k in ('ci95_lower','roc_auc','ci95_upper'))
            s.append(f'<path d="M {l} {yy} H {u} M {l} {yy-4} V {yy+4} M {u} {yy-4} V {yy+4}" fill="none" stroke="{c}" stroke-width="2"/><circle cx="{v}" cy="{yy}" r="4" fill="{c}"/>')
    s.append('<circle cx="260" cy="676" r="5" fill="#246c9b"/><text x="272" y="680" font-size="13">RoBERTa E4</text><circle cx="440" cy="676" r="5" fill="#bc7330"/><text x="452" y="680" font-size="13">TF-IDF + logistic regression</text></g></svg>\n')
    return ''.join(s)


def json_bytes(obj):
    return (json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n').encode()


def run(root, write):
    base=root/'results/stage7'; raw=checked_bytes(base/'predictions/raw/final_test_predictions.jsonl',PRED_SHA)
    report=json.loads(checked_bytes(base/'final_evaluation_report.json',REPORT_SHA))
    require(raw.endswith(b'\n') and b'\n\n' not in raw,'invalid JSONL framing')
    preds=[json.loads(line) for line in raw.split(b'\n')[:-1]]
    stage6=list(csv.DictReader(io.StringIO(checked_bytes(root/'results/stage6/stage6_operating_point_summary.csv',STAGE6_SHA).decode())))
    # The lock is not present in the source-only pre-lock snapshot; require it on the local repository.
    if (base/'final_evaluation_lock.json').is_file():
        lock=json.loads(checked_bytes(base/'final_evaluation_lock.json',LOCK_SHA))
        require(lock['dataset_sha256']==DATASET_SHA and lock['protocol_sha256']==PROTOCOL_SHA and
                lock['test_access']=='none' and lock['status']=='sealed' and
                len(lock['selected'])==2 and {r['model_id'] for r in lock['selected']}==set(MODELS), 'invalid lock metadata')
        for row in lock['selected']:
            m=row['model_id']
            require(row['baseline']['threshold']==THRESHOLDS[m,'tfidf_logreg'] and
                    row['roberta_e4']['threshold']==THRESHOLDS[m,'roberta_e4'] and
                    (row['roberta_e4']['seed'],row['roberta_e4']['epoch'])==(713,4), 'lock selection mismatch')
        lock_verified=True
    else:
        lock_verified=False
    checked_bytes(root/'configs/stage7/final_evaluation.json',PROTOCOL_SHA)
    result=audit(report,preds); joined=join_stage6(report,stage6)
    result.update(stage6_summary_sha256=STAGE6_SHA,joined_operating_points=len(joined),correlations=correlations(joined),lock_file_verified=lock_verified)
    buf=io.StringIO(newline=''); writer=csv.DictWriter(buf,fieldnames=list(joined[0])); writer.writeheader(); writer.writerows(joined)
    artifacts={'stage6_stage7_operating_points.csv':buf.getvalue().encode(),
               **{f'roc_auc_{v[0].lower()}.svg':forest_svg(report,m).encode() for m,v in MODELS.items()}}
    result['derived_sha256']={name:sha256(data).hexdigest() for name,data in artifacts.items()}
    artifacts['final_evaluation_audit.json']=json_bytes(result)
    if write:
        require(lock_verified,'local pre-test lock file required to write audit outputs')
        out=base/'analysis'
        for name,data in artifacts.items():
            p=out/name; require(not p.exists() or p.read_bytes()==data,f'refusing to overwrite differing artifact: {p}')
        out.mkdir(parents=True,exist_ok=True)
        for name,data in artifacts.items():
            p=out/name
            if not p.exists():
                with p.open('xb') as f: f.write(data)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]); parser.add_argument('--write-report',action='store_true'); args=parser.parse_args()
    result=run(args.root,args.write_report)
    print(json.dumps({k:v for k,v in result.items() if k!='correlations'},indent=2,sort_keys=True))

if __name__=='__main__':
    main()
