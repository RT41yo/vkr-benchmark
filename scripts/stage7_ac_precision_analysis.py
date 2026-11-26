#!/usr/bin/env python3
"""Analyze the frozen Stage-7 AC precision runs without inference or test access."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import mean, median
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLAN_SHA = "3bba2773c2e418bb1a718d06b48bd6b39ca0a7eb3af0a13839b3351a5e8f7b0e"
MODELS = ("meta-llama/Llama-3.2-3B", "Qwen/Qwen3-4B-Base")
PRECISIONS = (16, 32, 48)
METRICS = ("kl_stego_to_ref_mean_bits", "kl_stego_to_ref_median_bits",
           "tvd_mean", "tvd_median", "bits_per_token", "entropy_utilization",
           "ppl_raw_lm", "reference_entropy_mean_bits", "ber")
FILES = ("config.json", "result.json", "token_ids.json")

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def json_bytes(obj):
    return (json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2,
                       allow_nan=False) + "\n").encode()

def canonical_id(config):
    return sha(json.dumps(config, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), allow_nan=False).encode())[:16]

def common_prefix(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                min(len(a), len(b)))

def verify_plan(raw):
    require(sha(raw) == PLAN_SHA, "AC plan SHA-256 mismatch")
    plan = json.loads(raw)
    require(plan["schema_version"] == "stage7.ac_precision_plan.v1"
            and plan["role"] == "validation_only_diagnostic", "unexpected plan")
    groups = plan["groups"]
    require(len(groups) == 16 and all(g["partition"] == "validation" for g in groups)
            and len({g["article_id"] for g in groups}) == 16
            and len({g["prompt_id"] for g in groups}) == 16, "invalid groups")
    expected = {(m, g["article_id"], p) for m in MODELS for g in groups for p in PRECISIONS}
    rows = plan["rows"]
    require(len(rows) == 96 and len({r["run_id"] for r in rows}) == 96
            and {(r["model_id"], r["article_id"], r["precision"]) for r in rows}
            == expected, "incomplete plan")
    group_prompts = {g["article_id"]:g["prompt_id"] for g in groups}
    for row in rows:
        c = row["config"]
        require(canonical_id(c) == row["run_id"]
                and c["model"]["id"] == row["model_id"]
                and c["prompt_id"] == row["prompt_id"] == group_prompts[row["article_id"]]
                and c["method"]["id"] == "arithmetic_coding"
                and c["method"]["params"] == {"precision":row["precision"],"top_k":2048}
                and c["termination"] == {"mode":"fixed_carrier_tokens","target_carrier_tokens":256},
                "plan identity/control mismatch")
    for m in MODELS:
        for g in groups:
            configs = [r["config"] for r in rows if r["model_id"] == m and r["article_id"] == g["article_id"]]
            normalized = []
            for c in configs:
                c = json.loads(json.dumps(c))
                del c["method"]["params"]["precision"]
                normalized.append(c)
            require(all(c == normalized[0] for c in normalized), "unpaired configuration")
    return plan

def inspect_run(row, docs):
    c,r,t = (docs[n] for n in FILES)
    rid = row["run_id"]
    require(c == row["config"] and r["run_id"] == rid, f"{rid}: config/result mismatch")
    # This analysis is locked to the observed complete all-ok cohort.
    # Fail closed rather than silently drop a hard failure.
    require(r["status"] == "ok", f"{rid}: unexpected hard failure; needs explicit audit")
    a,b = t["sender_token_ids"],t["receiver_token_ids"]
    prefix = common_prefix(a,b)
    exact = a == b
    mismatch = None if exact else prefix
    require(len(a) == r["carrier_tokens"] == 256 and
            len(b)-len(a) == t["token_count_delta"] and
            exact == t["token_sequence_roundtrip_exact"] == r["token_sequence_roundtrip_exact"]
            and mismatch == t["first_token_mismatch"] == r["first_token_roundtrip_mismatch"],
            f"{rid}: token transport inconsistency")
    require(r["q_mode"] == "analytic_exact" and
            r["kl_ref_to_stego_finite_steps"] + r["kl_ref_to_stego_infinite_steps"] == 256
            and r["kl_stego_to_ref_finite_steps"] + r["kl_stego_to_ref_infinite_steps"] == 256,
            f"{rid}: KL accounting mismatch")
    for direction in ("ref_to_stego","stego_to_ref"):
        val = r[f"kl_{direction}_mean_bits"]
        require((val == "inf" and r[f"kl_{direction}_infinite_steps"] > 0) or
                (isinstance(val,(int,float)) and math.isfinite(val)
                 and r[f"kl_{direction}_infinite_steps"] == 0), f"{rid}: KL mean mismatch")
    require(all(isinstance(r[k],(int,float)) and math.isfinite(r[k]) for k in METRICS),
            f"{rid}: nonfinite metric")
    require(r["roundtrip_exact"] == (r["decoder_complete"] and r["bit_errors"] == 0
            and r["expected_length_bits"] == r["recovered_length_bits"]),
            f"{rid}: secret roundtrip inconsistency")
    require(math.isclose(r["bits_per_token"],r["payload_bits"]/256,abs_tol=1e-12),
            f"{rid}: capacity mismatch")
    return dict(run_id=rid,model_id=row["model_id"],article_id=row["article_id"],
                prompt_id=row["prompt_id"],precision=row["precision"],
                secret_id=c["secret_id"],status=r["status"],
                roundtrip_exact=r["roundtrip_exact"],decoder_complete=r["decoder_complete"],
                token_sequence_roundtrip_exact=exact,first_token_mismatch=mismatch,
                bit_errors=r["bit_errors"],payload_bits=r["payload_bits"],
                expected_length_bits=r["expected_length_bits"],
                recovered_length_bits=r["recovered_length_bits"],
                kl_ref_to_stego_mean_bits=r["kl_ref_to_stego_mean_bits"],
                kl_ref_to_stego_infinite_steps=r["kl_ref_to_stego_infinite_steps"],
                kl_stego_to_ref_infinite_steps=r["kl_stego_to_ref_infinite_steps"],
                **{k:r[k] for k in METRICS})

def summarize(rows):
    result = []
    for m in MODELS:
        for p in PRECISIONS:
            g = [r for r in rows if r["model_id"] == m and r["precision"] == p]
            require(len(g) == 16, "incomplete stratum")
            result.append(dict(model_id=m,precision=p,n=len(g),
                exact_count=sum(r["roundtrip_exact"] for r in g),
                soft_failure_count=sum(not r["roundtrip_exact"] for r in g),
                token_exact_count=sum(r["token_sequence_roundtrip_exact"] for r in g),
                decoder_complete_count=sum(r["decoder_complete"] for r in g),
                kl_ref_to_stego_infinite_steps=sum(r["kl_ref_to_stego_infinite_steps"] for r in g),
                kl_ref_to_stego_infinite_run_count=sum(r["kl_ref_to_stego_mean_bits"]=="inf" for r in g),
                kl_stego_to_ref_infinite_steps=sum(r["kl_stego_to_ref_infinite_steps"] for r in g),
                metrics={k:dict(mean=mean(r[k] for r in g),median=median(r[k] for r in g))
                         for k in METRICS}))
    return result

def paired(rows, tokens, groups):
    # Same article draws across all comparisons; descriptive, not confirmatory.
    draws = np.random.default_rng(731).integers(0,16,size=(2000,16))
    idx = {(r["model_id"],r["article_id"],r["precision"]):r for r in rows}
    result=[]
    for m in MODELS:
        for p in (16,48):
            a=[idx[m,g["article_id"],p] for g in groups]
            b=[idx[m,g["article_id"],32] for g in groups]
            prefixes=[common_prefix(tokens[x["run_id"]],tokens[y["run_id"]]) for x,y in zip(a,b)]
            metrics={}
            for k in METRICS:
                d=np.array([x[k]-y[k] for x,y in zip(a,b)],dtype=float)
                ci=np.quantile(np.mean(d[draws],axis=1),[.025,.975])
                metrics[k]=dict(mean_delta=float(d.mean()),median_delta=float(np.median(d)),
                    lower_count=int((d<0).sum()),higher_count=int((d>0).sum()),
                    equal_count=int((d==0).sum()),mean_delta_ci95=[float(v) for v in ci])
            result.append(dict(model_id=m,precision=p,reference_precision=32,n=16,
                               common_prefix_median=median(prefixes),common_prefix_max=max(prefixes),
                               identical_sender_count=prefixes.count(256),metrics=metrics))
    return result

def csv_bytes(rows):
    buf=io.StringIO(newline="")
    writer=csv.DictWriter(buf,fieldnames=list(rows[0]),lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode()

def svg_bytes(summary, metric, label):
    # Deterministic standalone vector figure; median of per-run means.
    values=[[next(r["metrics"][metric]["median"] for r in summary
                  if r["model_id"]==m and r["precision"]==p) for p in PRECISIONS] for m in MODELS]
    ymax=max(max(v) for v in values)*1.15
    y=lambda v: 250-v/ymax*195
    xs=[135,340,545]
    out=['<svg xmlns="http://www.w3.org/2000/svg" width="680" height="340" viewBox="0 0 680 340">',
         '<rect width="680" height="340" fill="white"/>',
         '<g font-family="sans-serif" font-size="13" fill="#202b38">',
         f'<text x="340" y="26" text-anchor="middle" font-size="17">{label}</text>',
         '<text x="340" y="46" text-anchor="middle">Median of 16 per-run means</text>']
    for j in range(5):
        v=ymax*j/4
        out += [f'<line x1="90" x2="590" y1="{y(v):.2f}" y2="{y(v):.2f}" stroke="#d9dfe5"/>',
                f'<text x="82" y="{y(v)+4:.2f}" text-anchor="end">{v:.4f}</text>']
    for x,p in zip(xs,PRECISIONS):
        out.append(f'<text x="{x}" y="270" text-anchor="middle">{p}</text>')
    out.append('<text x="340" y="291" text-anchor="middle">Precision (bits)</text>')
    for i, (v, color, name) in enumerate(zip(values,["#27638e","#be6b28"],["Llama-3.2-3B","Qwen3-4B-Base"])):
        pts=" ".join(f"{x},{y(a):.2f}" for x,a in zip(xs,v))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for x,a in zip(xs,v):
            out.append(f'<circle cx="{x}" cy="{y(a):.2f}" r="4" fill="{color}"/>')
        x=170+i*190
        out += [f'<line x1="{x}" x2="{x+24}" y1="315" y2="315" stroke="{color}" stroke-width="3"/>',
                f'<text x="{x+31}" y="319">{name}</text>']
    return ("\n".join(out+["</g></svg>"])+"\n").encode()

def analyze(plan_path, runs):
    raw=plan_path.read_bytes()
    plan=verify_plan(raw)
    require({p.name for p in runs.iterdir() if p.is_dir()} ==
            {r["run_id"] for r in plan["rows"]}, "run directory set differs from plan")
    rows=[]; tokens={}; hashes=[]
    for row in plan["rows"]:
        rid=row["run_id"]
        docs={}
        for n in FILES:
            b=(runs/rid/n).read_bytes()
            docs[n]=json.loads(b)
            hashes.append({"run_id":rid,"file":n,"sha256":sha(b)})
        rows.append(inspect_run(row,docs))
        tokens[rid]=docs["token_ids.json"]["sender_token_ids"]
    summary=summarize(rows)
    comparisons=paired(rows,tokens,plan["groups"])
    out={"ac_precision_runs.csv":csv_bytes(rows),
         "ac_precision_kl.svg":svg_bytes(summary,"kl_stego_to_ref_mean_bits","KL(P_stego || P_ref), bits"),
         "ac_precision_tvd.svg":svg_bytes(summary,"tvd_mean","Total variation distance")}
    report=dict(schema_version="stage7.ac_precision_analysis.v1",status="verified",
        role="validation_only_diagnostic",test_access="none",inference_repeated=False,
        plan_sha256=sha(raw),run_count=len(rows),article_groups=16,source_file_count=len(hashes),
        source_inventory_sha256=sha(json_bytes(hashes)),source_inventory=hashes,
        exact_count=sum(r["roundtrip_exact"] for r in rows),
        soft_failure_count=sum(not r["roundtrip_exact"] for r in rows),hard_failure_count=0,
        token_mismatch_count=sum(not r["token_sequence_roundtrip_exact"] for r in rows),
        summary=summary,paired_vs_p32=comparisons,
        bootstrap=dict(seed=731,resamples=2000,unit="article_id",statistic="mean paired difference",
                       confidence=.95,method="percentile",role="exploratory_unadjusted"),
        derived_sha256={k:sha(v) for k,v in out.items()})
    out["ac_precision_analysis.json"]=json_bytes(report)
    return report,out

def write_once(output, artifacts):
    # Check all existing outputs before creating any.
    for name, raw in artifacts.items():
        p=output/name
        require(not p.exists() or p.read_bytes()==raw,f"refuse overwrite: {p}")
    output.mkdir(parents=True,exist_ok=True)
    for name, raw in artifacts.items():
        p=output/name
        if not p.exists():
            with p.open("xb") as f:
                f.write(raw)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-report",action="store_true")
    args=ap.parse_args()
    output=ROOT/"results/stage7/analysis"
    report, artifacts=analyze(output/"ac_precision_plan.json",
              ROOT/"data/stage7/cache/ac_precision/execution/runs")
    if args.write_report:
        write_once(output,artifacts)
    print(json.dumps({k:report[k] for k in ("status","plan_sha256","run_count","article_groups",
          "source_file_count","source_inventory_sha256","exact_count","soft_failure_count",
          "hard_failure_count","token_mismatch_count","test_access","inference_repeated",
          "derived_sha256")}|{"report_sha256":sha(artifacts["ac_precision_analysis.json"]),
          "report_written":args.write_report},indent=2))

if __name__=="__main__":
    main()
