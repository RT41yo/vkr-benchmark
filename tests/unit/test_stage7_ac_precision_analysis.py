from __future__ import annotations
import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import stage7_ac_precision_analysis as ac


def sample():
    c={"secret_id":"s7-example"}
    rid=ac.canonical_id(c)
    row=dict(run_id=rid,model_id=ac.MODELS[0],article_id="a",
             prompt_id="p",precision=32,config=c)
    ids=list(range(256))
    r={k:0.0 for k in ac.METRICS}
    r.update(run_id=rid,status="ok",carrier_tokens=256,q_mode="analytic_exact",
             kl_ref_to_stego_mean_bits="inf",kl_ref_to_stego_finite_steps=0,
             kl_ref_to_stego_infinite_steps=256,kl_stego_to_ref_finite_steps=256,
             kl_stego_to_ref_infinite_steps=0,token_sequence_roundtrip_exact=True,
             first_token_roundtrip_mismatch=None,roundtrip_exact=True,
             decoder_complete=True,bit_errors=0,expected_length_bits=256,
             recovered_length_bits=256,payload_bits=256,bits_per_token=1.0)
    t=dict(sender_token_ids=ids,receiver_token_ids=ids.copy(),token_count_delta=0,
           token_sequence_roundtrip_exact=True,first_token_mismatch=None)
    return row,dict(zip(ac.FILES,[c,r,t]))


def test_common_prefix():
    assert ac.common_prefix([1,2],[1,3])==1
    assert ac.common_prefix([1,2],[1])==1
    assert ac.common_prefix([],[])==0
    assert ac.common_prefix(list(range(256)),list(range(256)))==256


def test_plan_hash_rejects_changed_bytes():
    with pytest.raises(ValueError,match="SHA-256 mismatch"):
        ac.verify_plan(b"{}")


def test_inspect_exact_and_reverse_kl():
    row,docs=sample()
    r=ac.inspect_run(row,docs)
    assert r["roundtrip_exact"]
    assert r["kl_ref_to_stego_mean_bits"]=="inf"
    assert r["kl_ref_to_stego_infinite_steps"]==256


def test_soft_failure_retained():
    row,docs=sample()
    r,t=docs["result.json"],docs["token_ids.json"]
    t["receiver_token_ids"][3]=999
    t.update(token_sequence_roundtrip_exact=False,first_token_mismatch=3)
    r.update(token_sequence_roundtrip_exact=False,first_token_roundtrip_mismatch=3,
             roundtrip_exact=False,bit_errors=1,ber=1/256)
    out=ac.inspect_run(row,docs)
    assert not out["roundtrip_exact"]
    assert out["first_token_mismatch"]==3
    assert out["ber"]==1/256


def test_inconsistent_token_flags_rejected():
    row,docs=sample()
    docs["token_ids.json"]["receiver_token_ids"][2]=999
    with pytest.raises(ValueError,match="token transport inconsistency"):
        ac.inspect_run(row,docs)


def test_hard_failure_not_silently_dropped():
    row,docs=sample()
    docs["result.json"]["status"]="error"
    with pytest.raises(ValueError,match="hard failure"):
        ac.inspect_run(row,docs)


def test_paired_differences():
    rows=[]; tokens={}
    groups=[{"article_id":str(i)} for i in range(16)]
    for m in ac.MODELS:
        for g in groups:
            for p in ac.PRECISIONS:
                rid=f"{m}-{g['article_id']}-{p}"
                rows.append(dict(run_id=rid,model_id=m,precision=p,
                    article_id=g["article_id"],**{k:float(p) for k in ac.METRICS}))
                tokens[rid]=list(range(256))
    results=ac.paired(rows,tokens,groups)
    assert len(results)==4
    for r in results:
        v=r["precision"]-32
        assert r["metrics"]["tvd_mean"]["mean_delta"]==v
        assert r["metrics"]["tvd_mean"]["mean_delta_ci95"]==[v,v]
        assert r["identical_sender_count"]==16


def test_write_once_checks_all_before_writing(tmp_path):
    (tmp_path/"existing").write_bytes(b"a")
    with pytest.raises(ValueError,match="refuse overwrite"):
        ac.write_once(tmp_path,{"new":b"b","existing":b"c"})
    assert not (tmp_path/"new").exists()
    ac.write_once(tmp_path,{"new":b"b","existing":b"a"})
    ac.write_once(tmp_path,{"new":b"b","existing":b"a"})
    assert (tmp_path/"new").read_bytes()==b"b"
