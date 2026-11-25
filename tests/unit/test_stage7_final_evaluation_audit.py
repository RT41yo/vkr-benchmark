from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import audit_stage7_final_evaluation as audit


def test_weighted_rank_auc_matches_literal_resampling_with_ties():
    y = np.array([0,1,0,1,0,1,0,1])
    s = np.array([.3,.3,.2,.9,.4,.7,.9,.1])
    g = np.array([0,0,1,1,1,1,2,2])
    draws = np.array([[1,1,1],[3,0,0],[0,2,1],[0,0,3],[2,1,0]])
    expected = [roc_auc_score(np.repeat(y,w[g]),np.repeat(s,w[g])) for w in draws]
    np.testing.assert_allclose(audit.weighted_rank_auc(y,s,g,draws),expected,rtol=0,atol=1e-14)


def test_rank_bootstrap_rejects_missing_class():
    with pytest.raises(ValueError,match='one class'):
        audit.weighted_rank_auc([0,1],[.2,.8],[0,1],[[2,0]])


def test_source_hash_guard(tmp_path):
    p=tmp_path/'input.json';p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='SHA-256 mismatch'):
        audit.checked_bytes(p,audit.REPORT_SHA)


def test_correlations_do_not_impute_infinite_kl():
    m=next(iter(audit.MODELS)); rows=[]
    for i in range(15):
        rows.append(dict(model_id=m,roberta_e4_roc_auc=i/15,tfidf_logreg_roc_auc=i/15,
            kl_stego_to_ref_mean_bits_median=str(i),tvd_mean_median=str(i),ppl_raw_lm_median=str(i),
            bits_per_token_median=str(i),kl_ref_to_stego_infinite_rate_ok=str(int(i>=3)),
            kl_ref_to_stego_finite_median_bits=str(i) if i<3 else ''))
    out=audit.correlations(rows)
    v=next(r for r in out if r['model_id']==m and r['detector']=='roberta_e4' and r['stage6_metric']=='kl_ref_to_stego_finite_median_bits')
    assert v['n']==3 and v['spearman_rho']==pytest.approx(1) and v['status']=='finite_subset_only'
