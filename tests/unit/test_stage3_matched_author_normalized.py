from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "stage3_matched_comparison_core.py"


def _module():
    spec = importlib.util.spec_from_file_location("stage3_matched_comparison_core", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config():
    return json.loads(
        (REPO_ROOT / "configs" / "reproducibility" / "matched_author_normalized.json").read_text(
            encoding="utf-8"
        )
    )


def test_frozen_config_validates() -> None:
    module = _module()
    module.validate_config(_config())


def test_config_rejects_metric_closeness_as_execution_gate() -> None:
    module = _module()
    config = _config()
    config["execution"]["metric_closeness_is_gate"] = True
    with pytest.raises(ValueError, match="closeness"):
        module.validate_config(config)


def test_rank_zero_secret_stream_matches_committed_figure3_hash() -> None:
    module = _module()
    bits = module.paper_sentence_bits(0, seed=1234, replicate=0, bit_count=16384)
    assert module.sha256_bits(bits) == "48dde28597585a7e92cc1921504f70607d714553a3fbc3b519cc7714b62db848"


def test_token_agreement_reports_exact_prefix_and_rate() -> None:
    module = _module()
    exact = module.token_agreement([1, 2, 3], [1, 2, 3])
    assert exact["exact"] is True
    assert exact["agreement_rate"] == 1.0
    assert exact["first_token_mismatch"] is None
    partial = module.token_agreement([1, 2, 3, 4], [1, 9, 3, 8])
    assert partial["exact"] is False
    assert partial["agreement_count"] == 2
    assert partial["agreement_rate"] == 0.5
    assert partial["common_prefix_tokens"] == 1
    assert partial["first_token_mismatch"] == 1


def _record(point: str, rank: int) -> dict:
    return {
        "point_id": point,
        "selection_rank": rank,
        "carrier_alignment": {"equal": True},
        "author": {
            "bits_per_token_author": 3.0,
            "kl_q_stego_to_p_lm_bits_author": 1.25,
        },
        "normalized": {
            "bits_per_token": 2.5,
            "kl_ref_to_stego_mean_bits": float("inf"),
            "kl_stego_to_ref_mean_bits": 0.75,
            "kl_ref_to_stego_infinite_steps": 1,
            "kl_stego_to_ref_infinite_steps": 0,
            "tvd_mean": 0.2,
            "exact_token_id_decode": True,
        },
        "token_sequence_agreement": {
            "exact": False,
            "agreement_rate": 0.5,
        },
    }


def test_aggregate_has_32_pairs_dual_kl_and_no_closeness_gate() -> None:
    module = _module()
    records = [
        _record(point, rank)
        for point in module.EXPECTED_POINT_IDS
        for rank in module.EXPECTED_CONTEXT_RANKS
    ]
    summary = module.aggregate_records(records, config=_config())
    assert summary["pair_count"] == 32
    assert summary["expected_pair_count"] == 32
    assert summary["all_carrier_lengths_matched"] is True
    assert summary["all_normalized_token_id_decodes_exact"] is True
    assert summary["dual_kl"]["all_reverse_kl_present"] is True
    assert summary["dual_kl"]["all_forward_kl_accounted"] is True
    assert summary["scientific_closeness_evaluated"] is False
    assert summary["ready_for_step_3_13_conformance_analysis"] is True
    assert len(summary["point_summaries"]) == 4
    assert summary["point_summaries"][0]["mean_delta_bits_per_token_normalized_minus_author"] == -0.5


def test_context_validation_rejects_text_hash_mismatch() -> None:
    module = _module()
    manifest = json.loads(
        (REPO_ROOT / "results" / "stage3" / "paper_reproduction" / "cnndm_context_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    contexts = []
    for item in manifest["contexts"][:8]:
        contexts.append(
            {
                "selection_rank": item["selection_rank"],
                "article_id": item["article_id"],
                "context_sha256": item["context_sha256"],
                "context": "not the frozen article context",
            }
        )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        module.validate_context_records(contexts, manifest=manifest)
