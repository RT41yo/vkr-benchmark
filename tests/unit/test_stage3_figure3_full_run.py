from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "reproducibility" / "figure3_full_run.json"
MATRIX_PATH = REPO_ROOT / "configs" / "reproducibility" / "paper_reproduction_matrix.json"
CORE_PATH = REPO_ROOT / "scripts" / "stage3_figure3_core.py"
PILOT_RUNNER_PATH = REPO_ROOT / "scripts" / "run_stage3_paper_sentence_pilot.py"
FIGURE_RUNNER_PATH = REPO_ROOT / "scripts" / "run_stage3_figure3_full.py"
AGGREGATOR_PATH = REPO_ROOT / "scripts" / "aggregate_stage3_figure3.py"
CHECKER_PATH = REPO_ROOT / "scripts" / "check_stage3_figure3_full.py"
DOC_PATH = REPO_ROOT / "docs" / "stages" / "stage3" / "figure3_full_run.md"
EXPECTED_MATRIX_SHA = "67b376590c7bb5a61dfb7565fd4699567880978c0a5b5bc4d741dafde5ff200b"
EXPECTED_SENTENCE_CORE_SHA = "b0b52366aece6e5ec406336c699c5a0a9d3d66dae1943e232e9ad367c82ff6c7"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


def test_full_run_config_matches_frozen_23_point_matrix() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    core = _load_module("stage3_figure3_core_test", CORE_PATH)
    plan = core.validate_full_config(config, matrix)
    assert plan == {"point_count": 23, "shard_count": 69, "runs_per_point": 240, "total_runs": 5520}
    assert hashlib.sha256(MATRIX_PATH.read_bytes()).hexdigest() == EXPECTED_MATRIX_SHA
    assert config["matrix_sha256"] == EXPECTED_MATRIX_SHA


def test_full_run_reuses_validated_sentence_core_and_stream() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    sentence_core = REPO_ROOT / config["paper_sentence_core_path"]
    assert hashlib.sha256(sentence_core.read_bytes()).hexdigest() == EXPECTED_SENTENCE_CORE_SHA
    assert config["paper_sentence_core_sha256"] == EXPECTED_SENTENCE_CORE_SHA

    pilot = _load_module("paper_sentence_pilot_stream", PILOT_RUNNER_PATH)
    first = pilot.paper_sentence_bits(7, seed=1234, replicate=0, bit_count=16384)
    again = pilot.paper_sentence_bits(7, seed=config["seed_base"], replicate=0, bit_count=config["secret_stream"]["bit_count"])
    replicate_one = pilot.paper_sentence_bits(7, seed=config["seed_base"], replicate=1, bit_count=config["secret_stream"]["bit_count"])
    assert first == again
    assert first != replicate_one


def test_full_run_predeclares_resume_and_non_scientific_execution_gate() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["storage"]["atomic_shard_write"] is True
    assert config["storage"]["resume_completed_shards"] is True
    assert config["storage"]["records_per_shard"] == 80
    assert config["execution"]["scientific_claims_are_not_execution_gates"] is True
    assert config["transport_recovery"]["role"] == "diagnostic_not_figure3_execution_gate"
    assert config["transport_recovery"]["retain_generation_sample_on_recovery_failure"] is True
    assert config["transport_recovery"]["paper_sentence_pilot_remains_strict"] is True
    assert config["sentence_stop"]["pilot_validated_safety_cap_tokens"] == 256
    assert config["sentence_stop"]["max_generated_tokens"] == 1024
    assert config["sentence_stop"]["adaptive_arithmetic_guard"]["cap_sequence_tokens"] == [1024, 2048, 4096, 8192]
    assert config["sentence_stop"]["adaptive_arithmetic_guard"]["apply_to_methods"] == ["arithmetic"]
    assert config["sentence_guard_escalation_amendment"]["max_escalated_guard_tokens"] == 8192
    assert config["sentence_stop"]["cap_role"] == "emergency_runaway_guard_not_paper_parameter"
    assert config["sentence_stop_amendment"]["new_emergency_cap_tokens"] == 1024
    assert config["zero_payload_policy"]["applies_to"] == ["arithmetic"]
    assert "arithmetic_t0.4_k300" in config["payload_zero_amendment"]["trigger"]
    assert "bins_b4" in config["sentence_stop_amendment"]["trigger"]
    assert config["aggregation"]["primary_standard_error"].startswith("sample standard deviation")
    assert "context" in config["aggregation"]["context_clustered_diagnostic"]


def test_full_runner_has_pilot_continuity_model_revision_and_atomic_shards() -> None:
    source = FIGURE_RUNNER_PATH.read_text(encoding="utf-8")
    assert "Step-3.9 committed result missing" in source
    assert "fixed-message mirror parity" in source
    assert "_resolved_model_provenance" in source
    assert "resolved_model_revision" in source
    assert "os.replace" in source
    assert ".failed.jsonl" in source
    assert "aggregate_completed_run" in source
    assert "_decode_with_diagnostics" in source
    assert "samples remain valid for Figure-3 generation metrics" in source
    assert "max_generated_tokens" in source


def test_aggregator_and_checker_defer_claim_interpretation_to_step311() -> None:
    aggregator = AGGREGATOR_PATH.read_text(encoding="utf-8")
    checker = CHECKER_PATH.read_text(encoding="utf-8")
    assert '"scientific_claims_evaluated": False' in aggregator
    assert "READY FOR STEP 3.11 INTERPRETATION" in aggregator
    assert "scientific paper claims are execution gates: False" in checker
    assert "--preflight" in checker


def test_documentation_states_5520_run_operationalization() -> None:
    source = DOC_PATH.read_text(encoding="utf-8")
    assert "5520" in source
    assert "23" in source
    assert "80" in source
    assert "3 replicates" in source
    assert "Step 3.11" in source
    assert "не является" in source and "Figure 3" in source
    assert "1024" in source
    assert "bins_b4" in source
    assert "zero-payload Arithmetic" in source


def test_point_aggregation_records_sentences_over_pilot_cap() -> None:
    core = _load_module("stage3_figure3_core_sentence_diag_test", CORE_PATH)
    point = {"id": "bins_b4", "method": "bins", "block_size_bits": 4}
    records = []
    for replicate in range(3):
        for rank in range(80):
            carrier = 300 if (replicate == 0 and rank == 78) else 20
            records.append({
                "selection_rank": rank,
                "replicate": replicate,
                "status": "ok",
                "generation": {"first_sentence_boundary_is_final_token": True, "text_token_roundtrip_exact": True},
                "recovery": {"exact_confirmed_payload_prefix_recovery": True, "status": "exact_prefix"},
                "secret_stream": {"used_implicit_zero_lookahead": False},
                "author_metrics": {
                    "bits_per_word_author": 4.0,
                    "kl_q_stego_to_p_lm_bits_author": 1.0,
                    "avg_nll_nats_author": 2.0,
                    "carrier_tokens": carrier,
                    "payload_bits_confirmed": carrier * 4,
                },
            })
    summary = core.aggregate_point(point, records)
    assert summary["sentence_length_diagnostic"]["over_step39_pilot_cap_256_count"] == 1
    assert summary["sentence_length_diagnostic"]["max_carrier_tokens"] == 300


def test_arithmetic_long_context_compatibility_is_explicit_and_narrow() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    compat = config["arithmetic_long_context_compatibility"]
    assert compat["max_cache_tokens"] == 1022
    assert compat["normalized_benchmark_unchanged"] is True
    assert compat["paper_sentence_core_unchanged"] is True
    assert "sequence axis" in compat["decision"]
    assert "DynamicCache" in compat["root_cause"]
    assert "1024" in compat["root_cause"]


def test_modern_gpt2_cache_limiter_trims_sequence_not_head_dim() -> None:
    import torch

    compat_path = REPO_ROOT / "scripts" / "stage3_arithmetic_cache_compat.py"
    compat = _load_module("stage3_arithmetic_cache_compat_test", compat_path)
    limiter = compat.ModernGPT2SequenceCacheLimiter(max_cache_tokens=1022)
    key = torch.arange(1 * 2 * 1030 * 4, dtype=torch.float32).reshape(1, 2, 1030, 4)
    value = key + 1
    limited = limiter(((key, value),))
    out_key, out_value = limited[0]
    assert tuple(out_key.shape) == (1, 2, 1022, 4)
    assert tuple(out_value.shape) == (1, 2, 1022, 4)
    assert torch.equal(out_key, key[:, :, -1022:, :])
    assert torch.equal(out_value, value[:, :, -1022:, :])
    assert limiter.trim_events == 1
    assert limiter.max_seen_sequence_tokens == 1030

    short_limiter = compat.ModernGPT2SequenceCacheLimiter(max_cache_tokens=1022)
    short_key = torch.zeros((1, 2, 100, 64))
    short_value = torch.ones((1, 2, 100, 64))
    short = short_limiter(((short_key, short_value),))
    assert tuple(short[0][0].shape) == (1, 2, 100, 64)
    assert short_limiter.trim_events == 0
    assert short_limiter.max_seen_sequence_tokens == 100


def test_arithmetic_adaptive_guard_sequence_is_monotonic_and_starts_at_initial_cap() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    runner = _load_module("stage3_figure3_runner_guard_test", FIGURE_RUNNER_PATH)
    caps = runner._arithmetic_guard_caps(config)
    assert caps == [1024, 2048, 4096, 8192]
    assert caps[0] == config["sentence_stop"]["max_generated_tokens"]
    assert caps == sorted(set(caps))


def test_arithmetic_adaptive_guard_reruns_only_on_cap_hit() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    runner = _load_module("stage3_figure3_runner_guard_behavior_test", FIGURE_RUNNER_PATH)
    seen_caps: list[int] = []

    def fake_encode(**kwargs):
        cap = int(kwargs["max_generated_tokens"])
        seen_caps.append(cap)
        terminal = "max_generated_tokens" if cap < 2048 else "sentence_boundary"
        return {
            "terminal_reason": terminal,
            "carrier_tokens": cap if terminal == "max_generated_tokens" else 1500,
        }

    runner.encode_arithmetic_mirror = fake_encode
    mirror, limiter = runner._encode_arithmetic_with_adaptive_sentence_guard(
        config=config,
        model=object(),
        enc=object(),
        utils=object(),
        stream=[0] * 16384,
        context_tokens=[1, 2, 3],
        precision=26,
        topk=300,
        temp=0.4,
        device="cpu",
    )
    assert seen_caps == [1024, 2048]
    assert mirror["terminal_reason"] == "sentence_boundary"
    assert mirror["sentence_guard"]["escalation_count"] == 1
    assert mirror["sentence_guard"]["final_guard_tokens"] == 2048
    assert limiter.max_cache_tokens == 1022


def test_arithmetic_final_guard_exhaustion_is_classified_not_extended() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    runner = _load_module("stage3_figure3_runner_final_guard_test", FIGURE_RUNNER_PATH)
    seen_caps: list[int] = []

    def fake_encode(**kwargs):
        cap = int(kwargs["max_generated_tokens"])
        seen_caps.append(cap)
        return {
            "terminal_reason": "max_generated_tokens",
            "carrier_tokens": cap,
        }

    runner.encode_arithmetic_mirror = fake_encode
    mirror, limiter = runner._encode_arithmetic_with_adaptive_sentence_guard(
        config=config,
        model=object(),
        enc=object(),
        utils=object(),
        stream=[0] * 16384,
        context_tokens=[1, 2, 3],
        precision=26,
        topk=300,
        temp=0.4,
        device="cpu",
    )
    assert seen_caps == [1024, 2048, 4096, 8192]
    assert mirror["terminal_reason"] == "max_generated_tokens"
    assert mirror["sentence_guard"]["final_guard_exhausted"] is True
    assert mirror["sentence_guard"]["reached_real_boundary"] is False
    assert mirror["sentence_guard"]["classification"] == "sentence_termination_failure"
    assert limiter.max_cache_tokens == 1022


def test_config_freezes_sentence_termination_failure_policy() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    policy = config["sentence_stop"]["termination_failure_policy"]
    assert policy["enabled"] is True
    assert policy["applies_to_methods"] == ["arithmetic"]
    assert policy["trigger_only_after_final_adaptive_guard_tokens"] == 8192
    assert policy["record_status"] == "sentence_termination_failure"
    assert config["execution"]["accepted_record_statuses"] == ["ok", "sentence_termination_failure"]
    assert config["execution"]["metric_eligible_status"] == "ok"
    assert "49, 50 and 58" in config["sentence_termination_failure_amendment"]["trigger"]


def test_record_builder_marks_final_guard_exhaustion_as_termination_failure() -> None:
    runner = _load_module("stage3_figure3_runner_record_failure_test", FIGURE_RUNNER_PATH)

    class DummyEnc:
        def decode(self, ids):
            return " X" * len(ids)
        def encode(self, text):
            raise AssertionError("termination-failure record must not retokenize the partial trajectory")

    class DummyUtils:
        @staticmethod
        def is_sent_finish(token_id, enc):
            return False

    mirror = {
        "generated_token_ids": [1, 1, 1],
        "payload_bits_confirmed": 2,
        "secret_bits_read": 28,
        "used_implicit_zero_lookahead": False,
        "avg_nll_nats_author": 2.0,
        "kl_q_stego_to_p_lm_bits_author": 0.3,
        "bits_per_word_author": 2 / 3,
        "words_per_bit_author": 1.5,
        "carrier_tokens": 3,
        "terminal_reason": "max_generated_tokens",
        "sentence_guard": {
            "initial_guard_tokens": 1024,
            "final_guard_tokens": 8192,
            "escalation_count": 3,
            "attempts": [],
            "reached_real_boundary": False,
            "final_guard_exhausted": True,
            "classification": "sentence_termination_failure",
        },
    }
    context = {
        "selection_rank": 49,
        "row_index": 1,
        "article_id": "x",
        "context_sha256": "y",
    }
    point = {
        "id": "arithmetic_t0.4_k300", "method": "arithmetic",
        "temperature": 0.4, "topk": 300, "precision": 26,
    }
    record = runner._record_from_mirror(
        point=point, replicate=1, context=context, stream=[0] * 100,
        mirror=mirror, enc=DummyEnc(), utils=DummyUtils(), recovered_bits=[],
        stegotext=" X X X", transport_recovery_attempted=False,
    )
    assert record["status"] == "sentence_termination_failure"
    assert record["recovery"]["status"] == "not_applicable_termination_failure"
    assert record["recovery"]["exact_confirmed_payload_prefix_recovery"] is None
    assert record["generation"]["retokenized_token_ids"] is None
    assert record["author_metrics"]["metric_eligibility"] == "excluded_sentence_termination_failure"
