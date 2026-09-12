from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "reproducibility" / "cnndm_context_source.json"
PREPARE_PATH = REPO_ROOT / "scripts" / "prepare_stage3_cnndm_contexts.py"
CHECK_PATH = REPO_ROOT / "scripts" / "check_stage3_cnndm_contexts.py"
MATRIX_PATH = REPO_ROOT / "configs" / "reproducibility" / "paper_reproduction_matrix.json"
EXPECTED_MATRIX_SHA = "67b376590c7bb5a61dfb7565fd4699567880978c0a5b5bc4d741dafde5ff200b"
EXPECTED_ARTIFACT_SHA = "04e322d2634a96dba76bf9a6294fbbe48e0b36abeae43f13d86ba2c3bebffe4e"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_cnndm_source_pin_is_immutable_and_explicit_about_split_uncertainty() -> None:
    cfg = _config()
    assert cfg["schema_version"] == "stage3.cnndm_context_source.v1"
    assert cfg["status"] == "pinned_before_gpt2_medium_pilot"
    assert cfg["paper_basis"]["paper_split_reported"] is False
    assert cfg["paper_basis"]["split_operationalization"] == "test"
    dataset = cfg["dataset"]
    assert dataset["repository"] == "abisee/cnn_dailymail"
    assert dataset["revision"] == "3adf6249f0cc8409a97a4d38471529ef5f7dc496"
    assert dataset["config"] == "3.0.0"
    assert dataset["split"] == "test"
    assert dataset["split_rows"] == 11490
    assert dataset["artifact_size_bytes"] == 30000471
    assert dataset["artifact_sha256"] == EXPECTED_ARTIFACT_SHA


def test_cnndm_pin_does_not_modify_frozen_step35_matrix() -> None:
    checker = _module(CHECK_PATH, "cnndm_checker_matrix")
    assert checker._sha256_file(MATRIX_PATH) == EXPECTED_MATRIX_SHA
    assert _config()["paper_matrix"]["sha256"] == EXPECTED_MATRIX_SHA


def test_stage3_sentence_splitter_handles_news_abbreviations_decimals_and_boundaries() -> None:
    prep = _module(PREPARE_PATH, "cnndm_prepare_splitter")
    article = (
        "Dr. Smith arrived at 3.14 p.m. He spoke to U.S. officials. "
        "The meeting ended! Another sentence follows."
    )
    assert prep.split_sentences(article) == [
        "Dr. Smith arrived at 3.14 p.m.",
        "He spoke to U.S. officials.",
        "The meeting ended!",
        "Another sentence follows.",
    ]


def test_stage3_sentence_splitter_normalizes_whitespace_but_preserves_case() -> None:
    prep = _module(PREPARE_PATH, "cnndm_prepare_normalize")
    article = "  First   sentence.\nSecond\tSentence?   Third sentence! Fourth sentence.  "
    sentences = prep.split_sentences(article)
    assert sentences[:4] == [
        "First sentence.",
        "Second Sentence?",
        "Third sentence!",
        "Fourth sentence.",
    ]


def test_sha256_rank_selection_is_frozen_for_test_split() -> None:
    prep = _module(PREPARE_PATH, "cnndm_prepare_rank")
    assert prep.ranked_indices(11490, seed=1234)[:12] == [
        11404,
        4791,
        6498,
        6721,
        11343,
        3988,
        4266,
        1281,
        1702,
        2999,
        9655,
        5612,
    ]


def test_context_builder_freezes_first_three_sentences_and_pilot_prefix() -> None:
    prep = _module(PREPARE_PATH, "cnndm_prepare_builder")
    articles = [
        f"Article {i} first. Article {i} second. Article {i} third. Article {i} fourth."
        for i in range(12)
    ]
    ids = [f"{i:040x}" for i in range(12)]
    records, rejected = prep.build_context_records(
        articles,
        ids,
        seed=1234,
        full_count=8,
        pilot_count=3,
        minimum_sentences=4,
    )
    assert rejected == 0
    assert len(records) == 8
    assert [record["pilot"] for record in records] == [True, True, True, False, False, False, False, False]
    for record in records:
        assert record["context"].count("Article") == 3
        assert record["human_next_sentence"].endswith("fourth.")
        assert len(record["context_sha256"]) == 64


def test_source_checker_accepts_pinned_config_without_generated_manifest() -> None:
    checker = _module(CHECK_PATH, "cnndm_checker_source")
    checker.validate_source_config(_config(), matrix_path=MATRIX_PATH)
