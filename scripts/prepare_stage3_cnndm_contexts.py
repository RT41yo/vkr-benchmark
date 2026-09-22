#!/usr/bin/env python3
"""Prepare the pinned CNN/DailyMail context set for Stage-3 paper reproduction.

This script downloads (or accepts) one immutable CNN/DailyMail Parquet shard,
verifies its SHA-256 and size, deterministically selects 80 valid articles, and
extracts the first three sentences from each. Article/context text is written
only to an ignored local JSONL file. The committed manifest contains identifiers
and hashes, not copyrighted article text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Iterable
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "reproducibility" / "cnndm_context_source.json"

# Frozen for stage3_news_sentence_splitter_v1. Keep this list deliberately
# small and explicit; changing it changes the context hashes and requires a new
# splitter_id.
ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.",
    "etc.", "e.g.", "i.e.", "u.s.", "u.k.", "u.n.", "sen.", "rep.",
    "gov.", "gen.", "sgt.", "lt.", "col.", "capt.", "cmdr.", "rev.",
    "inc.", "corp.", "co.", "ltd.", "no.", "fig.", "dept.", "jan.",
    "feb.", "mar.", "apr.", "jun.", "jul.", "aug.", "sep.", "sept.",
    "oct.", "nov.", "dec.",
}
CLOSERS = set('"\'”’)]}')
STARTERS = set('"\'“‘([')


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_article(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip(), flags=re.UNICODE)


def _preceding_token(text: str, dot_index: int) -> str:
    start = dot_index
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    return text[start : dot_index + 1].strip('"\'“‘([{')


def _looks_like_sentence_start(text: str, index: int) -> bool:
    if index >= len(text):
        return True
    ch = text[index]
    if ch in STARTERS:
        return True
    return ch.isupper() or ch.isdigit()


def _protected_period(text: str, index: int) -> bool:
    # Decimal number, e.g. 3.14.
    if index > 0 and index + 1 < len(text):
        if text[index - 1].isdigit() and text[index + 1].isdigit():
            return True

    token = _preceding_token(text, index)
    lowered = token.lower()
    if lowered in ABBREVIATIONS:
        return True

    # Single-letter initials ("J. Smith") and compact acronyms ("U.S.").
    if re.fullmatch(r"[A-Z]\.", token):
        return True
    if re.fullmatch(r"(?:[A-Z]\.){2,}", token):
        return True
    return False


def split_sentences(article: str) -> list[str]:
    """Deterministically segment normalized English news text.

    The goal is reproducibility, not to claim the exact unknown 2019 sentence
    tokenizer. The rule is frozen under ``stage3_news_sentence_splitter_v1``.
    """

    text = normalize_article(article)
    if not text:
        return []

    sentences: list[str] = []
    start = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch not in ".!?":
            i += 1
            continue

        if ch == "." and _protected_period(text, i):
            i += 1
            continue

        # Consume terminal punctuation sequences such as ?!, ... and closers.
        j = i + 1
        while j < len(text) and text[j] in ".!?":
            j += 1
        while j < len(text) and text[j] in CLOSERS:
            j += 1

        if j == len(text):
            piece = text[start:j].strip()
            if piece:
                sentences.append(piece)
            start = j
            break

        if not text[j].isspace():
            i += 1
            continue

        k = j
        while k < len(text) and text[k].isspace():
            k += 1
        if not _looks_like_sentence_start(text, k):
            i += 1
            continue

        piece = text[start:j].strip()
        if piece:
            sentences.append(piece)
        start = k
        i = k

    if start < len(text):
        tail = text[start:].strip()
        if tail:
            sentences.append(tail)
    return sentences


def selection_digest(row_index: int, *, seed: int) -> str:
    material = f"stage3-cnndm-test-v1|seed={seed}|row={row_index}".encode("ascii")
    return hashlib.sha256(material).hexdigest()


def ranked_indices(row_count: int, *, seed: int) -> list[int]:
    return sorted(range(row_count), key=lambda idx: selection_digest(idx, seed=seed))


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8")), sha256_bytes(raw)


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != "stage3.cnndm_context_source.v1":
        raise ValueError("unexpected context-source schema")
    dataset = config.get("dataset") or {}
    if dataset.get("repository") != "abisee/cnn_dailymail":
        raise ValueError("unexpected dataset repository")
    if dataset.get("config") != "3.0.0" or dataset.get("split") != "test":
        raise ValueError("Stage-3 context source must pin CNN/DailyMail 3.0.0 test")
    if dataset.get("artifact_sha256") != "04e322d2634a96dba76bf9a6294fbbe48e0b36abeae43f13d86ba2c3bebffe4e":
        raise ValueError("unexpected pinned Parquet checksum")
    extraction = config.get("context_extraction") or {}
    if extraction.get("splitter_id") != "stage3_news_sentence_splitter_v1":
        raise ValueError("unexpected sentence splitter id")
    selection = config.get("selection") or {}
    if selection.get("full_context_count") != 80 or selection.get("pilot_context_count") != 8:
        raise ValueError("context counts must remain 80 full / 8 pilot")


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "vkr-benchmark-stage3/1"})
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            with urlopen(request, timeout=120) as response:  # noqa: S310 - pinned HTTPS URL from config
                shutil.copyfileobj(response, tmp, length=1024 * 1024)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
    tmp_path.replace(destination)


def ensure_artifact(config: dict[str, Any], cache_path: Path, explicit_artifact: Path | None) -> Path:
    dataset = config["dataset"]
    expected_size = int(dataset["artifact_size_bytes"])
    expected_sha = str(dataset["artifact_sha256"])

    if explicit_artifact is not None:
        source = explicit_artifact.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        artifact = source
    else:
        artifact = cache_path
        if not artifact.is_file():
            print(f"Downloading pinned CNN/DailyMail artifact -> {artifact}", flush=True)
            _download(str(dataset["download_url"]), artifact)

    actual_size = artifact.stat().st_size
    actual_sha = sha256_file(artifact)
    if actual_size != expected_size:
        raise RuntimeError(f"artifact size mismatch: expected {expected_size}, got {actual_size}")
    if actual_sha != expected_sha:
        raise RuntimeError(f"artifact SHA-256 mismatch: expected {expected_sha}, got {actual_sha}")
    return artifact


def _iter_rows(artifact: Path) -> tuple[int, list[str], list[str]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required; install the project storage extra") from exc

    table = pq.read_table(artifact, columns=["article", "id"])
    articles = table.column("article").to_pylist()
    ids = table.column("id").to_pylist()
    if len(articles) != len(ids):
        raise RuntimeError("article/id column length mismatch")
    return len(articles), [str(x) for x in articles], [str(x) for x in ids]


def build_context_records(
    articles: list[str],
    article_ids: list[str],
    *,
    seed: int,
    full_count: int,
    pilot_count: int,
    minimum_sentences: int,
) -> tuple[list[dict[str, Any]], int]:
    accepted: list[dict[str, Any]] = []
    rejected = 0
    for row_index in ranked_indices(len(articles), seed=seed):
        sentences = split_sentences(articles[row_index])
        if len(sentences) < minimum_sentences:
            rejected += 1
            continue

        context_sentences = sentences[:3]
        context = " ".join(context_sentences)
        next_sentence = sentences[3]
        rank = len(accepted)
        accepted.append(
            {
                "selection_rank": rank,
                "pilot": rank < pilot_count,
                "row_index": row_index,
                "selection_digest": selection_digest(row_index, seed=seed),
                "article_id": article_ids[row_index],
                "context": context,
                "context_sha256": sha256_bytes(context.encode("utf-8")),
                "context_char_count": len(context),
                "context_sentence_char_counts": [len(item) for item in context_sentences],
                "human_next_sentence": next_sentence,
                "human_next_sentence_sha256": sha256_bytes(next_sentence.encode("utf-8")),
                "segmented_sentence_count": len(sentences),
            }
        )
        if len(accepted) == full_count:
            break

    if len(accepted) != full_count:
        raise RuntimeError(f"only {len(accepted)} valid contexts found; expected {full_count}")
    return accepted, rejected


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_context_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def prepare(config_path: Path, explicit_artifact: Path | None = None) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    config, config_sha = _load_config(config_path)
    _validate_config(config)

    paths = config["local_paths"]
    cache_path = (REPO_ROOT / paths["artifact_cache"]).resolve()
    text_path = (REPO_ROOT / paths["generated_context_text"]).resolve()
    manifest_path = (REPO_ROOT / paths["committed_manifest"]).resolve()

    artifact = ensure_artifact(config, cache_path, explicit_artifact)
    row_count, articles, article_ids = _iter_rows(artifact)
    expected_rows = int(config["dataset"]["split_rows"])
    if row_count != expected_rows:
        raise RuntimeError(f"row-count mismatch: expected {expected_rows}, got {row_count}")

    selection = config["selection"]
    extraction = config["context_extraction"]
    records, rejected = build_context_records(
        articles,
        article_ids,
        seed=int(selection["seed"]),
        full_count=int(selection["full_context_count"]),
        pilot_count=int(selection["pilot_context_count"]),
        minimum_sentences=int(extraction["minimum_sentences_required"]),
    )

    # Full text remains local/ignored. It is sufficient to reconstruct later
    # author-compatible runs without copying source news text into Git.
    _write_context_jsonl(text_path, records)

    public_records = []
    for record in records:
        public_records.append({key: value for key, value in record.items() if key not in {"context", "human_next_sentence"}})

    manifest = {
        "schema_version": "stage3.cnndm_context_manifest.v1",
        "status": "frozen",
        "source_config_path": str(config_path.relative_to(REPO_ROOT)),
        "source_config_sha256": config_sha,
        "dataset": {
            "repository": config["dataset"]["repository"],
            "revision": config["dataset"]["revision"],
            "config": config["dataset"]["config"],
            "split": config["dataset"]["split"],
            "artifact_path": config["dataset"]["artifact_path"],
            "artifact_size_bytes": artifact.stat().st_size,
            "artifact_sha256": sha256_file(artifact),
            "split_rows": row_count,
        },
        "extraction": {
            "splitter_id": extraction["splitter_id"],
            "sentence_count": extraction["sentence_count"],
            "minimum_sentences_required": extraction["minimum_sentences_required"],
            "normalization": extraction["normalization"],
        },
        "selection": {
            "algorithm_id": selection["algorithm_id"],
            "seed": selection["seed"],
            "full_context_count": selection["full_context_count"],
            "pilot_context_count": selection["pilot_context_count"],
            "invalid_rows_skipped_before_full_set": rejected,
        },
        "contexts": public_records,
        "go_no_go": {
            "dataset_artifact_verified": True,
            "context_manifest_frozen": True,
            "local_context_text_generated": True,
            "ready_for_gpt2_medium_pilot": True,
        },
    }
    _write_json(manifest_path, manifest)
    return {
        "artifact": artifact,
        "manifest": manifest_path,
        "text": text_path,
        "row_count": row_count,
        "rejected": rejected,
        "pilot": records[: int(selection["pilot_context_count"])],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=None,
        help="Use an already-downloaded Parquet file instead of the pinned URL.",
    )
    args = parser.parse_args()

    try:
        summary = prepare(args.config, args.artifact)
    except Exception as exc:
        print(f"Stage 3 CNN/DailyMail preparation: ERROR ({type(exc).__name__}: {exc})")
        return 1

    print("Stage 3 CNN/DailyMail context preparation")
    print(f"artifact: {summary['artifact']}")
    print(f"rows verified: {summary['row_count']}")
    print(f"invalid ranked rows skipped: {summary['rejected']}")
    print(f"manifest: {summary['manifest']}")
    print(f"local context text: {summary['text']} (gitignored)")
    print("pilot contexts: 8")
    print("full frozen context set: 80")
    print("ready for gpt2-medium pilot: True")
    print("Stage 3 CNN/DailyMail context preparation: READY")
    return 0


if __name__ == "__main__":
    sys.exit(main())
