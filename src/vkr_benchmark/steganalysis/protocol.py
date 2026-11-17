"""Stable Stage-7 grouping and randomness decisions, independent of LM runtime."""
from __future__ import annotations

import hashlib
from typing import Iterable

from vkr_benchmark.errors import ConfigurationError

DOMAIN = "VKR-BENCHMARK-STAGE7-v1"
SPLIT_SEED = 701
TRAIN_GROUP_COUNT = 96
VALIDATION_GROUP_COUNT = 32


def _digest(material: str) -> str:
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def grouped_split(
    prompt_article_ids: Iterable[tuple[str, str]],
    *,
    split_seed: int = SPLIT_SEED,
    train_count: int = TRAIN_GROUP_COUNT,
    validation_count: int = VALIDATION_GROUP_COUNT,
) -> dict[str, str]:
    """Assign each distinct article to train or validation, never individual texts.

    The returned mapping is keyed by prompt_id and independent of input order.
    """
    rows = list(prompt_article_ids)
    ids = [prompt_id for prompt_id, _ in rows]
    articles = [article for _, article in rows]
    if len(rows) != train_count + validation_count:
        raise ConfigurationError("Stage-7 train/validation group count mismatch")
    if any(not isinstance(x, str) or not x for x in ids + articles):
        raise ConfigurationError("prompt and article ids must be nonempty strings")
    if len(set(ids)) != len(ids) or len(set(articles)) != len(articles):
        raise ConfigurationError("duplicate prompt_id or article_id in Stage-7 candidates")
    if train_count <= 0 or validation_count <= 0:
        raise ConfigurationError("both train and validation must be nonempty")
    sorted_articles = sorted(
        articles,
        key=lambda article: (
            _digest(f"{DOMAIN}|split|seed={split_seed}|article_id={article}"),
            article,
        ),
    )
    train_articles = set(sorted_articles[:train_count])
    return {
        prompt_id: ("train" if article in train_articles else "validation")
        for prompt_id, article in rows
    }


def cover_seed(*, model_id: str, article_id: str, method_point_id: str, replicate_id: str) -> int:
    """Independent control RNG state, stable across machines/versions."""
    fields = (model_id, article_id, method_point_id, replicate_id)
    if not all(isinstance(x, str) and x for x in fields):
        raise ConfigurationError("cover seed requires four nonempty identifiers")
    material = (
        f"{DOMAIN}|role=cover|model={model_id}|article={article_id}"
        f"|point={method_point_id}|replicate={replicate_id}"
    )
    return int.from_bytes(bytes.fromhex(_digest(material))[:8], "big")


def secret_id_for_prompt(prompt_id: str, replicate_id: str = "r01") -> str:
    if not prompt_id or not replicate_id:
        raise ConfigurationError("prompt_id and replicate_id are required")
    return f"stage7-det-{prompt_id}-{replicate_id}"


def method_random_material(*, article_id: str, method_id: str, replicate_id: str) -> tuple[str | None, str | None]:
    """Use the already normalized method RNG/key roles, keyed per article.

    Across multiple operating points of the same method, the same article and
    replicate see the same method randomness. DAIR, Huffman and AC are
    deterministic from their secret bit stream and need no RNG/key.
    """
    roles = {"bins": "random_seed", "adg": "random_seed", "discop": "key",
             "rrc": "key", "huffman": None, "arithmetic_coding": None,
             "dairstega": None}
    if method_id not in roles:
        raise ConfigurationError(f"unknown Stage-7 method {method_id!r}")
    if not article_id or not replicate_id:
        raise ConfigurationError("method randomness requires article and replicate")
    role = roles[method_id]
    if role is None:
        return None, None
    material = (f"{DOMAIN}|role=method|article={article_id}|"
                f"replicate={replicate_id}|method={method_id}|material={role}")
    return (material, None) if role == "random_seed" else (None, material)
