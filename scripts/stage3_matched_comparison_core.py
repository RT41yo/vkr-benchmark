"""Pure helpers for Stage-3 Step 3.12 matched author/normalized comparison.

The module intentionally contains no model-loading code so configuration,
provenance, pairing, and aggregation can be unit-tested on CPU.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable

EXPECTED_SCHEMA = "stage3.matched_author_normalized.v1"
EXPECTED_MODEL = "gpt2-medium"
EXPECTED_MODEL_REVISION = "6dcaa7a952f72f9298047fd5137cd6e4f05f41da"
EXPECTED_REFERENCE_COMMIT = "14e982564aeaf9a33f7b4de440deda2184d17f12"
EXPECTED_POINT_IDS = (
    "bins_b3",
    "huffman_e3",
    "arithmetic_t0.9_k300",
    "arithmetic_t1.0_k50256",
)
EXPECTED_CONTEXT_RANKS = tuple(range(8))
EXPECTED_PAIR_COUNT = len(EXPECTED_POINT_IDS) * len(EXPECTED_CONTEXT_RANKS)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bits(bits: Iterable[int]) -> str:
    return hashlib.sha256("".join(str(int(bit)) for bit in bits).encode("ascii")).hexdigest()


def paper_sentence_bits(
    selection_rank: int,
    *,
    seed: int,
    replicate: int,
    bit_count: int,
) -> list[int]:
    """Exact Step-3.9/3.10 deterministic uniform-looking bit stream."""

    if selection_rank < 0 or replicate < 0 or bit_count <= 0:
        raise ValueError("invalid paper-sentence bit-stream parameters")
    bits: list[int] = []
    counter = 0
    while len(bits) < bit_count:
        material = (
            f"stage3-paper-sentence|seed={seed}|replicate={replicate}|"
            f"selection_rank={selection_rank}|counter={counter}"
        ).encode("ascii")
        for byte in hashlib.sha256(material).digest():
            for shift in range(7, -1, -1):
                bits.append((byte >> shift) & 1)
                if len(bits) == bit_count:
                    return bits
        counter += 1
    return bits


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record at {path}:{line_number} is not an object")
            out.append(value)
    return out


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != EXPECTED_SCHEMA:
        raise ValueError("unexpected Step-3.12 matched-comparison schema")
    if config.get("model_id") != EXPECTED_MODEL:
        raise ValueError("Step 3.12 must use GPT-2 Medium")
    if config.get("model_revision") != EXPECTED_MODEL_REVISION:
        raise ValueError("Step 3.12 GPT-2 Medium revision is not the frozen Stage-3 revision")
    if config.get("reference_commit") != EXPECTED_REFERENCE_COMMIT:
        raise ValueError("unexpected pinned author-reference commit")

    contexts = config.get("contexts") or {}
    ranks = tuple(int(x) for x in contexts.get("selection_ranks", []))
    if ranks != EXPECTED_CONTEXT_RANKS:
        raise ValueError("Step 3.12 must use frozen pilot selection ranks 0..7")
    if int(contexts.get("replicate", -1)) != 0:
        raise ValueError("Step 3.12 must use Figure-3 replicate 0 for paired author records")

    stream = config.get("secret_stream") or {}
    if stream.get("generator_id") != "paper_sentence_sha256_bitstream_v1":
        raise ValueError("Step 3.12 must reuse the frozen paper-sentence secret stream")
    if int(stream.get("seed", -1)) != 1234 or int(stream.get("replicate", -1)) != 0:
        raise ValueError("unexpected Step-3.12 secret stream seed/replicate")
    if int(stream.get("bit_count", 0)) < 4096:
        raise ValueError("Step 3.12 secret stream is too short")

    alignment = config.get("carrier_alignment") or {}
    if alignment.get("rule") != "normalized_fixed_tokens_equal_author_sentence_tokens":
        raise ValueError("unexpected carrier-alignment rule")
    if alignment.get("normalized_sentence_stopping") is not False:
        raise ValueError("normalized side must not add a sentence-stopping confound")

    points = config.get("points") or []
    point_ids = tuple(str(item.get("id")) for item in points)
    if point_ids != EXPECTED_POINT_IDS:
        raise ValueError("Step 3.12 representative points differ from the frozen order")
    if len({str(item.get("author_shard")) for item in points}) != len(points):
        raise ValueError("each matched point must reference a distinct author shard")

    execution = config.get("execution") or {}
    if int(execution.get("expected_pair_count", 0)) != EXPECTED_PAIR_COUNT:
        raise ValueError("Step 3.12 expected_pair_count must remain 32")
    if execution.get("require_normalized_exact_token_id_decode") is not True:
        raise ValueError("normalized token-id decode recovery must remain a hard gate")
    if execution.get("metric_closeness_is_gate") is not False:
        raise ValueError("metric closeness must not be a Step-3.12 execution gate")


def point_map(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    validate_config(config)
    return {str(item["id"]): dict(item) for item in config["points"]}


def load_author_records(
    *,
    repo_root: Path,
    config: dict[str, Any],
) -> dict[tuple[str, int], dict[str, Any]]:
    """Load the frozen author side from committed Figure-3 replicate-0 shards."""

    validate_config(config)
    wanted_ranks = set(EXPECTED_CONTEXT_RANKS)
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for point in config["points"]:
        point_id = str(point["id"])
        path = (repo_root / str(point["author_shard"])).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        records = load_jsonl(path)
        selected = [
            record
            for record in records
            if int(record.get("selection_rank", -1)) in wanted_ranks
        ]
        if len(selected) != len(wanted_ranks):
            raise ValueError(
                f"author shard {path} does not provide exactly eight pilot records"
            )
        for record in selected:
            rank = int(record["selection_rank"])
            if record.get("point_id") != point_id:
                raise ValueError(f"author point mismatch for {point_id} rank {rank}")
            if int(record.get("replicate", -1)) != 0:
                raise ValueError(f"author replicate mismatch for {point_id} rank {rank}")
            if record.get("status") != "ok":
                raise ValueError(
                    f"Step 3.12 requires a completed author sentence: {point_id} rank {rank}"
                )
            generation = record.get("generation") or {}
            if generation.get("terminal_reason") != "sentence_boundary":
                raise ValueError(f"author sentence boundary missing for {point_id} rank {rank}")
            if generation.get("first_sentence_boundary_is_final_token") is not True:
                raise ValueError(f"author first-boundary invariant failed for {point_id} rank {rank}")
            out[(point_id, rank)] = record

    if len(out) != EXPECTED_PAIR_COUNT:
        raise ValueError(f"expected {EXPECTED_PAIR_COUNT} author records, got {len(out)}")
    return out


def validate_context_records(
    contexts: list[dict[str, Any]],
    *,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return the eight frozen local context records after hash verification."""

    public = {
        int(item["selection_rank"]): item
        for item in manifest.get("contexts", [])
        if int(item.get("selection_rank", -1)) in EXPECTED_CONTEXT_RANKS
    }
    local = {
        int(item["selection_rank"]): item
        for item in contexts
        if int(item.get("selection_rank", -1)) in EXPECTED_CONTEXT_RANKS
    }
    if set(public) != set(EXPECTED_CONTEXT_RANKS) or set(local) != set(EXPECTED_CONTEXT_RANKS):
        raise ValueError("local/public context sets do not contain ranks 0..7")

    ordered: list[dict[str, Any]] = []
    for rank in EXPECTED_CONTEXT_RANKS:
        item = local[rank]
        context = str(item.get("context", ""))
        digest = hashlib.sha256(context.encode("utf-8")).hexdigest()
        expected = str(public[rank]["context_sha256"])
        if digest != expected or str(item.get("context_sha256")) != expected:
            raise ValueError(f"context SHA-256 mismatch at selection rank {rank}")
        if str(item.get("article_id")) != str(public[rank].get("article_id")):
            raise ValueError(f"article id mismatch at selection rank {rank}")
        ordered.append(item)
    return ordered


def first_mismatch_index(left: Iterable[int], right: Iterable[int]) -> int | None:
    a = tuple(int(x) for x in left)
    b = tuple(int(x) for x in right)
    for index, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return index
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def token_agreement(left: Iterable[int], right: Iterable[int]) -> dict[str, Any]:
    a = tuple(int(x) for x in left)
    b = tuple(int(x) for x in right)
    if len(a) != len(b):
        raise ValueError("matched token sequences must have equal carrier length")
    equal = sum(x == y for x, y in zip(a, b, strict=True))
    prefix = 0
    for x, y in zip(a, b, strict=True):
        if x != y:
            break
        prefix += 1
    rate = equal / len(a) if a else 1.0
    exact = a == b
    return {
        "exact": exact,
        "agreement_count": equal,
        "agreement_rate": rate,
        "common_prefix_tokens": prefix,
        "first_token_mismatch": first_mismatch_index(a, b),
        # Descriptive aliases retained for readability in downstream ad-hoc use.
        "exact_token_sequence_match": exact,
        "token_agreement_count": equal,
        "token_agreement_rate": rate,
    }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("cannot average empty values")
    return float(statistics.fmean(values))


def _median(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("cannot take median of empty values")
    return float(statistics.median(values))


def aggregate_records(
    records: list[dict[str, Any]],
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate completed matched pairs without imposing closeness thresholds."""

    if len(records) != EXPECTED_PAIR_COUNT:
        raise ValueError(f"expected {EXPECTED_PAIR_COUNT} matched records, got {len(records)}")
    pair_keys = {(str(r["point_id"]), int(r["selection_rank"])) for r in records}
    expected = {(point, rank) for point in EXPECTED_POINT_IDS for rank in EXPECTED_CONTEXT_RANKS}
    if pair_keys != expected:
        raise ValueError("matched records do not cover the frozen point/context grid exactly")

    if any((r.get("carrier_alignment") or {}).get("equal") is not True for r in records):
        raise ValueError("one or more matched carrier lengths differ")
    if any((r.get("normalized") or {}).get("exact_token_id_decode") is not True for r in records):
        raise ValueError("one or more normalized token-id decode checks failed")

    by_point: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_point[str(record["point_id"])].append(record)

    point_summaries: list[dict[str, Any]] = []
    for point_id in EXPECTED_POINT_IDS:
        group = sorted(by_point[point_id], key=lambda r: int(r["selection_rank"]))
        if len(group) != 8:
            raise ValueError(f"point {point_id} does not contain eight matched pairs")

        author_bpt = [float(r["author"]["bits_per_token_author"]) for r in group]
        normalized_bpt = [float(r["normalized"]["bits_per_token"]) for r in group]
        author_kl = [float(r["author"]["kl_q_stego_to_p_lm_bits_author"]) for r in group]
        normalized_kl = [float(r["normalized"]["kl_stego_to_ref_mean_bits"]) for r in group]
        tvd = [float(r["normalized"]["tvd_mean"]) for r in group]
        agreement = [float(r["token_sequence_agreement"]["agreement_rate"]) for r in group]

        point_summaries.append(
            {
                "point_id": point_id,
                "pair_count": len(group),
                "mean_author_bits_per_token": _mean(author_bpt),
                "mean_normalized_bits_per_token": _mean(normalized_bpt),
                "mean_delta_bits_per_token_normalized_minus_author": _mean(
                    n - a for a, n in zip(author_bpt, normalized_bpt, strict=True)
                ),
                "mean_author_kl_q_stego_to_p_lm_bits": _mean(author_kl),
                "mean_normalized_kl_stego_to_ref_bits": _mean(normalized_kl),
                "mean_delta_reverse_kl_bits_normalized_minus_author": _mean(
                    n - a for a, n in zip(author_kl, normalized_kl, strict=True)
                ),
                "mean_normalized_tvd": _mean(tvd),
                "mean_token_agreement_rate": _mean(agreement),
                "median_token_agreement_rate": _median(agreement),
                "exact_token_sequence_match_count": sum(
                    bool(r["token_sequence_agreement"]["exact"]) for r in group
                ),
                "normalized_ref_to_stego_infinite_run_count": sum(
                    int(r["normalized"]["kl_ref_to_stego_infinite_steps"]) > 0 for r in group
                ),
                "normalized_stego_to_ref_infinite_run_count": sum(
                    int(r["normalized"]["kl_stego_to_ref_infinite_steps"]) > 0 for r in group
                ),
                "normalized_exact_token_id_decode_count": sum(
                    bool(r["normalized"]["exact_token_id_decode"]) for r in group
                ),
            }
        )

    return {
        "schema_version": "stage3.matched_author_normalized_summary.v1",
        "pair_count": len(records),
        "expected_pair_count": EXPECTED_PAIR_COUNT,
        "point_count": len(point_summaries),
        "context_count": len(EXPECTED_CONTEXT_RANKS),
        "model_id": EXPECTED_MODEL,
        "model_revision": EXPECTED_MODEL_REVISION,
        "reference_commit": EXPECTED_REFERENCE_COMMIT,
        "config_schema_version": None if config is None else config.get("schema_version"),
        "all_normalized_token_id_decodes_exact": all(
            bool(r["normalized"]["exact_token_id_decode"]) for r in records
        ),
        "normalized_exact_decode_count": sum(
            bool(r["normalized"]["exact_token_id_decode"]) for r in records
        ),
        "all_carrier_lengths_matched": all(
            bool(r["carrier_alignment"]["equal"]) for r in records
        ),
        "same_secret_stream_verified": True,
        "dual_kl": {
            "all_reverse_kl_present": all(
                r["normalized"].get("kl_stego_to_ref_mean_bits") is not None for r in records
            ),
            "all_forward_kl_accounted": all(
                r["normalized"].get("kl_ref_to_stego_mean_bits") is not None
                or int(r["normalized"].get("kl_ref_to_stego_infinite_steps", 0)) > 0
                for r in records
            ),
            "comparison_direction": "D_KL(Q_stego || P_reference)",
            "benchmark_native_direction": "D_KL(P_reference || Q_stego)",
            "note": (
                "The normalized reverse direction is direction-matched to the author KL, "
                "but P_reference follows the normalized canonical generation policy. "
                "Step 3.13 interprets any remaining reference-policy differences."
            ),
        },
        "point_summaries": point_summaries,
        "token_sequence_parity_is_gate": False,
        "scientific_closeness_evaluated": False,
        "ready_for_step_3_13_conformance_analysis": True,
    }

def json_safe(value: Any) -> Any:
    """Convert non-finite floats to stable strings for strict JSON output."""

    if isinstance(value, float):
        if math.isnan(value):
            raise ValueError("NaN is not allowed in matched comparison output")
        if value == math.inf:
            return "inf"
        if value == -math.inf:
            return "-inf"
        return value
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value
