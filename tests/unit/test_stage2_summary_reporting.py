from __future__ import annotations

import math

from vkr_benchmark.reporting import build_markdown_table, normalize_summary_records


def _row(method: str, *, run_id: str, status: str = "ok") -> dict[str, object]:
    return {
        "run_id": run_id,
        "method_id": method,
        "method_params_hash": "params",
        "model_id": "model",
        "prompt_id": "p1",
        "secret_id": "s1",
        "bits_per_token": 2.0,
        "entropy_utilization": 0.5,
        "kl_mean_bits": math.inf,
        "tvd_mean": 0.25,
        "nll_raw_lm_nats_per_token": 1.5,
        "ppl_raw_lm": 4.5,
        "ber": 0.0,
        "encode_ms_per_token": 10.0,
        "decode_ms_per_token": 11.0,
        "status": status,
    }


def test_normalize_filters_status_by_default() -> None:
    rows = normalize_summary_records([
        _row("bins", run_id="b"),
        _row("huffman", run_id="h", status="error"),
    ])
    assert [row["method_id"] for row in rows] == ["bins"]


def test_normalize_can_filter_model_and_prompt() -> None:
    a = _row("bins", run_id="a")
    b = _row("huffman", run_id="b")
    b["model_id"] = "other"
    c = _row("arithmetic_coding", run_id="c")
    c["prompt_id"] = "p2"
    rows = normalize_summary_records([a, b, c], model_id="model", prompt_id="p1")
    assert [row["run_id"] for row in rows] == ["a"]


def test_normalize_order_is_deterministic() -> None:
    rows = normalize_summary_records([
        _row("huffman", run_id="2"),
        _row("bins", run_id="1"),
        _row("arithmetic_coding", run_id="3"),
    ])
    assert [row["method_id"] for row in rows] == [
        "arithmetic_coding",
        "bins",
        "huffman",
    ]


def test_markdown_formats_infinity_percent_and_metrics() -> None:
    table = build_markdown_table([_row("bins", run_id="1")])
    assert "| Метод | BPT | Использование энтропии, % |" in table
    assert "| bins | 2.0000 | 50.00 | inf | 0.2500 | 1.5000 | 4.5000 | 0.000000 | 10.000 | 11.000 | ok |" in table


def test_markdown_empty_table_is_explicit() -> None:
    assert build_markdown_table([]) == "_Нет подходящих запусков._\n"


def test_markdown_missing_values_use_em_dash() -> None:
    row = _row("bins", run_id="1")
    row["ppl_raw_lm"] = None
    table = build_markdown_table([row])
    assert "| — |" in table
