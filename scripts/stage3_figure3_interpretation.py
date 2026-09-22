"""Pure-stdlib helpers for Stage-3 Step 3.11 Figure-3 interpretation."""

from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path
from typing import Any, Iterable


FLOAT_FIELDS = {
    "termination_success_rate",
    "mean_bits_per_word",
    "se_bits_per_word_runs",
    "se_bits_per_word_context_clustered",
    "mean_kl_bits",
    "se_kl_bits_runs",
    "se_kl_bits_context_clustered",
    "mean_nll_nats",
    "mean_carrier_tokens",
    "mean_payload_bits",
    "zero_payload_rate",
    "transport_recovery_exact_prefix_rate",
}

INT_FIELDS = {
    "run_count",
    "terminated_sentence_run_count",
    "termination_failure_count",
    "sentence_over_pilot_cap_256_count",
    "max_carrier_tokens",
    "zero_payload_count",
    "arithmetic_cache_runs_with_encode_trim",
    "arithmetic_cache_total_encode_trim_events",
    "sentence_guard_runs_with_escalation",
    "sentence_guard_max_final_guard_tokens",
    "transport_recovery_failure_count",
}


def load_points_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = dict(raw)
            row["parameters"] = json.loads(row.pop("parameters_json"))
            for key in FLOAT_FIELDS:
                if key in row and row[key] != "":
                    row[key] = float(row[key])
            for key in INT_FIELDS:
                if key in row and row[key] != "":
                    row[key] = int(row[key])
            rows.append(row)
    return rows


def method_points(points: Iterable[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    return [point for point in points if point["method"] == method]


def arithmetic_k300_points(points: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        point
        for point in points
        if point["method"] == "arithmetic"
        and int(point["parameters"].get("topk", -1)) == 300
    ]
    return sorted(selected, key=lambda point: float(point["parameters"]["temperature"]))


def special_arithmetic_point(points: Iterable[dict[str, Any]]) -> dict[str, Any]:
    selected = [
        point
        for point in points
        if point["method"] == "arithmetic"
        and int(point["parameters"].get("topk", -1)) == 50256
    ]
    if len(selected) != 1:
        raise ValueError(f"expected exactly one Arithmetic k=50256 point, found {len(selected)}")
    return selected[0]


def strictly_increasing(values: Iterable[float]) -> bool:
    seq = list(values)
    return all(left < right for left, right in zip(seq, seq[1:]))


def nonincreasing(values: Iterable[float], *, atol: float = 0.0) -> bool:
    seq = list(values)
    return all(right <= left + atol for left, right in zip(seq, seq[1:]))


def _sort_xy(points: Iterable[dict[str, Any]]) -> list[tuple[float, float]]:
    return sorted(
        (float(point["mean_bits_per_word"]), float(point["mean_kl_bits"]))
        for point in points
    )


def linear_interpolate(curve: list[tuple[float, float]], x: float) -> float:
    if not curve:
        raise ValueError("curve is empty")
    if x < curve[0][0] or x > curve[-1][0]:
        raise ValueError(f"x={x} is outside curve domain [{curve[0][0]}, {curve[-1][0]}]")
    if x == curve[-1][0]:
        return curve[-1][1]
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return (y0 + y1) / 2.0
            weight = (x - x0) / (x1 - x0)
            return y0 + weight * (y1 - y0)
    raise AssertionError("interpolation interval not found")


def compare_mean_curves(
    arithmetic: Iterable[dict[str, Any]],
    baseline: Iterable[dict[str, Any]],
    *,
    grid_points: int,
) -> dict[str, Any]:
    arithmetic_curve = _sort_xy(arithmetic)
    baseline_curve = _sort_xy(baseline)
    low = max(arithmetic_curve[0][0], baseline_curve[0][0])
    high = min(arithmetic_curve[-1][0], baseline_curve[-1][0])
    if not low < high:
        raise ValueError("curves have no non-empty common bits/word interval")
    xs = [low + (high - low) * idx / (grid_points - 1) for idx in range(grid_points)]
    margins = [
        linear_interpolate(baseline_curve, x) - linear_interpolate(arithmetic_curve, x)
        for x in xs
    ]
    return {
        "common_bits_per_word_interval": [low, high],
        "grid_points": grid_points,
        "arithmetic_lower_at_every_grid_point": all(margin > 0 for margin in margins),
        "minimum_baseline_minus_arithmetic_kl_bits": min(margins),
        "maximum_baseline_minus_arithmetic_kl_bits": max(margins),
    }


def summarize_transport_by_method(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    totals: dict[str, dict[str, int]] = {}
    for point in summary["points"]:
        method = str(point["method"])
        target = totals.setdefault(method, {"applicable": 0, "exact": 0, "failures": 0})
        transport = point["transport_recovery"]
        target["applicable"] += int(transport["applicable_positive_payload_count"])
        target["exact"] += int(transport["exact_prefix_count"])
        target["failures"] += int(transport["failure_count"])
    result: dict[str, dict[str, Any]] = {}
    for method, values in totals.items():
        applicable = values["applicable"]
        result[method] = {
            **values,
            "exact_prefix_rate": (values["exact"] / applicable) if applicable else None,
        }
    return result


def build_interpretation(
    points: list[dict[str, Any]],
    summary: dict[str, Any],
    precision_interpretation: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    if len(points) != 23:
        raise ValueError(f"expected 23 Figure-3 points, found {len(points)}")
    if not summary.get("execution_gate_ready_for_step_3_11_interpretation"):
        raise ValueError("Step-3.10 execution gate is not READY FOR STEP 3.11 INTERPRETATION")
    if int(summary.get("run_count", -1)) != 5520:
        raise ValueError("expected completed Step-3.10 run_count=5520")

    bins = sorted(method_points(points, "bins"), key=lambda point: float(point["mean_bits_per_word"]))
    huffman = sorted(method_points(points, "huffman"), key=lambda point: float(point["mean_bits_per_word"]))
    arithmetic = arithmetic_k300_points(points)
    special = special_arithmetic_point(points)

    if len(bins) != 5 or len(huffman) != 8 or len(arithmetic) != 9:
        raise ValueError("Figure-3 method point counts do not match 5 Bins / 8 Huffman / 9 Arithmetic k=300")

    arithmetic_min = min(arithmetic, key=lambda point: float(point["mean_kl_bits"]))
    arithmetic_temps = [float(point["parameters"]["temperature"]) for point in arithmetic]
    arithmetic_kl = [float(point["mean_kl_bits"]) for point in arithmetic]
    min_index = arithmetic.index(arithmetic_min)
    decreases_to_min = nonincreasing(arithmetic_kl[: min_index + 1])
    increases_after_min = all(
        right > left for left, right in zip(arithmetic_kl[min_index:], arithmetic_kl[min_index + 1:])
    )

    bins_bpw = [float(point["mean_bits_per_word"]) for point in bins]
    bins_kl = [float(point["mean_kl_bits"]) for point in bins]
    huffman_bpw = [float(point["mean_bits_per_word"]) for point in huffman]
    huffman_kl = [float(point["mean_kl_bits"]) for point in huffman]

    grid_points = int(config["policy"]["dominance_check"]["grid_points"])
    dominance_huffman = compare_mean_curves(arithmetic, huffman, grid_points=grid_points)
    dominance_bins = compare_mean_curves(arithmetic, bins, grid_points=grid_points)

    paper_anchor_nats = float(config["paper"]["reported_unmodulated_kl_nats"])
    special_bits = float(special["mean_kl_bits"])
    special_nats = special_bits * math.log(2.0)
    precision40_nats = float(
        precision_interpretation["by_precision"]["40"]["mean_zero_padding_free_author_kl_nats"]
    )
    precision26_clean_nats = float(
        precision_interpretation["by_precision"]["26"]["mean_zero_padding_free_author_kl_nats"]
    )

    transport_by_method = summarize_transport_by_method(summary)
    arithmetic_completed = sum(
        int(point["terminated_sentence_run_count"])
        for point in summary["points"]
        if point["method"] == "arithmetic"
    )
    arithmetic_scheduled = sum(
        int(point["scheduled_run_count"])
        for point in summary["points"]
        if point["method"] == "arithmetic"
    )
    arithmetic_failures = arithmetic_scheduled - arithmetic_completed
    zero_payload = int(summary["zero_payload_diagnostic"]["count"])

    claims = [
        {
            "claim_id": "paper_orchestration",
            "paper_claim": "GPT-2 345M + CNN/DailyMail first-three-sentence context + one generated sentence under uniform random message; repeated Monte-Carlo samples.",
            "status": "partial_reproduction",
            "evidence": "GPT-2 Medium, pinned CNN/DailyMail contexts, first-boundary sentence protocol, 80 contexts x 3 replicates and all 23 tuning points were executed; the paper does not expose the exact split, Figure-3 sample count, or original batch driver.",
        },
        {
            "claim_id": "bins_curve",
            "paper_claim": "Block/Bins occupies the high-KL region of Figure 3 as capacity increases.",
            "status": "trend_reproduction",
            "evidence": f"Reproduced Bins means: bits/word {bins_bpw[0]:.1f}->{bins_bpw[-1]:.1f}, KL {bins_kl[0]:.3f}->{bins_kl[-1]:.3f} bits; all five means are in the high-KL region.",
        },
        {
            "claim_id": "huffman_curve",
            "paper_claim": "Huffman trades larger candidate pools / higher capacity for lower KL than Block/Bins.",
            "status": "trend_reproduction",
            "evidence": f"Huffman bits/word rises {huffman_bpw[0]:.3f}->{huffman_bpw[-1]:.3f} while KL falls {huffman_kl[0]:.3f}->{huffman_kl[-1]:.3f} bits; reproduced KL means are non-increasing across all 8 points.",
        },
        {
            "claim_id": "arithmetic_temperature_curve",
            "paper_claim": "Arithmetic temperature sweep reaches minimum KL around 4 bits/word at temperature=1.0.",
            "status": "trend_reproduction",
            "evidence": f"Minimum reproduced k=300 mean occurs at temperature={float(arithmetic_min['parameters']['temperature']):.1f}, {float(arithmetic_min['mean_bits_per_word']):.3f} bits/word and KL={float(arithmetic_min['mean_kl_bits']):.6f} bits; KL decreases through the minimum and rises again above temperature 1.0.",
        },
        {
            "claim_id": "arithmetic_dominates_baselines",
            "paper_claim": "Arithmetic gives the lowest Figure-3 KL trade-off relative to Huffman and Block/Bins.",
            "status": "trend_reproduction",
            "evidence": f"Piecewise-linear reproduced mean curves: Arithmetic is below Huffman at all {grid_points} grid points on {dominance_huffman['common_bits_per_word_interval'][0]:.3f}-{dominance_huffman['common_bits_per_word_interval'][1]:.3f} bits/word (minimum margin {dominance_huffman['minimum_baseline_minus_arithmetic_kl_bits']:.3f} bits) and below Bins on the common interval (minimum margin {dominance_bins['minimum_baseline_minus_arithmetic_kl_bits']:.3f} bits).",
        },
        {
            "claim_id": "unmodulated_near_zero_behavior",
            "paper_claim": "Unmodulated Arithmetic (temperature=1, k=50256) is near the language-model distribution and has near-zero KL.",
            "status": "partial_reproduction",
            "evidence": f"Reproduced special point: {float(special['mean_bits_per_word']):.3f} bits/word, {special_bits:.9f} KL bits = {special_nats:.9g} nats; this is {float(arithmetic_min['mean_kl_bits']) / special_bits:.1f}x lower than the reproduced k=300 temperature=1 point, but not at the paper's exact prose anchor.",
        },
        {
            "claim_id": "unmodulated_4e_minus_8_nats",
            "paper_claim": "The unmodulated language-model Arithmetic point has KL = 4e-8 nats.",
            "status": "not_reproducible",
            "evidence": f"Pinned executable precision=26 full sentence sweep gives {special_nats:.9g} nats, {special_nats / paper_anchor_nats:.1f}x the paper anchor. Independent zero-padding-free precision probe gives p26={precision26_clean_nats:.9g} nats and p40={precision40_nats:.9g} nats; p40 is the same order as 4e-8, but the public run_single.py sets precision=26 and the historical Figure-3 driver is unavailable, so precision=40 cannot be attributed to the authors.",
        },
        {
            "claim_id": "historical_mc_estimator",
            "paper_claim": "Figure 3 reports mean and standard error over repeated samples for each tuning point.",
            "status": "partial_reproduction",
            "evidence": "The reproduction reports run-level and context-clustered standard errors over a predeclared 80-context x 3-replicate design. The paper does not publish the original Figure-3 sample count or exact estimator implementation, so exact historical Monte-Carlo orchestration cannot be claimed.",
        },
    ]

    if not strictly_increasing(bins_bpw):
        raise AssertionError("Bins bits/word should strictly increase")
    if not strictly_increasing(huffman_bpw):
        raise AssertionError("Huffman bits/word should strictly increase")
    if not nonincreasing(huffman_kl):
        raise AssertionError("Huffman mean KL should be non-increasing across reproduced points")
    if float(arithmetic_min["parameters"]["temperature"]) != 1.0:
        raise AssertionError("Arithmetic k=300 minimum is not at temperature=1.0")
    if not (decreases_to_min and increases_after_min):
        raise AssertionError("Arithmetic k=300 curve does not decrease to tau=1.0 then rise")
    if not dominance_huffman["arithmetic_lower_at_every_grid_point"]:
        raise AssertionError("Arithmetic reproduced mean curve is not below Huffman on common interval")
    if not dominance_bins["arithmetic_lower_at_every_grid_point"]:
        raise AssertionError("Arithmetic reproduced mean curve is not below Bins on common interval")

    return {
        "schema_version": "stage3.figure3_interpretation.v1",
        "baseline_commit": str(config["baseline_commit"]),
        "paper": config["paper"],
        "scientific_claims_evaluated": True,
        "execution_input": {
            "scheduled_runs": int(summary["scheduled_run_count"]),
            "terminated_sentence_runs": int(summary["terminated_sentence_run_count"]),
            "termination_failures": int(summary["termination_failure_count"]),
            "termination_success_rate": float(summary["termination_success_rate"]),
            "point_count": int(summary["point_count"]),
        },
        "curves": {
            "bins": {
                "point_count": len(bins),
                "bits_per_word": bins_bpw,
                "kl_bits": bins_kl,
                "high_kl_tradeoff_reproduced": True,
            },
            "huffman": {
                "point_count": len(huffman),
                "bits_per_word": huffman_bpw,
                "kl_bits": huffman_kl,
                "kl_nonincreasing": True,
            },
            "arithmetic_k300": {
                "point_count": len(arithmetic),
                "temperatures": arithmetic_temps,
                "bits_per_word": [float(point["mean_bits_per_word"]) for point in arithmetic],
                "kl_bits": arithmetic_kl,
                "minimum": {
                    "temperature": float(arithmetic_min["parameters"]["temperature"]),
                    "bits_per_word": float(arithmetic_min["mean_bits_per_word"]),
                    "kl_bits": float(arithmetic_min["mean_kl_bits"]),
                    "se_kl_bits_runs": float(arithmetic_min["se_kl_bits_runs"]),
                },
                "decreases_to_tau_1_then_increases": True,
            },
            "arithmetic_k50256": {
                "bits_per_word": float(special["mean_bits_per_word"]),
                "kl_bits": special_bits,
                "kl_nats": special_nats,
                "paper_anchor_nats": paper_anchor_nats,
                "ratio_to_paper_anchor": special_nats / paper_anchor_nats,
                "kl_reduction_vs_k300_tau1_factor": float(arithmetic_min["mean_kl_bits"]) / special_bits,
            },
        },
        "curve_dominance_diagnostic": {
            "arithmetic_vs_huffman": dominance_huffman,
            "arithmetic_vs_bins": dominance_bins,
        },
        "precision_discrepancy": {
            "precision26_clean_kl_nats": precision26_clean_nats,
            "precision40_clean_kl_nats": precision40_nats,
            "precision40_to_paper_anchor_ratio": precision40_nats / paper_anchor_nats,
            "paper_anchor_not_reproduced_at_pinned_precision26": True,
            "do_not_infer_authors_used_precision40": True,
        },
        "reliability_diagnostics": {
            "transport_recovery_by_method": transport_by_method,
            "global_transport_recovery": summary["transport_recovery_diagnostic"],
            "arithmetic_scheduled_runs": arithmetic_scheduled,
            "arithmetic_completed_runs": arithmetic_completed,
            "arithmetic_termination_failures": arithmetic_failures,
            "arithmetic_termination_success_rate": arithmetic_completed / arithmetic_scheduled,
            "zero_payload_completed_arithmetic_count": zero_payload,
            "zero_payload_rate_among_completed_arithmetic": zero_payload / arithmetic_completed,
            "cache_compatibility": summary["arithmetic_cache_compatibility_diagnostic"],
            "sentence_guard": summary["sentence_guard_escalation_diagnostic"],
        },
        "claims": claims,
        "overall_assessment": {
            "status": "partial_reproduction",
            "summary": "The characteristic Figure-3 trade-off curves and method ordering are reproduced, including the Arithmetic minimum at temperature=1.0 near 4 bits/word. The exact unmodulated 4e-8-nat prose anchor is not numerically reproduced by the pinned public precision=26 executable, and exact historical Monte-Carlo orchestration is unavailable.",
            "ready_for_step_3_12_matched_author_vs_normalized": True,
        },
    }


def write_claims_csv(path: Path, interpretation: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["claim_id", "paper_claim", "status", "evidence"])
        writer.writeheader()
        writer.writerows(interpretation["claims"])


def _marker_svg(method: str, x: float, y: float, *, special: bool = False) -> str:
    if special:
        points = []
        for idx in range(10):
            angle = -math.pi / 2 + idx * math.pi / 5
            radius = 8 if idx % 2 == 0 else 3.5
            points.append(f"{x + radius * math.cos(angle):.2f},{y + radius * math.sin(angle):.2f}")
        return f'<polygon points="{" ".join(points)}" fill="white" stroke="black" stroke-width="1.5"/>'
    if method == "bins":
        return f'<rect x="{x-4:.2f}" y="{y-4:.2f}" width="8" height="8" fill="white" stroke="black" stroke-width="1.4"/>'
    if method == "huffman":
        pts = f"{x:.2f},{y-5:.2f} {x-5:.2f},{y+4:.2f} {x+5:.2f},{y+4:.2f}"
        return f'<polygon points="{pts}" fill="white" stroke="black" stroke-width="1.4"/>'
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="white" stroke="black" stroke-width="1.4"/>'


def render_svg(path: Path, points: list[dict[str, Any]]) -> None:
    width, height = 920, 620
    left, right, top, bottom = 90, 40, 55, 85
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_min, x_max = 0.8, 5.2
    y_min, y_max = 0.0, 3.5

    def sx(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return top + (y_max - y) / (y_max - y_min) * plot_h

    bins = sorted(method_points(points, "bins"), key=lambda point: float(point["mean_bits_per_word"]))
    huffman = sorted(method_points(points, "huffman"), key=lambda point: float(point["mean_bits_per_word"]))
    arithmetic = arithmetic_k300_points(points)
    special = special_arithmetic_point(points)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#111}.axis{stroke:#111;stroke-width:1.2}.grid{stroke:#ddd;stroke-width:1}.curve{fill:none;stroke:#111;stroke-width:1.8}.err{stroke:#666;stroke-width:1}.label{font-size:14px}.small{font-size:12px}.title{font-size:19px;font-weight:600}</style>',
        f'<text class="title" x="{width/2:.1f}" y="28" text-anchor="middle">Stage 3.11 — reproduced Figure-3 trade-off</text>',
    ]

    for y in [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
        py = sy(y)
        lines.append(f'<line class="grid" x1="{left}" y1="{py:.2f}" x2="{left+plot_w}" y2="{py:.2f}"/>')
        lines.append(f'<text class="small" x="{left-12}" y="{py+4:.2f}" text-anchor="end">{y:g}</text>')
    for x in [1, 2, 3, 4, 5]:
        px = sx(x)
        lines.append(f'<line class="grid" x1="{px:.2f}" y1="{top}" x2="{px:.2f}" y2="{top+plot_h}"/>')
        lines.append(f'<text class="small" x="{px:.2f}" y="{top+plot_h+24}" text-anchor="middle">{x}</text>')

    lines.append(f'<line class="axis" x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}"/>')
    lines.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/>')
    lines.append(f'<text class="label" x="{left+plot_w/2:.2f}" y="{height-28}" text-anchor="middle">Bits / word (author-compatible)</text>')
    lines.append(f'<text class="label" x="24" y="{top+plot_h/2:.2f}" text-anchor="middle" transform="rotate(-90 24 {top+plot_h/2:.2f})">KL(Q_stego || P_LM), bits</text>')

    for method, group, dash in [
        ("bins", bins, ""),
        ("huffman", huffman, "7 4"),
        ("arithmetic", arithmetic, "2 3"),
    ]:
        coords = " ".join(f"{sx(float(p['mean_bits_per_word'])):.2f},{sy(float(p['mean_kl_bits'])):.2f}" for p in group)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        lines.append(f'<polyline class="curve" points="{coords}"{dash_attr}/>')
        for point in group:
            x = float(point["mean_bits_per_word"])
            y = float(point["mean_kl_bits"])
            se_x = float(point["se_bits_per_word_runs"])
            se_y = float(point["se_kl_bits_runs"])
            px, py = sx(x), sy(y)
            x0, x1 = sx(max(x_min, x - se_x)), sx(min(x_max, x + se_x))
            y0, y1 = sy(min(y_max, y + se_y)), sy(max(y_min, y - se_y))
            lines.append(f'<line class="err" x1="{x0:.2f}" y1="{py:.2f}" x2="{x1:.2f}" y2="{py:.2f}"/>')
            lines.append(f'<line class="err" x1="{px:.2f}" y1="{y0:.2f}" x2="{px:.2f}" y2="{y1:.2f}"/>')
            lines.append(_marker_svg(method, px, py))

    spx, spy = sx(float(special["mean_bits_per_word"])), sy(float(special["mean_kl_bits"]))
    lines.append(_marker_svg("arithmetic", spx, spy, special=True))

    legend_x, legend_y = 650, 88
    legend = [
        ("Bins / Block", "bins", False),
        ("Huffman", "huffman", False),
        ("Arithmetic k=300", "arithmetic", False),
        ("Arithmetic k=50256", "arithmetic", True),
    ]
    for idx, (label, method, is_special) in enumerate(legend):
        y = legend_y + idx * 25
        lines.append(_marker_svg(method, legend_x, y, special=is_special))
        lines.append(f'<text class="small" x="{legend_x+18}" y="{y+4}">{html.escape(label)}</text>')

    lines.append(f'<text class="small" x="{left}" y="{height-8}">Error bars: run-level SE. Sentence means are conditional on successful termination.</text>')
    lines.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
