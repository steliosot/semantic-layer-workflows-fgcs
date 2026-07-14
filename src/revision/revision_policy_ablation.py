#!/usr/bin/env python3
"""Policy ablation and hardware stress analysis for the FGCS revision."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "outputs/heavy_paper_benchmark/heavy_all_models_20steps_5samples/heavy_paper_benchmark_metrics.json"
OUT_DIR = ROOT / "paper_fgcs_semantic_workflows/revision_outputs"
OUT_JSON = OUT_DIR / "policy_ablation.json"
OUT_MD = OUT_DIR / "policy_ablation_summary.md"


def semantic_cost(r: dict, k: int, det_scale: float = 1.0, diff_scale: float = 1.0, overhead: float = 0.0) -> float:
    return (
        det_scale * r["detector_seconds"]
        + diff_scale * r["semantic_v1_seconds"]
        + overhead
        + (k - 1) * (diff_scale * r["semantic_v2_cached_seconds"] + overhead)
    )


def naive_cost(r: dict, k: int, diff_scale: float = 1.0, overhead: float = 0.0) -> float:
    return diff_scale * r["naive_v1_seconds"] + overhead + (k - 1) * (diff_scale * r["naive_v2_seconds"] + overhead)


def break_even(r: dict, det_scale: float = 1.0, diff_scale: float = 1.0, overhead: float = 0.0) -> float:
    n1 = diff_scale * r["naive_v1_seconds"] + overhead
    n2 = diff_scale * r["naive_v2_seconds"] + overhead
    s1 = det_scale * r["detector_seconds"] + diff_scale * r["semantic_v1_seconds"] + overhead
    c = diff_scale * r["semantic_v2_cached_seconds"] + overhead
    denom = n2 - c
    if denom <= 0:
        return math.inf
    return 1 + (s1 - n1) / denom


def summarize_wins(records: list[dict], det_scale: float, diff_scale: float, overhead: float) -> dict:
    out = {}
    for k in [1, 2, 3, 5]:
        wins = sum(semantic_cost(r, k, det_scale, diff_scale, overhead) < naive_cost(r, k, diff_scale, overhead) for r in records)
        out[f"k{k}_wins_percent"] = 100 * wins / len(records)
        out[f"k{k}_semantic_mean"] = mean(semantic_cost(r, k, det_scale, diff_scale, overhead) for r in records)
        out[f"k{k}_naive_mean"] = mean(naive_cost(r, k, diff_scale, overhead) for r in records)
    be = [break_even(r, det_scale, diff_scale, overhead) for r in records]
    finite = [x for x in be if math.isfinite(x)]
    out["break_even_mean"] = mean(finite)
    out["break_even_median"] = median(finite)
    out["break_even_le2_percent"] = 100 * sum(x <= 2 for x in finite) / len(finite)
    out["break_even_le3_percent"] = 100 * sum(x <= 3 for x in finite) / len(finite)
    return out


def policy_choice(r: dict, policy: str, k: int) -> str:
    naive = naive_cost(r, k)
    sem = semantic_cost(r, k)
    if policy == "latency_only":
        return "semantic" if sem < naive else "naive"
    if policy == "preservation_first":
        return "semantic"
    if policy == "balanced":
        if r["outside_naive_v1_changed_percent"] > 50 and k >= 2:
            return "semantic"
        return "semantic" if sem < naive else "naive"
    if policy == "oracle":
        return "semantic" if sem < naive else "naive"
    raise ValueError(policy)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = json.loads(METRICS_PATH.read_text())["records"]
    records = [r for r in records if not r.get("error")]

    scenarios = {
        "edge_constrained": (1.80, 1.50, 0.0),
        "measured_mps": (1.00, 1.00, 0.0),
        "cuda_workstation": (0.35, 0.45, 0.0),
        "cloud_accelerator": (0.15, 0.20, 1.5),
        "fast_detector_slow_diffusion": (0.15, 1.00, 0.0),
        "slow_detector_fast_diffusion": (1.50, 0.25, 0.0),
    }
    scenario_summary = {
        name: summarize_wins(records, det, diff, overhead)
        for name, (det, diff, overhead) in scenarios.items()
    }

    policy_summary = {}
    for k in [1, 2, 3, 5]:
        policy_summary[f"k{k}"] = {}
        for policy in ["latency_only", "preservation_first", "balanced", "oracle"]:
            choices = [policy_choice(r, policy, k) for r in records]
            costs = [semantic_cost(r, k) if c == "semantic" else naive_cost(r, k) for r, c in zip(records, choices)]
            drift = [0.0 if c == "semantic" else r["outside_naive_v1_changed_percent"] for r, c in zip(records, choices)]
            policy_summary[f"k{k}"][policy] = {
                "semantic_selected_percent": 100 * choices.count("semantic") / len(choices),
                "mean_latency": mean(costs),
                "mean_outside_drift": mean(drift),
            }

    result = {
        "records": len(records),
        "scenarios": scenario_summary,
        "policy_ablation": policy_summary,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))

    lines = [
        "# Policy Ablation Summary",
        "",
        "## Hardware Stress Scenarios",
        "",
        "| Scenario | Median break-even k | k=1 wins | k=2 wins | k=3 wins | k=5 wins |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, vals in scenario_summary.items():
        lines.append(
            f"| {name} | {vals['break_even_median']:.2f} | {vals['k1_wins_percent']:.1f}% | "
            f"{vals['k2_wins_percent']:.1f}% | {vals['k3_wins_percent']:.1f}% | {vals['k5_wins_percent']:.1f}% |"
        )
    lines.extend(["", "## Policy Ablation", ""])
    for k, vals in policy_summary.items():
        lines.append(f"### {k}")
        lines.append("")
        lines.append("| Policy | Semantic selected | Mean latency | Mean outside drift |")
        lines.append("| --- | ---: | ---: | ---: |")
        for policy, row in vals.items():
            lines.append(
                f"| {policy} | {row['semantic_selected_percent']:.1f}% | "
                f"{row['mean_latency']:.3f}s | {row['mean_outside_drift']:.3f}% |"
            )
        lines.append("")
    OUT_MD.write_text("\n".join(lines))
    print(OUT_JSON)
    print(OUT_MD)


if __name__ == "__main__":
    main()
