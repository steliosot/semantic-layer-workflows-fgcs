#!/usr/bin/env python3
"""Aggregate latest cross-model semantic benchmark runs into one Markdown report."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parent
BENCH_ROOT = ROOT / "outputs" / "cross_model_semantic"


def load_latest_runs() -> list[dict[str, Any]]:
    candidates: dict[str, tuple[float, Path, dict[str, Any]]] = {}
    for metrics_path in BENCH_ROOT.glob("*/cross_model_metrics.json"):
        try:
            metrics = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            continue
        key = metrics.get("model", {}).get("key")
        if not key:
            continue
        mtime = metrics_path.stat().st_mtime
        current = candidates.get(key)
        if current is None or mtime > current[0]:
            candidates[key] = (mtime, metrics_path, metrics)
    return [item[2] for item in sorted(candidates.values(), key=lambda item: item[2].get("model", {}).get("name", ""))]


def fmt_seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}s"


def fmt_speedup(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}x"


def fmt_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}%"


def write_report(runs: list[dict[str, Any]], path: Path) -> None:
    lines: list[str] = [
        "# Cross-Model Semantic Benchmark Summary",
        "",
        "Latest run per model. Speedup compares full-image naive edit time against DINO+SAM2 masked local edit time; detector time is recorded separately and excluded from this speedup.",
        "",
        "## Models",
        "",
        "| Model | Family | Resolution | Mean naive | Mean local | Mean speedup | Report |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]

    by_case: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for run in runs:
        model = run["model"]
        cases = [case for case in run["cases"] if case.get("error") is None]
        mean_naive = mean(case["naive_seconds"] for case in cases) if cases else None
        mean_local = mean(case["semantic_local_seconds"] for case in cases) if cases else None
        mean_speedup = mean(case["speedup_vs_naive"] for case in cases) if cases else None
        report = Path(run["output_dir"]) / "cross_model_report.md"
        rel_report = report.relative_to(path.parent)
        lines.append(
            f"| {model['name']} | {model['family']} | {run['width']}x{run['height']} | "
            f"{fmt_seconds(mean_naive)} | {fmt_seconds(mean_local)} | {fmt_speedup(mean_speedup)} | [{model['key']}]({rel_report}) |"
        )
        for case in cases:
            by_case[case["label"]].append((model["name"], case))

    lines.extend(["", "## Case Tables", ""])
    for case_label in sorted(by_case):
        lines.extend(
            [
                f"### {case_label}",
                "",
                "| Model | Naive | DINO+SAM2 local | Speedup | Local outside changed | Detector |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        rows = sorted(by_case[case_label], key=lambda item: item[0])
        for model_name, case in rows:
            outside = case.get("semantic_local_projected", {}).get("preservation", {}).get("outside_target_changed_pixels_percent_gt10")
            lines.append(
                f"| {model_name} | {fmt_seconds(case['naive_seconds'])} | "
                f"{fmt_seconds(case['semantic_local_seconds'])} | {fmt_speedup(case['speedup_vs_naive'])} | "
                f"{fmt_percent(outside)} | {fmt_seconds(case.get('detector_seconds'))} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "- Detector time is shown separately because it can be cached/reused for repeated edits on the same image and semantic region.",
            "- Very high speedups can indicate an actually efficient local crop, but can also indicate that DINO+SAM2 selected a small or imperfect region. Always inspect the grid and mask overlay for those rows.",
            "- Outside-mask changed pixels are computed after compositing, so they should remain near zero by construction; seam and boundary metrics in each JSON are more informative for edge quality.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    BENCH_ROOT.mkdir(parents=True, exist_ok=True)
    runs = load_latest_runs()
    if not runs:
        raise SystemExit("No cross_model_metrics.json files found.")
    write_report(runs, BENCH_ROOT / "cross_model_summary.md")
    print(BENCH_ROOT / "cross_model_summary.md")


if __name__ == "__main__":
    main()
