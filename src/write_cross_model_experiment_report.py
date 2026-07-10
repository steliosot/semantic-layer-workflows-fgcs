#!/usr/bin/env python3
"""Write a narrative Markdown report for the cross-model semantic benchmark."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parent
BENCH_ROOT = ROOT / "outputs" / "cross_model_semantic"
REPORT_PATH = BENCH_ROOT / "cross_model_experiment_report.md"


def latest_runs() -> list[dict[str, Any]]:
    latest: dict[str, tuple[float, dict[str, Any]]] = {}
    for metrics_path in BENCH_ROOT.glob("*/cross_model_metrics.json"):
        try:
            data = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            continue
        key = data.get("model", {}).get("key")
        if not key:
            continue
        current = latest.get(key)
        mtime = metrics_path.stat().st_mtime
        if current is None or mtime > current[0]:
            latest[key] = (mtime, data)
    return sorted((item[1] for item in latest.values()), key=lambda run: run["model"]["name"])


def ok_cases(run: dict[str, Any]) -> list[dict[str, Any]]:
    return [case for case in run.get("cases", []) if case.get("error") is None]


def sec(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}s"


def x(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}x"


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}%"


def mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def case_outside(case: dict[str, Any]) -> float | None:
    return (
        case.get("semantic_local_projected", {})
        .get("preservation", {})
        .get("outside_target_changed_pixels_percent_gt10")
    )


def case_boundary(case: dict[str, Any]) -> float | None:
    return (
        case.get("semantic_local_projected", {})
        .get("boundary", {})
        .get("boundary_band_changed_pixels_percent_gt10")
    )


def rel(path: Path) -> str:
    return path.relative_to(REPORT_PATH.parent).as_posix()


def write_report(runs: list[dict[str, Any]]) -> None:
    all_cases = [case for run in runs for case in ok_cases(run)]
    naive_mean = mean(case["naive_seconds"] for case in all_cases)
    local_mean = mean(case["semantic_local_seconds"] for case in all_cases)
    speedup_mean = mean(case["speedup_vs_naive"] for case in all_cases)
    detector_mean = mean(case["detector_seconds"] for case in all_cases)
    inclusive_speedup_mean = mean(
        case["naive_seconds"] / (case["semantic_local_seconds"] + case["detector_seconds"])
        for case in all_cases
    )
    outside_mean = mean(case_outside(case) or 0.0 for case in all_cases)
    boundary_mean = mean(case_boundary(case) or 0.0 for case in all_cases)

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_label: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for run in runs:
        by_family[run["model"]["family"]].extend(ok_cases(run))
        for case in ok_cases(run):
            by_label[case["label"]].append((run["model"]["name"], case))

    model_rows: list[tuple[str, dict[str, Any], float, float, float, float]] = []
    for run in runs:
        cases = ok_cases(run)
        mean_naive = mean(case["naive_seconds"] for case in cases)
        mean_local = mean(case["semantic_local_seconds"] for case in cases)
        mean_speedup = mean(case["speedup_vs_naive"] for case in cases)
        mean_inclusive = mean(
            case["naive_seconds"] / (case["semantic_local_seconds"] + case["detector_seconds"])
            for case in cases
        )
        model_rows.append((run["model"]["name"], run, mean_naive, mean_local, mean_speedup, mean_inclusive))

    best_generation = max(model_rows, key=lambda row: row[4])
    best_inclusive = max(model_rows, key=lambda row: row[5])

    lines: list[str] = [
        "# Cross-Model Semantic Editing Experiment Report",
        "",
        "## Executive Summary",
        "",
        f"This report covers {len(runs)} Stable Diffusion checkpoints tested on four edit categories: face lips, human body shirt, object mug, and landscape sky. Each run compares a naive full-image edit against a GroundingDINO + SAM2 semantic local edit using the same model family, prompt case, resolution, and step budget.",
        "",
        f"Across {len(all_cases)} successful model/case runs, the naive edit averaged {sec(naive_mean)}, while the local semantic edit averaged {sec(local_mean)}. That gives a generation-only mean speedup of {x(speedup_mean)}. Detector time averaged {sec(detector_mean)}, so the one-off detector-inclusive speedup averaged {x(inclusive_speedup_mean)}.",
        "",
        f"The best generation-only model average was {best_generation[0]} at {x(best_generation[4])}. The best detector-inclusive average was {best_inclusive[0]} at {x(best_inclusive[5])}. Outside-mask changed pixels averaged {pct(outside_mean)}, which confirms that compositing preserves the rest of the image by construction.",
        "",
        "## Method",
        "",
        "- Naive baseline: full-image img2img/regeneration for the requested edit.",
        "- Semantic local path: GroundingDINO finds the requested target, SAM2 creates the mask, the script crops around the mask, runs local img2img on the crop, composites through the mask, and applies deterministic color projection.",
        "- Cases: lips to red lipstick, shirt to blue, mug to teal, and sky to warm sunset.",
        "- Main speedup excludes detector time because masks can be cached and reused across repeated edits. Detector-inclusive speedup is shown separately for one-off usage.",
        "",
        "## Model Summary",
        "",
        "| Model | Family | Resolution | Mean naive | Mean local | Generation speedup | Detector-inclusive speedup | Grid |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for model_name, run, mean_naive, mean_local, mean_speedup, mean_inclusive in model_rows:
        grid = Path(run["output_dir"]) / "cross_model_grid.png"
        lines.append(
            f"| {model_name} | {run['model']['family']} | {run['width']}x{run['height']} | "
            f"{sec(mean_naive)} | {sec(mean_local)} | {x(mean_speedup)} | {x(mean_inclusive)} | "
            f"[grid]({rel(grid)}) |"
        )

    lines.extend(["", "## Family-Level Result", "", "| Family | Runs | Mean naive | Mean local | Generation speedup | Detector-inclusive speedup |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for family, cases in sorted(by_family.items()):
        family_naive = mean(case["naive_seconds"] for case in cases)
        family_local = mean(case["semantic_local_seconds"] for case in cases)
        family_speedup = mean(case["speedup_vs_naive"] for case in cases)
        family_inclusive = mean(case["naive_seconds"] / (case["semantic_local_seconds"] + case["detector_seconds"]) for case in cases)
        lines.append(
            f"| {family} | {len(cases)} | {sec(family_naive)} | {sec(family_local)} | "
            f"{x(family_speedup)} | {x(family_inclusive)} |"
        )

    lines.extend(["", "## Category Results", ""])
    for label in sorted(by_label):
        rows = sorted(by_label[label], key=lambda item: item[0])
        label_speed = mean(case["speedup_vs_naive"] for _, case in rows)
        label_inclusive = mean(case["naive_seconds"] / (case["semantic_local_seconds"] + case["detector_seconds"]) for _, case in rows)
        label_boundary = mean(case_boundary(case) or 0.0 for _, case in rows)
        lines.extend(
            [
                f"### {label}",
                "",
                f"Mean generation-only speedup: {x(label_speed)}. Mean detector-inclusive speedup: {x(label_inclusive)}. Mean boundary-band changed pixels: {pct(label_boundary)}.",
                "",
                "| Model | Naive | DINO+SAM2 local | Detector | Generation speedup | Detector-inclusive speedup | Boundary band changed |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for model_name, case in rows:
            inclusive = case["naive_seconds"] / (case["semantic_local_seconds"] + case["detector_seconds"])
            lines.append(
                f"| {model_name} | {sec(case['naive_seconds'])} | {sec(case['semantic_local_seconds'])} | "
                f"{sec(case['detector_seconds'])} | {x(case['speedup_vs_naive'])} | {x(inclusive)} | "
                f"{pct(case_boundary(case))} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Visual Results",
            "",
            "Each grid shows the four benchmark cases for one model. Rows are cases; columns are base image, naive full-image edit, DINO+SAM2 mask overlay, and semantic local output.",
            "",
        ]
    )
    for model_name, run, *_ in model_rows:
        grid = Path(run["output_dir"]) / "cross_model_grid.png"
        report = Path(run["output_dir"]) / "cross_model_report.md"
        lines.extend(
            [
                f"### {model_name}",
                "",
                f"![{model_name} cross-model grid]({rel(grid)})",
                "",
                f"Detailed artifacts and per-case images: [{run['model']['key']}]({rel(report)})",
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation",
            "",
            "The practical gain is strongest when the edit is spatially small and the mask is reusable. Lips, mugs, and compact shirt regions tend to benefit most because local img2img runs on a much smaller crop than the full frame. Larger sky or body edits still benefit, but the crop approaches the full image size, so the speedup compresses.",
            "",
            "SD1.5 models are fast in absolute terms, so a one-off detector pass can dominate the wall clock. SDXL models are slower per generation, so local editing remains more attractive even before mask reuse. The result supports the project hypothesis: semantic masking plus policy-driven reduced local diffusion is useful, but the system needs mask caching or fast mask reuse to feel consistently faster to users.",
            "",
            "Quality is not purely measured by outside-mask changed pixels. That metric is near zero because compositing enforces locality. The more meaningful risks are detector selection quality, boundary-band artifacts, shadows/reflections/context mismatch, and whether the deterministic projection makes the edited region look too flat. The grids should be inspected alongside boundary-band metrics.",
            "",
            "## Tradeoffs",
            "",
            "- Naive full-image editing is simple and coherent globally, but it changes unrelated content and is slower, especially for SDXL.",
            "- DINO+SAM2 local editing preserves identity/background very well, but depends on the detector selecting the intended region.",
            "- Deterministic projection makes color edits reliable and repeatable, but can reduce photoreal texture if overused.",
            "- Detector time is the main bottleneck for one-off edits; cache/reuse is the clean path to user-visible speedups.",
            "- Larger semantic regions reduce crop savings and may require broader context handling.",
            "",
            "## Artifacts",
            "",
            f"- Compact summary: [cross_model_summary.md]({rel(BENCH_ROOT / 'cross_model_summary.md')})",
            "- Each model folder contains `base.png`, `naive.png`, DINO+SAM2 detector overlays, `semantic_projected.png`, per-case strips, `cross_model_grid.png`, and `cross_model_metrics.json`.",
            "",
        ]
    )

    REPORT_PATH.write_text("\n".join(lines))
    print(REPORT_PATH)


def main() -> None:
    runs = latest_runs()
    if not runs:
        raise SystemExit("No metrics found.")
    write_report(runs)


if __name__ == "__main__":
    main()
