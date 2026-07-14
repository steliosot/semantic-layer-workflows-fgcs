#!/usr/bin/env python3
"""Generate a top-5 cross-model report from the full heavy benchmark.

The GCP CUDA revision measures one SDXL checkpoint in detail. This script
summarizes the already completed 200-sample local benchmark across 10
compatible SD1.5/SDXL checkpoints and ranks the top five models by cached
semantic re-edit speedup.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT = ROOT.parent / "outputs" / "heavy_paper_benchmark" / "heavy_all_models_20steps_5samples" / "heavy_paper_benchmark_metrics.json"
OUT_DIR = ROOT / "revision_outputs" / "top5_cross_model_heavy"


def main() -> None:
    data = json.loads(INPUT.read_text())
    by_model = data["aggregates"]["by_model"]
    top = sorted(by_model, key=lambda row: row["mean_cached_reedit_speedup"], reverse=True)[:5]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for rank, row in enumerate(top, start=1):
        rows.append(
            {
                "rank": rank,
                "model_name": row["model_name"],
                "samples": row["count"],
                "mean_naive_v1_seconds": row["mean_naive_v1_seconds"],
                "mean_semantic_v1_seconds": row["mean_semantic_v1_seconds"],
                "mean_detector_seconds": row["mean_detector_seconds"],
                "mean_cached_reedit_seconds": row["mean_cached_reedit_seconds"],
                "mean_generation_speedup": row["mean_generation_speedup"],
                "mean_detector_inclusive_speedup": row["mean_detector_inclusive_speedup"],
                "mean_cached_reedit_speedup": row["mean_cached_reedit_speedup"],
                "mean_outside_naive_changed_percent": row["mean_outside_naive_changed_percent"],
                "mean_outside_semantic_changed_percent": row["mean_outside_semantic_changed_percent"],
                "mean_boundary_semantic_changed_percent": row["mean_boundary_semantic_changed_percent"],
            }
        )

    result = {
        "source": str(INPUT),
        "ranking_metric": "mean_cached_reedit_speedup",
        "note": "Top five from the completed 200-sample local heavy benchmark across 10 compatible checkpoints.",
        "rows": rows,
    }
    (OUT_DIR / "top5_cross_model_heavy.json").write_text(json.dumps(result, indent=2))

    with (OUT_DIR / "top5_cross_model_heavy.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Top-5 Cross-Model Heavy Benchmark",
        "",
        "Ranking metric: mean cached semantic re-edit speedup.",
        "",
        "| Rank | Model | N | Naive s | Local s | Detector s | Cached s | Gen. speedup | Incl. speedup | Cached speedup | Outside drift |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['rank']} | {row['model_name']} | {row['samples']} | "
            f"{row['mean_naive_v1_seconds']:.3f} | {row['mean_semantic_v1_seconds']:.3f} | "
            f"{row['mean_detector_seconds']:.3f} | {row['mean_cached_reedit_seconds']:.3f} | "
            f"{row['mean_generation_speedup']:.3f}x | {row['mean_detector_inclusive_speedup']:.3f}x | "
            f"{row['mean_cached_reedit_speedup']:.3f}x | {row['mean_outside_semantic_changed_percent']:.3f}% |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- OpenJourney v4 and DreamShaper 8 are SD1.5-family models and show strong local/cached speedups.",
            "- Juggernaut XL v2 is SDXL-family; its high cached speedup comes from very expensive full-image regeneration.",
            "- Detector-inclusive first edits are model dependent: DreamShaper 8, OpenJourney v4, and Juggernaut XL v2 are above 1x, while the faster SD1.5 baselines are better mainly for cached/repeated edits.",
            "- Semantic outside-mask drift remains 0.0% under mask compositing for all top-five rows.",
        ]
    )
    (OUT_DIR / "top5_cross_model_heavy.md").write_text("\n".join(lines))
    print(OUT_DIR / "top5_cross_model_heavy.md")


if __name__ == "__main__":
    main()
