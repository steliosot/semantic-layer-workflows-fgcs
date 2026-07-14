#!/usr/bin/env python3
"""Geometry-based dependency-risk analysis for semantic layer edits.

The benchmark stores one target mask per edit. This script estimates how often
the target lies near non-target high-change regions by measuring dilation-ring
contact and boundary disturbance. It is a lightweight proxy for dependency
edges when full DAAM/depth layers are unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median

import numpy as np
from PIL import Image
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "outputs/heavy_paper_benchmark/heavy_all_models_20steps_5samples/heavy_paper_benchmark_metrics.json"
OUT_DIR = ROOT / "paper_fgcs_semantic_workflows/revision_outputs"
OUT_JSON = OUT_DIR / "dependency_risk.json"
OUT_MD = OUT_DIR / "dependency_risk_summary.md"


def rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    im = Image.open(path).convert("L")
    if im.size != size:
        im = im.resize(size, Image.Resampling.NEAREST)
    return np.asarray(im) > 127


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = [r for r in json.loads(METRICS_PATH.read_text())["records"] if not r.get("error")]
    rows = []
    for rec in records:
        sample = Path(rec["artifact_dir"])
        base_p = sample / "base.png"
        sem_p = sample / "semantic_v1_projected.png"
        mask_p = sample / "target_mask_prepared.png"
        if not base_p.exists() or not sem_p.exists() or not mask_p.exists():
            continue
        base = rgb(base_p)
        sem = rgb(sem_p)
        m = mask(mask_p, (base.shape[1], base.shape[0]))
        changed = np.abs(base - sem).mean(axis=2) > 10
        outside_changed = changed & ~m
        dilated = ndimage.binary_dilation(m, iterations=12)
        ring = dilated & ~m
        ring_changed = outside_changed & ring
        ring_area = max(int(ring.sum()), 1)
        dependency_contact = 100 * float(ring_changed.sum()) / ring_area
        bbox_area = rec["bbox_area_percent"]
        mask_area = rec["mask_area_percent"]
        expansion_ratio = bbox_area / max(mask_area, 1e-6)
        risk_score = 0.5 * min(dependency_contact / 25.0, 1.0) + 0.3 * min(rec["boundary_semantic_v1_changed_percent"] / 50.0, 1.0) + 0.2 * min(expansion_ratio / 4.0, 1.0)
        rows.append(
            {
                "model_key": rec["model_key"],
                "family": rec["family"],
                "case_key": rec["case_key"],
                "sample_index": rec["sample_index"],
                "dependency_contact_percent": dependency_contact,
                "boundary_change_percent": rec["boundary_semantic_v1_changed_percent"],
                "bbox_mask_expansion_ratio": expansion_ratio,
                "dependency_risk_score": risk_score,
            }
        )

    by_case = {}
    for case in sorted({r["case_key"] for r in rows}):
        subset = [r for r in rows if r["case_key"] == case]
        by_case[case] = {
            "count": len(subset),
            "contact_mean": mean(r["dependency_contact_percent"] for r in subset),
            "contact_median": median(r["dependency_contact_percent"] for r in subset),
            "risk_mean": mean(r["dependency_risk_score"] for r in subset),
            "high_risk_percent": 100 * sum(r["dependency_risk_score"] > 0.7 for r in subset) / len(subset),
        }
    summary = {
        "records": len(rows),
        "contact_mean": mean(r["dependency_contact_percent"] for r in rows),
        "contact_median": median(r["dependency_contact_percent"] for r in rows),
        "risk_mean": mean(r["dependency_risk_score"] for r in rows),
        "high_risk_percent": 100 * sum(r["dependency_risk_score"] > 0.7 for r in rows) / len(rows),
    }
    result = {"summary": summary, "by_case": by_case, "rows": rows}
    OUT_JSON.write_text(json.dumps(result, indent=2))

    lines = [
        "# Dependency Risk Summary",
        "",
        f"Records: {summary['records']}",
        f"Mean dependency-contact ring change: {summary['contact_mean']:.3f}%",
        f"Median dependency-contact ring change: {summary['contact_median']:.3f}%",
        f"Mean dependency-risk score: {summary['risk_mean']:.3f}",
        f"High-risk samples (>0.7): {summary['high_risk_percent']:.1f}%",
        "",
        "| Case | Count | Contact mean | Contact median | Risk mean | High risk |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case, vals in by_case.items():
        lines.append(
            f"| {case} | {vals['count']} | {vals['contact_mean']:.3f}% | "
            f"{vals['contact_median']:.3f}% | {vals['risk_mean']:.3f} | {vals['high_risk_percent']:.1f}% |"
        )
    OUT_MD.write_text("\n".join(lines))
    print(OUT_JSON)
    print(OUT_MD)


if __name__ == "__main__":
    main()
