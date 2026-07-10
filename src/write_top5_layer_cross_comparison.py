#!/usr/bin/env python3
"""Write a top-5 cross comparison report for semantic layer quality and edit efficiency."""

from __future__ import annotations

import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
CROSS_ROOT = OUTPUTS / "cross_model_semantic"


TARGET_ALIASES = {
    "lips": ["lip", "mouth"],
    "shirt": ["shirt", "t-shirt", "clothing", "torso"],
    "mug": ["mug", "cup", "ceramic", "coffee"],
    "sky": ["sky", "cloud"],
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def latest_metrics(pattern: str, metric_name: str = "metrics.json") -> tuple[Path, dict[str, Any]]:
    candidates = sorted(OUTPUTS.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    for folder in candidates:
        path = folder / metric_name
        if path.exists():
            return folder, load_json(path)
    raise FileNotFoundError(f"No {metric_name} for {pattern}")


def latest_specific(path: str) -> tuple[Path, dict[str, Any]]:
    p = ROOT / path
    return p.parent, load_json(p)


def cross_model_runs() -> list[dict[str, Any]]:
    runs: dict[str, tuple[float, dict[str, Any]]] = {}
    for path in CROSS_ROOT.glob("*/cross_model_metrics.json"):
        data = load_json(path)
        key = data["model"]["key"]
        mtime = path.stat().st_mtime
        if key not in runs or mtime > runs[key][0]:
            runs[key] = (mtime, data)
    return [item[1] for item in runs.values()]


def ok_cases(run: dict[str, Any]) -> list[dict[str, Any]]:
    return [case for case in run.get("cases", []) if not case.get("error")]


def avg_speedup(run: dict[str, Any]) -> float:
    return mean(case["speedup_vs_naive"] for case in ok_cases(run))


def detector_inclusive_speedup(case: dict[str, Any]) -> float:
    return case["naive_seconds"] / (case["semantic_local_seconds"] + case["detector_seconds"])


def fmt_s(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}s"


def fmt_x(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}x"


def fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}%"


def outside_changed(case: dict[str, Any]) -> float:
    return (
        case.get("semantic_local_projected", {})
        .get("preservation", {})
        .get("outside_target_changed_pixels_percent_gt10", 0.0)
    )


def boundary_changed(case: dict[str, Any]) -> float:
    return (
        case.get("semantic_local_projected", {})
        .get("boundary", {})
        .get("boundary_band_changed_pixels_percent_gt10", 0.0)
    )


def name_accuracy_proxy(case: dict[str, Any]) -> bool:
    meta = case.get("detector_meta", {})
    label = str(meta.get("selected", {}).get("label", "")).lower()
    target = str(case.get("target", "")).lower()
    aliases = TARGET_ALIASES.get(target, [target])
    return any(alias in label for alias in aliases)


def detection_success(case: dict[str, Any]) -> bool:
    meta = case.get("detector_meta", {})
    bbox = meta.get("bbox")
    pixels = int(meta.get("mask_pixels") or 0)
    return bool(bbox and len(bbox) == 4 and pixels > 64)


def selection_score(case: dict[str, Any]) -> float | None:
    selected = case.get("detector_meta", {}).get("selected", {})
    value = selected.get("selection_score")
    return None if value is None else float(value)


def mask_area_percent(run: dict[str, Any], case: dict[str, Any]) -> float:
    pixels = float(case.get("detector_meta", {}).get("mask_pixels") or 0)
    return pixels / float(run["width"] * run["height"]) * 100.0


def review_flag(run: dict[str, Any], case: dict[str, Any]) -> str:
    flags = []
    area = mask_area_percent(run, case)
    score = selection_score(case)
    if not detection_success(case):
        flags.append("missing")
    if area < 0.10:
        flags.append("tiny mask")
    if score is not None and score < 0.30:
        flags.append("low score")
    if boundary_changed(case) > 50.0:
        flags.append("seam risk")
    return ", ".join(flags) if flags else "ok"


def copy_image(src: Path, image_dir: Path, name: str) -> str:
    image_dir.mkdir(parents=True, exist_ok=True)
    dst = image_dir / name
    if src.exists():
        shutil.copy2(src, dst)
    return f"images/{name}"


def image_ref(lines: list[str], title: str, rel_path: str) -> None:
    lines.extend([f"### {title}", "", f"![{title}]({rel_path})", ""])


def method_detector_rows(region_metrics: dict[str, Any]) -> list[str]:
    rows = []
    for key, label in [
        ("diffusion_attention", "DAAM attention"),
        ("grounding_sam2", "GroundingDINO + SAM2"),
        ("depth_segmentation_fusion", "Depth fusion"),
    ]:
        item = region_metrics["methods"].get(key, {})
        rows.append(
            f"| {label} | {fmt_s(item.get('cold_seconds'))} | {fmt_s(item.get('cached_reload_seconds'))} | "
            f"{item.get('detected_count', 'n/a')} | {', '.join(item.get('detected_regions', []))} |"
        )
    return rows


def make_gt_template(top_runs: list[dict[str, Any]], out_dir: Path) -> None:
    tasks = []
    for run in top_runs:
        for case in ok_cases(run):
            tasks.append(
                {
                    "model_key": run["model"]["key"],
                    "model_name": run["model"]["name"],
                    "case": case["label"],
                    "target": case["target"],
                    "image": str(Path(run["output_dir"]) / f"{ok_cases(run).index(case) + 1:02d}_{case['case_key']}" / "base.png"),
                    "labeler_target_name": case["target"],
                    "gt_bbox_xyxy": None,
                    "gt_mask_path": None,
                    "layer_type": None,
                    "parent_layer": None,
                    "depth_order": None,
                    "notes": "",
                }
            )
    (out_dir / "ground_truth_labeling_template.json").write_text(json.dumps({"tasks": tasks}, indent=2))


def write_report() -> Path:
    out_dir = OUTPUTS / f"top5_layer_cross_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    image_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = cross_model_runs()
    top_runs = sorted(runs, key=avg_speedup, reverse=True)[:5]

    detector_folder, detector_metrics = latest_specific(
        "outputs/region_detector_compare_20260430_233441/detector_comparison_metrics.json"
    )
    text_folder, text_metrics = latest_specific("outputs/sdxl_promo_text_autolayer_20260501_084853/metrics.json")
    face_folder, face_metrics = latest_specific("outputs/face_eye_edit_20260501_094643/metrics.json")
    landscape_folder, landscape_metrics = latest_specific("outputs/landscape_two_layer_autolayer_20260501_095114/metrics.json")
    manual_folder, manual_metrics = latest_specific("outputs/landscape_manual_inpaint_compare_20260501_095054/metrics.json")
    cache_folder, cache_metrics = latest_specific("outputs/semantic_cache_20260430_230543/semantic_cache_metrics.json")
    multi_folder, multi_metrics = latest_specific("outputs/multi_region_rollback_20260430_232001/multi_region_rollback_metrics.json")

    visual_refs: dict[str, str] = {}
    asset_manifest: list[dict[str, str]] = []
    for run in top_runs:
        key = run["model"]["key"]
        visual_refs[f"grid_{key}"] = copy_image(
            Path(run["output_dir"]) / "cross_model_grid.png",
            image_dir,
            f"top5_{key}_grid.png",
        )
        asset_manifest.append(
            {
                "kind": "top5_model_grid",
                "model": run["model"]["name"],
                "source": str(Path(run["output_dir"]) / "cross_model_grid.png"),
                "report_path": visual_refs[f"grid_{key}"],
            }
        )
        for case_index, case in enumerate(ok_cases(run), start=1):
            strip_src = Path(run["output_dir"]) / f"{case_index:02d}_{case['case_key']}" / "case_strip.png"
            strip_key = f"strip_{key}_{case['case_key']}"
            visual_refs[strip_key] = copy_image(
                strip_src,
                image_dir,
                f"top5_{key}_{case['case_key']}_strip.png",
            )
            asset_manifest.append(
                {
                    "kind": "case_strip",
                    "model": run["model"]["name"],
                    "case": case["label"],
                    "source": str(strip_src),
                    "report_path": visual_refs[strip_key],
                }
            )
    visual_refs["detectors"] = copy_image(detector_folder / "detector_overlay_comparison.png", image_dir, "detector_overlay_comparison.png")
    visual_refs["text"] = copy_image(text_folder / "promo_text_autolayer_comparison.png", image_dir, "text_layer_comparison.png")
    visual_refs["face"] = copy_image(face_folder / "face_eye_edit_comparison.png", image_dir, "face_eye_comparison.png")
    visual_refs["landscape"] = copy_image(landscape_folder / "landscape_two_layer_comparison.png", image_dir, "landscape_two_layer_comparison.png")
    visual_refs["manual"] = copy_image(manual_folder / "manual_vs_autolayer_comparison.png", image_dir, "manual_vs_autolayer_comparison.png")
    visual_refs["cache"] = copy_image(cache_folder / "semantic_cache_side_by_side.png", image_dir, "semantic_cache_side_by_side.png")
    visual_refs["multi"] = copy_image(multi_folder / "multi_region_rollback_side_by_side.png", image_dir, "multi_region_rollback_side_by_side.png")
    for key, source, description in [
        ("detectors", detector_folder / "detector_overlay_comparison.png", "Detector backend overlay comparison"),
        ("text", text_folder / "promo_text_autolayer_comparison.png", "Text layer Auto-Layer comparison"),
        ("face", face_folder / "face_eye_edit_comparison.png", "Face landmark eye-edit comparison"),
        ("landscape", landscape_folder / "landscape_two_layer_comparison.png", "Landscape two-layer Auto-Layer comparison"),
        ("manual", manual_folder / "manual_vs_autolayer_comparison.png", "Manual precise/rough inpaint versus Auto-Layer comparison"),
        ("cache", cache_folder / "semantic_cache_side_by_side.png", "Repeated semantic-cache edit comparison"),
        ("multi", multi_folder / "multi_region_rollback_side_by_side.png", "Multi-region rollback comparison"),
    ]:
        asset_manifest.append(
            {
                "kind": "scenario_visual",
                "description": description,
                "source": str(source),
                "report_path": visual_refs[key],
            }
        )

    make_gt_template(top_runs, out_dir)

    all_top_cases = [case for run in top_runs for case in ok_cases(run)]
    mean_generation_speedup = mean(case["speedup_vs_naive"] for case in all_top_cases)
    mean_inclusive_speedup = mean(detector_inclusive_speedup(case) for case in all_top_cases)
    mean_detector_time = mean(case["detector_seconds"] for case in all_top_cases)
    success_rate = mean(1.0 if detection_success(case) else 0.0 for case in all_top_cases)
    name_rate = mean(1.0 if name_accuracy_proxy(case) else 0.0 for case in all_top_cases)

    lines: list[str] = [
        "# Top-5 Semantic Layer Cross Comparison",
        "",
        "## Scope",
        "",
        "This report compares semantic annotation quality and editing efficiency using the strongest five models from the prior cross-model benchmark, ranked by generation-only speedup.",
        "",
        "Top-5 model set:",
        "",
    ]
    for idx, run in enumerate(top_runs, 1):
        lines.append(f"{idx}. {run['model']['name']} (`{run['model']['key']}`), mean generation speedup {fmt_x(avg_speedup(run))}")

    lines.extend(
        [
            "",
            "## Executive Result",
            "",
            f"Across the top-5 model set and four edit categories, DINO+SAM2 detection succeeded on {success_rate * 100:.1f}% of cases by bbox/mask existence, and the label-name proxy matched the requested target on {name_rate * 100:.1f}% of cases. The mean generation-only edit speedup was {fmt_x(mean_generation_speedup)}. Detector-inclusive speedup was {fmt_x(mean_inclusive_speedup)} because the detector costs about {fmt_s(mean_detector_time)} per fresh image/target.",
            "",
            "The central finding is not simply that local rendering is faster. The stronger result is that semantic layers make edits controllable: outside-target pixels remain stable, repeated edits become cheap after cache creation, rollback is effectively instant, and text/face/object layers can be routed to specialized renderers instead of forcing every edit through full-image diffusion.",
            "",
            "Important honesty note: bbox IoU, mask Dice/IoU, hierarchy accuracy, and depth-order accuracy require hand-labeled ground truth. This report therefore marks those metrics as not yet computed and writes a labeling template at `ground_truth_labeling_template.json`.",
            "",
            "## Assets Included",
            "",
            f"The report folder contains an `images/` directory with {len(asset_manifest)} copied visual assets. The assets include top-5 model grids, per-case strips for all top-5 model/case pairs, detector backend overlays, text-layer comparison, face-part edit comparison, manual inpaint comparison, multi-layer edit comparison, semantic cache comparison, and rollback comparison.",
            "",
            "## Semantic Annotation Quality",
            "",
            "### Detector Backend Comparison",
            "",
            "| Method | Cold time | Cached reload | Detected layers | Regions |",
            "| --- | ---: | ---: | ---: | --- |",
            *method_detector_rows(detector_metrics),
            "| Face landmarks | "
            + fmt_s(face_metrics["high_detail_detection_seconds"])
            + " | n/a | "
            + str(len(face_metrics["regions"]))
            + " | "
            + ", ".join(face_metrics["regions"].keys())
            + " |",
            "| OCR/layout | n/a | n/a | 1 | headline_text |",
            "| Auto-Layer ensemble | "
            + fmt_s(landscape_metrics["autolayer_detection_seconds"])
            + " | n/a | "
            + str(len(landscape_metrics["layers"]))
            + " | "
            + ", ".join(landscape_metrics["layers"].keys())
            + " |",
            "",
            "The backend comparison confirms the main architecture split: DAAM is essentially free once attention artifacts exist; GroundingDINO+SAM2 is more general but costly; depth fusion adds z-order rather than new object identity; face landmarks and OCR/layout are specialized but high-value for their layer types.",
            "",
            f"![Detector backend overlay]({visual_refs['detectors']})",
            "",
            "### Top-5 DINO+SAM2 Annotation Proxies",
            "",
            "| Model | Case | Target | Success | Name proxy | Selection score | Mask area | Boundary-risk proxy | Review flag |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )

    for run in top_runs:
        for case in ok_cases(run):
            score = selection_score(case)
            lines.append(
                f"| {run['model']['name']} | {case['label']} | {case['target']} | "
                f"{'yes' if detection_success(case) else 'no'} | {'yes' if name_accuracy_proxy(case) else 'no'} | "
                f"{'n/a' if score is None else f'{score:.3f}'} | {mask_area_percent(run, case):.3f}% | "
                f"{fmt_pct(boundary_changed(case))} | {review_flag(run, case)} |"
            )

    lines.extend(
        [
            "",
            "### Metrics Computability",
            "",
            "| Metric | Status in this run | Honest interpretation |",
            "| --- | --- | --- |",
            "| Detection success | computed | bbox and non-empty mask present |",
            "| Name accuracy | proxy computed | selected label contains target alias |",
            "| BBox quality IoU | not computed | requires hand-labeled bbox |",
            "| Mask quality IoU/Dice | not computed | requires hand-labeled mask |",
            "| Layer type accuracy | partially computed | known by scenario, not independently judged |",
            "| Hierarchy accuracy | not computed | requires labeled parent/child relationships |",
            "| Depth order accuracy | partially available | depth fusion reports order, but no GT order labels yet |",
            "",
            "### Why The Annotation Result Is Promising But Not Finished",
            "",
            "The current top-5 DINO+SAM2 proxy table is useful as a smoke test: the requested labels resolve to non-empty masks and the selected text labels contain the target names. However, this is not enough for a scientific quality claim. The object mug cases show why: a detector can return a named mug mask that is technically non-empty but too small or too narrow. That is why the report includes a review flag for tiny masks, low selection scores, and seam risk. The next required step is hand-labeling the generated scenes with boxes and masks.",
            "",
            "## Editing Efficiency: Top-5 Models",
            "",
            "| Model | Case | Naive | DINO+SAM2 local | Detector | Generation speedup | Detector-inclusive speedup | Outside-target change |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for run in top_runs:
        for case in ok_cases(run):
            lines.append(
                f"| {run['model']['name']} | {case['label']} | {fmt_s(case['naive_seconds'])} | "
                f"{fmt_s(case['semantic_local_seconds'])} | {fmt_s(case['detector_seconds'])} | "
                f"{fmt_x(case['speedup_vs_naive'])} | {fmt_x(detector_inclusive_speedup(case))} | "
                f"{fmt_pct(outside_changed(case))} |"
            )

    lines.extend(
        [
            "",
            "## Scenario Cross Comparison",
            "",
            "The scenario comparison is included because the layer type matters. A text layer, a face part, a rigid object, and a landscape region should not all be judged with one renderer. Auto-Layer is strongest when it routes each layer to the renderer that matches its structure.",
            "",
            "### A. Text Layer",
            "",
            f"Auto-Layer text rendering updated the headline in {fmt_s(text_metrics['ours_cached_text_layer_update_seconds'])}; cached naive SDXL regeneration took {fmt_s(text_metrics['naive_cached_full_regeneration_seconds'])}. Outside-text change was {fmt_pct(text_metrics['stability']['ours_cached_vs_base_ad']['outside_changed_pixels_gt_10_percent'])} for Auto-Layer versus {fmt_pct(text_metrics['stability']['naive_cached_vs_base_ad']['outside_changed_pixels_gt_10_percent'])} for naive.",
            "",
            f"![Text layer comparison]({visual_refs['text']})",
            "",
            "### B. Small Part Layer",
            "",
            f"Face-landmark eye editing took {fmt_s(face_metrics['eye_only_inpaint_seconds'])}; naive full green-eye regeneration took {fmt_s(face_metrics['naive_full_green_eyes_seconds'])}. Outside-eye change was {fmt_pct(face_metrics['stability']['eye_only_vs_base_outside_eyes']['outside_changed_pixels_gt_10_percent'])} for the local edit versus {fmt_pct(face_metrics['stability']['naive_vs_base_outside_eyes']['outside_changed_pixels_gt_10_percent'])} for naive.",
            "",
            f"![Face eye comparison]({visual_refs['face']})",
            "",
            "### C. Object Layer",
            "",
            f"For the landscape object edit, naive two-change regeneration took {fmt_s(manual_metrics['baselines_from_autolayer_run']['naive_full_two_changes_seconds'])}; Auto-Layer render took {fmt_s(manual_metrics['baselines_from_autolayer_run']['autolayer_two_layer_render_seconds'])}; manual precise inpaint took {fmt_s(manual_metrics['manual_precise']['seconds'])}; manual rough inpaint took {fmt_s(manual_metrics['manual_rough_box']['seconds'])}. Manual precise is faster for this one-off edit, while Auto-Layer preserves naming, persistence, and re-editability without hand masking.",
            "",
            f"![Manual vs Auto-Layer comparison]({visual_refs['manual']})",
            "",
            "### D. Multi-Layer Edit",
            "",
            f"The two-layer landscape Auto-Layer run edited tractor and house. Naive two-change regeneration took {fmt_s(landscape_metrics['naive_full_two_changes_seconds'])}; Auto-Layer render took {fmt_s(landscape_metrics['autolayer_two_layer_render_seconds'])}, plus {fmt_s(landscape_metrics['autolayer_detection_seconds'])} detection for a fresh scene.",
            "",
            f"![Landscape two-layer comparison]({visual_refs['landscape']})",
            "",
            "### E. Repeated Edit / Cache",
            "",
            f"The semantic cache run built the base/cache in {fmt_s(cache_metrics['semantic_cache']['base_total_seconds'])}. Repeated cached plain edits averaged {fmt_s(cache_metrics['cached_plain']['mean_edit_seconds'])}, with first edit {fmt_s(cache_metrics['cached_plain']['first_edit_seconds'])}. Multi-region rollback restored a cached right-only state in {fmt_s(multi_metrics['rollback']['cached_plain_right_only_seconds'])}.",
            "",
            f"![Semantic cache side by side]({visual_refs['cache']})",
            "",
            f"![Multi-region rollback]({visual_refs['multi']})",
            "",
            "## Per-Case Asset Gallery",
            "",
            "The following strips are copied into this report folder so visual review can happen without opening the original benchmark directories. Each strip shows base, naive edit, semantic mask overlay, and semantic local edit.",
            "",
        ]
    )

    for run in top_runs:
        key = run["model"]["key"]
        lines.extend([f"### {run['model']['name']} Case Strips", ""])
        for case in ok_cases(run):
            strip_key = f"strip_{key}_{case['case_key']}"
            lines.extend([f"#### {case['label']}", "", f"![{run['model']['name']} {case['label']}]({visual_refs[strip_key]})", ""])

    lines.extend(
        [
            "## Top-5 Visual Grids",
            "",
            "Each grid shows four cases for one model: base, naive full-image edit, DINO+SAM2 overlay, and semantic local output.",
            "",
        ]
    )

    for run in top_runs:
        key = run["model"]["key"]
        lines.extend(
            [
                f"### {run['model']['name']}",
                "",
                f"![{run['model']['name']} grid]({visual_refs[f'grid_{key}']})",
                "",
            ]
        )

    lines.extend(
        [
            "## Recommendations",
            "",
            "1. Use DINO+SAM2 as the broad object/part detector, but cache masks aggressively.",
            "2. Use specialized renderers for specialized layer types: text renderer for OCR/layout layers, face landmarks for face parts, and depth fusion for ordering.",
            "3. Treat manual precise inpaint as a strong one-off baseline for object edits; Auto-Layer should win on user effort, persistence, re-edits, and versioning.",
            "4. Add a hand-labeled 20-50 image set next. Without it, IoU/Dice and hierarchy/depth accuracy should not be claimed.",
            "5. For papers or demos, report both generation-only and detector-inclusive speedups. The former measures renderer policy; the latter measures user-visible one-off latency.",
            "",
            "## Ground Truth Next Step",
            "",
            "A starter labeling manifest has been written to `ground_truth_labeling_template.json`. Fill `gt_bbox_xyxy`, `gt_mask_path`, `layer_type`, `parent_layer`, and `depth_order`; then the same report can be extended with true IoU/Dice, hierarchy, and depth-order accuracy.",
            "",
            "## Asset Manifest",
            "",
            "| Kind | Description | Report asset | Source |",
            "| --- | --- | --- | --- |",
        ]
    )

    for asset in asset_manifest:
        description = asset.get("description") or " / ".join(
            part for part in [asset.get("model"), asset.get("case")] if part
        )
        lines.append(
            f"| {asset.get('kind', '')} | {description} | `{asset['report_path']}` | `{asset['source']}` |"
        )

    report_path = out_dir / "top5_layer_cross_comparison_report.md"
    report_path.write_text("\n".join(lines))
    (out_dir / "asset_manifest.json").write_text(json.dumps(asset_manifest, indent=2))
    (out_dir / "top5_layer_cross_comparison_metrics.json").write_text(
        json.dumps(
            {
                "top_models": [
                    {
                        "key": run["model"]["key"],
                        "name": run["model"]["name"],
                        "mean_generation_speedup": avg_speedup(run),
                    }
                    for run in top_runs
                ],
                "mean_generation_speedup": mean_generation_speedup,
                "mean_detector_inclusive_speedup": mean_inclusive_speedup,
                "mean_detector_time": mean_detector_time,
                "detection_success_rate": success_rate,
                "name_accuracy_proxy_rate": name_rate,
                "source_artifacts": {
                    "detector_comparison": str(detector_folder),
                    "text": str(text_folder),
                    "face": str(face_folder),
                    "landscape": str(landscape_folder),
                    "manual": str(manual_folder),
                    "cache": str(cache_folder),
                    "multi": str(multi_folder),
                },
            },
            indent=2,
        )
    )
    return report_path


def main() -> None:
    print(write_report())


if __name__ == "__main__":
    main()
