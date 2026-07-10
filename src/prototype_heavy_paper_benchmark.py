#!/usr/bin/env python3
"""Heavy paper-style benchmark for naive vs semantic layered editing.

The run is intentionally resumable. Each model/case/sample writes artifacts and
the aggregate JSON is updated after every completed sample.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw

from prototype_cross_model_semantic_benchmark import (
    CASES,
    MODEL_SPECS,
    BenchCase,
    ModelSpec,
    apply_color_projection,
    boundary_stats,
    cleanup,
    crop_for_mask,
    diff_stats,
    load_img2img,
    load_txt2img,
    mask_bbox,
    method_metrics,
    paste_crop,
    pick_device,
    prepare_mask,
    run_detector,
)
from prototype_face_detail_eye_edit import generator


ROOT = Path(__file__).resolve().parent
BENCH_ROOT = ROOT / "outputs" / "heavy_paper_benchmark"


MODEL_ORDER = [
    "juggernaut_reborn",
    "sd15_base",
    "realistic_vision_v6_fp16",
    "deliberate_v6",
    "dreamshaper_8",
    "openjourney_v4",
    "juggernaut_xl_v2",
    "sdxl_base_1_0_local",
    "realvisxl_v4",
    "dreamshaper_xl_1_0",
]


VERSION_2: dict[str, dict[str, Any]] = {
    "face_lips": {
        "name": "deep burgundy lipstick",
        "prompt": "same adult woman portrait, deep burgundy lipstick on closed lips only, preserve face identity and lighting, realistic makeup",
        "naive_prompt": "studio portrait photo of an adult woman, centered face, deep burgundy lipstick, clear eyes, soft neutral background, realistic photography",
        "rgb": (128, 14, 38),
        "projection_strength": 0.58,
    },
    "human_body_shirt": {
        "name": "forest green shirt",
        "prompt": "same person, forest green t-shirt only, keep face jeans shoes and background unchanged, realistic fabric",
        "naive_prompt": "realistic full body studio photo of an adult person standing, wearing a forest green t-shirt and blue jeans, simple gray background, sharp detail",
        "rgb": (28, 122, 72),
        "projection_strength": 0.50,
    },
    "object_mug": {
        "name": "glossy red mug",
        "prompt": "same product photograph, glossy red ceramic coffee mug only, keep table background and lighting unchanged, realistic glaze",
        "naive_prompt": "realistic product photo of a glossy red ceramic coffee mug on a wooden table, soft window light, clean background",
        "rgb": (205, 28, 32),
        "projection_strength": 0.50,
    },
    "landscape_sky": {
        "name": "stormy blue gray sky",
        "prompt": "same landscape photograph, stormy blue gray sky only, keep mountains lake and foreground unchanged, realistic lighting",
        "naive_prompt": "realistic landscape photograph, mountain lake valley, stormy blue gray sky, natural scene, crisp detail, 35mm lens",
        "rgb": (72, 92, 118),
        "projection_strength": 0.30,
    },
}


EXTRA_LOCAL_CHECKPOINTS = [
    {
        "path": "/Users/stelios/Documents/ComfyUI/models/checkpoints/epicrealism_v10-inpainting.safetensors",
        "reason": "inpainting checkpoint, not a txt2img/img2img checkpoint for this benchmark",
    },
    {
        "path": "/Users/stelios/Documents/ComfyUI/models/checkpoints/sd3_medium_incl_clips.safetensors",
        "reason": "SD3 checkpoint, incompatible with the SD1.5/SDXL diffusers pipelines used here",
    },
    {
        "path": "/Users/stelios/Documents/ComfyUI/models/checkpoints/ace_step_v1_3.5b.safetensors",
        "reason": "not an SD image-generation checkpoint for this benchmark",
    },
    {
        "path": "/Users/stelios/Documents/ComfyUI/models/checkpoints/tiny_test_download.safetensors",
        "reason": "tiny placeholder/test file",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run heavy paper-style semantic editing benchmark.")
    parser.add_argument("--outputs-dir", default=str(BENCH_ROOT))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--samples-per-case", type=int, default=5)
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--max-cases", type=int, default=4)
    parser.add_argument("--dino-threshold", type=float, default=0.12)
    parser.add_argument("--dino-text-threshold", type=float, default=0.10)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--disable-progress", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def now() -> float:
    return time.perf_counter()


def sec(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}s"


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}%"


def speed(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}x"


def mean_or_none(values: list[float]) -> float | None:
    return round(mean(values), 4) if values else None


def std_or_none(values: list[float]) -> float | None:
    return round(pstdev(values), 4) if len(values) > 1 else None


def selected_models(max_models: int | None) -> list[ModelSpec]:
    specs = [MODEL_SPECS[key] for key in MODEL_ORDER if key in MODEL_SPECS]
    if max_models is not None:
        specs = specs[:max_models]
    return specs


def available_model(spec: ModelSpec) -> tuple[bool, str | None]:
    path = Path(spec.path)
    if spec.loader.startswith("pretrained"):
        if (path / "model_index.json").exists():
            return True, None
        return False, f"missing diffusers model_index.json at {path}"
    if not path.exists() or path.stat().st_size < 1_000_000:
        return False, f"missing or invalid checkpoint at {path}"
    return True, None


def case_for_sample(case: BenchCase, sample_index: int) -> BenchCase:
    offset = sample_index * 10_000
    return replace(case, seed=case.seed + offset, edit_seed=case.edit_seed + offset)


def local_edit(
    img2img: Any,
    spec: ModelSpec,
    case: BenchCase,
    base: Image.Image,
    prepared_mask: Image.Image,
    prompt: str,
    rgb: tuple[int, int, int],
    projection_strength: float,
    out_dir: Path,
    prefix: str,
    device: str,
) -> dict[str, Any]:
    crop, crop_mask, crop_box = crop_for_mask(
        base,
        prepared_mask,
        min_crop=256 if spec.family == "sd15" else 384,
        max_crop=spec.max_crop_size,
    )
    crop.save(out_dir / f"{prefix}_crop.png")
    crop_mask.save(out_dir / f"{prefix}_crop_mask.png")

    started = now()
    raw_crop = img2img(
        prompt=prompt,
        negative_prompt=case.negative_prompt,
        image=crop,
        num_inference_steps=spec.steps,
        strength=case.strength,
        guidance_scale=spec.guidance_scale,
        generator=generator(case.edit_seed, device),
    ).images[0].convert("RGB")
    diffusion_seconds = now() - started

    raw = paste_crop(base, raw_crop, crop_mask, crop_box)
    raw.save(out_dir / f"{prefix}_raw.png")
    raw_crop.save(out_dir / f"{prefix}_raw_crop.png")

    started = now()
    projected = apply_color_projection(raw, prepared_mask, rgb, projection_strength)
    projection_seconds = now() - started
    projected.save(out_dir / f"{prefix}_projected.png")

    return {
        "image": projected,
        "raw_image": raw,
        "crop_box": crop_box,
        "crop_size": [crop.width, crop.height],
        "diffusion_seconds": round(diffusion_seconds, 4),
        "projection_seconds": round(projection_seconds, 6),
        "total_seconds": round(diffusion_seconds + projection_seconds, 4),
    }


def projection_only(
    base: Image.Image,
    mask: Image.Image,
    rgb: tuple[int, int, int],
    strength: float,
    out_dir: Path,
    prefix: str,
) -> dict[str, Any]:
    started = now()
    image = apply_color_projection(base, mask, rgb, strength)
    seconds = now() - started
    image.save(out_dir / f"{prefix}_projected.png")
    return {"image": image, "seconds": round(seconds, 6)}


def mask_area_percent(mask: Image.Image) -> float:
    arr = np.array(mask.convert("L")) > 8
    return round(float(arr.mean() * 100.0), 4)


def bbox_area_percent(mask: Image.Image, size: tuple[int, int]) -> float:
    try:
        x1, y1, x2, y2 = mask_bbox(mask)
    except RuntimeError:
        return 0.0
    return round(float(((x2 - x1 + 1) * (y2 - y1 + 1)) / (size[0] * size[1]) * 100.0), 4)


def save_sample_grid(path: Path, images: list[tuple[str, Image.Image]]) -> None:
    tile = 220
    header = 34
    canvas = Image.new("RGB", (tile * len(images), tile + header), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    for idx, (label, image) in enumerate(images):
        draw.text((idx * tile + 8, 11), label, fill=(245, 245, 245))
        im = image.convert("RGB")
        im.thumbnail((tile, tile), Image.Resampling.LANCZOS)
        canvas.paste(im, (idx * tile + (tile - im.width) // 2, header + (tile - im.height) // 2))
    canvas.save(path)


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "model_key",
        "model_name",
        "family",
        "case_key",
        "case_label",
        "sample_index",
        "width",
        "height",
        "steps",
        "base_seconds",
        "naive_v1_seconds",
        "naive_v2_seconds",
        "detector_seconds",
        "semantic_v1_seconds",
        "semantic_v2_cached_seconds",
        "projection_only_seconds",
        "version_rollback_seconds",
        "mask_area_percent",
        "bbox_area_percent",
        "crop_size",
        "crop_pixels_percent",
        "speedup_generation_v1",
        "speedup_detector_inclusive_v1",
        "speedup_cached_reedit_v2",
        "speedup_projection_only",
        "outside_naive_v1_changed_percent",
        "outside_semantic_v1_changed_percent",
        "outside_projection_only_changed_percent",
        "boundary_semantic_v1_changed_percent",
        "inside_semantic_v1_mean_abs_diff",
        "artifact_dir",
    }
    return {key: record.get(key) for key in keep}


def aggregate(records: list[dict[str, Any]], skips: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    ok = [record for record in records if record.get("error") is None]

    def groups(key: str) -> list[dict[str, Any]]:
        seen = sorted({record[key] for record in ok})
        return [
            {
                key: value,
                "count": len(rows := [record for record in ok if record[key] == value]),
                "mean_naive_v1_seconds": mean_or_none([r["naive_v1_seconds"] for r in rows]),
                "mean_semantic_v1_seconds": mean_or_none([r["semantic_v1_seconds"] for r in rows]),
                "mean_detector_seconds": mean_or_none([r["detector_seconds"] for r in rows if r.get("detector_seconds") is not None]),
                "mean_cached_reedit_seconds": mean_or_none([r["semantic_v2_cached_seconds"] for r in rows]),
                "mean_projection_only_seconds": mean_or_none([r["projection_only_seconds"] for r in rows]),
                "mean_generation_speedup": mean_or_none([r["speedup_generation_v1"] for r in rows]),
                "std_generation_speedup": std_or_none([r["speedup_generation_v1"] for r in rows]),
                "mean_detector_inclusive_speedup": mean_or_none([r["speedup_detector_inclusive_v1"] for r in rows]),
                "mean_cached_reedit_speedup": mean_or_none([r["speedup_cached_reedit_v2"] for r in rows]),
                "mean_projection_only_speedup": mean_or_none([r["speedup_projection_only"] for r in rows]),
                "mean_outside_naive_changed_percent": mean_or_none([r["outside_naive_v1_changed_percent"] for r in rows]),
                "mean_outside_semantic_changed_percent": mean_or_none([r["outside_semantic_v1_changed_percent"] for r in rows]),
                "mean_boundary_semantic_changed_percent": mean_or_none([r["boundary_semantic_v1_changed_percent"] for r in rows]),
            }
            for value in seen
        ]

    summary = {
        "output_dir": str(output_dir),
        "record_count": len(ok),
        "error_count": len(records) - len(ok),
        "skipped_model_count": len(skips),
        "mean_naive_v1_seconds": mean_or_none([r["naive_v1_seconds"] for r in ok]),
        "mean_semantic_v1_seconds": mean_or_none([r["semantic_v1_seconds"] for r in ok]),
        "mean_detector_seconds": mean_or_none([r["detector_seconds"] for r in ok if r.get("detector_seconds") is not None]),
        "mean_projection_only_seconds": mean_or_none([r["projection_only_seconds"] for r in ok]),
        "mean_generation_speedup": mean_or_none([r["speedup_generation_v1"] for r in ok]),
        "mean_detector_inclusive_speedup": mean_or_none([r["speedup_detector_inclusive_v1"] for r in ok]),
        "mean_cached_reedit_speedup": mean_or_none([r["speedup_cached_reedit_v2"] for r in ok]),
        "mean_projection_only_speedup": mean_or_none([r["speedup_projection_only"] for r in ok]),
        "mean_version_rollback_seconds": mean_or_none([r["version_rollback_seconds"] for r in ok]),
        "mean_outside_naive_changed_percent": mean_or_none([r["outside_naive_v1_changed_percent"] for r in ok]),
        "mean_outside_semantic_changed_percent": mean_or_none([r["outside_semantic_v1_changed_percent"] for r in ok]),
        "mean_boundary_semantic_changed_percent": mean_or_none([r["boundary_semantic_v1_changed_percent"] for r in ok]),
    }
    return {
        "summary": summary,
        "by_model": groups("model_name"),
        "by_case": groups("case_label"),
        "by_family": groups("family"),
    }


def write_report(metrics: dict[str, Any], path: Path) -> None:
    summary = metrics["aggregates"]["summary"]
    lines = [
        "# Heavy Paper Benchmark: Naive vs Semantic Layered Editing",
        "",
        "## Scope",
        "",
        "This benchmark runs the compatible local SD1.5/SDXL models on four edit categories with five seed variants per category. Each sample compares full-image naive editing against semantic local editing, cached-mask re-editing/versioning, and deterministic projection-only layer editing.",
        "",
        f"- Completed samples: `{summary['record_count']}`",
        f"- Errors: `{summary['error_count']}`",
        f"- Skipped incompatible local checkpoints: `{summary['skipped_model_count']}`",
        f"- Steps: `{metrics['config']['steps']}`",
        f"- Samples per case: `{metrics['config']['samples_per_case']}`",
        "",
        "## Overall Results",
        "",
        "| Metric | Mean |",
        "| --- | ---: |",
        f"| Naive full edit time | {sec(summary['mean_naive_v1_seconds'])} |",
        f"| Semantic local edit time, detector excluded | {sec(summary['mean_semantic_v1_seconds'])} |",
        f"| Detector time | {sec(summary['mean_detector_seconds'])} |",
        f"| Projection-only edit time | {sec(summary['mean_projection_only_seconds'])} |",
        f"| Generation-only semantic speedup | {speed(summary['mean_generation_speedup'])} |",
        f"| One-off detector-inclusive speedup | {speed(summary['mean_detector_inclusive_speedup'])} |",
        f"| Cached version re-edit speedup | {speed(summary['mean_cached_reedit_speedup'])} |",
        f"| Projection-only speedup | {speed(summary['mean_projection_only_speedup'])} |",
        f"| Rollback time | {sec(summary['mean_version_rollback_seconds'])} |",
        f"| Naive outside-target changed pixels | {pct(summary['mean_outside_naive_changed_percent'])} |",
        f"| Semantic outside-target changed pixels | {pct(summary['mean_outside_semantic_changed_percent'])} |",
        f"| Semantic boundary-band changed pixels | {pct(summary['mean_boundary_semantic_changed_percent'])} |",
        "",
        "## By Case",
        "",
        "| Case | N | Naive | Semantic local | Detector | Gen speedup | Inclusive speedup | Cached re-edit speedup | Semantic outside change | Boundary change |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics["aggregates"]["by_case"]:
        lines.append(
            f"| {row['case_label']} | {row['count']} | {sec(row['mean_naive_v1_seconds'])} | "
            f"{sec(row['mean_semantic_v1_seconds'])} | {sec(row['mean_detector_seconds'])} | "
            f"{speed(row['mean_generation_speedup'])} | {speed(row['mean_detector_inclusive_speedup'])} | "
            f"{speed(row['mean_cached_reedit_speedup'])} | {pct(row['mean_outside_semantic_changed_percent'])} | "
            f"{pct(row['mean_boundary_semantic_changed_percent'])} |"
        )

    lines.extend(
        [
            "",
            "## By Model",
            "",
            "| Model | N | Naive | Semantic local | Detector | Gen speedup | Inclusive speedup | Cached re-edit speedup |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in metrics["aggregates"]["by_model"]:
        lines.append(
            f"| {row['model_name']} | {row['count']} | {sec(row['mean_naive_v1_seconds'])} | "
            f"{sec(row['mean_semantic_v1_seconds'])} | {sec(row['mean_detector_seconds'])} | "
            f"{speed(row['mean_generation_speedup'])} | {speed(row['mean_detector_inclusive_speedup'])} | "
            f"{speed(row['mean_cached_reedit_speedup'])} |"
        )

    lines.extend(["", "## Sample Visuals", ""])
    for sample in metrics.get("sample_grids", [])[:20]:
        rel = Path(sample).relative_to(path.parent).as_posix()
        lines.extend([f"![sample grid]({rel})", ""])

    if metrics.get("skipped_models"):
        lines.extend(["", "## Skipped Local Checkpoints", "", "| Path | Reason |", "| --- | --- |"])
        for skip in metrics["skipped_models"]:
            lines.append(f"| `{skip['path']}` | {skip['reason']} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The detector-inclusive number is the honest one-off user experience when no layer cache exists. The cached re-edit and rollback numbers test the project's stronger claim: once the Image DOM/layer mask is persisted, trying variants of the same semantic layer should avoid repeated detection and preserve the rest of the image.",
            "",
            "Projection-only is the deterministic upper bound for simple color changes. It is fastest and most local, but it may look flatter than diffusion-assisted edits on textured or reflective targets. Semantic local diffusion is slower but can add texture, shading, and realism. Naive editing is globally coherent but tends to change unrelated pixels and loses identity/layout consistency.",
            "",
            "These results are still not a replacement for hand-labeled mask correctness. They are realistic timing, preservation, boundary, and versioning metrics over repeated generated samples.",
        ]
    )
    path.write_text("\n".join(lines))


def save_metrics(output_dir: Path, metrics: dict[str, Any]) -> None:
    records = metrics.get("records", [])
    metrics["aggregates"] = aggregate(records, metrics.get("skipped_models", []), output_dir)
    (output_dir / "heavy_paper_benchmark_metrics.json").write_text(json.dumps(metrics, indent=2))
    write_report(metrics, output_dir / "heavy_paper_benchmark_report.md")


def load_existing(output_dir: Path) -> dict[str, Any] | None:
    metrics_path = output_dir / "heavy_paper_benchmark_metrics.json"
    if not metrics_path.exists():
        return None
    return json.loads(metrics_path.read_text())


def run() -> None:
    args = parse_args()
    device = pick_device(args.device)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.outputs_dir).expanduser() / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.force:
        metrics = None
    else:
        metrics = load_existing(output_dir)
    if metrics is None:
        models = selected_models(args.max_models)
        skipped: list[dict[str, Any]] = []
        for extra in EXTRA_LOCAL_CHECKPOINTS:
            if Path(extra["path"]).exists():
                skipped.append(extra)
        metrics = {
            "output_dir": str(output_dir),
            "config": {
                "device": device,
                "steps": args.steps,
                "samples_per_case": args.samples_per_case,
                "max_models": args.max_models,
                "max_cases": args.max_cases,
                "dino_threshold": args.dino_threshold,
                "dino_text_threshold": args.dino_text_threshold,
            },
            "models_requested": [MODEL_SPECS[key].__dict__ for key in MODEL_ORDER if key in MODEL_SPECS],
            "skipped_models": skipped,
            "records": [],
            "sample_grids": [],
        }

    done_keys = {
        (record.get("model_key"), record.get("case_key"), int(record.get("sample_index", -1)))
        for record in metrics["records"]
        if record.get("error") is None
    }

    cases = CASES[: args.max_cases]
    for base_spec in selected_models(args.max_models):
        expected_keys = {
            (base_spec.key, case.key, sample_index + 1)
            for case in cases
            for sample_index in range(args.samples_per_case)
        }
        if expected_keys.issubset(done_keys) and not args.force:
            continue

        ok, reason = available_model(base_spec)
        if not ok:
            metrics["skipped_models"].append({"path": base_spec.path, "reason": reason or "unavailable"})
            save_metrics(output_dir, metrics)
            continue

        spec = replace(base_spec, steps=args.steps)
        model_dir = output_dir / spec.key
        model_dir.mkdir(parents=True, exist_ok=True)

        started = now()
        txt2img = load_txt2img(spec, spec.path, device, args.disable_progress)
        txt2img_load_seconds = round(now() - started, 4)
        try:
            started = now()
            img2img = load_img2img(spec, spec.path, device, args.disable_progress)
            img2img_load_seconds = round(now() - started, 4)
            try:
                for case_index, base_case in enumerate(cases, start=1):
                    for sample_index in range(args.samples_per_case):
                        record_key = (spec.key, base_case.key, sample_index + 1)
                        if record_key in done_keys and not args.force:
                            continue
                        case = case_for_sample(base_case, sample_index)
                        v2 = VERSION_2[case.key]
                        sample_dir = model_dir / f"{case_index:02d}_{case.key}" / f"sample_{sample_index + 1:02d}"
                        sample_dir.mkdir(parents=True, exist_ok=True)
                        record: dict[str, Any] = {
                            "model_key": spec.key,
                            "model_name": spec.name,
                            "family": spec.family,
                            "case_key": case.key,
                            "case_label": case.label,
                            "sample_index": sample_index + 1,
                            "width": spec.width,
                            "height": spec.height,
                            "steps": spec.steps,
                            "guidance_scale": spec.guidance_scale,
                            "txt2img_load_seconds": txt2img_load_seconds,
                            "img2img_load_seconds": img2img_load_seconds,
                            "artifact_dir": str(sample_dir),
                            "error": None,
                        }
                        try:
                            started = now()
                            base = txt2img(
                                prompt=case.base_prompt,
                                negative_prompt=case.negative_prompt,
                                width=spec.width,
                                height=spec.height,
                                num_inference_steps=spec.steps,
                                guidance_scale=spec.guidance_scale,
                                generator=generator(case.seed, device),
                            ).images[0].convert("RGB")
                            record["base_seconds"] = round(now() - started, 4)
                            base.save(sample_dir / "base.png")

                            started = now()
                            naive_v1 = txt2img(
                                prompt=case.naive_prompt,
                                negative_prompt=case.negative_prompt,
                                width=spec.width,
                                height=spec.height,
                                num_inference_steps=spec.steps,
                                guidance_scale=spec.guidance_scale,
                                generator=generator(case.edit_seed, device),
                            ).images[0].convert("RGB")
                            record["naive_v1_seconds"] = round(now() - started, 4)
                            naive_v1.save(sample_dir / "naive_v1.png")

                            started = now()
                            naive_v2 = txt2img(
                                prompt=v2["naive_prompt"],
                                negative_prompt=case.negative_prompt,
                                width=spec.width,
                                height=spec.height,
                                num_inference_steps=spec.steps,
                                guidance_scale=spec.guidance_scale,
                                generator=generator(case.edit_seed + 111, device),
                            ).images[0].convert("RGB")
                            record["naive_v2_seconds"] = round(now() - started, 4)
                            naive_v2.save(sample_dir / "naive_v2.png")

                            mask, det_meta = run_detector(
                                sample_dir / "base.png",
                                case,
                                sample_dir,
                                device,
                                args.dino_threshold,
                                args.dino_text_threshold,
                            )
                            record["detector_seconds"] = round(
                                float(det_meta.get("subprocess_wall_seconds") or det_meta.get("total_seconds") or 0.0), 4
                            )
                            record["detector_meta"] = det_meta

                            prepared = prepare_mask(mask)
                            prepared.save(sample_dir / "target_mask_prepared.png")
                            record["mask_area_percent"] = mask_area_percent(prepared)
                            record["bbox_area_percent"] = bbox_area_percent(prepared, base.size)

                            proj = projection_only(
                                base,
                                prepared,
                                case.projection_rgb,
                                case.projection_strength,
                                sample_dir,
                                "projection_only_v1",
                            )
                            projection_image = proj["image"]
                            record["projection_only_seconds"] = proj["seconds"]

                            semantic_v1 = local_edit(
                                img2img,
                                spec,
                                case,
                                base,
                                prepared,
                                case.edit_prompt,
                                case.projection_rgb,
                                case.projection_strength,
                                sample_dir,
                                "semantic_v1",
                                device,
                            )
                            semantic_v1_image = semantic_v1["image"]
                            record["semantic_v1_seconds"] = semantic_v1["total_seconds"]
                            record["semantic_v1_raw_diffusion_seconds"] = semantic_v1["diffusion_seconds"]
                            record["semantic_v1_projection_seconds"] = semantic_v1["projection_seconds"]
                            record["crop_size"] = semantic_v1["crop_size"]
                            record["crop_box"] = semantic_v1["crop_box"]
                            record["crop_pixels_percent"] = round(
                                float((semantic_v1["crop_size"][0] * semantic_v1["crop_size"][1]) / (spec.width * spec.height) * 100.0),
                                4,
                            )

                            semantic_v2 = local_edit(
                                img2img,
                                spec,
                                case,
                                base,
                                prepared,
                                v2["prompt"],
                                v2["rgb"],
                                v2["projection_strength"],
                                sample_dir,
                                "semantic_v2_cached",
                                device,
                            )
                            semantic_v2_image = semantic_v2["image"]
                            record["semantic_v2_cached_seconds"] = semantic_v2["total_seconds"]
                            record["semantic_v2_raw_diffusion_seconds"] = semantic_v2["diffusion_seconds"]
                            record["semantic_v2_projection_seconds"] = semantic_v2["projection_seconds"]

                            started = now()
                            rollback = semantic_v1_image.copy()
                            rollback.save(sample_dir / "rollback_to_v1.png")
                            record["version_rollback_seconds"] = round(now() - started, 6)

                            ImageChops.difference(base, naive_v1).save(sample_dir / "diff_naive_v1.png")
                            ImageChops.difference(base, semantic_v1_image).save(sample_dir / "diff_semantic_v1.png")
                            ImageChops.difference(base, projection_image).save(sample_dir / "diff_projection_only.png")

                            record["naive_v1_metrics"] = method_metrics(base, naive_v1, prepared)
                            record["semantic_v1_metrics"] = method_metrics(base, semantic_v1_image, prepared)
                            record["semantic_v2_cached_metrics"] = method_metrics(base, semantic_v2_image, prepared)
                            record["projection_only_metrics"] = method_metrics(base, projection_image, prepared)
                            record["naive_v2_metrics"] = method_metrics(base, naive_v2, prepared)
                            record["outside_naive_v1_changed_percent"] = record["naive_v1_metrics"]["preservation"][
                                "outside_target_changed_pixels_percent_gt10"
                            ]
                            record["outside_semantic_v1_changed_percent"] = record["semantic_v1_metrics"]["preservation"][
                                "outside_target_changed_pixels_percent_gt10"
                            ]
                            record["outside_projection_only_changed_percent"] = record["projection_only_metrics"]["preservation"][
                                "outside_target_changed_pixels_percent_gt10"
                            ]
                            record["boundary_semantic_v1_changed_percent"] = record["semantic_v1_metrics"]["boundary"][
                                "boundary_band_changed_pixels_percent_gt10"
                            ]
                            record["inside_semantic_v1_mean_abs_diff"] = record["semantic_v1_metrics"]["preservation"][
                                "inside_target_mean_abs_diff"
                            ]
                            record["speedup_generation_v1"] = round(
                                record["naive_v1_seconds"] / record["semantic_v1_seconds"], 4
                            )
                            record["speedup_detector_inclusive_v1"] = round(
                                record["naive_v1_seconds"] / (record["semantic_v1_seconds"] + record["detector_seconds"]), 4
                            )
                            record["speedup_cached_reedit_v2"] = round(
                                record["naive_v2_seconds"] / record["semantic_v2_cached_seconds"], 4
                            )
                            record["speedup_projection_only"] = round(
                                record["naive_v1_seconds"] / max(record["projection_only_seconds"], 1e-6), 4
                            )

                            grid_path = sample_dir / "sample_grid.png"
                            save_sample_grid(
                                grid_path,
                                [
                                    ("base", base),
                                    ("naive v1", naive_v1),
                                    ("projection", projection_image),
                                    ("semantic v1", semantic_v1_image),
                                    ("semantic v2", semantic_v2_image),
                                    ("rollback", rollback),
                                ],
                            )
                            if len(metrics["sample_grids"]) < 40:
                                metrics["sample_grids"].append(str(grid_path))
                        except Exception as exc:
                            record["error"] = repr(exc)
                        metrics["records"].append(record)
                        done_keys.add(record_key)
                        save_metrics(output_dir, metrics)
                        print(json.dumps(compact_record(record), indent=2), flush=True)
            finally:
                cleanup(img2img)
        finally:
            cleanup(txt2img)
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    save_metrics(output_dir, metrics)
    print(json.dumps({"output_dir": str(output_dir), "summary": metrics["aggregates"]["summary"]}, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    run()
