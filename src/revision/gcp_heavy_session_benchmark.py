#!/usr/bin/env python3
"""GCP CUDA multi-model session benchmark for semantic editing workflows.

This benchmark measures realistic editing sessions rather than one short
generation at a time. For each model and target case it:

1. generates one base image,
2. detects and segments the target layer once with GroundingDINO+SAM2,
3. runs many full-image naive variants until the session lasts several minutes,
4. runs the same number of semantic local variants using the cached layer mask,
5. reports detector-inclusive and cached session speedups plus preservation
   metrics for the first few variants.

The benchmark is designed for paper revision evidence: it makes the amortized
workflow speedup visible under long repeated-edit sessions.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
)
from PIL import Image, ImageFilter
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    family: str
    source: str
    ref: str
    width: int
    height: int
    steps: int
    guidance_scale: float
    min_crop: int
    max_crop: int
    target_seconds: int


@dataclass(frozen=True)
class BenchCase:
    key: str
    label: str
    base_prompt: str
    edit_template: str
    negative_prompt: str
    queries: list[str]
    prior_cx: float
    prior_cy: float
    prior_aspect: float
    strength: float


CASES = {
    "mug": BenchCase(
        key="mug",
        label="product mug recolor",
        base_prompt=(
            "realistic product photograph of a plain white ceramic coffee mug on a wooden table, "
            "soft window light, clean background, high detail"
        ),
        edit_template=(
            "same product photograph, {variant} ceramic coffee mug only, keep table background "
            "shadows and lighting unchanged, realistic glaze"
        ),
        negative_prompt="people, person, face, hands, logo, text, watermark, blurry, low quality, distorted object",
        queries=["coffee mug", "mug", "ceramic mug", "cup"],
        prior_cx=0.50,
        prior_cy=0.55,
        prior_aspect=0.9,
        strength=0.36,
    ),
    "shirt": BenchCase(
        key="shirt",
        label="shirt recolor",
        base_prompt=(
            "realistic full body studio photo of an adult person standing, wearing a plain white "
            "t-shirt and blue jeans, simple gray background, sharp detail"
        ),
        edit_template=(
            "same adult person, {variant} t-shirt only, keep face jeans shoes and background "
            "unchanged, realistic fabric"
        ),
        negative_prompt="child, teen, underage, extra limbs, deformed hands, bad anatomy, blurry, text, watermark",
        queries=["shirt", "t-shirt", "white shirt", "upper clothing", "torso clothing"],
        prior_cx=0.50,
        prior_cy=0.48,
        prior_aspect=0.9,
        strength=0.34,
    ),
}


VARIANTS = [
    "glossy teal",
    "deep cobalt blue",
    "matte forest green",
    "warm terracotta",
    "burgundy red",
    "soft lavender",
    "sunflower yellow",
    "charcoal black",
    "pearl gray",
    "mint green",
    "navy blue",
    "coral pink",
    "brushed copper",
    "cream white",
    "emerald green",
    "ruby red",
    "sky blue",
    "olive green",
    "mustard yellow",
    "plum purple",
]


def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def timed(label: str, fn):
    start = cuda_time()
    value = fn()
    seconds = cuda_time() - start
    print(f"{label}: {seconds:.3f}s", flush=True)
    return value, seconds


def clean_mask(mask: np.ndarray | None, min_area: int = 16) -> np.ndarray | None:
    if mask is None or not np.any(mask):
        return None
    arr = mask.astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    arr = cv2.morphologyEx(arr, cv2.MORPH_OPEN, kernel)
    arr = cv2.morphologyEx(arr, cv2.MORPH_CLOSE, kernel)
    num, labels, stats, _ = cv2.connectedComponentsWithStats((arr > 0).astype(np.uint8), 8)
    keep = np.zeros_like(arr, dtype=bool)
    for idx in range(1, num):
        if int(stats[idx, cv2.CC_STAT_AREA]) >= min_area:
            keep |= labels == idx
    return keep if np.any(keep) else mask.astype(bool)


def mask_bbox(mask: Image.Image) -> list[int]:
    arr = np.array(mask.convert("L")) > 8
    ys, xs = np.where(arr)
    if xs.size == 0:
        raise RuntimeError("Empty mask")
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def prepare_mask(mask: Image.Image, blur: float = 0.9) -> Image.Image:
    arr = np.array(mask.convert("L"))
    arr = cv2.morphologyEx(arr, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    out = Image.fromarray(arr.astype(np.uint8), mode="L")
    if blur > 0:
        out = out.filter(ImageFilter.GaussianBlur(blur))
    return out


def crop_for_mask(image: Image.Image, mask: Image.Image, min_crop: int, max_crop: int) -> tuple[Image.Image, Image.Image, list[int]]:
    x1, y1, x2, y2 = mask_bbox(mask)
    width, height = image.size
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    span = max(x2 - x1 + 1, y2 - y1 + 1)
    side = max(min_crop, min(max_crop, max(span + 32, int(span * 2.25))))
    side = min(width, height, int(math.ceil(side / 8) * 8))
    left = max(0, min(int(round(cx - side / 2)), width - side))
    top = max(0, min(int(round(cy - side / 2)), height - side))
    box = [left, top, left + side, top + side]
    return image.crop(box), mask.crop(box), box


def paste_crop(base: Image.Image, crop: Image.Image, mask: Image.Image, box: list[int]) -> Image.Image:
    out = base.copy()
    crop = crop.resize((box[2] - box[0], box[3] - box[1]))
    out.paste(crop, (box[0], box[1]), mask)
    return out


def box_score(box: list[float], score: float, size: tuple[int, int], case: BenchCase) -> float:
    width, height = size
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0 / max(1, width)
    cy = (y1 + y2) / 2.0 / max(1, height)
    bw = max(1.0, x2 - x1) / max(1, width)
    bh = max(1.0, y2 - y1) / max(1, height)
    location_prior = 1.0 - abs(cx - case.prior_cx) * 0.9 - abs(cy - case.prior_cy) * 1.4
    shape_prior = 1.0 - abs((bw / max(bh, 1e-3)) - case.prior_aspect) * 0.12
    area_penalty = -max(0.0, bw * bh - 0.12) * 2.0
    return float(score) + max(0.0, location_prior) * 0.18 + max(0.0, shape_prior) * 0.08 + area_penalty


def diff_stats(base: Image.Image, edited: Image.Image, mask: Image.Image) -> dict[str, float]:
    a = np.array(base.convert("RGB")).astype(np.int16)
    b = np.array(edited.convert("RGB")).astype(np.int16)
    diff = np.abs(a - b).mean(axis=2)
    inside = np.array(mask.convert("L")) > 8
    outside = ~inside
    return {
        "inside_target_mean_abs_diff": float(diff[inside].mean()) if inside.any() else 0.0,
        "outside_target_changed_pixels_percent_gt10": float((diff[outside] > 10).mean() * 100.0) if outside.any() else 0.0,
    }


class Detector:
    def __init__(self, dino_model: str, sam2_model: str, device: str):
        self.device = device
        start = cuda_time()
        self.processor = AutoProcessor.from_pretrained(dino_model)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model).to(device)
        self.model.eval()
        self.dino_load_seconds = cuda_time() - start
        start = cuda_time()
        self.sam_processor = Sam2Processor.from_pretrained(sam2_model)
        self.sam_model = Sam2Model.from_pretrained(sam2_model).to(device)
        self.sam_model.eval()
        self.sam_load_seconds = cuda_time() - start

    def detect(self, image: Image.Image, case: BenchCase, threshold: float, text_threshold: float) -> tuple[Image.Image, dict[str, Any]]:
        dino_start = cuda_time()
        inputs = self.processor(images=image, text=[case.queries], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        if hasattr(self.processor, "post_process_grounded_object_detection"):
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=threshold,
                text_threshold=text_threshold,
                target_sizes=[image.size[::-1]],
            )
        else:
            results = self.processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=[image.size[::-1]])
        dino_seconds = cuda_time() - dino_start
        result = results[0]
        boxes = [[float(v) for v in box.tolist()] for box in result.get("boxes", [])]
        scores = [float(score) for score in result.get("scores", [])]
        raw_labels = result.get("text_labels", result.get("labels", []))
        labels = [str(label) for label in raw_labels]
        candidates = [
            {"box": box, "score": score, "label": label, "selection_score": box_score(box, score, image.size, case)}
            for box, score, label in zip(boxes, scores, labels)
        ]
        candidates.sort(key=lambda item: item["selection_score"], reverse=True)
        if not candidates:
            raise RuntimeError(f"No detector candidate for {case.key}")
        selected = candidates[0]

        sam_start = cuda_time()
        sam_inputs = self.sam_processor(images=image, input_boxes=[[[float(v) for v in selected["box"]]]], return_tensors="pt").to(self.device)
        with torch.no_grad():
            sam_outputs = self.sam_model(**sam_inputs, multimask_output=False)
        post_masks = self.sam_processor.post_process_masks(sam_outputs.pred_masks.cpu(), sam_inputs["original_sizes"])[0]
        sam_seconds = cuda_time() - sam_start
        arr = np.asarray(np.squeeze(post_masks[0].detach().cpu().numpy()))
        if arr.ndim == 3:
            arr = arr[0]
        mask = clean_mask(arr > 0.0)
        if mask is None:
            raise RuntimeError(f"SAM2 empty mask for {case.key}")
        mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        return mask_img, {
            "dino_inference_seconds": dino_seconds,
            "sam2_inference_seconds": sam_seconds,
            "detector_seconds": dino_seconds + sam_seconds,
            "raw_detection_count": len(candidates),
            "selected": selected,
            "bbox": mask_bbox(mask_img),
            "mask_pixels": int(np.count_nonzero(mask)),
        }


def load_pipelines(spec: ModelSpec):
    common = {"torch_dtype": torch.float16}
    if spec.family == "sdxl_single_file":
        txt2img = StableDiffusionXLPipeline.from_single_file(spec.source, use_safetensors=True, **common)
        img2img = StableDiffusionXLImg2ImgPipeline.from_single_file(spec.source, use_safetensors=True, **common)
    elif spec.family == "sdxl":
        txt2img = StableDiffusionXLPipeline.from_pretrained(spec.source, use_safetensors=True, **common)
        img2img = StableDiffusionXLImg2ImgPipeline.from_pretrained(spec.source, use_safetensors=True, **common)
    elif spec.family == "sd15":
        txt2img = StableDiffusionPipeline.from_pretrained(
            spec.source,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False,
            **common,
        )
        img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
            spec.source,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False,
            **common,
        )
    else:
        raise ValueError(f"Unsupported model family: {spec.family}")

    txt2img.to("cuda")
    img2img.to("cuda")
    for pipe in (txt2img, img2img):
        pipe.set_progress_bar_config(disable=True)
        if hasattr(pipe, "vae"):
            pipe.vae.enable_slicing()
    return txt2img, img2img


def make_models(checkpoint: str, target_default: int, target_long: int) -> list[ModelSpec]:
    return [
        ModelSpec(
            key="juggernaut_x_bucket",
            label="Juggernaut X RunDiffusion bucket SDXL",
            family="sdxl_single_file",
            source=checkpoint,
            ref="RunDiffusion/Juggernaut-X-v10",
            width=1024,
            height=1024,
            steps=35,
            guidance_scale=5.0,
            min_crop=512,
            max_crop=704,
            target_seconds=target_long,
        ),
        ModelSpec(
            key="sdxl_base_1_0",
            label="Stable Diffusion XL Base 1.0",
            family="sdxl",
            source="stabilityai/stable-diffusion-xl-base-1.0",
            ref="stabilityai/stable-diffusion-xl-base-1.0",
            width=1024,
            height=1024,
            steps=35,
            guidance_scale=5.0,
            min_crop=512,
            max_crop=704,
            target_seconds=target_default,
        ),
        ModelSpec(
            key="dreamshaper_xl_1_0",
            label="DreamShaper XL 1.0",
            family="sdxl",
            source="Lykon/dreamshaper-xl-1-0",
            ref="Lykon/dreamshaper-xl-1-0",
            width=1024,
            height=1024,
            steps=35,
            guidance_scale=5.0,
            min_crop=512,
            max_crop=704,
            target_seconds=target_default,
        ),
        ModelSpec(
            key="dreamshaper_8",
            label="DreamShaper 8",
            family="sd15",
            source="Lykon/dreamshaper-8",
            ref="Lykon/dreamshaper-8",
            width=768,
            height=768,
            steps=35,
            guidance_scale=7.0,
            min_crop=384,
            max_crop=512,
            target_seconds=target_default,
        ),
        ModelSpec(
            key="openjourney_v4",
            label="OpenJourney v4",
            family="sd15",
            source="prompthero/openjourney-v4",
            ref="prompthero/openjourney-v4",
            width=768,
            height=768,
            steps=35,
            guidance_scale=7.0,
            min_crop=384,
            max_crop=512,
            target_seconds=target_default,
        ),
    ]


def prompt_for(case: BenchCase, index: int) -> str:
    return case.edit_template.format(variant=VARIANTS[index % len(VARIANTS)])


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % 1_000_000


def pipeline_call(pipe, prompt: str, negative_prompt: str, spec: ModelSpec, seed: int, image: Image.Image | None = None, strength: float | None = None):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": spec.steps,
        "guidance_scale": spec.guidance_scale,
        "generator": generator,
    }
    if image is None:
        kwargs.update({"height": spec.height, "width": spec.width})
    else:
        kwargs.update({"image": image, "strength": strength})
    return pipe(**kwargs).images[0].convert("RGB")


def summarize_sessions(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [s for s in sessions if not s.get("error")]
    if not ok:
        return {"completed_sessions": 0}

    def mean(key: str) -> float:
        return statistics.mean(float(s[key]) for s in ok)

    return {
        "completed_sessions": len(ok),
        "total_variants": sum(int(s["variants"]) for s in ok),
        "total_naive_session_seconds": sum(float(s["naive_session_seconds"]) for s in ok),
        "total_semantic_session_seconds": sum(float(s["semantic_detector_inclusive_session_seconds"]) for s in ok),
        "total_cached_session_seconds": sum(float(s["cached_session_seconds"]) for s in ok),
        "mean_naive_session_seconds": mean("naive_session_seconds"),
        "mean_semantic_detector_inclusive_session_seconds": mean("semantic_detector_inclusive_session_seconds"),
        "mean_cached_session_seconds": mean("cached_session_seconds"),
        "mean_detector_seconds": mean("detector_seconds"),
        "mean_detector_inclusive_session_speedup": mean("detector_inclusive_session_speedup"),
        "mean_cached_session_speedup": mean("cached_session_speedup"),
        "mean_outside_drift_naive": mean("mean_outside_drift_naive"),
        "mean_outside_drift_semantic": mean("mean_outside_drift_semantic"),
    }


def write_outputs(out_dir: Path, result: dict[str, Any]) -> None:
    (out_dir / "gcp_heavy_session_benchmark_partial.json").write_text(json.dumps(result, indent=2))
    sessions = [s for s in result["sessions"] if not s.get("error")]
    lines = [
        "# GCP Heavy Session Benchmark",
        "",
        f"GPU: {result['device'].get('gpu', 'unknown')}",
        f"Completed sessions: {len(sessions)} / {len(result['sessions'])}",
        "",
        "| Model | Case | Variants | Naive session s | Semantic incl. detector s | Cached s | Incl. speedup | Cached speedup | Outside drift naive / semantic |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in sessions:
        lines.append(
            f"| {s['model_label']} | {s['case_label']} | {s['variants']} | "
            f"{s['naive_session_seconds']:.1f} | {s['semantic_detector_inclusive_session_seconds']:.1f} | "
            f"{s['cached_session_seconds']:.1f} | {s['detector_inclusive_session_speedup']:.2f}x | "
            f"{s['cached_session_speedup']:.2f}x | {s['mean_outside_drift_naive']:.2f}% / {s['mean_outside_drift_semantic']:.4f}% |"
        )
    lines.extend(["", "## Summary", "", "```json", json.dumps(result["summary"], indent=2), "```"])
    (out_dir / "gcp_heavy_session_benchmark_summary.md").write_text("\n".join(lines))


def run_session(
    *,
    out_dir: Path,
    spec: ModelSpec,
    case: BenchCase,
    detector: Detector,
    txt2img,
    img2img,
    min_variants: int,
    max_variants: int,
    drift_variants: int,
    dino_threshold: float,
    dino_text_threshold: float,
) -> dict[str, Any]:
    session_dir = out_dir / spec.key / case.key
    session_dir.mkdir(parents=True, exist_ok=True)
    seed_base = stable_seed(spec.key, case.key)
    print(f"\n=== {spec.label} / {case.label} ===", flush=True)

    base, base_seconds = timed(
        f"{spec.key}/{case.key} base",
        lambda: pipeline_call(txt2img, case.base_prompt, case.negative_prompt, spec, seed_base),
    )
    base.save(session_dir / "base.png")

    mask, det_seconds_wall = timed(
        f"{spec.key}/{case.key} detector",
        lambda: detector.detect(base, case, dino_threshold, dino_text_threshold),
    )
    det_mask, det_meta = mask
    prepared_mask = prepare_mask(det_mask)
    crop, crop_mask, crop_box = crop_for_mask(base, prepared_mask, spec.min_crop, spec.max_crop)
    det_mask.save(session_dir / "target_mask.png")
    crop.save(session_dir / "crop.png")
    crop_mask.save(session_dir / "crop_mask.png")

    first_prompt = prompt_for(case, 0)
    _, naive_calibration_seconds = timed(
        f"{spec.key}/{case.key} calibration naive",
        lambda: pipeline_call(txt2img, first_prompt, case.negative_prompt, spec, seed_base + 100),
    )
    _, local_calibration_seconds = timed(
        f"{spec.key}/{case.key} calibration local",
        lambda: pipeline_call(img2img, first_prompt, case.negative_prompt, spec, seed_base + 200, image=crop, strength=case.strength),
    )

    variants = max(min_variants, math.ceil(spec.target_seconds / max(naive_calibration_seconds, 1e-6)))
    variants = min(max_variants, variants)
    print(
        f"{spec.key}/{case.key}: target={spec.target_seconds}s calibration_naive={naive_calibration_seconds:.3f}s "
        f"calibration_local={local_calibration_seconds:.3f}s variants={variants}",
        flush=True,
    )

    naive_times: list[float] = []
    local_times: list[float] = []
    naive_drift: list[float] = []
    semantic_drift: list[float] = []
    semantic_inside: list[float] = []
    sample_records: list[dict[str, Any]] = []

    for index in range(variants):
        prompt = prompt_for(case, index)
        naive, naive_seconds = timed(
            f"{spec.key}/{case.key} naive {index + 1}/{variants}",
            lambda p=prompt, i=index: pipeline_call(txt2img, p, case.negative_prompt, spec, seed_base + 1000 + i),
        )
        naive_times.append(naive_seconds)

        raw_local, local_seconds = timed(
            f"{spec.key}/{case.key} local {index + 1}/{variants}",
            lambda p=prompt, i=index: pipeline_call(
                img2img,
                p,
                case.negative_prompt,
                spec,
                seed_base + 2000 + i,
                image=crop,
                strength=case.strength,
            ),
        )
        local_times.append(local_seconds)
        semantic = paste_crop(base, raw_local, crop_mask, crop_box)

        if index < drift_variants:
            naive.save(session_dir / f"naive_{index + 1:03d}.png")
            semantic.save(session_dir / f"semantic_{index + 1:03d}.png")
            nd = diff_stats(base, naive, prepared_mask)
            sd = diff_stats(base, semantic, prepared_mask)
            naive_drift.append(nd["outside_target_changed_pixels_percent_gt10"])
            semantic_drift.append(sd["outside_target_changed_pixels_percent_gt10"])
            semantic_inside.append(sd["inside_target_mean_abs_diff"])
            grid = Image.new("RGB", (base.width * 3, base.height), "white")
            grid.paste(base, (0, 0))
            grid.paste(naive, (base.width, 0))
            grid.paste(semantic, (base.width * 2, 0))
            grid.save(session_dir / f"grid_{index + 1:03d}.png")
        sample_records.append({"index": index + 1, "prompt": prompt, "naive_seconds": naive_seconds, "local_seconds": local_seconds})

    naive_session = sum(naive_times)
    cached_session = sum(local_times)
    semantic_session = det_meta["detector_seconds"] + cached_session
    return {
        "model_key": spec.key,
        "model_label": spec.label,
        "model_family": spec.family,
        "model_source": spec.source,
        "model_ref": spec.ref,
        "case_key": case.key,
        "case_label": case.label,
        "width": spec.width,
        "height": spec.height,
        "steps": spec.steps,
        "guidance_scale": spec.guidance_scale,
        "strength": case.strength,
        "variants": variants,
        "target_naive_session_seconds": spec.target_seconds,
        "base_seconds": base_seconds,
        "detector_seconds": det_meta["detector_seconds"],
        "detector_wall_seconds": det_seconds_wall,
        "dino_inference_seconds": det_meta["dino_inference_seconds"],
        "sam2_inference_seconds": det_meta["sam2_inference_seconds"],
        "crop_box": crop_box,
        "crop_size": crop.size[0],
        "crop_area_percent": (crop.size[0] * crop.size[1]) / (spec.width * spec.height) * 100.0,
        "naive_calibration_seconds": naive_calibration_seconds,
        "local_calibration_seconds": local_calibration_seconds,
        "naive_session_seconds": naive_session,
        "semantic_detector_inclusive_session_seconds": semantic_session,
        "cached_session_seconds": cached_session,
        "mean_naive_variant_seconds": statistics.mean(naive_times),
        "mean_local_variant_seconds": statistics.mean(local_times),
        "detector_inclusive_session_speedup": naive_session / semantic_session,
        "cached_session_speedup": naive_session / cached_session,
        "mean_outside_drift_naive": statistics.mean(naive_drift) if naive_drift else 0.0,
        "mean_outside_drift_semantic": statistics.mean(semantic_drift) if semantic_drift else 0.0,
        "mean_semantic_inside_change": statistics.mean(semantic_inside) if semantic_inside else 0.0,
        "detector": det_meta,
        "samples": sample_records,
        "error": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--target-seconds-default", type=int, default=300)
    parser.add_argument("--target-seconds-long", type=int, default=600)
    parser.add_argument("--min-variants", type=int, default=12)
    parser.add_argument("--max-variants", type=int, default=160)
    parser.add_argument("--drift-variants", type=int, default=5)
    parser.add_argument("--cases", default="mug,shirt")
    parser.add_argument("--models", default="all")
    parser.add_argument("--shirt-models", default="juggernaut_x_bucket,dreamshaper_8")
    parser.add_argument("--dino-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--sam2-model", default="facebook/sam2.1-hiera-tiny")
    parser.add_argument("--dino-threshold", type=float, default=0.12)
    parser.add_argument("--dino-text-threshold", type=float, default=0.10)
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = make_models(args.checkpoint, args.target_seconds_default, args.target_seconds_long)
    if args.models != "all":
        requested_models = {m.strip() for m in args.models.split(",") if m.strip()}
        models = [model for model in models if model.key in requested_models]
    requested_cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    shirt_models = {m.strip() for m in args.shirt_models.split(",") if m.strip()}

    print("Loading detector once for all sessions...", flush=True)
    detector = Detector(args.dino_model, args.sam2_model, "cuda")
    sessions: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "gcp_heavy_session_benchmark",
        "device": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "config": vars(args),
        "detector_load": {
            "dino_load_seconds": detector.dino_load_seconds,
            "sam2_load_seconds": detector.sam_load_seconds,
        },
        "sessions": sessions,
        "summary": {},
    }

    for spec in models:
        try:
            load_start = cuda_time()
            txt2img, img2img = load_pipelines(spec)
            load_seconds = cuda_time() - load_start
            print(f"Loaded {spec.key} in {load_seconds:.1f}s", flush=True)
        except Exception as exc:
            sessions.append({"model_key": spec.key, "model_label": spec.label, "error": f"load failed: {exc!r}"})
            result["summary"] = summarize_sessions(sessions)
            write_outputs(out_dir, result)
            continue

        for case_key in requested_cases:
            if case_key == "shirt" and spec.key not in shirt_models:
                continue
            case = CASES[case_key]
            try:
                session = run_session(
                    out_dir=out_dir,
                    spec=spec,
                    case=case,
                    detector=detector,
                    txt2img=txt2img,
                    img2img=img2img,
                    min_variants=args.min_variants,
                    max_variants=args.max_variants,
                    drift_variants=args.drift_variants,
                    dino_threshold=args.dino_threshold,
                    dino_text_threshold=args.dino_text_threshold,
                )
                session["model_load_seconds"] = load_seconds
            except Exception as exc:
                session = {
                    "model_key": spec.key,
                    "model_label": spec.label,
                    "case_key": case_key,
                    "case_label": CASES[case_key].label,
                    "error": repr(exc),
                }
                print(f"ERROR {spec.key}/{case_key}: {exc!r}", flush=True)
            sessions.append(session)
            result["summary"] = summarize_sessions(sessions)
            write_outputs(out_dir, result)
            gc.collect()
            torch.cuda.empty_cache()

        del txt2img
        del img2img
        gc.collect()
        torch.cuda.empty_cache()

    result["completed_at"] = datetime.now(timezone.utc).isoformat()
    result["summary"] = summarize_sessions(sessions)
    (out_dir / "gcp_heavy_session_benchmark.json").write_text(json.dumps(result, indent=2))
    write_outputs(out_dir, result)
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
