#!/usr/bin/env python3
"""GCP CUDA detector-inclusive SDXL semantic workflow benchmark.

This is the cloud counterpart to the local Apple MPS heavy benchmark. It uses a
single SDXL checkpoint on a CUDA GPU, loads GroundingDINO and SAM2 once, and
measures per-sample:

* full-image SDXL base generation,
* full-image SDXL naive edit,
* GroundingDINO+SAM2 target detection and mask creation,
* local semantic crop img2img,
* cached local crop img2img.

The goal is not model parity with the 10-checkpoint local run; it is a measured
CUDA tier for the detector-inclusive scheduling argument.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
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
from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline
from PIL import Image, ImageDraw, ImageFilter
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor


@dataclass(frozen=True)
class BenchCase:
    key: str
    label: str
    base_prompt: str
    naive_prompt: str
    edit_prompt: str
    cached_prompt: str
    negative_prompt: str
    target: str
    queries: list[str]
    prior_cx: float
    prior_cy: float
    prior_aspect: float
    strength: float
    min_crop: int
    max_crop: int


CASES = [
    BenchCase(
        "face_lips",
        "face lips",
        "studio portrait photo of an adult woman, centered face, natural pink lips, clear eyes, soft neutral background, realistic photography",
        "studio portrait photo of an adult woman, centered face, vivid red lipstick, clear eyes, soft neutral background, realistic photography",
        "same adult woman portrait, vivid red lipstick on closed lips only, preserve face identity and lighting, realistic makeup",
        "same adult woman portrait, deep burgundy lipstick on closed lips only, preserve face identity and lighting, realistic makeup",
        "child, teen, underage, open mouth, teeth, deformed lips, bad anatomy, blurry, text, watermark",
        "lips",
        ["lips", "mouth", "closed lips", "pink lips", "woman lips"],
        0.50,
        0.59,
        2.6,
        0.28,
        256,
        384,
    ),
    BenchCase(
        "human_body_shirt",
        "human body shirt",
        "realistic full body studio photo of an adult person standing, wearing a plain white t-shirt and blue jeans, simple gray background, sharp detail",
        "realistic full body studio photo of an adult person standing, wearing a bright cobalt blue t-shirt and blue jeans, simple gray background, sharp detail",
        "same person, bright cobalt blue t-shirt only, keep face jeans shoes and background unchanged, realistic fabric",
        "same person, forest green t-shirt only, keep face jeans shoes and background unchanged, realistic fabric",
        "child, teen, underage, extra limbs, deformed hands, bad anatomy, blurry, text, watermark",
        "shirt",
        ["shirt", "t-shirt", "white shirt", "upper clothing", "torso clothing"],
        0.50,
        0.48,
        0.9,
        0.34,
        384,
        512,
    ),
    BenchCase(
        "object_mug",
        "object mug",
        "realistic product photo of a plain white ceramic coffee mug on a wooden table, soft window light, clean background",
        "realistic product photo of a teal ceramic coffee mug on a wooden table, soft window light, clean background",
        "same product photograph, teal ceramic coffee mug only, keep table background and lighting unchanged, realistic glaze",
        "same product photograph, glossy red ceramic coffee mug only, keep table background and lighting unchanged, realistic glaze",
        "people, person, face, hands, logo, text, watermark, blurry, low quality, distorted object",
        "mug",
        ["coffee mug", "mug", "ceramic mug", "cup"],
        0.50,
        0.55,
        0.9,
        0.36,
        384,
        512,
    ),
    BenchCase(
        "landscape_sky",
        "landscape sky",
        "realistic landscape photograph, mountain lake valley, broad pale blue sky, natural daylight, crisp detail, 35mm lens",
        "realistic landscape photograph, mountain lake valley, dramatic warm orange sunset sky, natural scene, crisp detail, 35mm lens",
        "same landscape photograph, warm orange sunset sky only, keep mountains lake and foreground unchanged, realistic lighting",
        "same landscape photograph, stormy blue gray sky only, keep mountains lake and foreground unchanged, realistic lighting",
        "people, person, buildings, logo, text, watermark, blurry, low quality, distorted landscape",
        "sky",
        ["sky", "blue sky", "open sky", "cloudy sky"],
        0.50,
        0.22,
        5.0,
        0.40,
        384,
        512,
    ),
]


def now() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def timed(label: str, fn):
    start = now()
    value = fn()
    seconds = now() - start
    print(f"{label}: {seconds:.4f}s", flush=True)
    return value, seconds


def box_score(box: list[float], score: float, size: tuple[int, int], case: BenchCase) -> float:
    width, height = size
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0 / max(1, width)
    cy = (y1 + y2) / 2.0 / max(1, height)
    bw = max(1.0, x2 - x1) / max(1, width)
    bh = max(1.0, y2 - y1) / max(1, height)
    location_prior = 1.0 - abs(cx - case.prior_cx) * 0.9 - abs(cy - case.prior_cy) * 1.4
    shape_prior = 1.0 - abs((bw / max(bh, 1e-3)) - case.prior_aspect) * 0.12
    area_penalty = -max(0.0, bw * bh - 0.08) * 3.0
    return float(score) + max(0.0, location_prior) * 0.18 + max(0.0, shape_prior) * 0.08 + area_penalty


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


def prepare_mask(mask: Image.Image, blur: float = 0.8) -> Image.Image:
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
    side = max(min_crop, min(max_crop, max(span + 16, int(span * 2.25))))
    side = min(width, height, int(math.ceil(side / 8) * 8))
    left = max(0, min(int(round(cx - side / 2)), width - side))
    top = max(0, min(int(round(cy - side / 2)), height - side))
    box = [left, top, left + side, top + side]
    return image.crop(box), mask.crop(box), box


def paste_crop(base: Image.Image, crop: Image.Image, mask: Image.Image, box: list[int]) -> Image.Image:
    out = base.copy()
    out.paste(crop.resize((box[2] - box[0], box[3] - box[1])), (box[0], box[1]), mask)
    return out


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
        start = now()
        self.processor = AutoProcessor.from_pretrained(dino_model)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model).to(device)
        self.model.eval()
        self.dino_load_seconds = now() - start
        start = now()
        self.sam_processor = Sam2Processor.from_pretrained(sam2_model)
        self.sam_model = Sam2Model.from_pretrained(sam2_model).to(device)
        self.sam_model.eval()
        self.sam_load_seconds = now() - start
        self.device = device

    def detect(self, image: Image.Image, case: BenchCase, threshold: float, text_threshold: float) -> tuple[Image.Image, dict[str, Any]]:
        dino_start = now()
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
            results = self.processor.post_process_object_detection(
                outputs,
                threshold=threshold,
                target_sizes=[image.size[::-1]],
            )
        dino_seconds = now() - dino_start
        result = results[0]
        boxes = [[float(v) for v in box.tolist()] for box in result.get("boxes", [])]
        scores = [float(score) for score in result.get("scores", [])]
        raw_labels = result.get("text_labels", result.get("labels", []))
        labels = [str(label) for label in raw_labels]
        candidates = []
        for box, score, label in zip(boxes, scores, labels):
            candidates.append({"box": box, "score": score, "label": label, "selection_score": box_score(box, score, image.size, case)})
        candidates.sort(key=lambda item: item["selection_score"], reverse=True)
        if not candidates:
            raise RuntimeError(f"No detector candidate for {case.key}")
        selected = candidates[0]

        sam_start = now()
        sam_inputs = self.sam_processor(images=image, input_boxes=[[[float(v) for v in selected["box"]]]], return_tensors="pt").to(self.device)
        with torch.no_grad():
            sam_outputs = self.sam_model(**sam_inputs, multimask_output=False)
        post_masks = self.sam_processor.post_process_masks(sam_outputs.pred_masks.cpu(), sam_inputs["original_sizes"])[0]
        sam_seconds = now() - sam_start
        arr = np.asarray(np.squeeze(post_masks[0].detach().cpu().numpy()))
        if arr.ndim == 3:
            arr = arr[0]
        mask = clean_mask(arr > 0.0)
        if mask is None:
            raise RuntimeError(f"SAM2 empty mask for {case.key}")
        mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        meta = {
            "dino_inference_seconds": dino_seconds,
            "sam2_inference_seconds": sam_seconds,
            "detector_seconds": dino_seconds + sam_seconds,
            "raw_detection_count": len(candidates),
            "selected": selected,
            "bbox": mask_bbox(mask_img),
            "mask_pixels": int(np.count_nonzero(mask)),
        }
        return mask_img, meta


def summarize(rows: list[dict[str, Any]], detector: Detector) -> dict[str, Any]:
    def mean(key: str) -> float:
        return statistics.mean(float(r[key]) for r in rows)

    def med(key: str) -> float:
        return statistics.median(float(r[key]) for r in rows)

    return {
        "samples": len(rows),
        "dino_load_seconds": detector.dino_load_seconds,
        "sam2_load_seconds": detector.sam_load_seconds,
        "mean_naive_seconds": mean("naive_seconds"),
        "mean_detector_seconds": mean("detector_seconds"),
        "median_detector_seconds": med("detector_seconds"),
        "mean_semantic_local_seconds": mean("semantic_local_seconds"),
        "mean_cached_local_seconds": mean("cached_local_seconds"),
        "mean_detector_inclusive_semantic_seconds": mean("semantic_detector_inclusive_seconds"),
        "mean_generation_only_speedup": statistics.mean(r["naive_seconds"] / r["semantic_local_seconds"] for r in rows),
        "mean_detector_inclusive_speedup": statistics.mean(r["naive_seconds"] / r["semantic_detector_inclusive_seconds"] for r in rows),
        "mean_cached_speedup": statistics.mean(r["naive_seconds"] / r["cached_local_seconds"] for r in rows),
        "mean_outside_drift_naive": mean("naive_outside_drift"),
        "mean_outside_drift_semantic": mean("semantic_outside_drift"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--samples-per-case", type=int, default=25)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--dino-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--sam2-model", default="facebook/sam2.1-hiera-tiny")
    parser.add_argument("--dino-threshold", type=float, default=0.12)
    parser.add_argument("--dino-text-threshold", type=float, default=0.10)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("Loading SDXL pipelines...", flush=True)
    txt2img = StableDiffusionXLPipeline.from_single_file(args.checkpoint, torch_dtype=torch.float16, use_safetensors=True).to("cuda")
    txt2img.set_progress_bar_config(disable=True)
    img2img = StableDiffusionXLImg2ImgPipeline.from_single_file(args.checkpoint, torch_dtype=torch.float16, use_safetensors=True).to("cuda")
    img2img.set_progress_bar_config(disable=True)

    print("Loading detector...", flush=True)
    detector = Detector(args.dino_model, args.sam2_model, "cuda")
    print(f"detector cold load: dino={detector.dino_load_seconds:.4f}s sam2={detector.sam_load_seconds:.4f}s", flush=True)

    rows: list[dict[str, Any]] = []
    for case_index, case in enumerate(CASES):
        for sample_index in range(args.samples_per_case):
            sample_id = sample_index + 1
            seed = 80000 + case_index * 1000 + sample_index
            sample_dir = out_dir / case.key / f"sample_{sample_id:03d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            try:
                base, base_seconds = timed(
                    f"{case.key} {sample_id} base",
                    lambda: txt2img(
                        prompt=case.base_prompt,
                        negative_prompt=case.negative_prompt,
                        height=args.height,
                        width=args.width,
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance_scale,
                        generator=torch.Generator(device="cuda").manual_seed(seed),
                    ).images[0].convert("RGB"),
                )
                base.save(sample_dir / "base.png")
                naive, naive_seconds = timed(
                    f"{case.key} {sample_id} naive",
                    lambda: txt2img(
                        prompt=case.naive_prompt,
                        negative_prompt=case.negative_prompt,
                        height=args.height,
                        width=args.width,
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance_scale,
                        generator=torch.Generator(device="cuda").manual_seed(seed + 101),
                    ).images[0].convert("RGB"),
                )
                naive.save(sample_dir / "naive.png")
                mask, det_meta = timed(
                    f"{case.key} {sample_id} detector",
                    lambda: detector.detect(base, case, args.dino_threshold, args.dino_text_threshold),
                )
                det_mask, det_values = mask
                det_mask.save(sample_dir / "target_mask.png")
                prepared = prepare_mask(det_mask)
                crop, crop_mask, box = crop_for_mask(base, prepared, case.min_crop, case.max_crop)
                crop.save(sample_dir / "crop.png")
                crop_mask.save(sample_dir / "crop_mask.png")
                raw_local, local_seconds = timed(
                    f"{case.key} {sample_id} local",
                    lambda: img2img(
                        prompt=case.edit_prompt,
                        negative_prompt=case.negative_prompt,
                        image=crop,
                        num_inference_steps=args.steps,
                        strength=case.strength,
                        guidance_scale=args.guidance_scale,
                        generator=torch.Generator(device="cuda").manual_seed(seed + 202),
                    ).images[0].convert("RGB"),
                )
                semantic = paste_crop(base, raw_local, crop_mask, box)
                semantic.save(sample_dir / "semantic.png")
                cached_crop, cached_seconds = timed(
                    f"{case.key} {sample_id} cached",
                    lambda: img2img(
                        prompt=case.cached_prompt,
                        negative_prompt=case.negative_prompt,
                        image=crop,
                        num_inference_steps=args.steps,
                        strength=case.strength,
                        guidance_scale=args.guidance_scale,
                        generator=torch.Generator(device="cuda").manual_seed(seed + 303),
                    ).images[0].convert("RGB"),
                )
                cached = paste_crop(base, cached_crop, crop_mask, box)
                cached.save(sample_dir / "cached.png")

                overlay = base.convert("RGBA")
                tint = Image.new("RGBA", base.size, (255, 0, 80, 0))
                tint.putalpha(det_mask.point(lambda p: min(120, p)))
                overlay = Image.alpha_composite(overlay, tint).convert("RGB")
                grid = Image.new("RGB", (base.width * 4, base.height), "white")
                for idx, im in enumerate([base, naive, overlay, semantic]):
                    grid.paste(im, (idx * base.width, 0))
                grid.save(sample_dir / "sample_grid.png")

                naive_drift = diff_stats(base, naive, prepared)["outside_target_changed_pixels_percent_gt10"]
                semantic_stats = diff_stats(base, semantic, prepared)
                row = {
                    "case_key": case.key,
                    "sample_index": sample_id,
                    "seed": seed,
                    "crop_box": box,
                    "crop_size": crop.size[0],
                    "base_seconds": base_seconds,
                    "naive_seconds": naive_seconds,
                    "detector_seconds": det_values["detector_seconds"],
                    "dino_inference_seconds": det_values["dino_inference_seconds"],
                    "sam2_inference_seconds": det_values["sam2_inference_seconds"],
                    "semantic_local_seconds": local_seconds,
                    "cached_local_seconds": cached_seconds,
                    "semantic_detector_inclusive_seconds": det_values["detector_seconds"] + local_seconds,
                    "generation_only_speedup": naive_seconds / local_seconds,
                    "detector_inclusive_speedup": naive_seconds / (det_values["detector_seconds"] + local_seconds),
                    "cached_speedup": naive_seconds / cached_seconds,
                    "naive_outside_drift": naive_drift,
                    "semantic_outside_drift": semantic_stats["outside_target_changed_pixels_percent_gt10"],
                    "semantic_inside_change": semantic_stats["inside_target_mean_abs_diff"],
                    "detector": det_values,
                    "error": None,
                }
            except Exception as exc:
                row = {"case_key": case.key, "sample_index": sample_id, "seed": seed, "error": repr(exc)}
                print(f"ERROR {case.key} {sample_id}: {exc!r}", flush=True)
            rows.append(row)
            ok_rows = [r for r in rows if not r.get("error")]
            partial = {"rows": rows, "summary": summarize(ok_rows, detector) if ok_rows else {}}
            (out_dir / "cuda_detector_inclusive_partial.json").write_text(json.dumps(partial, indent=2))
            gc.collect()
            torch.cuda.empty_cache()

    ok_rows = [r for r in rows if not r.get("error")]
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "gcp_cuda_detector_inclusive_benchmark",
        "device": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "config": vars(args),
        "summary": summarize(ok_rows, detector) if ok_rows else {},
        "rows": rows,
    }
    (out_dir / "cuda_detector_inclusive_benchmark.json").write_text(json.dumps(result, indent=2))
    lines = ["# GCP CUDA Detector-Inclusive Benchmark", "", f"GPU: {result['device']['gpu']}", f"Samples completed: {len(ok_rows)} / {len(rows)}", "", "| Metric | Value |", "| --- | ---: |"]
    for key, value in result["summary"].items():
        lines.append(f"| {key} | {value:.4f} |" if isinstance(value, float) else f"| {key} | {value} |")
    (out_dir / "cuda_detector_inclusive_benchmark_summary.md").write_text("\n".join(lines))
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
