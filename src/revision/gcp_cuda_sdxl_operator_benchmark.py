#!/usr/bin/env python3
"""Focused CUDA SDXL operator benchmark for the FGCS revision.

The full paper benchmark was produced on Apple MPS. This script measures the
same scheduling-relevant diffusion operators on a CUDA GPU without rerunning
the full detector-heavy 200-sample workflow:

* full-image SDXL txt2img for a naive edit,
* local-crop SDXL img2img for the semantic edit path,
* repeated local-crop SDXL img2img as the cached-edit proxy.

Detector and segmentation are intentionally not included here. The result is a
CUDA-class operator timing tier that can replace the synthetic diffusion scale
used in the sensitivity analysis while keeping the manuscript honest about
detector overhead.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline
from PIL import Image


@dataclass(frozen=True)
class BenchCase:
    key: str
    base_prompt: str
    naive_prompt: str
    edit_prompt: str
    cached_prompt: str
    negative_prompt: str
    crop_size: int
    strength: float


CASES = [
    BenchCase(
        "face_lips",
        "studio portrait photo of an adult woman, centered face, natural pink lips, clear eyes, soft neutral background, realistic photography",
        "studio portrait photo of an adult woman, centered face, vivid red lipstick, clear eyes, soft neutral background, realistic photography",
        "same adult woman portrait, vivid red lipstick on closed lips only, preserve face identity and lighting, realistic makeup",
        "same adult woman portrait, deep burgundy lipstick on closed lips only, preserve face identity and lighting, realistic makeup",
        "child, teen, underage, open mouth, teeth, deformed lips, bad anatomy, blurry, text, watermark",
        384,
        0.28,
    ),
    BenchCase(
        "human_body_shirt",
        "realistic full body studio photo of an adult person standing, wearing a plain white t-shirt and blue jeans, simple gray background, sharp detail",
        "realistic full body studio photo of an adult person standing, wearing a bright cobalt blue t-shirt and blue jeans, simple gray background, sharp detail",
        "same person, bright cobalt blue t-shirt only, keep face jeans shoes and background unchanged, realistic fabric",
        "same person, forest green t-shirt only, keep face jeans shoes and background unchanged, realistic fabric",
        "child, teen, underage, extra limbs, deformed hands, bad anatomy, blurry, text, watermark",
        512,
        0.34,
    ),
    BenchCase(
        "object_mug",
        "realistic product photo of a plain white ceramic coffee mug on a wooden table, soft window light, clean background",
        "realistic product photo of a teal ceramic coffee mug on a wooden table, soft window light, clean background",
        "same product photograph, teal ceramic coffee mug only, keep table background and lighting unchanged, realistic glaze",
        "same product photograph, glossy red ceramic coffee mug only, keep table background and lighting unchanged, realistic glaze",
        "people, person, face, hands, logo, text, watermark, blurry, low quality, distorted object",
        512,
        0.36,
    ),
    BenchCase(
        "landscape_sky",
        "realistic landscape photograph, mountain lake valley, clear blue sky, natural daylight, crisp detail, 35mm lens",
        "realistic landscape photograph, mountain lake valley, warm orange sunset sky, natural scene, crisp detail, 35mm lens",
        "same landscape photograph, warm orange sunset sky only, keep mountains lake and foreground unchanged, realistic lighting",
        "same landscape photograph, stormy blue gray sky only, keep mountains lake and foreground unchanged, realistic lighting",
        "cartoon, painting, low quality, blurry, text, watermark, buildings, people",
        512,
        0.32,
    ),
]


def sync_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def timed_call(label: str, fn) -> tuple[Image.Image, float]:
    torch.cuda.empty_cache()
    start = sync_time()
    image = fn().images[0].convert("RGB")
    seconds = sync_time() - start
    print(f"{label}: {seconds:.4f}s", flush=True)
    return image, seconds


def center_crop(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    left = max(0, (width - size) // 2)
    top = max(0, (height - size) // 2)
    return image.crop((left, top, left + size, top + size)).resize((size, size))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def vals(key: str) -> list[float]:
        return [float(r[key]) for r in rows]

    naive = vals("naive_seconds")
    local = vals("semantic_local_seconds")
    cached = vals("cached_local_seconds")
    return {
        "samples": len(rows),
        "mean_naive_seconds": statistics.mean(naive),
        "median_naive_seconds": statistics.median(naive),
        "mean_semantic_local_seconds": statistics.mean(local),
        "median_semantic_local_seconds": statistics.median(local),
        "mean_cached_local_seconds": statistics.mean(cached),
        "median_cached_local_seconds": statistics.median(cached),
        "mean_generation_only_speedup": statistics.mean(r["naive_seconds"] / r["semantic_local_seconds"] for r in rows),
        "median_generation_only_speedup": statistics.median(r["naive_seconds"] / r["semantic_local_seconds"] for r in rows),
        "mean_cached_speedup": statistics.mean(r["naive_seconds"] / r["cached_local_seconds"] for r in rows),
        "median_cached_speedup": statistics.median(r["naive_seconds"] / r["cached_local_seconds"] for r in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--samples-per-case", type=int, default=2)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("Loading SDXL txt2img pipeline...", flush=True)
    txt2img = StableDiffusionXLPipeline.from_single_file(
        args.checkpoint,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")
    txt2img.set_progress_bar_config(disable=True)

    print("Loading SDXL img2img pipeline...", flush=True)
    img2img = StableDiffusionXLImg2ImgPipeline.from_single_file(
        args.checkpoint,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")
    img2img.set_progress_bar_config(disable=True)

    print("Warmup...", flush=True)
    generator = torch.Generator(device="cuda").manual_seed(1234)
    _ = txt2img(
        prompt=CASES[0].base_prompt,
        negative_prompt=CASES[0].negative_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=4,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images[0]
    torch.cuda.synchronize()

    rows: list[dict[str, Any]] = []
    for case_index, case in enumerate(CASES):
        for sample_index in range(args.samples_per_case):
            seed = 50000 + case_index * 1000 + sample_index
            sample_dir = out_dir / case.key / f"sample_{sample_index + 1:02d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            base, base_seconds = timed_call(
                f"{case.key} sample {sample_index + 1} base",
                lambda: txt2img(
                    prompt=case.base_prompt,
                    negative_prompt=case.negative_prompt,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    generator=torch.Generator(device="cuda").manual_seed(seed),
                ),
            )
            base.save(sample_dir / "base.png")

            naive, naive_seconds = timed_call(
                f"{case.key} sample {sample_index + 1} naive",
                lambda: txt2img(
                    prompt=case.naive_prompt,
                    negative_prompt=case.negative_prompt,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    generator=torch.Generator(device="cuda").manual_seed(seed + 101),
                ),
            )
            naive.save(sample_dir / "naive.png")

            crop = center_crop(base, case.crop_size)
            crop.save(sample_dir / "local_crop.png")
            local, local_seconds = timed_call(
                f"{case.key} sample {sample_index + 1} local",
                lambda: img2img(
                    prompt=case.edit_prompt,
                    negative_prompt=case.negative_prompt,
                    image=crop,
                    num_inference_steps=args.steps,
                    strength=case.strength,
                    guidance_scale=args.guidance_scale,
                    generator=torch.Generator(device="cuda").manual_seed(seed + 202),
                ),
            )
            local.save(sample_dir / "semantic_local_crop.png")

            cached, cached_seconds = timed_call(
                f"{case.key} sample {sample_index + 1} cached",
                lambda: img2img(
                    prompt=case.cached_prompt,
                    negative_prompt=case.negative_prompt,
                    image=crop,
                    num_inference_steps=args.steps,
                    strength=case.strength,
                    guidance_scale=args.guidance_scale,
                    generator=torch.Generator(device="cuda").manual_seed(seed + 303),
                ),
            )
            cached.save(sample_dir / "cached_local_crop.png")

            row = {
                "case_key": case.key,
                "sample_index": sample_index + 1,
                "seed": seed,
                "crop_size": case.crop_size,
                "base_seconds": base_seconds,
                "naive_seconds": naive_seconds,
                "semantic_local_seconds": local_seconds,
                "cached_local_seconds": cached_seconds,
                "generation_only_speedup": naive_seconds / local_seconds,
                "cached_speedup": naive_seconds / cached_seconds,
            }
            rows.append(row)
            (out_dir / "cuda_sdxl_operator_benchmark_partial.json").write_text(json.dumps({"rows": rows}, indent=2))

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "gcp_cuda_sdxl_operator_benchmark",
        "checkpoint": args.checkpoint,
        "device": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "driver_visible": True,
        },
        "config": {
            "steps": args.steps,
            "samples_per_case": args.samples_per_case,
            "height": args.height,
            "width": args.width,
            "guidance_scale": args.guidance_scale,
        },
        "summary": summarize(rows),
        "rows": rows,
    }
    (out_dir / "cuda_sdxl_operator_benchmark.json").write_text(json.dumps(result, indent=2))

    lines = [
        "# GCP CUDA SDXL Operator Benchmark",
        "",
        f"GPU: {result['device']['gpu']}",
        f"Torch: {result['device']['torch']} / CUDA {result['device']['cuda']}",
        f"Steps: {args.steps}",
        f"Samples: {len(rows)}",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in result["summary"].items():
        if key == "samples":
            lines.append(f"| {key} | {value} |")
        else:
            lines.append(f"| {key} | {value:.4f} |")
    lines.extend(["", "| Case | Sample | Crop | Naive | Local | Cached | Gen speedup | Cached speedup |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in rows:
        lines.append(
            f"| {row['case_key']} | {row['sample_index']} | {row['crop_size']} | "
            f"{row['naive_seconds']:.4f} | {row['semantic_local_seconds']:.4f} | "
            f"{row['cached_local_seconds']:.4f} | {row['generation_only_speedup']:.4f} | {row['cached_speedup']:.4f} |"
        )
    (out_dir / "cuda_sdxl_operator_benchmark_summary.md").write_text("\n".join(lines))
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
