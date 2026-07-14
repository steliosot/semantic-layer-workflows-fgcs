#!/usr/bin/env python3
"""CUDA SDXL operator sweep for semantic workflow scheduling.

This benchmark expands the focused L4 run by varying local crop size and
denoising steps. It measures the scheduler-facing question directly: how much
does local img2img save relative to full-image txt2img on the same GPU tier?
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline
from PIL import Image


BASE_PROMPT = (
    "realistic product photograph of a ceramic coffee mug on a wooden desk, "
    "soft window light, clean background, high detail"
)
EDIT_PROMPT = (
    "same product photograph, glossy teal ceramic mug only, preserve desk, "
    "background, shadows, and lighting"
)
NEGATIVE_PROMPT = "people, hands, logo, text, watermark, blurry, low quality, distorted object"


def sync_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def timed(label: str, fn) -> tuple[Image.Image, float]:
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
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["steps"], row["crop_size"]), []).append(row)

    by_condition = []
    for (steps, crop_size), subset in sorted(grouped.items()):
        full = [r["full_seconds"] for r in subset]
        local = [r["local_seconds"] for r in subset]
        by_condition.append(
            {
                "steps": steps,
                "crop_size": crop_size,
                "crop_area_percent": (crop_size * crop_size) / (768 * 768) * 100.0,
                "repeats": len(subset),
                "mean_full_seconds": statistics.mean(full),
                "mean_local_seconds": statistics.mean(local),
                "mean_speedup": statistics.mean(r["full_seconds"] / r["local_seconds"] for r in subset),
                "median_speedup": statistics.median(r["full_seconds"] / r["local_seconds"] for r in subset),
            }
        )

    return {
        "samples": len(rows),
        "conditions": len(grouped),
        "mean_full_seconds": statistics.mean(r["full_seconds"] for r in rows),
        "mean_local_seconds": statistics.mean(r["local_seconds"] for r in rows),
        "mean_speedup": statistics.mean(r["full_seconds"] / r["local_seconds"] for r in rows),
        "median_speedup": statistics.median(r["full_seconds"] / r["local_seconds"] for r in rows),
        "min_speedup": min(r["full_seconds"] / r["local_seconds"] for r in rows),
        "max_speedup": max(r["full_seconds"] / r["local_seconds"] for r in rows),
        "by_condition": by_condition,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--steps", default="10,20,30")
    parser.add_argument("--crop-sizes", default="256,384,512,640,768")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--strength", type=float, default=0.34)
    parser.add_argument("--cpu-offload", action="store_true", help="Use Accelerate CPU offload for smaller GPUs.")
    parser.add_argument("--vae-tiling", action="store_true", help="Enable VAE tiling to reduce decode memory.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps_values = [int(v) for v in args.steps.split(",") if v.strip()]
    crop_sizes = [int(v) for v in args.crop_sizes.split(",") if v.strip()]

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    txt2img = StableDiffusionXLPipeline.from_single_file(args.checkpoint, torch_dtype=torch.float16, use_safetensors=True)
    if args.cpu_offload:
        txt2img.enable_model_cpu_offload()
    else:
        txt2img.to("cuda")
    if args.vae_tiling:
        txt2img.vae.enable_tiling()
        txt2img.vae.enable_slicing()
    txt2img.set_progress_bar_config(disable=True)
    img2img = StableDiffusionXLImg2ImgPipeline.from_single_file(args.checkpoint, torch_dtype=torch.float16, use_safetensors=True)
    if args.cpu_offload:
        img2img.enable_model_cpu_offload()
    else:
        img2img.to("cuda")
    if args.vae_tiling:
        img2img.vae.enable_tiling()
        img2img.vae.enable_slicing()
    img2img.set_progress_bar_config(disable=True)

    print("warmup", flush=True)
    _ = txt2img(
        prompt=BASE_PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        height=args.height,
        width=args.width,
        num_inference_steps=4,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator(device="cuda").manual_seed(7000),
    ).images[0]
    torch.cuda.synchronize()

    rows: list[dict[str, Any]] = []
    for steps in steps_values:
        for repeat in range(args.repeats):
            seed = 91000 + steps * 100 + repeat
            full, full_seconds = timed(
                f"steps {steps} repeat {repeat + 1} full",
                lambda: txt2img(
                    prompt=EDIT_PROMPT,
                    negative_prompt=NEGATIVE_PROMPT,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=steps,
                    guidance_scale=args.guidance_scale,
                    generator=torch.Generator(device="cuda").manual_seed(seed),
                ),
            )
            if repeat == 0:
                full.save(out_dir / f"full_steps_{steps}.png")

            for crop_size in crop_sizes:
                crop = center_crop(full, crop_size)
                local, local_seconds = timed(
                    f"steps {steps} repeat {repeat + 1} crop {crop_size}",
                    lambda: img2img(
                        prompt=EDIT_PROMPT,
                        negative_prompt=NEGATIVE_PROMPT,
                        image=crop,
                        num_inference_steps=steps,
                        strength=args.strength,
                        guidance_scale=args.guidance_scale,
                        generator=torch.Generator(device="cuda").manual_seed(seed + crop_size),
                    ),
                )
                if repeat == 0:
                    local.save(out_dir / f"local_steps_{steps}_crop_{crop_size}.png")
                row = {
                    "steps": steps,
                    "repeat": repeat + 1,
                    "crop_size": crop_size,
                    "crop_area_percent": (crop_size * crop_size) / (args.height * args.width) * 100.0,
                    "full_seconds": full_seconds,
                    "local_seconds": local_seconds,
                    "speedup": full_seconds / local_seconds,
                }
                rows.append(row)
                (out_dir / "cuda_operator_sweep_partial.json").write_text(json.dumps({"rows": rows}, indent=2))

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "gcp_cuda_operator_sweep",
        "device": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "config": vars(args),
        "summary": summarize(rows),
        "rows": rows,
    }
    (out_dir / "cuda_operator_sweep.json").write_text(json.dumps(result, indent=2))

    lines = ["# GCP CUDA Operator Sweep", "", f"GPU: {result['device']['gpu']}", "", "| Steps | Crop | Area % | Full s | Local s | Speedup |", "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for item in result["summary"]["by_condition"]:
        lines.append(
            f"| {item['steps']} | {item['crop_size']} | {item['crop_area_percent']:.1f} | "
            f"{item['mean_full_seconds']:.3f} | {item['mean_local_seconds']:.3f} | {item['mean_speedup']:.2f}x |"
        )
    (out_dir / "cuda_operator_sweep_summary.md").write_text("\n".join(lines))
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
