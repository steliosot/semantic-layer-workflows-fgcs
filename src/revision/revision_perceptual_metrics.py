#!/usr/bin/env python3
"""Perceptual scoring for the FGCS major-revision experiment.

This script reuses the existing 200 benchmark artifacts. It does not rerun
diffusion. It computes LPIPS distances for naive and semantic outputs, with
outside-mask and target-crop LPIPS variants. CLIP text-image alignment is used
when the model is locally available; otherwise the script writes LPIPS-only
results instead of failing the whole revision analysis.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import lpips
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "outputs/heavy_paper_benchmark/heavy_all_models_20steps_5samples/heavy_paper_benchmark_metrics.json"
OUT_DIR = ROOT / "paper_fgcs_semantic_workflows/revision_outputs"
OUT_JSON = OUT_DIR / "perceptual_metrics.json"
OUT_MD = OUT_DIR / "perceptual_metrics_summary.md"

CASE_PROMPTS = {
    "face_lips": "a portrait with natural red lipstick on the lips",
    "human_body_shirt": "a person wearing a blue shirt",
    "object_mug": "a mug changed to teal color",
    "landscape_sky": "a landscape with a warm sunset sky",
}

OUTPUTS = {
    "naive": "naive_v1.png",
    "semantic": "semantic_v1_projected.png",
    "cached": "semantic_v2_cached_projected.png",
    "projection": "projection_only_v1_projected.png",
}


def pil_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def pil_mask(path: Path, size: tuple[int, int]) -> Image.Image:
    mask = Image.open(path).convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    return mask


def outside_masked(image: Image.Image, mask: Image.Image) -> Image.Image:
    gray = Image.new("RGB", image.size, (127, 127, 127))
    return Image.composite(gray, image, mask)


def bbox_from_mask(mask: Image.Image) -> tuple[int, int, int, int]:
    box = mask.getbbox()
    if box is None:
        return (0, 0, mask.width, mask.height)
    pad = 8
    x0, y0, x1, y1 = box
    return (max(0, x0 - pad), max(0, y0 - pad), min(mask.width, x1 + pad), min(mask.height, y1 + pad))


def lpips_tensor(image: Image.Image) -> torch.Tensor:
    image = image.resize((256, 256), Image.Resampling.BICUBIC)
    arr = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    arr = arr.view(image.height, image.width, 3).permute(2, 0, 1).float() / 127.5 - 1.0
    return arr.unsqueeze(0)


def clip_scores(model: CLIPModel, processor: CLIPProcessor, images: list[Image.Image], prompts: list[str]) -> list[float]:
    with torch.no_grad():
        inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
        image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
        text_features = model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return (image_features * text_features).sum(dim=-1).cpu().tolist()


def try_load_clip() -> tuple[CLIPModel | None, CLIPProcessor | None, str]:
    if os.environ.get("SKIP_CLIP") == "1":
        return None, None, "skipped_by_environment"
    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
        return model, processor, "openai/clip-vit-base-patch32"
    except Exception as exc:  # pragma: no cover - depends on local model cache/network.
        return None, None, f"unavailable: {type(exc).__name__}: {exc}"


def flush_clip_batch(
    rows: list[dict],
    batch_images: list[Image.Image],
    batch_prompts: list[str],
    batch_refs: list[dict],
    clip_model: CLIPModel | None,
    clip_processor: CLIPProcessor | None,
) -> None:
    if not batch_refs:
        return
    if clip_model is not None and clip_processor is not None:
        scores = clip_scores(clip_model, clip_processor, batch_images, batch_prompts)
        for ref, score in zip(batch_refs, scores):
            ref["clip_alignment"] = float(score)
            rows.append(ref)
    else:
        for ref in batch_refs:
            ref["clip_alignment"] = None
            rows.append(ref)
    batch_images.clear()
    batch_prompts.clear()
    batch_refs.clear()


def aggregate(rows: list[dict]) -> dict:
    summary: dict[str, dict[str, float]] = {}
    for variant in OUTPUTS:
        subset = [r for r in rows if r["variant"] == variant]
        if not subset:
            continue
        clip_values = [r["clip_alignment"] for r in subset if r.get("clip_alignment") is not None]
        item = {
            "lpips_full_mean": mean(r["lpips_full"] for r in subset),
            "lpips_outside_mean": mean(r["lpips_outside"] for r in subset),
            "lpips_target_mean": mean(r["lpips_target"] for r in subset),
        }
        if clip_values:
            item["clip_mean"] = mean(clip_values)
            item["clip_median"] = median(clip_values)
        summary[variant] = item
    return summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(METRICS_PATH.read_text())
    records = [r for r in data["records"] if not r.get("error")]

    print("Loading CLIP model...")
    clip_model, clip_processor, clip_status = try_load_clip()
    print("CLIP status:", clip_status)

    print("Loading LPIPS model...")
    lpips_model = lpips.LPIPS(net="alex")
    lpips_model.eval()

    rows: list[dict] = []
    batch_images: list[Image.Image] = []
    batch_prompts: list[str] = []
    batch_refs: list[dict] = []

    for rec in records:
        sample_dir = Path(rec["artifact_dir"])
        base_path = sample_dir / "base.png"
        mask_path = sample_dir / "target_mask_prepared.png"
        if not base_path.exists() or not mask_path.exists():
            continue
        base = pil_rgb(base_path)
        mask = pil_mask(mask_path, base.size)
        box = bbox_from_mask(mask)
        prompt = CASE_PROMPTS[rec["case_key"]]
        base_full = lpips_tensor(base)
        base_out = lpips_tensor(outside_masked(base, mask))
        base_crop = lpips_tensor(base.crop(box))

        for variant, name in OUTPUTS.items():
            out_path = sample_dir / name
            if not out_path.exists():
                continue
            image = pil_rgb(out_path)
            with torch.no_grad():
                full = float(lpips_model(base_full, lpips_tensor(image)).item())
                outside = float(lpips_model(base_out, lpips_tensor(outside_masked(image, mask))).item())
                target = float(lpips_model(base_crop, lpips_tensor(image.crop(box))).item())
            row = {
                "model_key": rec["model_key"],
                "model_name": rec["model_name"],
                "family": rec["family"],
                "case_key": rec["case_key"],
                "case_label": rec["case_label"],
                "sample_index": rec["sample_index"],
                "variant": variant,
                "lpips_full": full,
                "lpips_outside": outside,
                "lpips_target": target,
            }
            batch_images.append(image)
            batch_prompts.append(prompt)
            batch_refs.append(row)

            if len(batch_images) == 32:
                flush_clip_batch(rows, batch_images, batch_prompts, batch_refs, clip_model, clip_processor)
                print("scored", len(rows), "variant images")

    if batch_images:
        flush_clip_batch(rows, batch_images, batch_prompts, batch_refs, clip_model, clip_processor)

    summary = aggregate(rows)
    by_case: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for case in CASE_PROMPTS:
        case_rows = [r for r in rows if r["case_key"] == case]
        by_case[case] = aggregate(case_rows)

    result = {
        "records": len(records),
        "variant_rows": len(rows),
        "clip_model": clip_status,
        "lpips_model": "alex",
        "summary": summary,
        "by_case": by_case,
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))

    lines = [
        "# Perceptual Metrics Summary",
        "",
        f"Records: {len(records)}",
        f"Variant images scored: {len(rows)}",
        f"CLIP status: {clip_status}",
        "",
        "| Variant | LPIPS full | LPIPS outside | LPIPS target | CLIP mean |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for variant, vals in summary.items():
        clip_text = f"{vals['clip_mean']:.4f}" if "clip_mean" in vals else "n/a"
        lines.append(
            f"| {variant} | {vals['lpips_full_mean']:.4f} | "
            f"{vals['lpips_outside_mean']:.4f} | {vals['lpips_target_mean']:.4f} | {clip_text} |"
        )
    lines.extend(["", "## By Case", ""])
    for case, agg in by_case.items():
        lines.append(f"### {case}")
        lines.append("")
        lines.append("| Variant | LPIPS full | LPIPS outside | LPIPS target | CLIP mean |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for variant, vals in agg.items():
            clip_text = f"{vals['clip_mean']:.4f}" if "clip_mean" in vals else "n/a"
            lines.append(
                f"| {variant} | {vals['lpips_full_mean']:.4f} | "
                f"{vals['lpips_outside_mean']:.4f} | {vals['lpips_target_mean']:.4f} | {clip_text} |"
            )
        lines.append("")
    OUT_MD.write_text("\n".join(lines))
    print(OUT_JSON)
    print(OUT_MD)


if __name__ == "__main__":
    main()
