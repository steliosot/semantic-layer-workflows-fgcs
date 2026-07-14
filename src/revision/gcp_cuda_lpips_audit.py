#!/usr/bin/env python3
"""LPIPS audit for cloud detector-inclusive outputs."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lpips
import numpy as np
import torch
from PIL import Image


def image_tensor(path: Path, mask: Image.Image | None = None) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 127.5 - 1.0
    if mask is not None:
        m = np.asarray(mask.convert("L")).astype(np.float32) / 255.0
        arr = arr * m[..., None]
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def mask_crop(mask: Image.Image, pad: int = 16) -> tuple[int, int, int, int]:
    arr = np.asarray(mask.convert("L")) > 8
    ys, xs = np.where(arr)
    if xs.size == 0:
        return (0, 0, mask.width, mask.height)
    return (
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(mask.width, int(xs.max()) + pad + 1),
        min(mask.height, int(ys.max()) + pad + 1),
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"samples": len(rows)}
    variants = sorted({r["variant"] for r in rows})
    for variant in variants:
        subset = [r for r in rows if r["variant"] == variant]
        for key in ["lpips_full", "lpips_outside", "lpips_target_crop"]:
            values = [float(r[key]) for r in subset]
            summary[f"{variant}_{key}_mean"] = statistics.mean(values)
            summary[f"{variant}_{key}_median"] = statistics.median(values)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    metric = lpips.LPIPS(net="alex").to(device).eval()

    rows: list[dict[str, Any]] = []
    for sample_dir in sorted(input_dir.glob("*/*")):
        if not sample_dir.is_dir():
            continue
        base_path = sample_dir / "base.png"
        mask_path = sample_dir / "target_mask.png"
        if not base_path.exists() or not mask_path.exists():
            continue
        mask = Image.open(mask_path).convert("L")
        outside = Image.fromarray((255 - np.asarray(mask)).astype(np.uint8), mode="L")
        crop_box = mask_crop(mask)

        for variant in ["naive", "semantic", "cached"]:
            variant_path = sample_dir / f"{variant}.png"
            if not variant_path.exists():
                continue
            with torch.no_grad():
                full = metric(image_tensor(base_path).to(device), image_tensor(variant_path).to(device)).item()
                outside_score = metric(
                    image_tensor(base_path, outside).to(device),
                    image_tensor(variant_path, outside).to(device),
                ).item()
                base_crop = Image.open(base_path).convert("RGB").crop(crop_box)
                variant_crop = Image.open(variant_path).convert("RGB").crop(crop_box)
                tmp_base = out_dir / "_tmp_base_crop.png"
                tmp_variant = out_dir / "_tmp_variant_crop.png"
                base_crop.save(tmp_base)
                variant_crop.save(tmp_variant)
                target = metric(image_tensor(tmp_base).to(device), image_tensor(tmp_variant).to(device)).item()
            rows.append(
                {
                    "sample_dir": str(sample_dir.relative_to(input_dir)),
                    "variant": variant,
                    "lpips_full": full,
                    "lpips_outside": outside_score,
                    "lpips_target_crop": target,
                }
            )

    for tmp in [out_dir / "_tmp_base_crop.png", out_dir / "_tmp_variant_crop.png"]:
        if tmp.exists():
            tmp.unlink()

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "gcp_cuda_lpips_audit",
        "device": str(device),
        "summary": summarize(rows),
        "rows": rows,
    }
    (out_dir / "cuda_lpips_audit.json").write_text(json.dumps(result, indent=2))
    lines = ["# GCP CUDA LPIPS Audit", "", "| Metric | Value |", "| --- | ---: |"]
    for key, value in result["summary"].items():
        lines.append(f"| {key} | {value:.6f} |" if isinstance(value, float) else f"| {key} | {value} |")
    (out_dir / "cuda_lpips_audit.md").write_text("\n".join(lines))
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
