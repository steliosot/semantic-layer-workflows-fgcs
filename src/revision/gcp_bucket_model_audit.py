#!/usr/bin/env python3
"""Audit bucket model files for benchmark compatibility.

The paper benchmark uses an SDXL Diffusers path. The project bucket also holds
Flux, Wan, Qwen, GGUF, video, and ComfyUI-specific weights. This audit records
what can be run in the SDXL benchmark without changing the execution backend,
and what should be treated as a separate future runtime tier.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def classify(path: str) -> dict[str, Any]:
    lower = path.lower()
    if "juggernaut" in lower and lower.endswith(".safetensors"):
        return {
            "family": "SDXL checkpoint",
            "paper_runtime_status": "measured",
            "reason": "Compatible with the single-file SDXL Diffusers pipeline used by the CUDA benchmarks.",
        }
    if lower.endswith(".gguf"):
        return {
            "family": "GGUF diffusion/LLM-style weight",
            "paper_runtime_status": "not_comparable",
            "reason": "Requires a GGUF-capable ComfyUI or custom runtime rather than the SDXL Diffusers pipeline.",
        }
    if "flux" in lower:
        return {
            "family": "Flux checkpoint",
            "paper_runtime_status": "separate_runtime",
            "reason": "Requires a Flux pipeline/runtime and different scheduler assumptions.",
        }
    if "wan" in lower:
        return {
            "family": "Wan video/image-to-video checkpoint",
            "paper_runtime_status": "separate_runtime",
            "reason": "Targets a video or Wan-specific pipeline, not paired SDXL image editing.",
        }
    if "qwen" in lower:
        return {
            "family": "Qwen image edit checkpoint",
            "paper_runtime_status": "separate_runtime",
            "reason": "Uses a different image-edit model family and runtime interface.",
        }
    if "z-image" in lower or "z_image" in lower:
        return {
            "family": "Z-image checkpoint",
            "paper_runtime_status": "separate_runtime",
            "reason": "Not directly loadable by the SDXL single-file benchmark path.",
        }
    return {
        "family": "unknown",
        "paper_runtime_status": "requires_manual_triage",
        "reason": "Model family could not be inferred from the object name.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default="gs://bbk-q2-comfyui-models")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(["gsutil", "ls", "-l", "-r", args.bucket], check=True, text=True, capture_output=True)
    rows = []
    for line in proc.stdout.splitlines():
        fields = line.split()
        if len(fields) < 2 or not fields[-1].startswith("gs://"):
            continue
        path = fields[-1]
        if path.endswith("/"):
            continue
        size = int(fields[0]) if fields[0].isdigit() else None
        info = classify(path)
        rows.append({"path": path, "size_bytes": size, **info})

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["paper_runtime_status"]] = counts.get(row["paper_runtime_status"], 0) + 1
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bucket": args.bucket,
        "summary": {"objects": len(rows), "status_counts": counts},
        "rows": rows,
    }
    (out_dir / "bucket_model_runtime_audit.json").write_text(json.dumps(result, indent=2))

    lines = ["# Bucket Model Runtime Audit", "", f"Bucket: `{args.bucket}`", "", "| Status | Count |", "| --- | ---: |"]
    for status, count in sorted(counts.items()):
        lines.append(f"| {status} | {count} |")
    lines.extend(["", "| Model object | Family | Paper-runtime status |", "| --- | --- | --- |"])
    for row in rows:
        lines.append(f"| `{row['path'].replace(args.bucket + '/', '')}` | {row['family']} | {row['paper_runtime_status']} |")
    (out_dir / "bucket_model_runtime_audit.md").write_text("\n".join(lines))
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
