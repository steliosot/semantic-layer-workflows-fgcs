# Semantic Layer Workflows for Adaptive Generative AI Editing

This repository contains the reproducibility artifact for the FGCS manuscript:

**Semantic Layer Workflows for Adaptive Generative AI Editing in the Compute Continuum**

Author: Stelios Sotiriadis

## Repository contents

- `paper/`: LaTeX source, bibliography, figures, and highlights.
- `src/`: benchmark and aggregation scripts used for the semantic layer workflow experiments.
- `src/revision/`: revision-analysis scripts for policy ablation, dependency-risk auditing, LPIPS perceptual scoring, top-five cross-model summaries, GCP CUDA detector-inclusive benchmarking, L4/T4 operator sweeps, long L4 session benchmarking, and bucket model-runtime auditing.
- `reports/`: benchmark reports, model-by-case CSV matrix, and revision-analysis outputs.
- `CITATION.cff`: citation metadata for the artifact.
- `CHECKSUMS.txt`: SHA-256 checksums for the release ZIP assets.
- GitHub release asset: compiled manuscript PDFs, full metrics JSON, and the 200 sample-grid images from the 10-model, 4-category, 5-seed benchmark.

## Benchmark scope

The reported experiment covers 10 diffusion checkpoints, four edit categories, and five seed variants, giving 200 paired workflow executions at 20 denoising steps. The sample grids in the release archive show the base image, detector mask or overlay, naive edits, semantic local outputs, deterministic or cached variants where applicable, and rollback/version artifacts.

The revision analyses reuse the stored benchmark metrics and image outputs. They add a scheduling-policy ablation, a heterogeneous break-even stress test, a geometry-only dependency-contact audit, LPIPS full-image/outside-mask/target-crop scoring over 800 local output images, a top-five cross-model summary from the 200-sample heavy benchmark, a 100-sample GCP NVIDIA L4 detector-inclusive SDXL workflow benchmark, a 45-sample L4 crop/step operator sweep, a compact NVIDIA T4 offload tier, a long L4 session benchmark with 303 variants across five model families, a bucket model-runtime audit, and cloud LPIPS scoring over the L4 outputs.

## External assets

Large diffusion checkpoints and external detector weights are not redistributed in this repository. They are subject to their own licenses and should be obtained from their original sources. The manuscript bibliography and scripts document the local checkpoint filenames, Hugging Face model-card identifiers, and detector identifiers used in the benchmark.

## Verification

Use `CHECKSUMS.txt` to verify downloaded release ZIP files:

```bash
shasum -a 256 -c CHECKSUMS.txt
```

## Reproducing the workflow

Install the Python dependencies listed in `src/requirements.txt` and `src/requirements_vision.txt`, then inspect `src/prototype_heavy_paper_benchmark.py` for paths and model configuration. The primary benchmark was run locally on an Apple MPS environment, with additional measured GCP NVIDIA L4 and T4 tiers under `src/revision/`; timings may differ on other CUDA, cloud GPU, edge, or HPC deployments.
