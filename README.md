# Semantic Layer Workflows for Adaptive Generative AI Editing

This repository contains the reproducibility artifact for the FGCS manuscript:

**Semantic Layer Workflows for Adaptive Generative AI Editing in the Compute Continuum**

Author: Stelios Sotiriadis

## Repository contents

- `paper/`: LaTeX source, bibliography, figures, and highlights.
- `src/`: benchmark and aggregation scripts used for the semantic layer workflow experiments.
- `src/revision/`: revision-analysis scripts for policy ablation, dependency-risk auditing, and LPIPS perceptual scoring.
- `reports/`: benchmark reports, model-by-case CSV matrix, and revision-analysis outputs.
- `CITATION.cff`: citation metadata for the artifact.
- `CHECKSUMS.txt`: SHA-256 checksums for the release ZIP assets.
- GitHub release asset: compiled manuscript PDFs, full metrics JSON, and the 200 sample-grid images from the 10-model, 4-category, 5-seed benchmark.

## Benchmark scope

The reported experiment covers 10 diffusion checkpoints, four edit categories, and five seed variants, giving 200 paired workflow executions at 20 denoising steps. The sample grids in the release archive show the base image, detector mask or overlay, naive edits, semantic local outputs, deterministic or cached variants where applicable, and rollback/version artifacts.

The revision analyses reuse the stored benchmark metrics and image outputs. They add a scheduling-policy ablation, a heterogeneous break-even stress test, a geometry-only dependency-contact audit, and LPIPS full-image/outside-mask/target-crop scoring over 800 output images.

## External assets

Large diffusion checkpoints and external detector weights are not redistributed in this repository. They are subject to their own licenses and should be obtained from their original sources. The scripts and reports document the model identifiers used in the benchmark.

## Verification

Use `CHECKSUMS.txt` to verify downloaded release ZIP files:

```bash
shasum -a 256 -c CHECKSUMS.txt
```

## Reproducing the workflow

Install the Python dependencies listed in `src/requirements.txt` and `src/requirements_vision.txt`, then inspect `src/prototype_heavy_paper_benchmark.py` for paths and model configuration. The benchmark was run locally on an Apple MPS environment; timings may differ on CUDA, cloud GPU, edge, or HPC deployments.
