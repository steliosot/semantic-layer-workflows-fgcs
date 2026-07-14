# GCP CUDA SDXL Operator Benchmark

GPU: NVIDIA L4
Torch: 2.5.1+cu121 / CUDA 12.1
Steps: 20
Samples: 20

| Metric | Value |
| --- | ---: |
| samples | 20 |
| mean_naive_seconds | 4.7411 |
| median_naive_seconds | 4.7410 |
| mean_semantic_local_seconds | 0.9263 |
| median_semantic_local_seconds | 0.9572 |
| mean_cached_local_seconds | 0.9192 |
| median_cached_local_seconds | 0.9554 |
| mean_generation_only_speedup | 5.2272 |
| median_generation_only_speedup | 4.9520 |
| mean_cached_speedup | 5.2772 |
| median_cached_speedup | 4.9594 |

| Case | Sample | Crop | Naive | Local | Cached | Gen speedup | Cached speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| face_lips | 1 | 384 | 4.6738 | 0.7742 | 0.7045 | 6.0367 | 6.6345 |
| face_lips | 2 | 384 | 4.7349 | 0.7095 | 0.7084 | 6.6733 | 6.6842 |
| face_lips | 3 | 384 | 4.7790 | 0.7107 | 0.7033 | 6.7241 | 6.7954 |
| face_lips | 4 | 384 | 4.7711 | 0.7078 | 0.7071 | 6.7409 | 6.7478 |
| face_lips | 5 | 384 | 4.7452 | 0.7103 | 0.7079 | 6.6808 | 6.7030 |
| human_body_shirt | 1 | 512 | 4.7600 | 1.0114 | 0.9550 | 4.7063 | 4.9841 |
| human_body_shirt | 2 | 512 | 4.7286 | 0.9543 | 0.9583 | 4.9550 | 4.9343 |
| human_body_shirt | 3 | 512 | 4.7344 | 0.9580 | 0.9549 | 4.9420 | 4.9581 |
| human_body_shirt | 4 | 512 | 4.7348 | 0.9575 | 0.9540 | 4.9451 | 4.9630 |
| human_body_shirt | 5 | 512 | 4.7458 | 0.9582 | 0.9568 | 4.9527 | 4.9602 |
| object_mug | 1 | 512 | 4.7575 | 1.0589 | 1.0598 | 4.4927 | 4.4891 |
| object_mug | 2 | 512 | 4.7414 | 1.0629 | 1.0562 | 4.4609 | 4.4889 |
| object_mug | 3 | 512 | 4.7490 | 1.0578 | 1.0603 | 4.4895 | 4.4789 |
| object_mug | 4 | 512 | 4.7400 | 1.0578 | 1.0556 | 4.4811 | 4.4904 |
| object_mug | 5 | 512 | 4.7335 | 1.0554 | 1.0581 | 4.4849 | 4.4735 |
| landscape_sky | 1 | 512 | 4.7414 | 0.9537 | 0.9534 | 4.9718 | 4.9732 |
| landscape_sky | 2 | 512 | 4.7406 | 0.9556 | 0.9553 | 4.9609 | 4.9624 |
| landscape_sky | 3 | 512 | 4.7424 | 0.9555 | 0.9640 | 4.9631 | 4.9195 |
| landscape_sky | 4 | 512 | 4.7302 | 0.9592 | 0.9567 | 4.9313 | 4.9445 |
| landscape_sky | 5 | 512 | 4.7378 | 0.9569 | 0.9555 | 4.9513 | 4.9587 |