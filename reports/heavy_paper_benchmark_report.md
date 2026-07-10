# Heavy Paper Benchmark: Naive vs Semantic Layered Editing

## Scope

This benchmark runs the compatible local SD1.5/SDXL models on four edit categories with five seed variants per category. Each sample compares full-image naive editing against semantic local editing, cached-mask re-editing/versioning, and deterministic projection-only layer editing.

- Completed samples: `200`
- Errors: `0`
- Skipped incompatible local checkpoints: `4`
- Steps: `20`
- Samples per case: `5`

## Overall Results

| Metric | Mean |
| --- | ---: |
| Naive full edit time | 35.546s |
| Semantic local edit time, detector excluded | 30.132s |
| Detector time | 13.543s |
| Projection-only edit time | 0.031s |
| Generation-only semantic speedup | 1.788x |
| One-off detector-inclusive speedup | 0.809x |
| Cached version re-edit speedup | 7.041x |
| Projection-only speedup | 1227.276x |
| Rollback time | 0.188s |
| Naive outside-target changed pixels | 93.9542% |
| Semantic outside-target changed pixels | 0.0000% |
| Semantic boundary-band changed pixels | 42.3008% |

## By Case

| Case | N | Naive | Semantic local | Detector | Gen speedup | Inclusive speedup | Cached re-edit speedup | Semantic outside change | Boundary change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| face lips | 50 | 35.796s | 28.063s | 13.871s | 2.313x | 0.877x | 11.055x | 0.0000% | 42.0660% |
| human body shirt | 50 | 38.372s | 30.619s | 13.517s | 1.958x | 0.886x | 6.085x | 0.0000% | 40.6253% |
| landscape sky | 50 | 31.858s | 31.358s | 13.636s | 1.211x | 0.634x | 4.966x | 0.0000% | 44.2351% |
| object mug | 50 | 36.160s | 30.488s | 13.151s | 1.669x | 0.838x | 6.058x | 0.0000% | 42.2767% |

## By Model

| Model | N | Naive | Semantic local | Detector | Gen speedup | Inclusive speedup | Cached re-edit speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Deliberate v6 | 20 | 12.213s | 8.180s | 12.954s | 1.601x | 0.580x | 6.744x |
| DreamShaper 8 | 20 | 31.388s | 8.298s | 13.292s | 3.970x | 1.484x | 7.762x |
| DreamShaper XL 1.0 | 20 | 39.873s | 55.469s | 14.677s | 0.723x | 0.570x | 4.699x |
| Juggernaut Reborn | 20 | 11.390s | 4.973s | 12.778s | 2.392x | 0.643x | 6.872x |
| Juggernaut XL v2 | 20 | 118.220s | 75.622s | 15.298s | 1.567x | 1.302x | 10.514x |
| OpenJourney v4 | 20 | 23.941s | 7.496s | 12.384s | 3.537x | 1.246x | 12.229x |
| RealVisXL V4.0 | 20 | 53.656s | 62.119s | 14.488s | 0.865x | 0.700x | 3.172x |
| Realistic Vision V6.0 B1 fp16 | 20 | 10.540s | 9.139s | 12.194s | 1.334x | 0.507x | 6.429x |
| SDXL Base 1.0 local safetensors | 20 | 43.568s | 60.456s | 14.151s | 0.727x | 0.587x | 5.589x |
| Stable Diffusion 1.5 Base | 20 | 10.675s | 9.567s | 13.220s | 1.164x | 0.471x | 6.401x |

## Sample Visuals

![sample grid](juggernaut_reborn/01_face_lips/sample_01/sample_grid.png)

![sample grid](juggernaut_reborn/01_face_lips/sample_02/sample_grid.png)

![sample grid](juggernaut_reborn/01_face_lips/sample_03/sample_grid.png)

![sample grid](juggernaut_reborn/01_face_lips/sample_04/sample_grid.png)

![sample grid](juggernaut_reborn/01_face_lips/sample_05/sample_grid.png)

![sample grid](juggernaut_reborn/02_human_body_shirt/sample_01/sample_grid.png)

![sample grid](juggernaut_reborn/02_human_body_shirt/sample_02/sample_grid.png)

![sample grid](juggernaut_reborn/02_human_body_shirt/sample_03/sample_grid.png)

![sample grid](juggernaut_reborn/02_human_body_shirt/sample_04/sample_grid.png)

![sample grid](juggernaut_reborn/02_human_body_shirt/sample_05/sample_grid.png)

![sample grid](juggernaut_reborn/03_object_mug/sample_01/sample_grid.png)

![sample grid](juggernaut_reborn/03_object_mug/sample_02/sample_grid.png)

![sample grid](juggernaut_reborn/03_object_mug/sample_03/sample_grid.png)

![sample grid](juggernaut_reborn/03_object_mug/sample_04/sample_grid.png)

![sample grid](juggernaut_reborn/03_object_mug/sample_05/sample_grid.png)

![sample grid](juggernaut_reborn/04_landscape_sky/sample_01/sample_grid.png)

![sample grid](juggernaut_reborn/04_landscape_sky/sample_02/sample_grid.png)

![sample grid](juggernaut_reborn/04_landscape_sky/sample_03/sample_grid.png)

![sample grid](juggernaut_reborn/04_landscape_sky/sample_04/sample_grid.png)

![sample grid](juggernaut_reborn/04_landscape_sky/sample_05/sample_grid.png)


## Skipped Local Checkpoints

| Path | Reason |
| --- | --- |
| `/Users/stelios/Documents/ComfyUI/models/checkpoints/epicrealism_v10-inpainting.safetensors` | inpainting checkpoint, not a txt2img/img2img checkpoint for this benchmark |
| `/Users/stelios/Documents/ComfyUI/models/checkpoints/sd3_medium_incl_clips.safetensors` | SD3 checkpoint, incompatible with the SD1.5/SDXL diffusers pipelines used here |
| `/Users/stelios/Documents/ComfyUI/models/checkpoints/ace_step_v1_3.5b.safetensors` | not an SD image-generation checkpoint for this benchmark |
| `/Users/stelios/Documents/ComfyUI/models/checkpoints/tiny_test_download.safetensors` | tiny placeholder/test file |

## Interpretation

The detector-inclusive number is the honest one-off user experience when no layer cache exists. The cached re-edit and rollback numbers test the project's stronger claim: once the Image DOM/layer mask is persisted, trying variants of the same semantic layer should avoid repeated detection and preserve the rest of the image.

Projection-only is the deterministic upper bound for simple color changes. It is fastest and most local, but it may look flatter than diffusion-assisted edits on textured or reflective targets. Semantic local diffusion is slower but can add texture, shading, and realism. Naive editing is globally coherent but tends to change unrelated pixels and loses identity/layout consistency.

These results are still not a replacement for hand-labeled mask correctness. They are realistic timing, preservation, boundary, and versioning metrics over repeated generated samples.