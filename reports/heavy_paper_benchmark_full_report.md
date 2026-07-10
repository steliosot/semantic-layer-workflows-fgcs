# Heavy Paper Benchmark Full Report

## Executive Result

This run completed **200 paired samples**: 10 compatible SD/SDXL models x 4 edit cases x 5 seed variants, at **20 steps**. There were **0 per-sample errors**. Four incompatible local checkpoints were skipped.

| Claim | Result | Interpretation |
| --- | ---: | --- |
| Generation-only semantic speedup | 1.788x | Local rendering is faster than full naive generation when detector/cache overhead is excluded. |
| One-off detector-inclusive speedup | 0.809x | First edit is slower on average because GroundingDINO+SAM2 costs about 13.543s. |
| Cached semantic re-edit speedup | 7.041x | Reusing the semantic layer/mask is the strongest realistic speed claim. |
| Projection-only deterministic speedup | 1227.276x | Simple color edits are effectively instantaneous if realism demands are low. |
| Rollback time | 0.188s | Versioning/rollback is near-interactive because it restores cached layer state. |
| Naive outside-target drift | 93.9542% | Full-image regeneration changes most non-target pixels. |
| Semantic outside-target drift | 0.0000% | The compositing path preserved non-target pixels exactly under the pixel-change metric. |

## Case Averages

| Case | N | Naive | Semantic local | Detector | Gen speedup | Inclusive speedup | Cached re-edit | Naive outside | Semantic outside |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| face lips | 50 | 35.796s | 28.063s | 13.871s | 2.313x | 0.877x | 11.055x | 90.3554% | 0.0000% |
| human body shirt | 50 | 38.372s | 30.619s | 13.517s | 1.958x | 0.886x | 6.085x | 91.9078% | 0.0000% |
| landscape sky | 50 | 31.858s | 31.358s | 13.636s | 1.211x | 0.634x | 4.966x | 98.7448% | 0.0000% |
| object mug | 50 | 36.160s | 30.488s | 13.151s | 1.669x | 0.838x | 6.058x | 94.8088% | 0.0000% |

## Model Averages

| Model | N | Naive | Semantic local | Detector | Gen speedup | Inclusive speedup | Cached re-edit | Naive outside | Semantic outside |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Deliberate v6 | 20 | 12.213s | 8.180s | 12.954s | 1.601x | 0.580x | 6.744x | 93.4625% | 0.0000% |
| DreamShaper 8 | 20 | 31.388s | 8.298s | 13.292s | 3.970x | 1.484x | 7.762x | 95.5151% | 0.0000% |
| DreamShaper XL 1.0 | 20 | 39.873s | 55.469s | 14.677s | 0.723x | 0.570x | 4.699x | 92.0015% | 0.0000% |
| Juggernaut Reborn | 20 | 11.390s | 4.973s | 12.778s | 2.392x | 0.643x | 6.872x | 93.9576% | 0.0000% |
| Juggernaut XL v2 | 20 | 118.220s | 75.622s | 15.298s | 1.567x | 1.302x | 10.514x | 91.8132% | 0.0000% |
| OpenJourney v4 | 20 | 23.941s | 7.496s | 12.384s | 3.537x | 1.246x | 12.229x | 97.5011% | 0.0000% |
| RealVisXL V4.0 | 20 | 53.656s | 62.119s | 14.488s | 0.865x | 0.700x | 3.172x | 91.3233% | 0.0000% |
| Realistic Vision V6.0 B1 fp16 | 20 | 10.540s | 9.139s | 12.194s | 1.334x | 0.507x | 6.429x | 94.8154% | 0.0000% |
| SDXL Base 1.0 local safetensors | 20 | 43.568s | 60.456s | 14.151s | 0.727x | 0.587x | 5.589x | 91.1468% | 0.0000% |
| Stable Diffusion 1.5 Base | 20 | 10.675s | 9.567s | 13.220s | 1.164x | 0.471x | 6.401x | 98.0053% | 0.0000% |

## Model x Case Matrix

| Model | Case | N | Gen speedup | Inclusive speedup | Cached re-edit | Semantic local | Detector | Mask area | Crop pixels | Boundary change |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Deliberate v6 | face lips | 5 | 1.983x | 0.598x | 10.814x | 5.505s | 12.330s | 0.7070% | 25.0000% | 36.9513% |
| Deliberate v6 | human body shirt | 5 | 1.736x | 0.599x | 6.913x | 6.794s | 12.118s | 5.8872% | 41.0010% | 42.4182% |
| Deliberate v6 | landscape sky | 5 | 1.437x | 0.604x | 4.303x | 11.298s | 14.952s | 4.6744% | 48.5205% | 41.7164% |
| Deliberate v6 | object mug | 5 | 1.247x | 0.517x | 4.945x | 9.122s | 12.416s | 6.7548% | 46.8213% | 40.6806% |
| DreamShaper 8 | face lips | 5 | 5.833x | 1.799x | 14.314x | 6.109s | 14.087s | 0.7655% | 25.0000% | 43.7743% |
| DreamShaper 8 | human body shirt | 5 | 4.381x | 1.793x | 5.487x | 9.035s | 12.738s | 6.9003% | 54.4238% | 43.3555% |
| DreamShaper 8 | landscape sky | 5 | 1.762x | 0.608x | 5.454x | 8.027s | 14.105s | 4.1303% | 47.3291% | 41.7238% |
| DreamShaper 8 | object mug | 5 | 3.904x | 1.734x | 5.793x | 10.021s | 12.237s | 6.9640% | 52.0508% | 41.2055% |
| DreamShaper XL 1.0 | face lips | 5 | 0.813x | 0.630x | 7.138x | 50.854s | 14.883s | 0.6760% | 25.0000% | 43.0873% |
| DreamShaper XL 1.0 | human body shirt | 5 | 0.765x | 0.598x | 2.953x | 55.118s | 15.364s | 6.0337% | 44.4444% | 42.5256% |
| DreamShaper XL 1.0 | landscape sky | 5 | 0.635x | 0.511x | 3.982x | 58.093s | 14.214s | 11.0301% | 43.1098% | 41.5692% |
| DreamShaper XL 1.0 | object mug | 5 | 0.677x | 0.543x | 4.722x | 57.810s | 14.246s | 10.2317% | 42.3611% | 43.7679% |
| Juggernaut Reborn | face lips | 5 | 3.298x | 0.662x | 13.084x | 3.465s | 13.849s | 0.8794% | 25.0000% | 44.3207% |
| Juggernaut Reborn | human body shirt | 5 | 2.156x | 0.646x | 5.234x | 5.381s | 12.602s | 6.9233% | 53.1104% | 43.2842% |
| Juggernaut Reborn | landscape sky | 5 | 1.952x | 0.628x | 3.893x | 5.787s | 12.208s | 9.8304% | 55.3320% | 45.9681% |
| Juggernaut Reborn | object mug | 5 | 2.161x | 0.636x | 5.279x | 5.257s | 12.453s | 9.7805% | 50.0000% | 42.4948% |
| Juggernaut XL v2 | face lips | 5 | 1.376x | 1.165x | 2.917x | 85.087s | 15.355s | 0.8146% | 25.0000% | 42.8572% |
| Juggernaut XL v2 | human body shirt | 5 | 1.795x | 1.488x | 10.520x | 79.936s | 16.868s | 6.4345% | 44.4444% | 42.6949% |
| Juggernaut XL v2 | landscape sky | 5 | 1.439x | 1.184x | 12.794x | 66.315s | 14.220s | 7.0084% | 44.4444% | 46.9574% |
| Juggernaut XL v2 | object mug | 5 | 1.658x | 1.373x | 15.825x | 71.151s | 14.748s | 15.1503% | 42.3611% | 44.2363% |
| OpenJourney v4 | face lips | 5 | 4.354x | 1.423x | 28.999x | 6.373s | 13.668s | 1.0284% | 25.6445% | 40.4256% |
| OpenJourney v4 | human body shirt | 5 | 5.165x | 1.669x | 11.003x | 6.223s | 11.464s | 4.5251% | 45.3369% | 32.8530% |
| OpenJourney v4 | landscape sky | 5 | 1.223x | 0.506x | 4.088x | 8.810s | 12.330s | 5.0783% | 56.2500% | 47.1467% |
| OpenJourney v4 | object mug | 5 | 3.405x | 1.385x | 4.827x | 8.578s | 12.073s | 10.4461% | 51.3281% | 40.6277% |
| RealVisXL V4.0 | face lips | 5 | 0.873x | 0.692x | 4.043x | 55.826s | 14.258s | 0.8353% | 25.0000% | 42.9296% |
| RealVisXL V4.0 | human body shirt | 5 | 0.726x | 0.590x | 3.487x | 62.142s | 14.448s | 6.2134% | 44.4444% | 41.8574% |
| RealVisXL V4.0 | landscape sky | 5 | 0.980x | 0.805x | 2.009x | 68.872s | 14.822s | 14.7839% | 43.3680% | 43.5955% |
| RealVisXL V4.0 | object mug | 5 | 0.880x | 0.711x | 3.149x | 61.637s | 14.422s | 13.6000% | 42.6063% | 43.7795% |
| Realistic Vision V6.0 B1 fp16 | face lips | 5 | 2.260x | 0.634x | 11.652x | 4.877s | 11.917s | 0.6416% | 25.0000% | 43.3030% |
| Realistic Vision V6.0 B1 fp16 | human body shirt | 5 | 1.089x | 0.475x | 4.511x | 10.267s | 12.217s | 7.3271% | 56.2500% | 42.6676% |
| Realistic Vision V6.0 B1 fp16 | landscape sky | 5 | 0.949x | 0.446x | 3.924x | 11.148s | 12.509s | 5.2620% | 53.0908% | 46.5107% |
| Realistic Vision V6.0 B1 fp16 | object mug | 5 | 1.038x | 0.472x | 5.627x | 10.263s | 12.131s | 3.5452% | 44.0039% | 42.6783% |
| SDXL Base 1.0 local safetensors | face lips | 5 | 0.826x | 0.653x | 6.643x | 55.276s | 14.473s | 0.5772% | 25.0000% | 42.5688% |
| SDXL Base 1.0 local safetensors | human body shirt | 5 | 0.681x | 0.554x | 6.090x | 61.203s | 13.877s | 5.4537% | 42.8168% | 42.3994% |
| SDXL Base 1.0 local safetensors | landscape sky | 5 | 0.731x | 0.597x | 4.437x | 64.477s | 14.240s | 10.9155% | 44.4444% | 40.4068% |
| SDXL Base 1.0 local safetensors | object mug | 5 | 0.669x | 0.543x | 5.186x | 60.869s | 14.016s | 9.3593% | 42.1202% | 43.6493% |
| Stable Diffusion 1.5 Base | face lips | 5 | 1.515x | 0.518x | 10.949x | 7.255s | 13.889s | 1.0016% | 29.4531% | 40.4423% |
| Stable Diffusion 1.5 Base | human body shirt | 5 | 1.090x | 0.453x | 4.654x | 10.088s | 13.471s | 4.5473% | 56.2500% | 32.1975% |
| Stable Diffusion 1.5 Base | landscape sky | 5 | 0.999x | 0.451x | 4.773x | 10.756s | 12.755s | 5.2265% | 48.6377% | 46.7562% |
| Stable Diffusion 1.5 Base | object mug | 5 | 1.050x | 0.463x | 5.228x | 10.171s | 12.764s | 9.3178% | 50.0000% | 39.6467% |

## Representative Assets

Each image below is the sample grid for sample 1 of a model/case pair. The full folder contains all base images, masks, naive edits, semantic edits, cached re-edits, rollback outputs, and grids for all 200 samples.

### Juggernaut Reborn

**face lips**

![Juggernaut Reborn face lips](juggernaut_reborn/01_face_lips/sample_01/sample_grid.png)

**human body shirt**

![Juggernaut Reborn human body shirt](juggernaut_reborn/02_human_body_shirt/sample_01/sample_grid.png)

**object mug**

![Juggernaut Reborn object mug](juggernaut_reborn/03_object_mug/sample_01/sample_grid.png)

**landscape sky**

![Juggernaut Reborn landscape sky](juggernaut_reborn/04_landscape_sky/sample_01/sample_grid.png)

### Stable Diffusion 1.5 Base

**face lips**

![Stable Diffusion 1.5 Base face lips](sd15_base/01_face_lips/sample_01/sample_grid.png)

**human body shirt**

![Stable Diffusion 1.5 Base human body shirt](sd15_base/02_human_body_shirt/sample_01/sample_grid.png)

**object mug**

![Stable Diffusion 1.5 Base object mug](sd15_base/03_object_mug/sample_01/sample_grid.png)

**landscape sky**

![Stable Diffusion 1.5 Base landscape sky](sd15_base/04_landscape_sky/sample_01/sample_grid.png)

### Realistic Vision V6.0 B1 fp16

**face lips**

![Realistic Vision V6.0 B1 fp16 face lips](realistic_vision_v6_fp16/01_face_lips/sample_01/sample_grid.png)

**human body shirt**

![Realistic Vision V6.0 B1 fp16 human body shirt](realistic_vision_v6_fp16/02_human_body_shirt/sample_01/sample_grid.png)

**object mug**

![Realistic Vision V6.0 B1 fp16 object mug](realistic_vision_v6_fp16/03_object_mug/sample_01/sample_grid.png)

**landscape sky**

![Realistic Vision V6.0 B1 fp16 landscape sky](realistic_vision_v6_fp16/04_landscape_sky/sample_01/sample_grid.png)

### Deliberate v6

**face lips**

![Deliberate v6 face lips](deliberate_v6/01_face_lips/sample_01/sample_grid.png)

**human body shirt**

![Deliberate v6 human body shirt](deliberate_v6/02_human_body_shirt/sample_01/sample_grid.png)

**object mug**

![Deliberate v6 object mug](deliberate_v6/03_object_mug/sample_01/sample_grid.png)

**landscape sky**

![Deliberate v6 landscape sky](deliberate_v6/04_landscape_sky/sample_01/sample_grid.png)

### DreamShaper 8

**face lips**

![DreamShaper 8 face lips](dreamshaper_8/01_face_lips/sample_01/sample_grid.png)

**human body shirt**

![DreamShaper 8 human body shirt](dreamshaper_8/02_human_body_shirt/sample_01/sample_grid.png)

**object mug**

![DreamShaper 8 object mug](dreamshaper_8/03_object_mug/sample_01/sample_grid.png)

**landscape sky**

![DreamShaper 8 landscape sky](dreamshaper_8/04_landscape_sky/sample_01/sample_grid.png)

### OpenJourney v4

**face lips**

![OpenJourney v4 face lips](openjourney_v4/01_face_lips/sample_01/sample_grid.png)

**human body shirt**

![OpenJourney v4 human body shirt](openjourney_v4/02_human_body_shirt/sample_01/sample_grid.png)

**object mug**

![OpenJourney v4 object mug](openjourney_v4/03_object_mug/sample_01/sample_grid.png)

**landscape sky**

![OpenJourney v4 landscape sky](openjourney_v4/04_landscape_sky/sample_01/sample_grid.png)

### Juggernaut XL v2

**face lips**

![Juggernaut XL v2 face lips](juggernaut_xl_v2/01_face_lips/sample_01/sample_grid.png)

**human body shirt**

![Juggernaut XL v2 human body shirt](juggernaut_xl_v2/02_human_body_shirt/sample_01/sample_grid.png)

**object mug**

![Juggernaut XL v2 object mug](juggernaut_xl_v2/03_object_mug/sample_01/sample_grid.png)

**landscape sky**

![Juggernaut XL v2 landscape sky](juggernaut_xl_v2/04_landscape_sky/sample_01/sample_grid.png)

### SDXL Base 1.0 local safetensors

**face lips**

![SDXL Base 1.0 local safetensors face lips](sdxl_base_1_0_local/01_face_lips/sample_01/sample_grid.png)

**human body shirt**

![SDXL Base 1.0 local safetensors human body shirt](sdxl_base_1_0_local/02_human_body_shirt/sample_01/sample_grid.png)

**object mug**

![SDXL Base 1.0 local safetensors object mug](sdxl_base_1_0_local/03_object_mug/sample_01/sample_grid.png)

**landscape sky**

![SDXL Base 1.0 local safetensors landscape sky](sdxl_base_1_0_local/04_landscape_sky/sample_01/sample_grid.png)

### RealVisXL V4.0

**face lips**

![RealVisXL V4.0 face lips](realvisxl_v4/01_face_lips/sample_01/sample_grid.png)

**human body shirt**

![RealVisXL V4.0 human body shirt](realvisxl_v4/02_human_body_shirt/sample_01/sample_grid.png)

**object mug**

![RealVisXL V4.0 object mug](realvisxl_v4/03_object_mug/sample_01/sample_grid.png)

**landscape sky**

![RealVisXL V4.0 landscape sky](realvisxl_v4/04_landscape_sky/sample_01/sample_grid.png)

### DreamShaper XL 1.0

**face lips**

![DreamShaper XL 1.0 face lips](dreamshaper_xl_1_0/01_face_lips/sample_01/sample_grid.png)

**human body shirt**

![DreamShaper XL 1.0 human body shirt](dreamshaper_xl_1_0/02_human_body_shirt/sample_01/sample_grid.png)

**object mug**

![DreamShaper XL 1.0 object mug](dreamshaper_xl_1_0/03_object_mug/sample_01/sample_grid.png)

**landscape sky**

![DreamShaper XL 1.0 landscape sky](dreamshaper_xl_1_0/04_landscape_sky/sample_01/sample_grid.png)

## Method Notes

- Naive baseline is full-image regeneration/editing and is measured as the user-visible full render time for the first edit and a second variant edit.
- Semantic local editing uses detected target masks, local crop rendering, and compositing back through the semantic mask.
- Cached re-edit excludes repeated detector work and tests the Image DOM/versioning idea: once a layer exists, new variants can reuse it.
- Projection-only is deterministic color/layer math. It is a speed upper bound for color-only edits, not a realism upper bound.
- Outside-target changed pixels are exact pixel-change metrics against the base outside the mask. They do not replace human quality ratings or hand-labeled mask IoU.
- The run used realistic local MPS execution. SDXL memory fragmentation required restart/resume between some models; persisted records were reused rather than recomputed.

## Outputs

- Metrics JSON: `/Users/stelios/Desktop/semantic-sd/outputs/heavy_paper_benchmark/heavy_all_models_20steps_5samples/heavy_paper_benchmark_metrics.json`
- Model-case CSV: `/Users/stelios/Desktop/semantic-sd/outputs/heavy_paper_benchmark/heavy_all_models_20steps_5samples/heavy_paper_benchmark_model_case_matrix.csv`
- Main generated report: `/Users/stelios/Desktop/semantic-sd/outputs/heavy_paper_benchmark/heavy_all_models_20steps_5samples/heavy_paper_benchmark_report.md`
