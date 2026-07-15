# GCP Heavy Session Combined Results

| Model | Case | Variants | Naive session s | Semantic incl. detector s | Cached s | Incl. speedup | Cached speedup | Outside drift naive / semantic |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DreamShaper 8 | product mug recolor | 58 | 305.6 | 54.4 | 53.9 | 5.62x | 5.67x | 95.81% / 0.0000% |
| DreamShaper 8 | shirt recolor | 58 | 305.8 | 50.7 | 50.2 | 6.03x | 6.09x | 92.73% / 0.0000% |
| DreamShaper XL 1.0 | product mug recolor | 21 | 300.5 | 54.1 | 53.6 | 5.55x | 5.61x | 95.59% / 0.0000% |
| Juggernaut X RunDiffusion bucket SDXL | product mug recolor | 45 | 641.4 | 131.0 | 130.5 | 4.90x | 4.91x | 97.97% / 0.0000% |
| Juggernaut X RunDiffusion bucket SDXL | shirt recolor | 44 | 628.9 | 120.4 | 119.9 | 5.22x | 5.25x | 98.78% / 0.0000% |
| OpenJourney v4 | product mug recolor | 56 | 304.3 | 56.5 | 56.0 | 5.38x | 5.43x | 94.52% / 0.0000% |
| Stable Diffusion XL Base 1.0 | product mug recolor | 21 | 300.2 | 61.7 | 61.0 | 4.87x | 4.92x | 93.41% / 0.0000% |

## Summary

```json
{
  "completed_sessions": 7,
  "models": [
    "dreamshaper_8",
    "dreamshaper_xl_1_0",
    "juggernaut_x_bucket",
    "openjourney_v4",
    "sdxl_base_1_0"
  ],
  "cases": [
    "mug",
    "shirt"
  ],
  "total_variants": 303,
  "total_naive_session_seconds": 2786.666936453999,
  "total_semantic_detector_inclusive_session_seconds": 528.8529460209954,
  "total_cached_session_seconds": 525.1538296829954,
  "mean_naive_session_seconds": 398.0952766362855,
  "mean_semantic_detector_inclusive_session_seconds": 75.55042086014221,
  "mean_cached_session_seconds": 75.02197566899933,
  "mean_detector_seconds": 0.5284451911428667,
  "mean_detector_inclusive_session_speedup": 5.367015550267592,
  "median_detector_inclusive_session_speedup": 5.384464977638836,
  "min_detector_inclusive_session_speedup": 4.865235613369563,
  "max_detector_inclusive_session_speedup": 6.026476025805032,
  "mean_cached_session_speedup": 5.411076147247453,
  "median_cached_session_speedup": 5.432343876241666,
  "mean_outside_drift_naive": 95.54409152061116,
  "mean_outside_drift_semantic": 0.0
}
```