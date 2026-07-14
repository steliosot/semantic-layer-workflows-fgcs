# Policy Ablation Summary

## Hardware Stress Scenarios

| Scenario | Median break-even k | k=1 wins | k=2 wins | k=3 wins | k=5 wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| edge_constrained | 2.02 | 24.0% | 48.5% | 92.5% | 99.5% |
| measured_mps | 1.79 | 25.5% | 60.0% | 98.0% | 99.5% |
| cuda_workstation | 1.61 | 26.5% | 71.5% | 99.5% | 99.5% |
| cloud_accelerator | 1.58 | 26.5% | 71.5% | 99.5% | 99.5% |
| fast_detector_slow_diffusion | 0.89 | 54.0% | 96.0% | 100.0% | 99.5% |
| slow_detector_fast_diffusion | 5.55 | 0.0% | 9.0% | 24.0% | 46.0% |

## Policy Ablation

### k1

| Policy | Semantic selected | Mean latency | Mean outside drift |
| --- | ---: | ---: | ---: |
| latency_only | 25.5% | 30.780s | 69.973% |
| preservation_first | 100.0% | 43.675s | 0.000% |
| balanced | 25.5% | 30.780s | 69.973% |
| oracle | 25.5% | 30.780s | 69.973% |

### k2

| Policy | Semantic selected | Mean latency | Mean outside drift |
| --- | ---: | ---: | ---: |
| latency_only | 60.0% | 49.543s | 38.102% |
| preservation_first | 100.0% | 51.791s | 0.000% |
| balanced | 100.0% | 51.791s | 0.000% |
| oracle | 60.0% | 49.543s | 38.102% |

### k3

| Policy | Semantic selected | Mean latency | Mean outside drift |
| --- | ---: | ---: | ---: |
| latency_only | 98.0% | 59.839s | 1.773% |
| preservation_first | 100.0% | 59.906s | 0.000% |
| balanced | 100.0% | 59.906s | 0.000% |
| oracle | 98.0% | 59.839s | 1.773% |

### k5

| Policy | Semantic selected | Mean latency | Mean outside drift |
| --- | ---: | ---: | ---: |
| latency_only | 99.5% | 75.967s | 0.455% |
| preservation_first | 100.0% | 76.137s | 0.000% |
| balanced | 100.0% | 76.137s | 0.000% |
| oracle | 99.5% | 75.967s | 0.455% |
