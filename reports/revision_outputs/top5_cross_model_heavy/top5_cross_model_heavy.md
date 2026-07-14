# Top-5 Cross-Model Heavy Benchmark

Ranking metric: mean cached semantic re-edit speedup.

| Rank | Model | N | Naive s | Local s | Detector s | Cached s | Gen. speedup | Incl. speedup | Cached speedup | Outside drift |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | OpenJourney v4 | 20 | 23.941 | 7.496 | 12.384 | 3.618 | 3.537x | 1.246x | 12.229x | 0.000% |
| 2 | Juggernaut XL v2 | 20 | 118.220 | 75.622 | 15.298 | 26.529 | 1.567x | 1.302x | 10.514x | 0.000% |
| 3 | DreamShaper 8 | 20 | 31.388 | 8.298 | 13.292 | 4.883 | 3.970x | 1.484x | 7.762x | 0.000% |
| 4 | Juggernaut Reborn | 20 | 11.390 | 4.973 | 12.778 | 2.109 | 2.392x | 0.643x | 6.872x | 0.000% |
| 5 | Deliberate v6 | 20 | 12.213 | 8.180 | 12.954 | 2.610 | 1.601x | 0.580x | 6.744x | 0.000% |

Interpretation:

- OpenJourney v4 and DreamShaper 8 are SD1.5-family models and show strong local/cached speedups.
- Juggernaut XL v2 is SDXL-family; its high cached speedup comes from very expensive full-image regeneration.
- Detector-inclusive first edits are model dependent: DreamShaper 8, OpenJourney v4, and Juggernaut XL v2 are above 1x, while the faster SD1.5 baselines are better mainly for cached/repeated edits.
- Semantic outside-mask drift remains 0.0% under mask compositing for all top-five rows.