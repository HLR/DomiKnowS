# Full test rescoring with original labels

Retained champion with grounded Complex planning and visual QA. All 450 saved predictions were rescored; no inference rerun or dataset changes. Hash-verified pre-repair backups restore nine changed density references in this test split. Corrected-label aggregation reproduces the previous report exactly.

| Metric | Original labels | Corrected labels |
|---|---:|---:|
| Overall progress | 78.00 | 78.54 |
| Legacy progress | 65.91 | 66.26 |
| Skill recall | 98.18 | 98.18 |
| Parameter recall | 66.59 | 67.48 |
| Skill-parameter recall | 70.15 | 71.04 |
| Graph matching | 50.81 | 51.70 |
| Six-dimension macro | 77.11 | 77.78 |

| Dimension | Episodes | Original-label score |
|---|---:|---:|
| M&T | 113 | 78.89 |
| Spatial | 53 | 72.17 |
| ComSense | 84 | 81.73 |
| Semantic | 84 | 85.54 |
| PhysLaw | 60 | 57.00 |
| Complex | 56 | 87.35 |

Zero generation failures and search-limit failures in the original generation. Only scoring labels changed. Metadata-assisted results are not included in this rescore. Input-interface and split/metric equivalence must still be established before comparing published baselines.
