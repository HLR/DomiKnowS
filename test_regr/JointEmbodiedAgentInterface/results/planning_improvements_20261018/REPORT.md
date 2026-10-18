# Planning improvements: full 450-episode comparisons

The retained champion with the new grounded policy is recommended. Every run evaluated all 450 episodes with zero generation/search failures.

| Condition | Overall | Macro | Complex | PhysLaw |
|---|---:|---:|---:|---:|
| Previous visual | 72.72 | 70.93 | 54.20 | 56.00 |
| Representation/input repairs | 74.47 | 72.34 | 54.70 | 61.00 |
| Grounded visual QA, champion | **78.54** | **77.78** | **87.35** | **61.00** |
| Metadata-assisted QA, champion | **82.67** | **82.95** | **87.35** | **92.00** |
| Adapted head, visual (rejected) | 63.81 | 64.88 | 87.81 | 66.00 |
| Adapted head, metadata-assisted (rejected) | 67.01 | 68.88 | 87.81 | 90.00 |

Visual and metadata-assisted QA use different inputs. Complex grounding uses exposed language/entity names and public task knowledge. The visual condition does not mean raw-image-only planning. No simulator rollout was measured.

The retained model answers 21/60 QA buttons correctly in visual mode and 52/60 with metadata assistance. All 14 reflection cases were evaluated, with six correct. Book rearrangement, cooking and hammer/picture tasks score 100; chemistry scores 90.95; Texas hold'em remains weak at 41.36.

The three-epoch head-only prototype was selected by validation loss, but its generated QA validation accuracy declined from 40.74% to 35.19% (54 examples). It was rejected using validation evidence; test results did not select the checkpoint. Its worse overall test score confirms that lower teacher-forced loss is insufficient. The training workflow now selects by generated validation macro with retention of all measured dimensions and keeps the original model as fallback.

All 4,500 reference plans pass the final source audit. All 450 retained visual plans pass independent completed-plan verification and exact score aggregation. Current-policy replay preserves all 4,641 recorded deterministic choices. Broad regressions: 670 passed, 13 pre-existing failures, one skipped. Relative to the previous visual run, ComSense declines 1.90 and Spatial 2.55 points; M&T and Semantic improve.

See `summary.json`, `selection.json`, individual reports, run/source manifests, and the task-level verification report in this directory. Detailed logs, epoch checkpoints and complete source snapshots remain under `/home/auszok/evaluation_runs/planning_improvements_20261018T174007Z/` on gpu2. The prototype's original loss-based protocol is preserved in its provenance, rather than relabeled as the newer retention protocol.
