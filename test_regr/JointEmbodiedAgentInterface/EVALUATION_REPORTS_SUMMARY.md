# Summary of Evaluation Reports 1–3

This document summarizes the three EAI/VLABench evaluation reports. End-to-end
goal or simulator success is the primary measure; graph matching, DFA validity,
and training return are diagnostics rather than substitutes for task success.

| Report | Main configuration | EAI only, Stage 1 → Stage 2 | VLABench only, Stage 1 → Stage 2 | Joint result |
|---|---|---:|---:|---|
| [Report 1](EVALUATION_REPORT.md) | Earlier mixed runs; VLABench and Joint included resumed runs and a different evaluation protocol | 77.3% → 79.5% | 76.67% → 83.33% | VLABench success 80.0% → 80.0%; Joint EAI was not reported as a comparable primary result |
| [Report 2](EVALUATION_REPORT2.md) | Fresh full runs; EAI used Qwen3-8B, VLABench/Joint used Qwen2.5-VL-3B | 80.7% → **89.8%** | 100% → 100% | EAI 12.5% → 12.5%; VLABench 100% → 100% |
| [Report 3](EVALUATION_REPORT3.md) | Fresh full runs; all three used Qwen3-VL-8B-Instruct | 79.5% → **85.2%** | 100% → 100% | EAI 12.5% → 12.5%; VLABench 100% → 100% |

## Main findings

- Standalone EAI benefited from RL in every report. The largest retained gain
  was in Report 2: 80.7% to 89.8%. With the common Qwen3-VL backbone in Report
  3, EAI improved by five examples, from 70/88 to 75/88, but remained 4.6
  percentage points below Report 2's final result.
- Reports 2 and 3 are the most directly comparable because both are fresh,
  full-data runs from the same commit and use the same fixed-seed VLABench
  evaluation. VLABench was already 10/10 before RL in both reports, so the
  unchanged 10/10 result does not demonstrate an RL improvement.
- The common Qwen3-VL backbone substantially improved Joint VLABench plan
  imitation: exact-graph match increased from 62.63% in Report 2 to 90.19% in
  Report 3. This did not improve the already saturated simulator score.
- Joint EAI remains the main weakness. Reports 2 and 3 both retained only 4/32
  goal successes (12.5%) and 22.63% goal recall, with no held-out gain from
  Joint RL. A stronger shared vision-language model alone did not repair the
  cross-domain EAI branch.
- Controller RL evidence is weak. VLABench PPO updates were repeatedly rolled
  back by the trust-region checks, and later Joint epochs improved training
  rollout return without changing held-out success. Training return should not
  be interpreted as an evaluation gain.

## Interpretation and next step

Report 1 established that the two-stage pipeline could improve standalone EAI
and VLABench under the earlier setup, but its resumed runs and different
evaluation protocol make it unsuitable for precise comparison with Reports 2
and 3. The fresh runs show that standalone EAI is learnable, VLABench's current
ten-episode evaluation is too easy to distinguish checkpoints, and Joint EAI
does not yet benefit from sharing the backbone.

The next evaluation should score the standalone and Joint planners on exactly
the same EAI examples, expand VLABench to multiple seeds and rollouts per task,
and report unassisted controller success separately. At least three training
seeds are needed before ranking Qwen3-8B, Qwen2.5-VL-3B, and Qwen3-VL-8B.
