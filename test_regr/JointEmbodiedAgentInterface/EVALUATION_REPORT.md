# EAI and VLABench Training Evaluation Report

## Evaluation objective

This report compares three base training settings:

1. EmbodiedAgentInterface (EAI) only
2. VLABench only
3. Joint EAI and VLABench

Each base setting is evaluated both without reinforcement learning (the restored
Stage 1 supervised checkpoint) and with reinforcement learning (the restored
Stage 2 checkpoint), for six settings in total. End-to-end task success is the
primary comparison metric. Sequence matching, graph/DFA validity, constraint
satisfaction, and reward components are reported as supporting diagnostics.

## Results status

| Training setting | Without RL | With RL | Status |
|---|---:|---:|---|
| EAI only | 77.3% goal success | 79.5% goal success | Complete |
| VLABench only | Pending | Pending | Run in progress |
| Joint EAI and VLABench | Pending | Pending | Awaiting final results |

## EAI-only experiment

### Evaluation protocol

- Validation examples: 88
- Stage 1: supervised exact-match pretraining with `SolverPOIProgram`
- Stage 2: reinforcement learning with dense goal and constraint-modulated reward
- Stage 1 learning rate: `1e-4`
- Stage 2 learning rate: `1e-5`
- Stage 1 selected checkpoint: epoch 3 of 5
- Stage 2 selected checkpoint: epoch 1 of 3
- DFA validity remained 100% in every reported epoch

The checkpoint paths produced by the run are:

- Without RL: `test_regr/EmbodiedAgentInterface/models/report_eai/eai_without_rl.pth`
- With RL: `test_regr/EmbodiedAgentInterface/models/report_eai/eai_with_rl.pth`

### Primary comparison

| Metric | Without RL | With RL | Absolute change |
|---|---:|---:|---:|
| Goal success | 77.3% | 79.5% | +2.2 percentage points |
| Goal-state recall | 83.4% | 84.2% | +0.8 percentage points |
| Temporal progress | 89.2% | 91.5% | +2.3 percentage points |
| Positive-reward rate | 88.6% | 92.0% | +3.4 percentage points |
| Aggregate RL reward | 0.804 | 0.817 | +0.013 |
| Exact action sequence | 17.0% | 22.7% | +5.7 percentage points |
| Token accuracy | 49.7% | 53.0% | +3.3 percentage points |
| DFA validity | 100.0% | 100.0% | 0.0 percentage points |
| Applicable world-constraint score | 1.000 | 1.000 | 0.000 |
| Average predicted plan length | 15.33 | 16.34 | +1.01 actions |

The goal-success change corresponds to approximately 68 successful examples
without RL and 70 with RL. Therefore, the current evidence supports a modest
improvement from reinforcement learning, not a conclusive improvement across
the task distribution. Multiple seeded runs are required for uncertainty and
significance estimates.

### Stage 1 supervised results

| Epoch | Exact sequence | Goal success | Goal recall | Temporal progress | Positive reward | RL reward |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 14.8% | 70.5% | 76.0% | 85.8% | 83.0% | 0.737 |
| 2 | 18.2% | 68.2% | 76.7% | 88.6% | 86.4% | 0.740 |
| 3 | 17.0% | 77.3% | 83.4% | 89.2% | 88.6% | 0.804 |
| 4 | 20.5% | 75.0% | 83.2% | 85.8% | 85.2% | 0.797 |
| 5 | 17.0% | 72.7% | 79.1% | 90.3% | 89.8% | 0.773 |

Epoch 3 was restored because it had the strongest balanced semantic result.
Later supervised epochs improved individual diagnostics but reduced goal
success and aggregate reward.

### Stage 2 reinforcement-learning results

| Epoch | Exact sequence | Goal success | Goal recall | Temporal progress | Positive reward | RL reward |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 22.7% | 79.5% | 84.2% | 91.5% | 92.0% | 0.817 |
| 2 | 22.7% | 78.4% | 84.1% | 89.2% | 89.8% | 0.814 |
| 3 | 23.9% | 78.4% | 85.2% | 89.2% | 87.5% | 0.813 |

Epoch 1 was restored because it produced the best balanced semantic outcome.
Although epoch 3 produced the highest exact-sequence score and goal recall,
epoch 1 had higher goal success, temporal progress, positive-reward rate, and
aggregate reward.

### EAI-only finding

On this 88-example validation set, reinforcement learning improved EAI goal
success from 77.3% to 79.5%, temporal progress from 89.2% to 91.5%, and the
positive-reward rate from 88.6% to 92.0%. Exact sequence accuracy also improved
from 17.0% to 22.7%, while DFA validity and applicable world-constraint
satisfaction remained at 100%. The result is directionally positive, but the
small absolute success difference and single training seed require cautious
interpretation.

## VLABench-only experiment

Results will be added after the corrected standalone run completes. The primary
metric will be simulator task success, accompanied by return, plan validity,
rollout efficiency, IK recovery/truncation rates, and planner exact-graph match.

## Joint EAI and VLABench experiment

Results will be added after the joint run completes. EAI goal success and
VLABench simulator success will remain separate domain-local metrics; the two
rewards are not combined. The primary joint comparison will report both domain
success rates and their balanced minimum and mean.

## Limitations and remaining evaluation

- The current EAI comparison contains one seed and 88 validation examples.
- The approximately 2.2-point goal-success gain is two additional successful
  examples and has not been tested for paired statistical significance.
- Final conclusions require the VLABench-only and joint results.
- Each of the six settings should ideally be repeated with at least three seeds,
  reporting mean, standard deviation, and a paired comparison where possible.
