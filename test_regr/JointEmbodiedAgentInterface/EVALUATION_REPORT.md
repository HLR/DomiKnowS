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
| VLABench only | Stage 1 restored checkpoint | 76.7% simulator success | Complete; best RL checkpoint retained |
| Joint EAI and VLABench | Stage 1 validation complete | 77.5% VLABench success; 10 successful tasks | Complete; best Stage 2 checkpoint retained |

The VLABench-only and Joint runs were resumed on GPU 4 and GPU 5 respectively from saved snapshots after the original processes were killed. The resumed runs used commit `db7e11b7`, which initializes the Stage 2 rollout `progress` accumulator that previously caused both runs to terminate with `KeyError: 'progress'`.
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

### Evaluation protocol

- Physical GPU: GPU 4
- Resumed run log: `test_regr/VLABenchAgentInterface/results/vlabench_full_gpu4_resume_20260912_153549.log`
- Resume source: `test_regr/VLABenchAgentInterface/checkpoints/vlabench_full_gpu4_restart_20260911_142209/agent_stage1.pt`
- Best retained checkpoint: `test_regr/VLABenchAgentInterface/checkpoints/vlabench_full_gpu4_resume_20260912_153549/agent_rl_best.pt`
- Ten primitive VLABench tasks were evaluated with eight rollouts per reinforcement update.
- The run used the corrected camera mapping, controller frame v2, task contracts, and simulator diagnostics.

### Final retained result

| Metric | Best retained VLABench RL checkpoint |
|---|---:|
| Simulator success rate | 76.7% |
| Successful tasks | 9/10 |
| Positive-return rate | 80.0% |
| Mean return | 0.729 |
| Mean progress | 0.729 |
| Valid/executable rate | 80.0% |
| Mean episode steps | 77.3 |
| IK truncation rate | 0.0% in the retained evaluation summary |

The final fixed-seed retention check rejected a later RL checkpoint because task signal or controller feasibility degraded. The run restored the earlier best checkpoint from RL epoch 0 and saved it as `agent_rl_best.pt`. Individual simulator physics failures still occurred in tasks such as `insert_flower`, `select_fruit`, and `select_toy`; these were recorded as invalid rollouts and are included in the aggregate validity rate.
## Joint EAI and VLABench experiment

### Evaluation protocol

- Physical GPU: GPU 5
- Resumed run log: `test_regr/JointEmbodiedAgentInterface/results/joint_full_gpu5_resume_20260912_153549.log`
- Resume source: `test_regr/JointEmbodiedAgentInterface/checkpoints/joint_full_gpu5_restart_20260911_142209/joint_stage2_progress.pt`
- Best retained checkpoint: `test_regr/JointEmbodiedAgentInterface/checkpoints/joint_full_gpu5_resume_20260912_153549/joint_stage2_best.pt`
- EAI training examples: 438 (`394` train, `44` validation)
- VLABench planning examples: 4,500 (`3,600` train, `450` validation, `450` test)
- Stage 2 used ten rounds per epoch and eight VLABench rollouts per update.

### Final Stage 2 result

| Metric | Joint Stage 2 retained result |
|---|---:|
| VLABench simulator success rate | 77.5% |
| Successful VLABench tasks | 10 |
| VLABench positive-return/valid rate | 88.75% |
| VLABench mean return | 0.738 |
| VLABench mean episode steps | 99.3 |
| EAI and VLABench retention gate | Eligible |

The retained checkpoint was produced from `joint_stage2_epoch_000.pt`. VLABench success and EAI goal success remain separate domain-local metrics; they are not merged into one reward or one success percentage. Physics failures affected some VLABench rollouts, but the Joint Stage 2 checkpoint remained retention-eligible under the configured thresholds.
## Limitations and remaining evaluation

- The EAI comparison contains one seed and 88 validation examples.
- The VLABench-only and Joint results are single resumed runs, so they do not provide multi-seed uncertainty estimates.
- VLABench simulator physics failures and invalid rollouts remain concentrated in several task families and should be audited before treating the aggregate success rates as robust.
- The VLABench-only run retained an earlier RL checkpoint after a later fixed-seed evaluation lost task signal; checkpoint selection therefore matters in the reported result.
- Each setting should ideally be repeated with at least three seeds, reporting mean, standard deviation, and paired comparisons where possible.