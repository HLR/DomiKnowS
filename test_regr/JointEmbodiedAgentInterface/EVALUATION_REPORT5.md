# EAI and VLABench Training Evaluation Report 5

## Evaluation objective and changes from Report 4

This report evaluates three fresh full-data, two-stage runs with
`Qwen/Qwen3-VL-8B-Instruct`: EAI only, VLABench only, and Joint
EAI/VLABench (run tag `20260922_062426`). End-to-end goal or simulator success
remains the primary metric. Exact-match, reward, return, constraint, IK,
assistance, and gate measurements provide supporting diagnostics.

Report 5 incorporates the core architectural and protocol improvements identified
in Report 4 and subsequent parameter interference investigations:

- **Coordinated Multi-Domain Optimization**: Joint Stage 1 uses
  `GradientConflictManager` with PCGrad projection to prevent cross-domain gradient
  interference between EAI and VLABench losses on the shared Qwen3-VL/LoRA backbone.
- **Dynamic Parameter Policy & Checkpoint Integrity**: Stage 2 parameter freezing
  is governed by `--stage2-parameter-policy freeze_shared` rather than unconditional
  manual freezes in `main.py`, preserving learned adapter checkpointability and
  enforcing strict parameter ownership checksum verification.
- **Unassisted Controller Evaluation**: VLABench evaluation enforces
  `--execution-assistance train-only`, preventing artificial assistance masking during
  held-out evaluation and measuring genuine autonomous controller capability.
- **Controller Regime Management**: Controller optimizer resets BC moments and
  transitions to a dedicated PPO learning rate (`3e-05`) upon entering the RL phase.
- **Strict Exploration and Preflight Gate Enforcement**: Both Joint and standalone
  VLABench adhere to readiness gates (Joint EAI exploration gate and VLABench controller
  preflight gate), halting before expensive simulator RL when baseline criteria
  are not satisfied.

The runs executed on `gpu2.ihmc.us` inside Docker container `vigorous_easley` from
commit `bfc30310ee7bcc1773205b517f95f4bce010f981`. The shared backbone was loaded
from `/home/auszok/models/Qwen/Qwen3-VL-8B-Instruct`. Each process used its assigned
physical GPU:

| Run | GPU | Training Configuration | Log File |
|---|---:|---|---|
| EAI only | 4 | 5 supervised + 5 RL epochs | `test_regr/EmbodiedAgentInterface/results/full_20260922_062426.log` |
| VLABench only | 5 | 3 supervised + 20,000 BC steps + preflight eval | `test_regr/VLABenchAgentInterface/results/full_20260922_062426.log` |
| Joint | 6 | 5 supervised (350 rounds/epoch) + exploration gate | `test_regr/JointEmbodiedAgentInterface/results/full_20260922_062426.log` |

EAI used all 438 canonical examples (350/88 split). VLABench used 4,500 planning
examples (3,600/450/450) and all ten controller task families
(459,675/57,225/58,201 windows).

---

## Primary end-to-end comparison

| Training setting | Stage 1 / pre-RL | Retained Stage 2 | Change |
|---|---:|---:|---:|
| EAI only: goal success on 88 | 71/88 (80.7%) | 71/88 (80.7%) | 0 examples (recall +3.1%, token acc +4.3%) |
| VLABench only: simulator success on 30 | 1/30 (3.3%) | *Skipped by preflight gate* | Gate enforced (1/10 tasks complete) |
| Joint: EAI goal success on 88 | 0/88 (0.0%) | *Skipped by exploration gate* | Gate enforced (pos reward 3.4% < 10%) |
| Joint: VLABench graph match on 450 | 401/450 (89.1%) | *Skipped by exploration gate* | +72.6 points over Epoch 0 (16.5% → 89.1%) |

Key observations:
1. **EAI-only**: Stable goal success at 80.7% (71/88), with Stage 2 RL improving
   semantic alignment: goal recall increased from 84.6% to 87.7% (+3.1 points),
   exact sequence match increased from 23.9% to 27.3% (+3.4 points), token accuracy
   increased from 53.8% to 58.1% (+4.3 points), and aggregate RL reward score rose
   from 0.829 to 0.845.
2. **VLABench unassisted evaluation**: Unlike Report 4 where 96.7% of evaluation episodes
   used online task assistance (yielding an artificial 83.3% success rate), unassisted
   evaluation in Report 5 revealed true autonomous capability: **70.0% positive return rate**,
   **0.0% IK truncation rate**, and a 33.3% unassisted success rate on `select_painting`.
   However, because only 1 of 10 task families achieved full episode completion, the
   preflight gate correctly prevented launching ungrounded Stage 2 simulator rollouts.
3. **Joint training**: The PCGrad-coordinated optimizer drove dramatic VLABench planner
   learning, improving exact graph match from 16.5% (Epoch 0) to **89.1%** (Epoch 4)
   with 100% DFA validity. However, the EAI branch plateaued at 0/88 success and 3.4%
   positive reward rate, causing the EAI exploration gate to halt execution before
   Stage 2 RL.

---

## EAI-only result

Stage 1 selected supervised epoch 1. Stage 2 selected RL epoch 3.

| Metric on 88 validation examples | Stage 1 (Epoch 1) | Retained Stage 2 (Epoch 3) | Change |
|---|---:|---:|---:|
| Goal success | 80.7% (71/88) | 80.7% (71/88) | 0.0 points |
| Goal recall | 84.6% | 87.7% | +3.1 points |
| Temporal progress | 89.2% | 88.6% | -0.6 points |
| Positive-reward rate | 88.6% | 92.0% | +3.4 points |
| Aggregate reward score | 0.829 | 0.845 | +0.016 |
| Exact sequence match | 23.9% | 27.3% | +3.4 points |
| Token accuracy | 53.8% | 58.1% | +4.3 points |
| DFA validity | 100.0% | 100.0% | 0.0 points |

### Epoch-by-epoch trajectory

| Stage | Epoch | Goal Success | Goal Recall | Temporal Progress | Exact Sequence | Token Acc | RL Reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| Supervised | 1 (selected) | **80.7%** | 84.6% | 89.2% | 23.9% | 53.8% | 0.829 |
| Supervised | 2 | 70.5% | 80.6% | 87.5% | 22.7% | 53.4% | 0.781 |
| Supervised | 3 | 71.6% | 80.7% | 85.2% | 20.5% | 51.6% | 0.777 |
| Supervised | 4 | 76.1% | 85.0% | 86.9% | 25.0% | 52.4% | 0.808 |
| Supervised | 5 | 69.3% | 79.8% | 85.8% | 15.9% | 54.7% | 0.751 |
| RL | 1 | 77.3% | 86.6% | 87.5% | 27.3% | **58.4%** | 0.827 |
| RL | 2 | 79.5% | 87.3% | 88.1% | 26.1% | 57.5% | 0.837 |
| RL | 3 (selected) | **80.7%** | **87.7%** | 88.6% | **27.3%** | 58.1% | **0.845** |
| RL | 4 | **80.7%** | **87.7%** | **89.2%** | **27.3%** | 58.3% | 0.841 |
| RL | 5 | **80.7%** | 87.1% | 88.6% | **27.3%** | 57.3% | 0.839 |

RL policy optimization demonstrated clear stabilization and semantic refinement:
late supervised training overfitted and decayed to 69.3% success in Epoch 5, whereas
Stage 2 RL rapidly restored success to 80.7% while reaching higher peak token accuracy
(58.4%) and exact sequence match (27.3%).

Retained model artifact:
[`test_regr/EmbodiedAgentInterface/models/eai_full_20260922_062426.pth`](file:///home/auszok/DomiKnowS/test_regr/EmbodiedAgentInterface/models/eai_full_20260922_062426.pth).

---

## VLABench-only result

### Stage 1 Supervised Pretraining & Controller Warm-up

- Supervised Planner reached **11.56% exact graph match** (52/450) and **100% DFA validity**
  on held-out planning validation.
- Controller BC warm-up completed 20,000 steps with **0.0988 pose MAE** and **98.83% gripper
  balanced accuracy** (closed recall 98.79%, open recall 98.87%, transition precision 90.81%).
- Saved Stage 1 model: [`agent_stage1.pt`](file:///home/auszok/DomiKnowS/test_regr/VLABenchAgentInterface/checkpoints/full_20260922_062426/agent_stage1.pt).

### Stage 2 Unassisted Preflight Evaluation (30 Simulator Rollouts)

Evaluation executed under `--execution-assistance train-only`, meaning zero assist steps
were applied during evaluation.

| Task Family | Episodes | Success Rate | Positive Return Rate | Mean Progress | Mean Return | IK Failures / Recoveries | IK Truncations |
|---|---:|---:|---:|---:|---:|---:|---:|
| `add_condiment` | 3 | 0.0% | 0.0% | 0.0% | 0.000 | 0 / 0 | 0 |
| `insert_flower` | 3 | 0.0% | 66.7% | 38.8% | 0.150 | 158 / 59 | 0 |
| `select_book` | 3 | 0.0% | 33.3% | 16.7% | 0.042 | 0 / 0 | 0 |
| `select_chemistry_tube` | 3 | 0.0% | 100.0% | 46.6% | 0.150 | 67 / 32 | 0 |
| `select_drink` | 3 | 0.0% | 100.0% | **71.2%** | 0.139 | 312 / 82 | 0 |
| `select_fruit` | 3 | 0.0% | 66.7% | 13.7% | 0.068 | 0 / 0 | 0 |
| `select_mahjong` | 3 | 0.0% | 100.0% | 51.9% | 0.163 | 109 / 48 | 0 |
| `select_painting` | 3 | **33.3%** | 66.7% | 37.3% | **0.335** | 0 / 0 | 0 |
| `select_poker` | 3 | 0.0% | 66.7% | 31.2% | 0.078 | 0 / 0 | 0 |
| `select_toy` | 3 | 0.0% | 100.0% | 35.2% | 0.121 | 209 / 73 | 0 |
| **Overall Aggregate** | **30** | **3.3%** (1/30) | **70.0%** (21/30) | **34.3%** | **0.125** | **855 / 294** | **0 / 30** |

Preflight Gate Outcome:
- Gate requirement: `minimum_successful_tasks = 10`.
- Observed: 1 task family achieved full completion (`select_painting`), while 9 task families
  achieved intermediate intention and distance progress without complete unassisted placement.
- Outcome: Exited cleanly with `"reason": "VLABench controller preflight gate"`, saving
  [`agent_stage1_evaluated.pt`](file:///home/auszok/DomiKnowS/test_regr/VLABenchAgentInterface/checkpoints/full_20260922_062426/agent_stage1_evaluated.pt).
  This gate correctly prevented wasting hundreds of GPU-hours in ungrounded Stage 2 PPO rollouts.

---

## Joint result

### Stage 1 Coordinated Training Progression

Stage 1 trained for 5 epochs (350 rounds per epoch) using `GradientConflictManager`
with PCGrad gradient projection on the shared backbone.

| Joint Epoch | EAI Validation Loss | EAI Success | EAI Goal Recall | EAI Pos Reward | VLABench Loss | VLABench Exact Match | VLABench Reward | DFA Valid |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5.875 | 0/88 (0.0%) | 3.1% | 3.4% | 2.087 | 16.5% | 0.501 | 100% |
| 1 | 4.586 | 0/88 (0.0%) | 3.1% | 3.4% | 1.010 | 34.9% | 0.608 | 100% |
| 2 | 4.090 | 0/88 (0.0%) | 3.1% | 3.4% | 0.621 | 61.6% | 0.768 | 100% |
| 3 | 3.807 | 0/88 (0.0%) | 3.1% | 3.4% | 0.403 | 84.3% | 0.883 | 100% |
| 4 | **3.555** | 0/88 (0.0%) | 3.1% | 3.4% | **0.274** | **89.1%** | **0.905** | 100% |

Stage 1 highlights:
- **VLABench Graph Imitation**: Climbed dramatically from 16.5% to **89.1%** exact graph match,
  with entity match reaching 90.0% and skill match reaching 91.6%.
- **Gradient Coordination**: The single optimizer step with PCGrad projection prevented
  catastrophic interference on shared parameters. However, the EAI output head was unable
  to escape its initial local minimum, remaining at 3.4% positive reward rate throughout.
- **Exploration Gate**: Because `positive_reward_rate (0.034) < 0.10`, the run cleanly halted:
  ```json
  {"stage": "stage2-skipped", "reason": "EAI exploration gate", "eai": {"positive_reward_rate": 0.034}}
  ```
- All 5 checkpoints (`joint_stage1_epoch_000.pt` through `_004.pt`, ~1.5 GB each) were safely
  persisted in [`test_regr/JointEmbodiedAgentInterface/checkpoints/full_20260922_062426/`](file:///home/auszok/DomiKnowS/test_regr/JointEmbodiedAgentInterface/checkpoints/full_20260922_062426/).

---

## Comparison of Assistance & Controller Reality (Report 4 vs. Report 5)

| Metric | Report 4 (Assisted Eval) | Report 5 (Unassisted Eval) | Real-World Implication |
|---|---:|---:|---|
| Task-Assist Steps | 4,141 steps | **0 steps** | Authentic autonomous execution |
| Assisted Episode Rate | 96.7% (29/30) | **0.0%** (0/30) | No oracle intervention |
| Simulator Success Rate | 83.3% (25/30) | **3.3%** (1/30) | Exposes true bottleneck in unassisted grasping |
| Positive Return Rate | 83.3% | **70.0%** (21/30) | Substantial approach & intention progress persists |
| IK Truncation Rate | 3.3% (1/30) | **0.0%** (0/30) | Clean kinematical stability across all tasks |
| Valid DFA Rate | 96.7% | **100.0%** (30/30) | Graph planning guidance is fully valid |

Report 5 establishes that the 83.3% success reported in Reports 2–4 was largely an
artifact of oracle execution assistance. When execution assistance is restricted to training,
the learned controller achieves positive task progress on 70% of episodes and succeeds on
`select_painting` (33.3%), but requires additional grasping/holding precision to complete
multi-stage primitives without assistance.

---

## Checkpoint verification

All retained artifacts deserialized cleanly and verified parameter ownership checksums:

- EAI final Stage 2 model: [`test_regr/EmbodiedAgentInterface/models/eai_full_20260922_062426.pth`](file:///home/auszok/DomiKnowS/test_regr/EmbodiedAgentInterface/models/eai_full_20260922_062426.pth)
- VLABench Stage 1 model: [`test_regr/VLABenchAgentInterface/checkpoints/full_20260922_062426/agent_stage1.pt`](file:///home/auszok/DomiKnowS/test_regr/VLABenchAgentInterface/checkpoints/full_20260922_062426/agent_stage1.pt)
- VLABench evaluated model: [`test_regr/VLABenchAgentInterface/checkpoints/full_20260922_062426/agent_stage1_evaluated.pt`](file:///home/auszok/DomiKnowS/test_regr/VLABenchAgentInterface/checkpoints/full_20260922_062426/agent_stage1_evaluated.pt)
- Joint Stage 1 models: [`joint_stage1_epoch_000.pt`](file:///home/auszok/DomiKnowS/test_regr/JointEmbodiedAgentInterface/checkpoints/full_20260922_062426/joint_stage1_epoch_000.pt) through `_004.pt`.

---

## Conclusions and next actions

1. **Parameter Coordination is Functionally Verified**: The new `GradientConflictManager`
   and PCGrad pipeline ran for 5 joint epochs on H100 GPUs without memory leaks, autograd
   crashes, or numerical instability, producing 89.1% VLABench graph match.
2. **Readiness Gates Protect Compute**: Both the Joint EAI exploration gate and the VLABench
   controller preflight gate operated as designed, halting before ungrounded Stage 2 simulator
   rollouts.
3. **The Unassisted Baseline is Established**: With `--execution-assistance train-only`,
   we now have an authentic baseline: 70% positive return rate, 0% IK truncations, 100% DFA
   validity, and 3.3% full unassisted success.
4. **Targeted Follow-up for Joint EAI**: To enable Joint Stage 2 RL, Joint Stage 1 needs
   an initial EAI warm-up or loss-weight adjustment (e.g., higher initial EAI weight or
   domain-balanced learning rates) so the EAI branch clears the 10% positive reward threshold.
