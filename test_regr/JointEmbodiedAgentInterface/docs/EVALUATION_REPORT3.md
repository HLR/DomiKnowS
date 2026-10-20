# EAI and VLABench Training Evaluation Report 3

## Evaluation objective and provenance

This report evaluates three fresh, full-data, two-stage runs using the same
`Qwen/Qwen3-VL-8B-Instruct` base model: EAI only, VLABench only, and Joint
EAI/VLABench. End-to-end goal or simulator success is the primary metric;
sequence, graph, constraint, and return measurements are supporting
diagnostics.

The runs executed concurrently on `gpu2.ihmc.us` from commit
`75918bd818e951e04bbe964d3a0b75d99e0ff7e4`. The common base model was loaded
from `/home/auszok/models/Qwen/Qwen3-VL-8B-Instruct`. Each run used a new
output path and no `--resume` option. All three processes exited with code 0.

| Run | Physical GPU | Training stages | Log |
|---|---:|---|---|
| EAI only | 4 | 5 supervised + 5 RL epochs | `test_regr/EmbodiedAgentInterface/results/full_qwen3vl8b_20260920.log` |
| VLABench only | 5 | 3 supervised epochs + 20,000 controller-BC steps + 3 RL epochs | `test_regr/VLABenchAgentInterface/results/full_qwen3vl8b_20260920.log` |
| Joint | 6 | 5 supervised epochs + 20,000 controller-BC steps + 3 RL epochs | `test_regr/JointEmbodiedAgentInterface/results/full_qwen3vl8b_20260920.log` |

Each process saw its assigned physical GPU as `cuda:0` through
`CUDA_VISIBLE_DEVICES`. EAI used the Qwen3-VL language path with LoRA.
VLABench and Joint used 4-bit Qwen3-VL LoRA plus a SigLIP controller vision
encoder. EAI used all 438 examples (350 training and 88 validation). Joint
used a 394/44 split and scored 32 EAI validation examples because of its
configured validation limit. VLABench used 4,500 planning examples
(3,600/450/450) and all ten controller tasks, comprising
459,675/57,225/58,201 train/validation/test windows. Seed 42 was used for
training; fixed-seed simulator evaluation used seed 100042 and one rollout
per task.

## Primary end-to-end comparison

| Training setting / outcome | Selected Stage 1, without RL | Retained Stage 2, with RL | Change |
|---|---:|---:|---:|
| EAI only: benchmark-goal success | 70/88 (79.5%) | 75/88 (85.2%) | +5.7 percentage points |
| VLABench only: fixed-seed simulator success | 10/10 (100%) | 10/10 (100%) | 0 points |
| Joint: EAI benchmark-goal success | 4/32 (12.5%) | 4/32 (12.5%) | 0 points |
| Joint: fixed-seed VLABench simulator success | 10/10 (100%) | 10/10 (100%) | 0 points |

EAI success evaluates the benchmark `tl_goal` after simulated actions, not
reference-plan exact match. EAI-only and Joint EAI use different validation
subsets and sample counts and therefore are not a paired cross-model test.
EAI and VLABench rewards remain domain-local and are never numerically
combined. The ten-episode VLABench evaluation was already at its observed
ceiling before RL, so it cannot demonstrate an improvement in simulator
success.

## EAI-only experiment

The supervised run selected epoch 1, and RL selected epoch 3. Checkpoints:

- Stage 1: `test_regr/EmbodiedAgentInterface/models/eai_full_qwen3vl8b_20260920.stage1.pth`
- Stage 2: `test_regr/EmbodiedAgentInterface/models/eai_full_qwen3vl8b_20260920.pth`

| Validation metric (88 examples) | Stage 1 selected | Stage 2 selected | Change |
|---|---:|---:|---:|
| Goal success | 79.5% | 85.2% | +5.7 points (5 examples) |
| Goal-state recall | 87.5% | 89.9% | +2.4 points |
| Temporal progress | 89.2% | 91.5% | +2.3 points |
| Positive-reward rate | 93.2% | 94.3% | +1.1 points |
| Aggregate RL reward | 0.842 | 0.880 | +0.038 |
| Exact action sequence | 25.0% | 20.5% | -4.5 points |
| Token accuracy | 51.4% | 53.8% | +2.4 points |
| DFA validity | 100% | 100% | 0 points |
| Applicable world-constraint score | 0.978 | 0.957 | -0.021 |

| Stage | Epoch | Goal success | Goal recall | Temporal progress | Exact sequence | Reward |
|---|---:|---:|---:|---:|---:|---:|
| Supervised | 1 (selected) | 79.5% | 87.5% | 89.2% | 25.0% | 0.842 |
| Supervised | 2 | 72.7% | 82.9% | 87.5% | 23.9% | 0.797 |
| Supervised | 3 | 67.0% | 78.5% | 85.8% | 20.5% | 0.746 |
| Supervised | 4 | 76.1% | 82.4% | 88.6% | 13.6% | 0.798 |
| Supervised | 5 | 72.7% | 82.6% | 86.9% | 20.5% | 0.794 |
| RL | 1 | 83.0% | 87.8% | 90.9% | 15.9% | 0.852 |
| RL | 2 | 80.7% | 87.6% | 89.2% | 14.8% | 0.850 |
| RL | 3 (selected) | 85.2% | 89.9% | 91.5% | 20.5% | 0.880 |
| RL | 4 | 79.5% | 87.4% | 87.5% | 17.0% | 0.837 |
| RL | 5 | 84.1% | 90.0% | 90.9% | 19.3% | 0.877 |

RL improved semantic end-to-end success by five examples, but exact-sequence
accuracy and the applicable constraint score declined. The result therefore
supports a semantic improvement on this seed, not a general improvement in
reference imitation or constraint satisfaction.

## VLABench-only experiment

The selected supervised checkpoint is
`test_regr/VLABenchAgentInterface/checkpoints/full_qwen3vl8b_20260920/agent_stage1_evaluated.pt`.
Planner validation exact-graph match was 11.56% on 450 examples with 100%
graph validity. The 20,000-step controller warm-up ended at loss 0.1887.
Controller validation pose MAE was 0.0407 on 512 actions. Reported gripper
accuracy was 100%, but all 512 scored targets were open, so this does not
establish closed-gripper performance.

The supervised fixed-seed simulator baseline and every RL checkpoint scored
10/10 success, mean return 0.9470, mean 103.9 steps, 100% valid/executable,
and zero evaluation IK failures or truncations. The retained checkpoint is
the first RL epoch:

`test_regr/VLABenchAgentInterface/checkpoints/full_qwen3vl8b_20260920/agent_rl_best.pt`

| RL epoch | Training success | Mean training return | IK-truncated episodes | IK failure / recovery events | PPO update rounds rolled back |
|---:|---:|---:|---:|---:|---:|
| 1 (retained) | 86.25% | 0.8413 | 0/80 | 0 / 0 | 10/10 |
| 2 | 85.00% | 0.8234 | 1/80 | 1,149 / 446 | 10/10 |
| 3 | 83.75% | 0.8049 | 1/80 | 1,765 / 593 | 10/10 |

The later training epochs had lower rollout success and return. More
importantly, all fixed-seed evaluations tied at 100%, so the selection key
retained the earliest eligible epoch. Every PPO round was rolled back by the
trust-region check, meaning the reported RL checkpoint selection does not
show a successful controller PPO improvement. The two IK-truncated episodes
occurred during training, not the selected ten-episode evaluation.

## Joint EAI/VLABench experiment

Stage 1 selected the fifth supervised epoch and then completed the dedicated
controller warm-up. The selected supervised checkpoint is
`test_regr/JointEmbodiedAgentInterface/checkpoints/full_qwen3vl8b_20260920/joint_controller_warmup.pt`.
Its controller warm-up loss was 0.1935 after 20,000 steps.

| Stage 1 epoch | EAI goal success (32) | EAI goal recall | VLABench exact graph (450) | VLABench graph validity |
|---:|---:|---:|---:|---:|
| 1 | 0.0% | 10.13% | 20.48% | 100% |
| 2 | 0.0% | 10.13% | 31.15% | 100% |
| 3 | 6.25% | 16.38% | 69.93% | 100% |
| 4 | 12.5% | 22.63% | 86.59% | 100% |
| 5 (selected) | 12.5% | 22.63% | 90.19% | 100% |

The pre-RL fixed-seed simulator baseline was 10/10 success with mean return
0.9470. All three RL-epoch evaluations retained exactly the same held-out
scores: EAI goal success 4/32, EAI goal recall 22.63%, VLABench success
10/10, VLABench return 0.9470, 100% valid/executable, and zero evaluation IK
failures or truncations. Thus Joint RL did not improve either domain's
held-out end-to-end metric.

The retained checkpoint is the first RL epoch:

`test_regr/JointEmbodiedAgentInterface/checkpoints/full_qwen3vl8b_20260920/joint_stage2_best.pt`

| RL epoch | EAI sampled-training success | VLABench training success (80 rollouts) | VLABench mean training return | IK failure / recovery events | IK-truncated episodes |
|---:|---:|---:|---:|---:|---:|
| 1 (retained) | 2.5% | 81.25% | 0.7970 | 3,438 / 1,586 | 0 |
| 2 | 2.5% | 91.25% | 0.8691 | 497 / 242 | 0 |
| 3 | 5.0% | 92.50% | 0.8782 | 0 / 0 | 0 |

Epochs 2 and 3 had stronger training-rollout results, but checkpoint selection
uses held-out balanced domain performance, not training return. Their held-out
EAI and VLABench scores tied epoch 1, so the earliest eligible epoch remained
selected. The growing VLABench training return cannot substitute for the
unchanged EAI evaluation or establish a held-out VLABench gain.

## Comparison with Evaluation Report 2

Report 2 used Qwen3-8B for standalone EAI and Qwen2.5-VL-3B for VLABench and
Joint. Report 3 uses Qwen3-VL-8B for all three.

| Retained result | Report 2 | Report 3 | Difference |
|---|---:|---:|---:|
| EAI-only goal success | 89.8% | 85.2% | -4.6 points |
| VLABench fixed-seed success | 100% | 100% | 0 points |
| Joint EAI goal success | 12.5% | 12.5% | 0 points |
| Joint VLABench fixed-seed success | 100% | 100% | 0 points |
| Joint VLABench Stage 1 exact graph | 62.63% | 90.19% | +27.56 points |

The common Qwen3-VL backbone substantially improved Joint VLABench graph-plan
imitation on this seed, but that gain did not change simulator success because
the fixed evaluation was already saturated. It also did not repair the Joint
EAI branch. The standalone EAI retained score was lower than Report 2, so the
common backbone is not uniformly better across the three workflows.

## Checkpoint verification and limitations

- The EAI Stage 1 and final Stage 2 files, VLABench `agent_rl_best.pt`, and
  Joint `joint_stage2_best.pt` each deserialized successfully with
  `torch.load(..., map_location="cpu", mmap=True)`. This checks file integrity,
  not reproduction of the reported scores.
- The retained checkpoint sizes are approximately 17 GB for each EAI file,
  1.3 GB for VLABench, and 1.5 GB for Joint.
- The simulator evaluation uses one fixed rollout per task. Its persistent
  10/10 ceiling is insufficient for robustness claims. Evaluation should be
  repeated across multiple seeds, objects, and scenes.
- Controller diagnostics record task-specific pick, grasp, lift, place,
  insert, pull, pour, or press assistance. Simulator success measures this
  assisted execution pipeline, not the learned controller alone. Camera
  mapping was configured from dataset aliases but remained logged as
  `unverified`.
- VLABench logged two training IK truncations and one QACC physics warning.
  Joint logged no IK truncations and three QACC warnings. Selected held-out
  evaluations had no IK failures or truncations.
- EAI-only scored 88 validation examples while Joint scored 32. Their
  percentages are not a controlled paired comparison. A proper transfer test
  must score both frozen models on the same EAI examples.
- These are one-seed experiments. Report per-task distributions and
  uncertainty over at least three seeds before making a model-ranking claim.
