# EAI and VLABench Training Evaluation Report 2

## Evaluation objective and provenance

This report compares supervised Stage 1 with the retained reinforcement-learning
checkpoint for three **fresh, full-data** runs: EAI only, VLABench only, and
Joint EAI/VLABench. End-to-end goal or simulator success is the primary metric;
sequence, graph, constraint, and reward scores are supporting diagnostics. The
runs were concurrent on `gpu2.ihmc.us`, from clean commit `75918bd8`, with no
`--resume` option and no reuse of the checkpoints in
[`EVALUATION_REPORT.md`](EVALUATION_REPORT.md). Each process exited with code 0.

| Run | Physical GPU | Training stages | Log |
|---|---:|---|---|
| EAI only | 4 | 5 supervised + 5 RL epochs | `test_regr/EmbodiedAgentInterface/results/full_gpu2_20260920.log` |
| VLABench only | 5 | 3 supervised epochs + 20,000 controller-BC steps + 3 RL epochs | `test_regr/VLABenchAgentInterface/results/full_gpu2_20260920.log` |
| Joint | 6 | 5 supervised epochs + 20,000 controller-BC steps + 3 RL epochs | `test_regr/JointEmbodiedAgentInterface/results/full_gpu2_20260920.log` |

Each process saw its assigned physical card as `cuda:0` via
`CUDA_VISIBLE_DEVICES`. EAI used Qwen3-8B/LoRA; VLABench and Joint used
Qwen2.5-VL-3B, 4-bit LoRA, and a SigLIP controller vision encoder. VLABench and
Joint used batch size 1. The EAI-only run used all 438 examples (350 training,
88 validation). Joint used the same 438 EAI examples in a 394/44 split and
evaluated 32 EAI validation examples because of its validation limit. The
VLABench planning corpus contained 4,500 examples (3,600/450/450), and the
control loaders indexed all ten tasks with 459,675/57,225/58,201
train/validation/test windows. VLABench simulator evaluation used fixed seed
`100042` and **one rollout per task** (ten episodes), before RL and after each
RL epoch. No dataset limit was imposed, but the bounded 20,000-step controller
warm-ups did not necessarily visit every indexed control window. These are not
multi-seed reliability estimates.

## Primary end-to-end comparison

| Training setting / outcome | Selected Stage 1, without RL | Retained Stage 2, with RL | Change |
|---|---:|---:|---:|
| EAI only: benchmark-goal success | 71/88 (80.7%) | 79/88 (89.8%) | +9.1 percentage points |
| VLABench only: simulator success | 10/10 (100%) | 10/10 (100%) | 0 points |
| Joint: EAI benchmark-goal success | 4/32 (12.5%) | 4/32 (12.5%) | 0 points |
| Joint: VLABench simulator success | 10/10 (100%) | 10/10 (100%) | 0 points |

EAI success evaluates the benchmark `tl_goal` after simulated actions, not
reference-plan exact match. The EAI-only and Joint EAI percentages use
different validation subsets and sample sizes; they are **not** a paired
cross-model comparison. The two domains' rewards and success rates are never
combined into one metric. On the fixed-seed VLABench evaluation, both
supervised baselines were already at the observed ceiling; these runs do not
show an RL improvement in simulator success.

## EAI-only experiment

The supervised run restored epoch 1; RL restored epoch 5. Both the Stage 1
checkpoint and final two-stage checkpoint were produced under
`test_regr/EmbodiedAgentInterface/models/`:

- Without RL: `eai_full_gpu2_20260920.stage1.pth`
- With RL: `eai_full_gpu2_20260920.pth`

| Validation metric (88 examples) | Stage 1 selected | Stage 2 selected | Change |
|---|---:|---:|---:|
| Goal success | 80.7% | 89.8% | +9.1 points (8 examples) |
| Goal-state recall | 86.5% | 91.9% | +5.4 points |
| Temporal progress | 89.8% | 94.9% | +5.1 points |
| Positive-reward rate | 94.3% | 96.6% | +2.3 points |
| Aggregate RL reward | 0.834 | 0.909 | +0.075 |
| Exact action sequence | 17.0% | 17.0% | 0 points |
| Token accuracy | 49.9% | 53.7% | +3.8 points |
| DFA validity | 100% | 100% | 0 points |
| Applicable world-constraint score | 1.000 | 0.976 | -0.024 |

| Stage | Epoch | Goal success | Goal recall | Temporal progress | Exact sequence | Reward |
|---|---:|---:|---:|---:|---:|---:|
| Supervised | 1 (selected) | 80.7% | 86.5% | 89.8% | 17.0% | 0.834 |
| Supervised | 2 | 70.5% | 79.7% | 89.2% | 19.3% | 0.771 |
| Supervised | 3 | 71.6% | 78.4% | 84.7% | 17.0% | 0.746 |
| Supervised | 4 | 75.0% | 83.2% | 86.4% | 18.2% | 0.799 |
| Supervised | 5 | 78.4% | 83.4% | 86.4% | 27.3% | 0.812 |
| RL | 1 | 80.7% | 87.5% | 88.1% | 20.5% | 0.837 |
| RL | 2 | 85.2% | 89.9% | 89.8% | 22.7% | 0.869 |
| RL | 3 | 86.4% | 91.1% | 90.9% | 19.3% | 0.879 |
| RL | 4 | 87.5% | 90.8% | 92.0% | 20.5% | 0.894 |
| RL | 5 (selected) | 89.8% | 91.9% | 94.9% | 17.0% | 0.909 |

The eight-example success gain is encouraging on this one seed, but exact
sequence accuracy did not improve, and the applicable world-constraint score
fell slightly. A repeated-seed, paired evaluation is needed before calling the
gain robust.

## VLABench-only experiment

The supervised checkpoint is
`test_regr/VLABenchAgentInterface/checkpoints/full_gpu2_20260920/agent_stage1_evaluated.pt`.
Its planner validation exact-graph match was 11.56% on 450 examples, with 100%
graph/DFA validity. The 20,000-step controller-BC checkpoint reported pose MAE
0.0441 on 512 validation actions. Its reported 100% gripper accuracy is not
evidence for both gripper classes: all 512 scored targets were **open**.

The pre-RL supervised simulator baseline, all three RL-epoch fixed-seed
evaluations, and retained `agent_rl_best.pt` each scored 10/10 success, mean
return 0.9470, mean 103.9 steps, 100% valid/executable, and no IK failures or
truncations in those ten evaluation episodes. The retained best checkpoint is
the first RL epoch (`agent_rl_epoch_000.pt`), not the final epoch.

Training-rollout results tell a less saturated story. Each RL epoch used 80
rollouts (eight per task); these are **not** the fixed-seed evaluation results:

| RL epoch | Training success | Mean training return | IK-truncated episodes | IK failure / recovery events | PPO update rounds rolled back |
|---:|---:|---:|---:|---:|---:|
| 1 (retained) | 86.25% | 0.8413 | 0/80 | 0 / 0 | 9/10 |
| 2 | 85.00% | 0.8234 | 1/80 | 1,149 / 446 | 10/10 |
| 3 | 83.75% | 0.8049 | 1/80 | 1,765 / 593 | 10/10 |

The two truncated episodes were `select_toy` rollouts stopped after three
rejected inverse-kinematics chunks. Failure/recovery figures count IK events,
not failed episodes. The high PPO rollback count and falling training success
do not support a claim that RL improved this simulator policy, despite the
unchanged 10/10 fixed-seed evaluation.

## Joint EAI/VLABench experiment

The selected supervised model after controller warm-up is
`test_regr/JointEmbodiedAgentInterface/checkpoints/full_gpu2_20260920/joint_controller_warmup.pt`.
Its EAI validation goal success was 4/32 (12.5%), goal recall 22.63%, and
exact sequence 0%. Its VLABench planner exact-graph match was 62.63% on 450
examples, with 100% graph validity. Controller validation pose MAE was 0.0416
on 512 actions; again, all scored gripper targets were open. The pre-RL
fixed-seed simulator baseline was 10/10 success, mean return 0.9470.

All three RL epochs retained the same fixed-seed scores: EAI goal success
4/32, EAI goal recall 22.63%, VLABench success 10/10, VLABench mean return
0.9470, 100% valid/executable, and zero evaluation IK failures or truncations.
The selected checkpoint,
`test_regr/JointEmbodiedAgentInterface/checkpoints/full_gpu2_20260920/joint_stage2_best.pt`,
comes from the first RL epoch (`joint_stage2_epoch_000.pt`). The final epoch
checkpoint was also saved as `joint_stage2_epoch_002.pt`.

| RL epoch | EAI sampled-training success | VLABench training success (80 rollouts) | VLABench mean training return | IK failure / recovery events | IK-truncated episodes |
|---:|---:|---:|---:|---:|---:|
| 1 (retained) | 5.0% | 83.75% | 0.8141 | 2,106 / 838 | 0 |
| 2 | 7.5% | 83.75% | 0.8191 | 2,835 / 1,236 | 0 |
| 3 | 0.0% | 90.00% | 0.8558 | 0 / 0 | 0 |

Joint RL did not improve either domain's **fixed-seed** end-to-end success in
this run. Its sampled training rewards and simulator outcomes are separate
domain-local measurements; a higher VLABench training return in epoch 3 does
not establish an EAI improvement or override the retained-checkpoint result.

## Verification, limitations, and next evaluation

- The three processes exited with code 0. The EAI final model, VLABench
  `agent_rl_best.pt`, and Joint `joint_stage2_best.pt` were each deserialized
  successfully with `torch.load(..., map_location="cpu")`. This verifies file
  loadability, not independent reproduction of the scores.
- The ten VLABench evaluation episodes use one fixed seed and one rollout per
  task. Perfect 10/10 results at Stage 1 and every RL epoch cannot establish
  robustness or measure improvement above this evaluation ceiling. Test more
  seeds, objects, scenes, and unassisted execution conditions.
- Rollout diagnostics record task-specific pick/grasp/place/pull assistance.
  The simulator success numbers evaluate this **assisted execution pipeline**;
  they should not be attributed solely to the learned controller. Camera
  mapping was configured by dataset aliases but logged as `unverified`.
- During training, VLABench logged one physics-instability (`QACC`) warning
  and two IK-truncated `select_toy` episodes; Joint logged three QACC warnings.
  The selected ten-episode evaluations report zero IK failures or truncations;
  the QACC log lines were not independently attributed to training versus
  evaluation.
- The Joint EAI validation limit of 32 differs from EAI-only's 88; direct
  comparison of their percentages is not a controlled transfer experiment.
  Compare both models on the same held-out EAI examples before making a
  cross-domain claim.
- Each setting used one training seed. Repeat with at least three seeds and
  report per-task success distributions and uncertainty, not only aggregate
  success or planner validity.
