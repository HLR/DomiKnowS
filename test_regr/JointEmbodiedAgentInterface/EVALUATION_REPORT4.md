# EAI and VLABench Training Evaluation Report 4

## Evaluation objective and changes from Report 3

This report evaluates three fresh full-data, two-stage runs with
`Qwen/Qwen3-VL-8B-Instruct`: EAI only, VLABench only, and Joint
EAI/VLABench. End-to-end goal or simulator success is the primary metric.
Exact-match, reward, return, constraint, IK, assistance, and PPO measurements
are supporting diagnostics.

Report 4 implements the main evaluation changes recommended in Report 3:

- standalone and Joint EAI now use the same canonical 350/88 train/validation
  split, and all 88 validation examples are scored;
- VLABench evaluation uses three fixed-seed rollouts per task, 30 episodes
  total, instead of one rollout per task;
- simulator summaries expose assistance, IK recovery/truncation, PPO
  approximate KL, accepted parameter changes, and rollback counts;
- Qwen3-8B and Qwen3-VL-8B are also compared on the same 88 frozen EAI
  examples, including missed-goal/action inspection.

The runs used seed 43 and executed on `gpu2.ihmc.us` from commit
`75918bd818e951e04bbe964d3a0b75d99e0ff7e4`. The shared model was loaded
from `/home/auszok/models/Qwen/Qwen3-VL-8B-Instruct`. Each process used a
separate new output path and its assigned physical GPU:

| Run | GPU | Training | Log |
|---|---:|---|---|
| EAI only | 4 | 5 supervised + 5 RL epochs | `test_regr/EmbodiedAgentInterface/results/full_qwen3vl8b_report4_seed43_20260921.log` |
| VLABench only | 5 | 3 supervised + 20,000 BC steps + 3 RL epochs | `test_regr/VLABenchAgentInterface/results/full_qwen3vl8b_report4_seed43_20260921.log` |
| Joint | 6 | 5 supervised + 20,000 BC steps + 3 RL epochs | `test_regr/JointEmbodiedAgentInterface/results/full_qwen3vl8b_report4_seed43_20260921.log` |

EAI used all 438 examples (350/88). VLABench used 4,500 planning examples
(3,600/450/450) and all ten controller tasks
(459,675/57,225/58,201 train/validation/test windows). All three runs
completed. The Joint Stage 1 EAI exploration gate initially rejected the
model; Stage 2 was then resumed from the completed epoch-5 Stage 1 checkpoint
with the EAI gate thresholds explicitly relaxed to zero. Joint Stage 2 is
therefore a diagnostic continuation, not evidence that its supervised EAI
branch passed the normal readiness gate.

## Primary end-to-end comparison

| Training setting | Stage 1 / pre-RL | Retained Stage 2 | Change |
|---|---:|---:|---:|
| EAI only: goal success on 88 | 73/88 (83.0%) | 78/88 (88.6%) | +5 examples, +5.7 points |
| VLABench only: simulator success on 30 | 25/30 (83.3%) | 25/30 (83.3%) | 0 |
| Joint: EAI goal success on 88 | 0/88 (0.0%) | 1/88 (1.1%) | +1 example, +1.1 points |
| Joint: VLABench simulator success on 30 | 25/30 (83.3%) | 25/30 (83.3%) | 0 |

The EAI-only result shows a semantic Stage 2 gain on this seed. Neither
VLABench setting improves held-out simulator success or return: the same
fixed-seed evaluation remains at 25/30 success and 0.7948 mean return before
and after RL. The Joint EAI branch remains a failed transfer result despite
the single extra success at its retained Stage 2 checkpoint.

EAI and VLABench rewards are domain-local. They are not added or averaged;
only their alternating gradients affect the shared Joint backbone.

## EAI-only result

Stage 1 selected supervised epoch 1. Stage 2 selected RL epoch 3.

| Metric on 88 examples | Stage 1 | Retained Stage 2 | Change |
|---|---:|---:|---:|
| Goal success | 83.0% | 88.6% | +5.7 points |
| Goal recall | 87.6% | 90.9% | +3.3 points |
| Temporal progress | 89.2% | 93.8% | +4.6 points |
| Positive-reward rate | 90.9% | 95.5% | +4.6 points |
| Aggregate reward | 0.852 | 0.895 | +0.043 |
| Exact sequence | 23.9% | 21.6% | -2.3 points |
| Token accuracy | 52.2% | 55.1% | +2.9 points |
| DFA validity | 100% | 100% | 0 |

| Stage | Epoch | Success | Recall | Temporal progress | Exact sequence | Reward |
|---|---:|---:|---:|---:|---:|---:|
| Supervised | 1 (selected) | 83.0% | 87.6% | 89.2% | 23.9% | 0.852 |
| Supervised | 2 | 70.5% | 79.8% | 83.0% | 23.9% | 0.770 |
| Supervised | 3 | 77.3% | 83.5% | 88.6% | 27.3% | 0.815 |
| Supervised | 4 | 75.0% | 84.3% | 85.2% | 19.3% | 0.788 |
| Supervised | 5 | 76.1% | 83.9% | 86.9% | 23.9% | 0.810 |
| RL | 1 | 86.4% | 89.5% | 91.5% | 19.3% | 0.879 |
| RL | 2 | 88.6% | 89.6% | 92.0% | 17.0% | 0.883 |
| RL | 3 (selected) | 88.6% | 90.9% | 93.8% | 21.6% | 0.895 |
| RL | 4 | 80.7% | 87.7% | 89.2% | 22.7% | 0.846 |
| RL | 5 | 79.5% | 87.4% | 87.5% | 25.0% | 0.836 |

The five-example gain is semantic rather than reference-imitation gain:
exact sequence declined while goal success, recall, and temporal progress
increased. The late-epoch regression also confirms that retaining the best
semantic checkpoint matters.

Retained model:
`test_regr/EmbodiedAgentInterface/models/eai_full_qwen3vl8b_report4_seed43_20260921.pth`.

## Why Qwen3-VL was worse than Qwen3 on EAI

The paired diagnostic scores the prior frozen Qwen3-8B and Qwen3-VL-8B EAI
models on the identical canonical 88 examples:

| Frozen model | Success | Goal recall | Mean predicted actions |
|---|---:|---:|---:|
| Qwen3-8B | 79/88 (89.8%) | 91.93% | 14.99 |
| Qwen3-VL-8B | 75/88 (85.2%) | 89.94% | 15.03 |

Both models succeed on 75 examples. Qwen3-8B alone succeeds on rows 351, 420,
431, and 435; Qwen3-VL-8B has no exclusive successes. The regression is not
caused by shorter generation. It is concentrated in action/entity grounding:

- row 351 misses the required left-hand hold of the novel;
- row 420 omits the coffee-maker plug-in action;
- row 431 misbinds the dish-soap and plate entities across grab/wash/rinse;
- row 435 substitutes `water_glass` for `drinking_glass` and omits the
  required soap grab and glass wash.

Thus the larger vision-language model is not automatically a better
text-only EAI planner. Its multimodal pretraining and different language
representation do not guarantee stronger exact action/entity binding after
the same low-rank adaptation. Report 4's new seed improves Qwen3-VL from
75/88 to 78/88, only one example behind the earlier Qwen3-8B result. This
large seed effect is another reason not to infer a stable architecture
ranking from one run.

Paired evidence:
`test_regr/JointEmbodiedAgentInterface/results/paired_eai_qwen3_8b_report4.json`
and
`test_regr/JointEmbodiedAgentInterface/results/paired_eai_qwen3vl_8b_report4.json`.

## VLABench-only result

The selected supervised planner reached 13.11% exact graph match on 450
examples. Controller BC warm-up ended at loss 0.1859 after 20,000 steps.
The pre-RL 30-episode simulator baseline was already 25/30 success with mean
return 0.7948.

| RL epoch | Training success (80) | Mean return | IK failures / recoveries | IK truncations | PPO rollbacks |
|---:|---:|---:|---:|---:|---:|
| 1 (retained) | 80.0% | 0.7826 | 1,993 / 753 | 1/80 | 10/10 |
| 2 | 87.5% | 0.8481 | 677 / 339 | 0/80 | 10/10 |
| 3 | 87.5% | 0.8396 | 166 / 85 | 0/80 | 10/10 |

All 30 standalone PPO attempts were rolled back: 29 by early-stop KL checks
and one by the final-drift check. Approximate KL values were above the 0.03
limit (observed roughly 0.058 to 0.232). Therefore the standalone controller
did not receive an accepted PPO parameter update. The better epoch-2/3
training rollouts cannot be attributed to retained controller PPO learning;
rollout stochasticity, task instances, planner updates, and execution
assistance remain alternative explanations.

Every Stage 1 and RL fixed-seed evaluation tied exactly at 25/30 success,
0.7948 return, and 133.7 mean steps. The retained file points to epoch 1
because held-out selection tied:
`test_regr/VLABenchAgentInterface/checkpoints/full_qwen3vl8b_report4_seed43_20260921/agent_rl_best.pt`.

## Joint result

The selected Stage 1/warm-up checkpoint had 0/88 EAI success, 3.13% goal
recall, 0% exact sequence, and 54.96% VLABench exact graph match. Controller
warm-up loss was 0.1874. This is why the normal EAI exploration gate stopped
the original run. The diagnostic resume preserved Stage 1 and relaxed only
the gate thresholds.

| Joint RL epoch | EAI eval success | EAI recall | VLABench training success (80) | Training return | PPO accepted / attempted | Actor delta L2 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0/88 | 3.13% | 82.5% | 0.8033 | 8/10 | 2.702 |
| 2 | 0/88 | 3.13% | 91.25% | 0.8703 | 10/10 | 2.822 |
| 3 (retained) | 1/88 | 4.26% | 85.0% | 0.8174 | 10/10 | 3.078 |

Mean per-attempt approximate KL was 0.00986, 0.00713, and 0.00647 for the
three epochs. Epoch 1 rolled back two updates; epochs 2 and 3 accepted all
ten. Unlike standalone VLABench, Joint therefore provides direct evidence
that PPO changed controller parameters: 28/30 attempted rounds were accepted,
all accepted rounds report nonzero parameter change, and the accumulated
actor-delta norms are nonzero.

This is evidence of controller optimization, not evidence of held-out
controller improvement. All three Joint checkpoints still scored the same
25/30 fixed-seed simulator success and 0.7948 return as the pre-RL baseline.
The retained epoch 3 is selected because its EAI evaluation improves from
zero to one success while VLABench remains tied.

Retained model:
`test_regr/JointEmbodiedAgentInterface/checkpoints/full_qwen3vl8b_report4_seed43_20260921/joint_stage2_best.pt`.

## Assistance and execution evidence

The 30-episode fixed-seed VLABench evaluation recorded:

| Measurement | Value |
|---|---:|
| Simulator success | 25/30 (83.3%) |
| Mean return | 0.7948 |
| Valid/executable | 29/30 (96.7%) |
| IK failures / recoveries | 178 / 115 |
| IK-truncated episodes | 1/30 |
| Task-assist steps | 4,141 |
| Episodes with assistance | 29/30 (96.7%) |
| Unassisted successes | 0/30 |

The five failures are concentrated in `insert_flower`, `select_poker`,
and `select_toy`; the latter contains all 178 evaluation IK failures and
115 recoveries. Most importantly, every successful evaluation episode used
task-specific assistance. The 83.3% figure is therefore success of the
assisted planner/controller/execution pipeline, not success of the learned
controller in isolation.

Report 4 improves observability but does not yet provide causal unassisted
controller evidence. A future evaluation must disable pick/grasp/place/
insert/pull/pour/lift/press assistance and compare the same fixed seeds before
and after PPO.

## Checkpoint verification

The following retained artifacts deserialized successfully with
`torch.load(..., map_location="cpu", weights_only=False, mmap=True)`:

- EAI final Stage 2 model;
- VLABench Stage 1 evaluated checkpoint and `agent_rl_best.pt`;
- Joint controller-warm-up checkpoint and `joint_stage2_best.pt`.

The checkpoint metadata confirms VLABench retained RL epoch 1 and Joint
retained RL epoch 3. Deserialization verifies file integrity and metadata
availability, not independent reproduction of the reported metrics.

## Conclusions and next actions

1. EAI-only RL again improves end-to-end semantic success, reaching 78/88 on
   seed 43, but the gain remains seed-sensitive and exact-match does not
   improve.
2. The controlled same-88 comparison explains the earlier Qwen3-VL deficit
   as action/entity grounding regressions, not decoding length or evaluation
   mismatch.
3. Standalone VLABench produces no accepted controller PPO updates. Its KL
   gate is doing useful safety work, but the controller learning rate,
   minibatching, or update schedule must be reduced before another run.
4. Joint PPO demonstrably changes controller parameters, yet held-out
   simulator success and return do not improve.
5. VLABench success remains dominated by execution assistance. The next
   decisive experiment is a pre/post-PPO no-assistance evaluation over
   multiple seeds, followed by at least three training seeds for uncertainty.
6. Joint EAI transfer remains the largest failure: 1/88 retained success.
   It should not enter expensive simulator RL until its shared-backbone EAI
   branch passes the normal exploration gate.
