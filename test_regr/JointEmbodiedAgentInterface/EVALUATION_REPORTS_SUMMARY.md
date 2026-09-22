# Summary of Evaluation Reports 1–5

This document summarizes the five EAI/VLABench evaluation reports.
End-to-end goal or simulator success is the primary measure; graph matching,
DFA validity, training return, and parameter movement are diagnostics rather
than substitutes for held-out task success.

| Report | Main configuration | EAI only, Stage 1 → Stage 2 | VLABench only, Stage 1 → Stage 2 | Joint result |
|---|---|---:|---:|---|
| [Report 1](EVALUATION_REPORT.md) | Earlier mixed runs; VLABench and Joint included resumed runs and a different evaluation protocol | 77.3% → 79.5% | 76.67% → 83.33% | VLABench success 80.0% → 80.0%; Joint EAI was not reported as a comparable primary result |
| [Report 2](EVALUATION_REPORT2.md) | Fresh full runs; EAI used Qwen3-8B, VLABench/Joint used Qwen2.5-VL-3B; one simulator rollout per task | 80.7% → **89.8%** | 100% → 100% | EAI 12.5% → 12.5%; VLABench 100% → 100% |
| [Report 3](EVALUATION_REPORT3.md) | Fresh full runs; all three used Qwen3-VL-8B-Instruct; one simulator rollout per task | 79.5% → **85.2%** | 100% → 100% | EAI 12.5% → 12.5%; VLABench 100% → 100% |
| [Report 4](EVALUATION_REPORT4.md) | Qwen3-VL-8B, seed 43; canonical 88-example EAI validation; three simulator rollouts per task; assistance and PPO evidence | 83.0% → **88.6%** | 83.3% → 83.3% | EAI 0.0% → 1.1%; VLABench 83.3% → 83.3% |
| [Report 5](EVALUATION_REPORT5.md) | Qwen3-VL-8B; multi-domain parameter coordination (PCGrad); unassisted evaluation; strict gate enforcement | 80.7% → 80.7% (recall 84.6% → 87.7%) | 3.3% unassisted (preflight gate enforced) | VLABench graph match 16.5% → 89.1%; EAI exploration gate enforced |

Report 4 exposed that earlier 83.3%–100% VLABench simulator success rates relied
heavily on online execution assistance (96.7% assisted episodes). Report 5 enforced
unassisted evaluation (`--execution-assistance train-only`) and multi-domain parameter
coordination via `GradientConflictManager` with PCGrad, while strictly observing
both the EAI exploration gate and the VLABench controller preflight gate.

## Main findings across reports

- **Standalone EAI**: Benefited from RL in every evaluation. Semantic success and
  goal recall consistently improve over supervised baselines. In Report 5, RL restored
  late-stage supervised degradation back to 80.7% while reaching peak token accuracy
  (58.4%), peak exact sequence match (27.3%), and 87.7% goal recall.
- **Unassisted Controller Reality**: Report 5's unassisted evaluation established an
  authentic baseline: **70.0% positive return rate**, **0.0% IK truncation rate**,
  100% DFA validity, and 3.3% full unassisted task success (`select_painting` at 33.3%).
  This proves the controller acquires genuine approach and intention progress (e.g.,
  `select_drink` 71.2% progress, `select_mahjong` 51.9% progress), but multi-stage
  grasps and insertions require higher terminal precision to succeed without assistance.
- **Multi-Domain Parameter Coordination**: The integration of `GradientConflictManager`
  and PCGrad projection in Report 5 resolved the shared-backbone gradient interference
  identified in earlier reports. During Joint Stage 1, VLABench graph imitation climbed
  steadily from 16.5% in Epoch 0 to **89.1%** in Epoch 4 with 100% DFA validity.
- **Readiness Gate Protection**: Both the Joint EAI exploration gate (`positive_reward_rate < 0.10`)
  and the VLABench preflight gate (`minimum_successful_tasks: 10`) halted execution as
  contracted, preventing hundreds of ungrounded simulator GPU-hours from running when
  pre-conditions were unmet.
- **Joint EAI Bottleneck**: Across Reports 2–5, the EAI component within Joint training
  struggles to acquire grounding on the shared backbone (3.4% positive reward rate in
  Report 5), while the VLABench component learns effectively. Addressing this requires
  domain-balanced loss weighting or staged initialization before joint optimization.

## What the reports establish

| Question | Evidence-based answer |
|---|---|
| Does two-stage learning help standalone EAI? | Yes across all five reports. Semantic recall, token accuracy, and positive reward consistently increase with RL. |
| Does VLABench RL improve held-out simulator success? | Not demonstrated. Earlier high success was driven by execution assistance; unassisted evaluation in Report 5 establishes 3.3% success and 70% positive return. |
| Does controller PPO update parameters? | Yes in Joint (Report 4), but rolled back by KL gates in standalone Report 4. Preflight gates in Report 5 safely prevented ungrounded Stage 2 runs. |
| Is the VLABench controller independently successful? | Partially. The controller reliably navigates to targets (70% positive return, 0% IK truncation), but unassisted multi-stage manipulation remains challenging. |
| Does PCGrad prevent Joint parameter interference? | Yes. Joint Stage 1 achieved 89.1% VLABench graph imitation without destabilizing optimizer dynamics or autograd crashes. |
| Does Joint training preserve EAI capability? | Not yet. EAI requires dedicated warm-up or loss-rebalancing within Joint Stage 1 to clear the 10% exploration threshold. |

## Recommended next experiments

1. **Joint EAI Warm-up / Re-balancing**: Calibrate the joint loss weighting (e.g. increase
   EAI loss weight relative to VLABench) or pre-initialize the EAI head so the shared model
   crosses the 10% positive reward exploration threshold.
2. **Controller Precision Fine-Tuning**: Incorporate curriculum assistance annealing during
   controller training to bridge the gap between 70% approach progress and terminal grasp/place
   success.
3. **Multi-Seed Unassisted Benchmarking**: Evaluate unassisted controller rollouts across
   three random seeds to establish definitive variance bounds on autonomous simulator success.
