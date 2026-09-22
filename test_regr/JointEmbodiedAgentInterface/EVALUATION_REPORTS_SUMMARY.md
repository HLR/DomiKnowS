# Summary of Evaluation Reports 1–4

This document summarizes the four EAI/VLABench evaluation reports.
End-to-end goal or simulator success is the primary measure; graph matching,
DFA validity, training return, and parameter movement are diagnostics rather
than substitutes for held-out task success.

| Report | Main configuration | EAI only, Stage 1 → Stage 2 | VLABench only, Stage 1 → Stage 2 | Joint result |
|---|---|---:|---:|---|
| [Report 1](EVALUATION_REPORT.md) | Earlier mixed runs; VLABench and Joint included resumed runs and a different evaluation protocol | 77.3% → 79.5% | 76.67% → 83.33% | VLABench success 80.0% → 80.0%; Joint EAI was not reported as a comparable primary result |
| [Report 2](EVALUATION_REPORT2.md) | Fresh full runs; EAI used Qwen3-8B, VLABench/Joint used Qwen2.5-VL-3B; one simulator rollout per task | 80.7% → **89.8%** | 100% → 100% | EAI 12.5% → 12.5%; VLABench 100% → 100% |
| [Report 3](EVALUATION_REPORT3.md) | Fresh full runs; all three used Qwen3-VL-8B-Instruct; one simulator rollout per task | 79.5% → **85.2%** | 100% → 100% | EAI 12.5% → 12.5%; VLABench 100% → 100% |
| [Report 4](EVALUATION_REPORT4.md) | Qwen3-VL-8B, seed 43; canonical 88-example EAI validation; three simulator rollouts per task; assistance and PPO evidence | 83.0% → **88.6%** | 83.3% → 83.3% | EAI 0.0% → 1.1%; VLABench 83.3% → 83.3% |

Report 4's VLABench percentages use 30 simulator episodes rather than the ten
episodes used by Reports 2 and 3, so the lower percentage is not a
like-for-like regression. Its Joint Stage 2 is also diagnostic: the normal
EAI exploration gate rejected Stage 1, after which Stage 2 was explicitly
resumed with the gate thresholds relaxed.

## Main findings

- Standalone EAI benefited from RL in every report. Report 2 has the highest
  retained score at 79/88 (89.8%). Report 4 reaches 78/88 (88.6%) on a new
  seed, improving by five examples over its own supervised checkpoint.
  Across reports, semantic success improves more consistently than exact
  action-sequence match.
- A paired same-88 diagnostic explains why the earlier Qwen3-VL model was
  worse than Qwen3-8B: Qwen3-8B scored 79/88, Qwen3-VL scored 75/88, and the
  four regressions were action/entity-grounding errors. Qwen3-VL had no
  exclusive successes and did not produce shorter plans. Report 4's 78/88
  Qwen3-VL result shows substantial seed sensitivity, so one run is
  insufficient for a stable backbone ranking.
- Reports 2 and 3 saturated the ten-episode VLABench evaluation at 10/10.
  Report 4's stronger 30-episode protocol scored 25/30 both before and after
  RL, with identical 0.7948 mean return. No report yet demonstrates a
  held-out VLABench improvement caused by RL.
- Report 4 makes the controller limitation explicit: the 30-episode
  evaluation used 4,141 task-assist steps, 29/30 episodes used assistance,
  and there were 0/30 unassisted successes. The reported 83.3% is success of
  the assisted planner/controller/execution pipeline, not the learned
  controller alone.
- Standalone VLABench still lacks controller-RL evidence. All 30 PPO attempts
  in Report 4 were rolled back by the KL safety gate, so later training
  returns cannot be attributed to accepted controller PPO updates.
- Joint PPO did update controller parameters in Report 4: 28/30 updates were
  accepted, with nonzero actor parameter deltas and mean approximate KL below
  the configured limit. Held-out VLABench success and return nevertheless
  stayed unchanged. Parameter movement is evidence of optimization, not
  evidence of improved control.
- Joint EAI remains the dominant failure. Using the same 88-example
  validation set as standalone EAI, Report 4 retained only 1/88 success and
  4.26% goal recall. This removes the earlier 32-versus-88 evaluation
  ambiguity and shows that the shared-backbone EAI branch is genuinely
  failing, not merely being scored on a different subset.
- The common Qwen3-VL backbone can improve Joint VLABench graph imitation
  while harming or failing to transfer EAI action/entity grounding. A stronger
  shared backbone alone is therefore insufficient; the domain heads,
  scheduling, and gradient interference need separate diagnosis.

## What the reports establish

| Question | Evidence-based answer |
|---|---|
| Does two-stage learning help standalone EAI? | Yes on all four reported runs, although the magnitude is seed-sensitive and exact sequence need not improve. |
| Does VLABench RL improve held-out simulator success? | Not demonstrated. Reports 2–3 were saturated; Report 4 stayed at 25/30 with unchanged return. |
| Does controller PPO actually update parameters? | Not in standalone Report 4, where 30/30 attempts rolled back. Yes in Joint Report 4, where 28/30 were accepted, but without a held-out gain. |
| Is the VLABench controller independently successful? | Not established. Report 4 recorded zero unassisted successes and extensive task-specific assistance. |
| Does Joint training preserve EAI capability? | No. The controlled 88-example Report 4 evaluation retained only one success and failed the normal exploration gate. |
| Is Qwen3-VL-8B better than Qwen3-8B for EAI? | Not established. The paired frozen-model comparison favors Qwen3-8B by four examples, while the new Qwen3-VL seed nearly closes the gap. |

## Recommended next experiments

1. Run pre/post-PPO VLABench evaluation with all task-specific execution
   assistance disabled, using the same seeds and at least three rollouts per
   task.
2. Reduce standalone controller PPO step size or update intensity until some
   updates pass the KL gate, then require both nonzero parameter change and a
   held-out no-assistance improvement.
3. Do not launch expensive Joint simulator RL unless the Joint EAI branch
   passes the normal exploration gate on all 88 canonical validation
   examples.
4. Diagnose Joint EAI gradient interference by comparing frozen-backbone,
   domain-specific LoRA, and shared-LoRA variants on the same EAI examples,
   especially the action/entity-binding failures.
5. Repeat the full comparison over at least three training seeds and report
   confidence intervals or per-seed counts before ranking backbones or
   training regimes.
