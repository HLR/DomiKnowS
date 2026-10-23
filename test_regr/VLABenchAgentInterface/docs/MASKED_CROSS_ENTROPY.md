# Masked cross-entropy learning (Stage 1 of the Joint planner)

Stage 1 of the Joint agent trains the planner with teacher forcing. By default it now minimises **cross entropy
under the DFA mask**: at each step the softmax is restricted to the labels the DFA allows there. This document
describes the method, where each part is implemented, and what it does not do. Measured results are in
[DFA_LOSS_SMOKE.md](DFA_LOSS_SMOKE.md).

## 1. The idea

Let `p(.)` be the planner's softmax over all labels at step `t` (given the instruction, the images and the target
prefix), `y_t` the gold label, and `A_t` the set of labels the DFA allows at that step along the gold plan.

| Loss | Per-step value | What it trains |
| --- | --- | --- |
| Plain cross entropy | `-log p(y_t)` | `y_t` against every other label, allowed or not |
| **Masked cross entropy** | `-log [ p(y_t) / P(A_t) ]  =  -log p(y_t) + log P(A_t)`, with `P(A_t) = sum of p(a) over a in A_t` | `y_t` against the other *allowed* labels; the probability the model puts outside `A_t` is not penalised |

The masked form is exactly the distribution a DFA-masked decoder samples from (logits outside `A_t` set to
`-inf`, then softmax), so training and masked decoding use the same distribution. Because `y_t` is in `A_t`,
`P(A_t) >= p(y_t)` and the masked loss never exceeds plain cross entropy.

The earlier "allowed-mass" term, `-log P(A_t)`, is bounded by `-log p(y_t)` on the same steps, so on teacher-forced
data it adds no signal beyond plain cross entropy (see the smoke report). It is kept as an option and is off by
default.

**Fallback.** If the gold label is not in `A_t` (the gold plan disagrees with the constraints at that step), the
step keeps the plain cross entropy, and the step is counted. The count is reported as `masked_ce_misses` (the share
of scored steps whose gold label the DFA disallows).

## 2. Data flow and where each step lives

1. **Gold labels are valid plans.** Plans are serialised to labels (`skill:*`, `arg:*`, `obj:*`, end label) by
   [`plan_to_tokens`](../plan_codec.py#L183) and padded with the end label in
   [`prepare_planner_program_examples`](../training.py#L188).
2. **One planner forward per item** produces logits for every step:
   [`sequence_logits`](../../JointEmbodiedAgentInterface/models.py#L444) (the plain-cross-entropy path is
   [`supervised_loss`](../../JointEmbodiedAgentInterface/models.py#L504), the reference for what the masked loss
   replaces). With the primal-dual constraint loss on, the logits are captured from the same forward in
   [`JointPlannerDomainView.forward`](../../JointEmbodiedAgentInterface/models.py#L815).
3. **Loss assembly** is [`JointSolverPOIProgram._domain_loss`](../../JointEmbodiedAgentInterface/program.py#L249):
   - keep the positions up to and including the first end label ([L255](../../JointEmbodiedAgentInterface/program.py#L255));
   - get the logits ([L256-L268](../../JointEmbodiedAgentInterface/program.py#L256));
   - bind the DFA to the item with `runtime.dfa_for` ([L269](../../JointEmbodiedAgentInterface/program.py#L269);
     implemented in [`utils/generation_runtime.py:94`](../../JointEmbodiedAgentInterface/utils/generation_runtime.py#L94),
     which binds VLABench through [`bind_vlabench_contextual_dfa`](../generation_runtime.py#L123) and EAI through
     [`bind_eai_policy`](../../EmbodiedAgentInterface/utils/constraint_compiler.py#L371));
   - compute the allowed sets along the gold path ([L277](../../JointEmbodiedAgentInterface/program.py#L277));
   - apply the masked loss ([L280-L283](../../JointEmbodiedAgentInterface/program.py#L280));
   - combine with the optional allowed-mass and primal-dual terms and report `masked_ce_misses`
     ([L286-L294](../../JointEmbodiedAgentInterface/program.py#L286)).
4. **The three loss helpers** are in the learners package:
   [`allowed_label_sets`](../../../domiknows/generation/learners/common/losses.py#L140) walks the DFA along the gold
   path (the state advances at every step; `lookahead` chooses between "valid now" and "valid and still completable"),
   [`allowed_log_mass`](../../../domiknows/generation/learners/common/losses.py#L170) gives `log P(A_t)`, and
   [`masked_cross_entropy`](../../../domiknows/generation/learners/common/losses.py#L179) returns the loss and the
   miss count (the restriction is the `logsumexp` at
   [L199](../../../domiknows/generation/learners/common/losses.py#L199), the fallback is
   [L195-L197](../../../domiknows/generation/learners/common/losses.py#L195)). They are exported from
   `domiknows.generation.learners`.
5. **Where it is called per round:** the Stage 1 round calls `_domain_loss` for each domain
   ([L361](../../JointEmbodiedAgentInterface/program.py#L361) in the single-domain step, and
   [L477](../../JointEmbodiedAgentInterface/program.py#L477) / [L482](../../JointEmbodiedAgentInterface/program.py#L482)
   for EAI and VLABench in `train_round`), then the two domain gradients are combined with PCGrad as before.

## 3. Configuration

Config class [`DFALossConfig`](../../JointEmbodiedAgentInterface/program.py#L46):
`masked_ce_domains` ([L58](../../JointEmbodiedAgentInterface/program.py#L58)) and `mass_weight`
([L59](../../JointEmbodiedAgentInterface/program.py#L59)). Built from the command line in
[`_dfa_loss_config`](../../JointEmbodiedAgentInterface/main.py#L307) and passed at
[main.py:455](../../JointEmbodiedAgentInterface/main.py#L455). Flags of `train-agent`
([main.py:1030-1045](../../JointEmbodiedAgentInterface/main.py#L1030)):

| Flag | Default | Meaning |
| --- | --- | --- |
| `--stage1-dfa-loss on/off` | on | off = plain cross entropy only |
| `--stage1-masked-ce-domains` | vlabench | domains trained on the masked loss described here; the others keep plain cross entropy; `none` disables it |
| `--stage1-dfa-mass-weight` | 0.0 | weight of the (redundant) allowed-mass term |
| `--stage1-dfa-domains` | eai,vlabench | domains trained on the DFA |
| `--stage1-dfa-lookahead-domains` | vlabench | domains whose sets also require a reachable completion (a search that costs seconds per EAI item) |
| `--stage1-pmd-domains` | vlabench | domains that also get the primal-dual constraint loss (about ten times slower; `none` disables it) |

## 4. What it does and does not give

It is a per-domain choice. VLABench uses it by default: quality under the mask rose in the smoke test. EAI keeps plain
cross entropy by default because the smoke test showed no measured gain there while its unmasked probability halved
(see [DFA_LOSS_SMOKE.md](DFA_LOSS_SMOKE.md) and the EAI comparison below it).

- It trains the distribution used when decoding through the DFA, which is how the planner is used.
- It does **not** make the planner mask-independent: the probability on valid labels without the mask is lower than
  with plain cross entropy (0.44 vs 0.83 on EAI, 0.035 vs 0.16 on VLABench after 100 rounds), because outside
  probability is no longer penalised. Free decoding without the DFA is therefore not expected to work.
- A high `masked_ce_misses` would mean the gold plans disagree with the active constraints, which is worth checking
  before trusting either.

Diagnostic for the other direction: `dfa_mass_report`
([program.py:297](../../JointEmbodiedAgentInterface/program.py#L297), logged per epoch at
[main.py:482](../../JointEmbodiedAgentInterface/main.py#L482)) measures the unmasked probability on valid labels.

## 5. Tests

- Helpers: [`test_masked_cross_entropy_restricts_the_softmax_and_falls_back_when_the_target_is_disallowed`](../../generation/test_automata_heads.py#L203)
  (hand-computed values, fallback and miss count) and
  [`test_allowed_label_sets_follow_the_path_and_respect_score_steps`](../../generation/test_automata_heads.py#L222).
- Program: [`test_masked_cross_entropy_is_never_above_the_full_one_and_reports_disagreements`](../../JointEmbodiedAgentInterface/test/test_stage1_dfa_loss.py#L143) and
  [`test_masked_cross_entropy_applies_only_to_the_listed_domains`](../../JointEmbodiedAgentInterface/test/test_stage1_dfa_loss.py#L157).

## 6. Example

```bash
python -u -m test_regr.JointEmbodiedAgentInterface.main train-agent --two-stage \
  --stage1-masked-ce-domains vlabench --stage1-pmd-domains none ...   # masked cross entropy on VLABench only, no primal-dual term
python -u -m test_regr.JointEmbodiedAgentInterface.main train-agent --two-stage \
  --stage1-dfa-loss off ...                             # plain cross entropy, for comparison
```

Line numbers refer to the commit that introduced `--stage1-masked-ce-domains` and its successors until those files change.
