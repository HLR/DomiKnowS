# Joint_Operation

This document describes the current joint EAI/VLABench training lifecycle in
`main.py train-agent --two-stage`. One Qwen-VL backbone and LoRA adapter serve
two domain-specific label heads. Stage 1 combines supervised domain losses
with gradient conflict projection and controller behavior cloning. Stage 2
alternates EAI plan reinforcement learning and VLABench simulator updates,
keeping their rewards separate.

The structure follows [EAI_Operation.md](../../EmbodiedAgentInterface/docs/EAI_Operation.md)
and [VLABench_Operation.md](../../VLABenchAgentInterface/docs/VLABench_Operation.md).
The joint workflow has its own planner architecture, loss defaults, parameter
policies, gates, and checkpoint format. Historical reports and README examples
can describe earlier configurations; the implementation locations below are
the authority for this operational description.

## Operation at a glance

```text
EAI rows                       VLABench planning rows + control demonstrations
   |                              |
   +-- instruction, TL goal        +-- instruction, images, scene entities
   +-- reference action labels     +-- reference skill/argument/pointer labels
   +-- task context                +-- demonstration image/state/action windows
   |                              |
   +---------------+--------------+
                   |
                   v
One joint root graph + one shared Qwen/LoRA + two linear label heads
                   |
                   v
EAI-only supervised warm-up (default 350 steps)
                   |
                   v
Stage 1: paired supervised rounds
   +-- EAI full-vocabulary CE, weighted 3.0 by default
   +-- VLABench DFA-masked CE + primal-dual constraint loss by default
   +-- PCGrad on shared gradients; one planner step for the pair
   +-- one controller behavior-cloning step
                   |
                   v
Restore best joint Stage 1 checkpoint and apply EAI exploration gate
   +-- fail: report stage2-skipped and return
   +-- pass: dedicated controller-only warm-up (default 20,000 steps)
                   |
                   v
Fixed-seed baseline evaluation and VLABench preflight report
                   |
                   v
Stage 2: alternating domain updates
   +-- EAI: eight sampled plans, REINFORCE + supervised anchor
   +-- VLABench: eight simulator episodes, planner REINFORCE + controller PPO
   +-- shared backbone frozen within domain scopes by default
   +-- round progress checkpoints
                   |
                   v
Fixed-seed evaluation, regression/retention gates, eligible best restoration
```

The VLABench baseline preflight currently reports its result without stopping
the RL loop on failure. Post-epoch retention checks do stop training. This
distinction is detailed below.

## Shared graph, parameters, and task representation

### One root, separate domain semantics

`build_joint_world_graph()` creates shared `embodied_episode`,
`embodied_entity`, and `embodied_operation` concepts. EAI and VLABench world
concepts inherit from that spine. `build_joint_runtime()` attaches both
generation graphs and compiles their policies:

```text
joint root
  shared episode/entity/operation concepts
  EAI world graph                 EAI generation graph and policy
  VLABench world graph            VLABench generation graph and policy
```

The domains retain their own actions, argument roles, effects, logical
constraints, vocabularies, and checksums. Shared ancestry does not make an EAI
action and a VLABench skill interchangeable.

`runtime.domain_scope(domain, parameter_policy=...)` activates the selected
world/generation profile and restores previous activation on exit, including
exceptions. A reentrant lock serializes scopes on the same runtime. Switching
domains does not reconstruct the graph, model, controller, or optimizers.

Graph activation and parameter ownership are separate controls. The parameter
policy determines which shared/domain parameters can receive gradients; graph
activation determines which concepts and constraints execute.

### One causal planner, two linear heads

`JointQwenVLPlanner` registers the shared model once. Non-owning domain views
adapt standalone planner APIs to this object. Its current learned output
modules are `label_heads["eai"]` and `label_heads["vlabench"]`, both linear
projections from Qwen hidden states to the corresponding compact vocabulary.

Defaults are `Qwen/Qwen3-VL-8B-Instruct`, 4-bit NF4 loading, and LoRA with rank
16, alpha 32, and dropout 0.05 on attention and MLP projections. When the
tokenizer and input embeddings are available, label weights are initialized
from averaged input-token embeddings, normalized in scale, with zero bias.

The current joint planner has no separate GRU planner decoder. The retained
`--planner-decoder-hidden-dim` option is stored as configuration/compatibility
metadata; it does not introduce a recurrent decoder in this implementation.
The continuous controller still has its own temporal GRU.

During teacher forcing, the prompt plus gold label prefix is processed in one
causal backbone pass per sequence row. Boundary hidden states predict each
next compact label. During autoregressive generation, `sample_labels()`
rebuilds the full prompt/prefix and calls Qwen at every label with
`use_cache=False`. `EncodedContext` carries a context mapping, but
`sample_labels_from_context()` delegates to this full-prefix sampler; it does
not reuse a cached context vector to avoid later backbone calls.

### Domain inputs and labels

EAI uses text instructions and the SimpleTL goal. `planner_context()` prefers
the prepared `causal_prompt_text`, supplies the goal separately, and retains
permitted observation fields for contextual constraints. Targets are EAI
action/entity labels followed by EOS, for example:

```text
grab, novel_1000, read, novel_1000, <eos>
```

VLABench uses an instruction, images, and a numbered entity table. Targets are
explicit skill, argument-role, and pointer labels, for example:

```text
skill:pick, arg:target_entity_name, obj:0,
skill:place, arg:target_container_name, obj:1, <eos>
```

`planner_context()` merges row observations, nested `planner_context`, and
explicit context overrides in that order. It does not copy target plans into
the context. EAI's vocabulary and compatibility sets nevertheless originate
from the loaded reference trajectories, and its reward grounding also uses
reference information; context filtering does not remove those dependencies.

EAI uses the standalone ordered 80/20 train/dev split. VLABench uses a seeded
80/10/10 planning split; held-out task partitions are removed before splitting.
Control demonstrations are split by episode before windows are assigned to
loaders. Default budgets are 60 EAI labels, 64 VLABench entity slots, and 12
VLABench operations. EAI semantic validation defaults to its full holdout;
`--validation-limit` can cap it.

## Stage 1: supervised joint training

### Objective

Each paired round computes two planner losses:

```text
L_eai = EOS-masked cross-entropy over EAI labels

L_vla = EOS-masked DFA-restricted cross-entropy
        + pmd_weight * primal_dual_constraint_loss
        + mass_weight * allowed_mass_loss

shared update = PCGrad(eai_loss_weight * grad(L_eai), grad(L_vla))
```

The last line describes gradient coordination rather than a scalar reward or
ordinary summed loss. Default `eai_loss_weight` is 3.0, `pmd_weight` is 1.0,
and `mass_weight` is 0.0. Domains have one turn each per round, but their loss
weights are not equal by default.

### 1. Attach both sensor branches once

`JointSolverPOIProgram` attaches both planner views to graph reader/learner
sensors and constructs one `SolverPOIProgram` lifecycle. The active domain
selects the applicable EOS-masked objective. Gold labels are shifted right
with EOS as the start sentinel; padding after the first EOS is excluded.

### 2. Run the EAI-only warm-up

Before the first joint Stage 1 epoch, 350 supervised EAI optimizer steps run
by default. This updates shared trainable planner parameters and the EAI head
under `freeze_inactive`, without a paired VLABench gradient or controller
update. `--stage1-eai-warmup-steps 0` disables it.

This phase is distinct from the optional text-only VLABench warm-up available
in the standalone EAI CLI; the joint command does not run that phase.

### 3. Apply the configured generation losses

`--stage1-dfa-loss on` is the default. Its domain lists determine which parts
apply:

| Component | Default domains | Behavior |
|---|---|---|
| DFA training machinery | EAI, VLABench | Inspect allowed labels along gold prefixes |
| Masked cross-entropy | VLABench | Normalize over DFA-allowed labels |
| Productive-completion lookahead | VLABench | Require a reachable completion in allowed sets |
| Primal-dual constraint loss | VLABench | Evaluate generation-graph logical constraints |
| Allowed-mass loss | Weight zero | Computed/reported but adds no gradient at the default weight |

If the DFA disallows a teacher target, masked CE falls back to the full
softmax for that position and records a `masked_ce_misses` diagnostic. EAI
retains ordinary CE by default. Primal-dual programs reuse the planner's
forward logits and update constraint multipliers by ascent on their schedule:
ten warm-up rounds, then every ten rounds. `--stage1-dfa-loss off` selects plain
supervised CE for both domains.

### 4. Coordinate a paired planner update with PCGrad

The program evaluates the EAI and VLABench losses sequentially inside their
domain scopes, with inactive domain parameters frozen. `GradientConflictManager`
captures their gradients, projects conflicts on shared parameters, and performs
one planner optimizer step. Domain-specific heads retain their domain gradients.
Gradient norm clipping uses 1.0.

The planner optimizer is AdamW at `2e-5` by default. PCGrad reduces conflicting
updates; it does not guarantee that either domain's evaluation score improves.

### 5. Train the controller on the VLABench turn

Each paired round also takes one controller behavior-cloning update. The
controller consumes two-frame multi-view image/state history, the official
language task index, and plan conditioning. The task index preserves distinct
requested objects; it is not replaced by a primitive skill-pattern ID.

Pose supervision compares wrapped, normalized local deltas against the policy's
reachable step range. Gripper supervision uses binary cross-entropy on logits.
The standard visual encoder is frozen SigLIP; trainable control layers include
attention pooling, temporal fusion, actor, and critic components. Controller
entity conditioning uses the padding row because demonstrations lack stable
segmentation-to-graph pointer labels.

An epoch defaults to the smaller of the EAI and VLABench planning training
lengths. Streams cycle when `--stage1-rounds-per-epoch` requests more rounds.
Default Stage 1 duration is five epochs.

### 6. Select and restore the best Stage 1 checkpoint

After each epoch, EAI plans are decoded greedily with the bound policy and
scored for goal success/recall. VLABench plans are decoded and scored on the
planning validation split. Checkpoints are ranked by:

1. minimum of EAI goal success and VLABench `exact_graph_match`;
2. mean of those two scores;
3. EAI goal recall;
4. VLABench validity;
5. lower `validation_loss`.

Despite its name, this `validation_loss` is the mean of the two reported
training planner losses. It is not held-out teacher-forced loss. Also,
VLABench `exact_graph_match` is a partial dependency-node match score, not
binary equality of the complete plan.

Every epoch is saved as `joint_stage1_epoch_<epoch:03d>.pt`; the selected file
is reloaded, including controller and optimizer state.

### 7. Apply the EAI gate and warm up the controller

Before entering the controller-only warm-up and Stage 2, the selected EAI
metrics must meet these defaults:

- positive-reward rate at least `0.05`;
- goal recall at least `0.20`;
- goal success at least `0.10`.

Failure reports `stage2-skipped` and returns. After a pass, the controller gets
20,000 dedicated BC updates by default at `3e-4`, without updating the planner.
The command validates the controller and saves `joint_controller_warmup.pt`.
`--controller-warmup-steps 0` disables this dedicated phase.

## Stage 2: alternating reinforcement learning

### 1. Reuse the shared model with an explicit parameter policy

`JointReinforcementProgram` reuses the exact joint runtime, planner, domain
views, and controller. The controller optimizer clears its BC moments and
switches to `3e-5` when first entering RL; the planner optimizer is reused.

| `--stage2-parameter-policy` | Effect |
|---|---|
| `freeze_shared` (default) | Freeze shared Qwen/LoRA within domain scopes; train the active linear label head |
| `freeze_inactive` | Keep shared parameters trainable and freeze the inactive head |
| `none` | Apply no scoped parameter-freezing policy |
| `pcgrad_protected_eai` | Experimental trainable-shared mode with an EAI-update projection against its supervised anchor |

The last option's explicit projection is in `train_eai_update()`. The inherited
VLABench planner update does not perform the paired Stage 1 PCGrad operation.
These alternatives should not be described as equivalent protection against
cross-domain interference. The deprecated freeze-shared Boolean flag can
override the selected policy.

### 2. Record fixed-seed baselines and the preflight result

The command evaluates the post-warm-up model on ten simulator rollouts per
selected task by default, with seed base `seed + 100000`. It also records EAI
holdout success/recall and optional static VLABench planning scores.

VLABench preflight defaults require positive-return rate at least `0.01` and
IK truncation at most `0.50`. Minimum success and successful-task coverage both
default to zero. **Current control flow only logs `stage2-skipped` when this
preflight fails; it does not return before the RL epoch loop.** This differs
from the EAI gate and from the standalone VLABench command.

Setting `--stage2-eval-rollouts-per-task 0` disables this baseline evaluation
and the corresponding fixed-seed post-epoch evaluations. Selection then uses
training metrics, and baseline references may be absent.

### 3. Train the EAI head with sampled-plan reward

Each round chooses a random EAI training item, samples eight constrained plans
by default, and computes one reward per complete trajectory. The update uses:

```text
A_i = R_i - mean(R_0, ..., R_N-1)
L_eai_RL = -mean_i[A_i * log p(sampled_labels_i | context, DFA)]
           + 0.5 * L_random_gold_EAI_example
```

Plans are sampled under `no_grad`; nonzero-advantage trajectories are replayed
one at a time for gradients. A random EAI reference example supplies the
supervised anchor. This is the joint causal planner's replay path, rather than
the standalone EAI KV-cached sampler and microbatch option.

EAI reward defaults to dense grounded final-state recall multiplied by ordered
temporal progress, modulated by applicable world constraints with default
weight 0.25. Binary mode uses evaluator success instead. The evaluator parses
`tl_goal`, but reference trajectories still ground variables/types and help
infer initial facts. It does not execute the full PDDL transition model or a
live BEHAVIOR/VirtualHome environment. These limits are explained in
[EAI_Operation.md](../../EmbodiedAgentInterface/docs/EAI_Operation.md).

### 4. Collect VLABench simulator episodes

The same round chooses the next primitive task by round-robin cursor and
collects eight episodes by default. Each replan samples four graph-constrained
candidates. The inherited collector executes the first valid candidate;
constraint-invalid candidates may become zero-return planner evidence.

The controller predicts a 16-action chunk, executes up to four controls, and
then replans. Local pose targets are converted to robot commands through the
simulator adapter and IK. Default limits are 2 cm translation, 0.10 radians
rotation per step, and 400 steps per episode. Repeated rejected IK chunks can
truncate execution. An invalid plan never reaches the controller.

Task success comes from the simulator task predicate. Return includes:

```text
efficiency = success * max(0, 1 - steps / max_steps)

R_vla = clip(0.60 * success + 0.25 * final_progress + 0.10 * final_intention
             + 0.05 * efficiency + milestone_credit - IK_penalty, 0, 1)
```

The inherited collector can award unsuccessful episodes milestone credit of
0.15 for grasp, 0.15 for lift, and 0.05 for container approach, and subtract
0.20 for IK truncation/failure. A terminal correction reconciles intermediate
shaping with this bounded total. Invalid episodes receive zero return.
Reference-plan similarity is a separate evaluation metric.

### 5. Update the VLABench planner and controller

Planner decisions receive simulator return-to-go. Mean-centered decision
returns weight replayed sampled-plan log probabilities. The joint CLI adds a
reference-plan anchor with weight **0.25**, controlled by
`--stage2-planner-anchor-weight`.

The controller uses PPO with GAE: gamma 0.99, lambda 0.95, clip 0.2, two PPO
epochs, value weight 0.5, and entropy weight 0.01. Its demonstration BC anchor
also defaults to **0.25**, controlled by `--stage2-controller-bc-weight`.
These joint defaults differ from the standalone VLABench defaults.

Only batches with positive task return enable actor updates. Wholly zero-return
batches train the detached critic. Mixed batches contrast executed positive-
and zero-return behavior. Additional terms handle infeasible actions and
available assisted-imitation targets. Probability-ratio and approximate-KL
checks can roll back a controller update and reduce its learning rate.

Default `--execution-assistance train-only` enables training scaffolding while
disabling it in fixed-seed evaluation. Assistance follows a cosine curriculum
from 1.0 toward 0.0. `on`, component ablations, shadow imitation, handover,
hindsight relabeling, and learning-progress scheduling change the experiment
contract and should be reported explicitly. Controller evidence records
accepted PPO epochs, rollback status, KL, and measured parameter changes.

Default Stage 2 duration is three epochs with ten alternating rounds per epoch.
EAI and VLABench rewards remain separate; the workflow does not average them
into one training reward.

### 6. Evaluate and enforce retention

With fixed-seed evaluation enabled, each epoch reevaluates the EAI holdout and
the same VLABench task/seed schedule. Static VLABench planning validation is
enabled by default but remains diagnostic rather than a simulator reward.

An epoch must retain the VLABench task-signal/IK preflight requirements and
respect the recorded baseline regressions:

- VLABench success loss no greater than `0.0` by default;
- EAI goal-success loss no greater than `0.0`;
- EAI goal-recall loss no greater than `0.02`.

Failure saves the epoch as rejected, restores an available eligible best or
Stage 1/controller-warm-up fallback, and aborts further RL epochs. The loader
rejects Stage 2 checkpoints explicitly marked `retention_eligible: false`.

### 7. Select the best eligible joint RL checkpoint

Additional best-candidate defaults require VLABench success at least `0.10`,
success in at least three task families, and IK truncation no greater than
`0.25`. For single-task experiments, adjust the successful-task threshold to
the chosen task count.

Eligible epochs are ranked by:

1. minimum of EAI success and VLABench success-rate lower bound;
2. mean of those two values;
3. EAI goal recall;
4. mean VLABench simulator return;
5. fewer simulator steps.

The VLABench success-rate lower bound falls back to observed success when
unavailable. Selected weights and optimizer state are restored and saved to
`joint_stage2_best.pt`. If no candidate is eligible, the command reports
`stage2-best-skipped`; training completion alone does not produce a best file.

## Checkpoints and resuming

Version-8 joint checkpoints atomically save learned planner parameters once,
including shared LoRA even when temporarily frozen, both label heads, controller
state, planner/controller optimizers, stage/epoch/cursor, metrics, and
Python/NumPy/PyTorch/CUDA random states. Immutable quantized base weights remain
external. Compatibility records include domain/runtime checksums, vocabularies,
DFA configurations, parameter ownership, model/controller configuration, and a
separate logical-knowledge definition/activation manifest.

| File or saved stage | Resume behavior |
|---|---|
| `joint_stage1_epoch_*.pt` / `stage1` | Continue at the next Stage 1 epoch |
| `joint_controller_warmup.pt` / `controller_warmup` | Skip completed Stage 1 and controller warm-up |
| `joint_stage2_progress.pt` | Resume the next recorded round with partial metrics and prior-best metadata |
| `joint_stage2_epoch_*.pt` / `stage2` | Continue at the next RL epoch |
| `joint_stage2_best.pt` | Load the selected eligible epoch; further training follows its saved position |

`--resume` loads a combined checkpoint; `--resume-adapter` initializes only the
planner adapter/head files. Keep compatible data, vocabulary, model, and round
settings. Epoch counts are total target counts, not numbers of additional
epochs to append. A resumed Stage 2 run skips a new baseline preflight and
reuses baseline/fallback metadata when present.

Stage 1 primal-dual multiplier states, their optimizers, and their update-round
counter are not included in the joint checkpoint payload. Stage 1 resume
therefore rebuilds those objects rather than exactly continuing all auxiliary
optimization state. Optional hindsight/replay experiment buffers are likewise
not a promise of exact process continuation.

Knowledge-definition drift is checked separately from intentional activation
ablations. Historical checkpoints without a manifest report
`unrecorded_legacy`. Supported controller migrations can require fresh warm-up;
older Stage 2 versions predating the robot-frame contract are rejected. The
planning evaluator's explicit schema-migration mode is a warm start, not an
exact optimizer/run resume.

## Evaluation and current implementation limits

Report EAI abstract goal success and VLABench simulator end-to-end success
separately, with task coverage, return, IK statistics, steps, and assistance.
Static SR/PR/SPR/PM scores assess planning against references without executing
controls. Their four-term total, legacy total, dimensions, observation modes,
label revisions, and knowledge tiers must be matched across comparisons.
The joint static planning track is distinct from the standalone official VLM
protocol. Use `eval_planning.py` for full planning reports and
`utils/paired_eai_eval.py` for explicit EAI comparison selections.

Two implementation details limit claims about rollout probability replay:
`prepare_replay_context()` currently returns processor tensors, while subsequent
joint sampling/replay helpers rebuild a prompt from the mapping as though it
contained observation fields. The original instruction/image/entity fields
are not explicitly retained in that returned mapping. In addition, joint
training sampling and replay do not impose the standalone EAI sampler's
dropout-disabled evaluation-mode block. Reapplying a DFA and obtaining a
gradient therefore do not, by themselves, establish identical observation
conditioning or probabilities between collection and replay.

The current unit tests verify graph activation, parameter ownership, losses,
gradient flow, checkpoint compatibility, and lightweight pipeline behavior.
They do not establish real-model prompt replay fidelity, live simulator task
success, or GPU training performance. The baseline preflight control-flow and
auxiliary checkpoint-state limits described above also remain implementation
behavior rather than repaired features.

## Principal implementation locations

- [`main.py`](../main.py): data preparation, stage lifecycle, defaults, baseline
  evaluation, gates, selection, and progress checkpoint callbacks.
- [`world_graph.py`](../world_graph.py): shared semantic spine and domain world
  graph construction.
- [`utils/generation_runtime.py`](../utils/generation_runtime.py): generation
  graph compilation, contextual policies, activation, and parameter scopes.
- [`utils/planner_context.py`](../utils/planner_context.py): observation merging
  for row, nested-context, and keyword planner APIs.
- [`models.py`](../models.py): shared causal Qwen/LoRA, linear heads, domain
  views, prompt boundaries, teacher forcing, sampling, and replay.
- [`program.py`](../program.py): supervised losses, PCGrad rounds, controller
  warm-up, EAI RL, and alternating VLABench updates.
- [`checkpoint.py`](../checkpoint.py) and
  [`utils/knowledge_manifest.py`](../utils/knowledge_manifest.py): compact state,
  resume compatibility, migrations, and knowledge provenance.
- [`eval_planning.py`](../eval_planning.py) and
  [`utils/paired_eai_eval.py`](../utils/paired_eai_eval.py): static planning and
  paired abstract EAI evaluation.
- [`VLABench simulation program`](../../VLABenchAgentInterface/simulation/program.py):
  inherited candidate selection, physical execution, rewards, REINFORCE,
  controller PPO, and fixed-seed simulator evaluation.
- [`EAI reward evaluator`](../../EmbodiedAgentInterface/reward.py): SimpleTL
  parsing, reference-dependent grounding, abstract state transitions, and reward.
- [Joint logical constraints](JointLogicalConstraints.md): declared knowledge,
  observation bindings, activation, and checkpoint provenance contracts.
