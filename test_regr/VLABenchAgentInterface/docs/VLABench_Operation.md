# VLABench_Operation

This document describes the operational flow of the VLABench hierarchical
agent's two-stage training pipeline. Stage 1 learns symbolic plans from
reference sequences and continuous controls from demonstrations. Stage 2
reuses both policies and refines them through simulator execution: REINFORCE
updates the planner, while PPO with generalized advantage estimation (GAE)
updates the controller. Both retain supervised anchors.

The canonical entry point is `train-agent --two-stage` in
[`main.py`](../main.py). This follows the explanatory structure of
[`EAI_Operation.md`](../../EmbodiedAgentInterface/docs/EAI_Operation.md), with
the VLABench-specific representations, objectives, and checkpoint behavior.

## Operation at a glance

```text
Planning examples                         Control demonstrations
   |                                         |
   +-- instruction and images                +-- multi-view image history
   +-- scene entity table                    +-- robot end-effector state
   +-- reference operation sequence          +-- task/plan conditioning
   +-- dependency metadata                   +-- demonstrated action chunks
   |                                         |
   v                                         v
Stage 1: planner teacher forcing + controller behavior cloning
   |
   +-- Qwen-VL context encoder and compact graph-token GRU decoder
   +-- EOS-masked planner cross-entropy
   +-- controller pose-delta and gripper losses
   +-- planning/controller validation; save agent_stage1.pt
   |
   v
Fixed-seed simulator preflight
   |
   +-- fail: save evaluated Stage 1 and skip reinforcement learning
   +-- pass: reuse planner, controller, vocabulary, and graph runtime
   |
   v
Stage 2: joint simulator reinforcement learning
   |
   +-- sample graph-constrained candidate plans
   +-- execute the first valid candidate with receding-horizon controls
   +-- obtain simulator success, progress, intention, and feasibility signals
   +-- planner: return-to-go REINFORCE + reference-plan anchor
   +-- controller: PPO/GAE + demonstration and assistance losses
   |
   v
Fixed-seed evaluation, retention gates, and resumable checkpoints
   |
   +-- restore eligible best RL checkpoint as agent_rl_best.pt
   +-- abort and restore best/fallback if retention fails
```

## Shared task representation

### Planning examples

A planning example supplies an instruction, ordinary and/or numbered segmented
images, a scene entity table, a reference operation sequence, and dependency
metadata. The graph planner prefers segmented images when available.

For a simple pick-and-place task, the representation is conceptually:

```text
Instruction: Put the apple in the bowl.
Objects:
  0: apple
  1: bowl

Reference operations:
  pick(target_entity_name=0)
  place(target_container_name=1)

Graph labels:
  skill:pick, arg:target_entity_name, obj:0,
  skill:place, arg:target_container_name, obj:1, <eos>
```

Argument roles are explicit labels. Entity pointers index the example's entity
table; `obj:0` has no permanent object identity across scenes. The vocabulary
comes from the world-domain skill and argument declarations, with a configured
entity capacity, rather than being inferred from whichever gold plans happen
to be loaded.

The planner uses Qwen-VL to encode the instruction, images, and object table
once per observation. Its final context hidden state initializes a compact
GRU decoder through a trainable projection. Token embeddings, the GRU, and an
output layer predict the graph vocabulary. Qwen does not generate each graph
label through its native language head. In the default LoRA configuration,
the base model is frozen while LoRA and the compact decoder remain trainable;
4-bit model loading is enabled by default.

### Control examples

The controller consumes a short history of camera views and robot state,
conditioned on the demonstration instruction's task index and graph plan
context. The standard vision encoder is frozen SigLIP; a tiny encoder is
available for smoke runs. Attention pooling and a temporal GRU feed the action
and value heads.

Actions describe end-effector pose and gripper state. The learned pose policy
uses bounded local deltas in robot coordinates; action chunks are accumulated
from the current pose. The environment adapter converts targets to world-frame
inverse kinematics (IK) and robot commands. Offline demonstrations have no
segmentation-to-pointer correspondence, so their controller entity conditioning
uses the padding row; online collection follows that convention.

The standalone planning loader uses a seeded 80/10/10 train/validation/test
split, with adjustments for small collections. Control loaders construct their
own demonstration splits. These internal splits and live simulator evaluation
are separate evaluation contracts.

## Stage 1: supervised planner and controller training

### Objective

Stage 1 has two supervised objectives, applied to separate policies:

```text
L_planner = mean_t[-log p(gold_label_t | observation, gold_prefix_<t)]

L_controller = normalized pose-delta SmoothL1 + gripper BCEWithLogits
```

The planner's "exact-match" objective is token-level cross-entropy. Controller
behavior cloning fits demonstrated controls rather than symbolic plan labels.

### 1. Build graph-owned gold labels

`prepare_planner_program_examples()` converts each reference plan into skill,
argument, and object-pointer labels, appends EOS, and pads the remaining token
positions with EOS. Plans exceeding the configured token budget raise an error.
The runtime defaults to 64 entity slots, 12 operations, and a token budget of
`max_operations * 5 + 1`.

### 2. Shift the targets for teacher forcing

The planner shifts gold labels right, using EOS as its internal start sentinel:

```text
Input:   [START, skill:pick, arg:target_entity_name, obj:0, ...]
Target:  [skill:pick, arg:target_entity_name, obj:0, skill:place, ...]
```

The observation context stays constant. At every decoder position, the prefix
contains the correct preceding labels. One Qwen-VL context encoding and one
causal recurrent decoder pass supply the sequence logits.

### 3. Apply EOS-masked cross-entropy through SolverPOI

Graph reader sensors provide the context, token positions, and target labels.
A `ModuleLearner` wraps the planner, and `SolverPOIProgram` uses
`EOSMaskedCrossEntropyLoss` with `local/argmax` inference.

Loss includes the meaningful labels and the first EOS; later EOS padding is
excluded. This Stage 1 loss uses the full label logits. It does not apply the
autoregressive DFA mask to the teacher-forced cross-entropy.

### 4. Warm up the controller by behavior cloning

The controller predicts a demonstrated action chunk from image/state history
and conditioning. Pose supervision compares normalized incremental deltas,
wraps angular differences, and clamps target deltas to the deployed policy's
reachable step range. Gripper supervision uses binary cross-entropy on logits.

By default, `train-agent` runs 20,000 controller warm-up optimizer updates.
Setting `--controller-warmup-steps 0` selects full passes controlled by
`--controller-epochs`. Controller warm-up uses its own optimizer and learning
rate, separately from planner supervised training.

### 5. Validate and save Stage 1

After planner training and controller warm-up, the command evaluates generated
plans on the planning validation split and controller losses on held-out
demonstrations. Planning evaluation decodes autoregressively with the compiled
graph policy, then reports component scores, validity, and grounding coverage.

The standalone command saves the resulting combined state as `agent_stage1.pt`.
It does not implement EAI's explicit semantic ranking and restoration of a best
Stage 1 epoch. Planning similarity and low controller imitation loss also do not
establish that the combined agent succeeds in the simulator.

### 6. Apply the simulator preflight gate

Before RL, fixed-seed simulator rollouts evaluate every selected primitive task.
The baseline and gate result are saved as `agent_stage1_evaluated.pt`.
Default preflight requirements are:

- success on at least one evaluation episode in every adapter primitive task;
- positive-return rate of at least `0.01`;
- IK truncation rate no greater than `0.50`;
- overall success rate at least the configured minimum, which defaults to `0.0`.

Task coverage is an independent requirement even when the overall minimum is
zero. A single-task run must adjust the task-count thresholds to one. A failed
gate retains Stage 1 and prints `reinforcement-skipped`.

The default is ten evaluation rollouts per task. Setting
`--eval-rollouts-per-task 0` disables fixed-seed preflight and retention checking;
checkpoint eligibility then uses training metrics.

## Stage 2: joint reinforcement learning

### Objective

The two policies receive different gradients from the same executed episodes:

```text
L_planner = -mean_j[A_j * log p(sampled_plan_j | observation_j, DFA_j)]
            + supervised_weight * L_gold_plan

L_controller = L_PPO + value_weight * L_value - entropy_weight * entropy
               + feasibility_loss + assisted_imitation_loss
               + controller_bc_weight * L_demonstration
```

Default anchor weights are `0.1` for the planner and `0.05` for controller
demonstrations. The supervised planner anchor draws a random training example;
it need not be the live simulator task.

### 1. Reuse the Stage 1 policies and enter the PPO regime

Stage 2 reuses the planner, controller, vocabulary, world graph, and compiled
generation runtime. The controller optimizer switches from the warm-up learning
rate (`3e-4`) to the PPO learning rate (`3e-5`) and clears its behavior-cloning
optimizer moments. Planner RL defaults to `2e-5`.

The standalone command cycles primitive task descriptors across rounds. Each
round collects `--rollouts-per-update` episodes, default eight, before updating
both policies. Defaults are three RL epochs and ten rounds per epoch.

### 2. Sample candidate plans under the graph policy

At each planning decision, current images and scene entities bind the world
constraints into the generation policy. For primitive simulator tasks, an
additional task-pattern restriction enforces the expected skill sequence and
available target restrictions. A productive-state wrapper removes dead ends.

The DFA masks illegal next labels before categorical sampling. Declarations
cover skill schemas, required and optional arguments, entity pointers, complete
subtask patterns, EOS boundaries, and operation limits. Task/scene constraints
can further restrict continuations when the necessary observations are exposed.
An accepted symbolic plan still requires successful physical execution.

The planner samples `--rl-num-samples` candidates, default four, sharing one
encoded observation. Collection stores replayable CPU inputs and sampled labels
without retaining a Qwen gradient graph. The **first valid candidate** is
selected. Other valid candidates are not executed as sibling simulator trials;
constraint-invalid candidates can be retained as zero-return planner evidence.
If all candidates are rejected, the episode terminates as `invalid_plan`.

### 3. Execute controls and replan

The controller samples an action chunk, default horizon 16. Up to four controls
are executed before a fresh planning decision. The live adapter tracks operation
progress and task-specific execution conditions rather than assuming that an
approach movement completed a grasp or placement.

Translation and rotation limits default to 2 cm and 0.10 radians per step.
IK failures trigger smaller target attempts; repeated rejected chunks truncate
the episode. PPO stores the policy's sampled actions and their old probabilities,
while executed safety/assistance transforms belong to the environment transition.
The default episode budget is 400 simulator steps.

### 4. Calculate simulator return

Task success comes from the simulator's task predicate. Task progress and
intention provide shaping; geometric diagnostics can provide fallback progress
when upstream progress is unavailable or remains zero.

In `collect_episode()`, the bounded target return is:

```text
efficiency = success * max(0, 1 - steps / max_steps)

R = clip(0.60 * success + 0.25 * final_progress + 0.10 * final_intention
         + 0.05 * efficiency + milestone_credit - IK_penalty, 0, 1)
```

For unsuccessful episodes, milestone credit can add `0.15` for grasp,
`0.15` for lift, and `0.05` for approaching the container. IK truncation or an
IK-failure termination incurs a `0.20` penalty. Invalid episodes have zero return.

Intermediate progress/intention deltas and milestone increments are reconciled
by a terminal correction so stored transition rewards sum to the bounded episode
return. The simpler `simulation/rollout_reward.py` accumulator used by the runtime
agent has the base success/progress/intention/efficiency formula without these
training-specific milestone and IK terms.

Reference-plan similarity is a separate diagnostic. It is not the reward used
by the canonical simulator RL pipeline.

### 5. Update the planner using return-to-go

Each executed planning decision receives the sum of transition rewards from
that decision onward. Rejected plans receive zero. Across the decision/return
pairs collected for an update:

```text
baseline = mean(return_to_go_j)
A_j = return_to_go_j - baseline
```

The baseline spans the collected decisions, rather than independent simulator
scores for every candidate at one observation. Equal returns give zero policy
advantages; the supervised anchor can still supply a gradient when decision
pairs exist. If no planner evidence exists, the update returns without an anchor.

Stored decisions are replayed sequentially with gradients through Qwen LoRA and
the compact decoder. Replay teacher-forces the sampled labels and applies the
same DFA used during sampling. Only one replay's vision-language autograd graph
needs to be resident at a time. The gold-plan anchor adds ordinary teacher-forced
cross-entropy, followed by gradient clipping and a planner optimizer step.

### 6. Update the controller with PPO and GAE

GAE computes transition advantages and value targets from stored rewards,
old values, and terminal flags. PPO compares current and stored probabilities
for the executed prefix of each sampled chunk and clips the importance ratio.
The value head uses SmoothL1 loss; entropy encourages exploration.

A batch must contain positive task return before actor updates are enabled.
Then executed transitions, including zero-return episodes, contribute evidence.
A wholly zero-return batch updates only the detached critic. Advantages are
normalized when multiple informative transitions exist; a singleton retains its
raw advantage.

Additional terms penalize persistently infeasible sampled targets, imitate
available assisted controls, and anchor the controller to demonstrations.
Non-finite loss/gradient checks, bounded log ratios, and post-update trust-region
checks can skip or roll back a controller update. Optional shadow imitation,
handover, hindsight relabeling, and advantage-weighted replay are experiment
variants controlled by explicit flags.

Evaluators restore the previous train/eval mode. Planner and controller updates
also explicitly enter training mode before recurrent replay and backpropagation.

### 7. Keep execution assistance explicit

The default `--execution-assistance train-only` permits deterministic motion and
grasp scaffolding during training while disabling it in fixed-seed evaluation.
The default assistance curriculum decreases linearly from 1.0 toward 0.0 over
training rounds. `off` disables scaffolding, while `on` includes it in evaluation.
Report assisted evaluation separately from learned-controller performance and
inspect assistance counters when experiment overrides are enabled.

### 8. Evaluate, retain, and restore checkpoints

After each RL epoch, the command evaluates the same task descriptors with the
same seed schedule. It saves `agent_rl_epoch_<epoch:03d>.pt` and checks whether
task signal and feasibility still satisfy the preflight thresholds. A retention
failure aborts RL and restores an available eligible best or evaluated Stage 1
fallback.

For an epoch to become the best RL candidate, default additional requirements
are success rate at least `0.10`, success in at least three task families, and
IK truncation rate at most `0.25`. Eligible checkpoints are ranked by:

1. success-rate lower bound, when available, otherwise observed success rate;
2. number of successful task families;
3. mean simulator return;
4. execution-complete rate;
5. lower IK truncation rate;
6. fewer steps.

The selected checkpoint is restored and saved as `agent_rl_best.pt`. If no epoch
is eligible, the command reports that no best checkpoint was selected. Completing
an executable rollout or reaching the step limit does not establish task success.

## Checkpoints and resuming

Joint checkpoints atomically save trainable planner state, controller state,
both optimizer states, domain/vocabulary checksums, model configurations,
metrics, and Python/NumPy/PyTorch/CUDA random states. The frozen planner backbone
must remain available separately.

`agent_rl_progress.pt` is overwritten after each completed RL round. It records
the next round, completed round metrics, prior best candidate, and fallback
location. `--resume` on `train-agent` resumes the standalone combined checkpoint;
keep the original round count and compatible model/domain settings. A supervised
checkpoint skips completed Stage 1 training, while an RL checkpoint resumes its
recorded RL position. Recognized legacy controller migrations may require a
fresh controller warm-up.

`--resume-adapter` instead loads a planner adapter directory with its compact
decoder; it does not restore the combined agent's controller or training position.
The separate `train-planner` command exports `planner_stage1/`, containing the
adapter, `graph_decoder.pt`, and `vocab.json`.

## How the two stages differ

| Property | Stage 1 | Stage 2 |
|---|---|---|
| Planner signal | Gold graph labels at each position | Simulator return-to-go plus gold-plan anchor |
| Planner prefix | Gold labels | Sampled labels, replayed for gradients |
| Controller signal | Demonstrated action chunks | PPO/GAE, feasibility, imitation, and BC anchor |
| Graph policy | Autoregressive planning validation | Candidate sampling, replay, and validation |
| Simulator affects gradients | No | Through detached rewards and advantages |
| Physical task success | Measured by simulator preflight | Measured after each epoch |
| Stage checkpoint | Resulting supervised state | Eligible best simulator checkpoint restored |

## Evaluation interpretation

Use simulator end-to-end success as the main measure of the complete agent.
Report per-task success and task coverage alongside return, positive-return rate,
IK failures/recoveries, IK truncation, execution completeness, steps, and assistance.

Static planning evaluation reports SR (skill recall), PR (argument-role/entity
recall), SPR (skill/argument-tuple recall), and PM (matching dependency-graph-node
fraction). PM is partial credit rather than binary whole-plan equality. The
`planning_metrics` component scores use a 0-100 scale:

```text
score        = 0.4 * SR + 0.4 * PR + 0.1 * SPR + 0.1 * PM
legacy_score = 0.4 * SR + 0.4 * PR
```

The separate validity-gated planner reward is on a 0-1 scale. Planning metrics
score readable predictions independently of world-validity gating. Neither
metric executes controls. Keep internal split results, metadata-assisted results,
and the standard `--protocol official` VLM evaluation in separately labeled
tracks. Observation mode, label revision, task mapping, and split must match for
comparisons; see the [README](../README.md) and
[baseline and evaluation guide](VLABENCH_BASELINE_AND_EVALUATION_GUIDE.md).

## Principal implementation locations

- [`main.py`](../main.py): CLI, two-stage lifecycle, simulator preflight,
  retention/best gates, round checkpoints, and restoration.
- [`dataset.py`](../dataset.py): planning examples, images/entities, internal
  splits, demonstration windows, and controller conditioning.
- [`plan_codec.py`](../plan_codec.py): stable graph vocabulary and conversions
  between operations, pointers, tokens, and labels.
- [`graph.py`](../graph.py) and [`generation_runtime.py`](../generation_runtime.py):
  generation declarations, compilation, and context-bound DFA construction.
- [`world_graph.py`](../world_graph.py), [`world_domain.py`](../world_domain.py),
  and [`knowledge/`](../knowledge/): skill schemas, semantic constraints, task
  patterns, and task contracts.
- [`models.py`](../models.py): Qwen-VL/LoRA context encoder, compact GRU planner,
  teacher forcing, constrained sampling/replay, and adapter serialization.
- [`training.py`](../training.py): runtime and program construction, prepared
  examples, evaluation, supervised helpers, and combined checkpoint I/O.
- [`simulation/controller.py`](../simulation/controller.py): multi-view
  actor-critic, bounded action representation, and behavior-cloning objective.
- [`simulation/program.py`](../simulation/program.py): SolverPOI loss/sensors,
  candidate selection, simulator collection, reward shaping, planner REINFORCE,
  controller PPO/GAE, assistance, and fixed-seed evaluation.
- [`simulation/environment.py`](../simulation/environment.py): simulator setup,
  camera/frame handling, IK conversion, and action execution.
- [`simulation/agent.py`](../simulation/agent.py) and
  [`simulation/rollout_reward.py`](../simulation/rollout_reward.py): runtime
  hierarchical agent and its simpler rollout reward accumulator.
- [`reward.py`](../reward.py) and [`planning_metrics.py`](../planning_metrics.py):
  reference-plan diagnostics, validity-gated planner reward, and planning metrics.
