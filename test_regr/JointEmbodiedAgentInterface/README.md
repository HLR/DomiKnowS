# Unified EAI/VLABench two-stage agent

This package is the canonical workflow for training EmbodiedAgentInterface
(EAI) and VLABench together. It creates one DomiKnowS root graph, one
Qwen3-VL-8B-Instruct backbone with one 4-bit LoRA adapter, two compact label heads,
and one sequential program lifecycle. The standalone EAI and VLABench CLIs
remain supported for component debugging.

The planner encodes each text or text-plus-vision observation with Qwen once.
Separate EAI and VLABench graph-token embeddings and recurrent decoders reuse
that differentiable context for the complete teacher-forced sequence or
autoregressive trajectory. Multiple Stage 2 candidates for the same EAI item
or VLABench observation also share that context. This avoids one full Qwen
execution per prefix or sampled candidate. Qwen's non-reentrant layer
checkpointing bounds activation memory during its
single context pass. `--planner-decoder-hidden-dim` controls the compact
decoder width and defaults to `512`.

## Combined graph and activation

The root owns a small semantic spine and four sibling subgraphs:

```text
joint_embodied_world
├── embodied_episode
├── embodied_entity
├── embodied_operation
├── EAI world graph
├── VLABench world graph
├── EAI generation graph
└── VLABench generation graph
```

The EAI and VLABench episode, entity, and operation concepts inherit from the
three shared concepts. Domain semantics remain separate: each world graph is
still authoritative for its own actions or skills, roles, transitions,
validation, logical constraints, vocabulary, and checksum. Dataset records
validate these definitions; they do not redefine them.

Before every sequential batch or rollout group, `JointDomainRuntime` enters
`domain_scope("eai")` or `domain_scope("vlabench")`. The scope calls
`root.set_active_concepts(...)` with Concept objects from the selected world
and generation profiles. Required `is_a` ancestors and every graph constraint
concept are added automatically. Inactive properties, sensors, and logical
constraints are skipped. The previous domain is restored on normal exit and
on exceptions; the outermost scope restores the default in which all concepts
are active.

Names and Concept objects may be mixed through
`runtime.activate_domain(domain, extra_concepts=...)`. Equal short names in
sibling generation graphs use qualified names internally, while the runtime
uses Concept identity to avoid process-global naming suffix problems.

Activation is mutable graph state. `domain_scope` holds a reentrant lock for
the entire scope, so joint execution is serialized. Do not run two domain
updates concurrently against the same runtime. Switching domains never
rebuilds the graph, program, model, heads, controller, or optimizers.

## Training stages

Stage 1 constructs one `JointSolverPOIProgram(SolverPOIProgram)` with both
sensor branches attached. Each round performs one EAI update and one VLABench
update, giving the domains equal weight:

1. EAI teacher-forced exact action/entity sequence learning with EOS-masked
   cross entropy.
2. VLABench teacher-forced compact graph-token plan learning with EOS-masked
   cross entropy, followed by one controller behavior-cloning update.

The two planner losses are differentiated together. Conflicting gradients on
shared Qwen/LoRA parameters are projected with PCGrad and the pair produces
one optimizer step, so update order cannot let one domain immediately
overwrite the other. Domain-only decoders retain their own gradients. The
controller is updated only during the VLABench turn. An epoch defaults to the
smaller loader length; override it
with `--stage1-rounds-per-epoch`. Stage 1 checkpoint selection first maximizes
the minimum of EAI goal success and VLABench exact graph match, then their
mean, EAI recall, VLABench validity, and validation loss. The EAI exploration
gate is checked before Stage 2.

After the selected Stage 1 checkpoint is restored, the controller receives a
dedicated behavior-cloning warm-up (20,000 updates by default) without changing
either planner head or the shared LoRA. This is intentionally separate from
the balanced domain rounds so extra controller supervision does not increase
VLABench planner weight. Pose supervision uses wrapped local deltas normalized
by the same translation/rotation scales as the deployed controller, rather
than regressing absolute end-effector coordinates. The completed warm-up is saved as
`joint_controller_warmup.pt`; resume from that file skips both Stage 1 and the
already-completed warm-up. Configure or disable it with
`--controller-warmup-steps N` (use `0` to disable).

Stage 2 constructs one
`JointReinforcementProgram(ReinforcementProgram)` over the same root and the
exact same `JointQwenVLPlanner`. By default Stage 2 freezes the shared
Qwen/LoRA and trains only the separate EAI and VLABench graph-token decoders;
this prevents one domain's policy gradient from erasing the other domain's
representation. Use `--no-stage2-freeze-shared-backbone` only as an explicit
shared-adapter ablation:

1. EAI samples eight prefix-conditioned, DFA-masked trajectories. Its
   SimpleTL/final-state/world-constraint reward trains the planner with
   REINFORCE, plus a `0.5` teacher-forced EAI anchor.
2. VLABench performs eight simulator rollouts. Each replan samples four
   constrained graph plans. Simulator return-to-go trains the planner with
   REINFORCE plus a `0.1` supervised planner anchor. Controller actions use
   PPO/GAE with a `0.05` behavior-cloning anchor.

The controller uses the official LeRobot `task_index` for all 128 language
instructions from `meta/tasks.parquet`; it does not collapse distinct target
objects into a shared primitive-pattern ID. It also consumes the active
graph-plan skill and operation position. The entity slot stays on its padding
row because demonstrations do not provide stable segmentation-to-graph
pointer labels. Stage 1
derives the operation from normalized demonstration phase; Stage 2 uses the
simulator progress cursor with normalized episode phase as a fallback when
progress remains flat. During rollout, the environment
instruction must resolve to exactly one of those IDs before controller
execution. The controller uses tanh-transformed Normal distributions for six end-effector coordinates, a
Bernoulli gripper, learned log standard deviation, and a value head. Its actor
predicts bounded local xyz/Euler increments and cumulatively integrates them
around the last observed end-effector pose. The public actions remain absolute
coordinates, but a biased head therefore cannot make the robot repeatedly
walk toward one remote, unreachable pose. Translation and rotation use
separate exploration-noise scales. It runs four-action receding-horizon chunks. Each
`[x,y,z,roll,pitch,yaw,gripper]` action is converted with
`get_qpos_from_ee_pos`; the binary gripper becomes two `0.04` (open) or `0.0`
(closed) finger commands. PPO uses `gamma=0.99`, GAE `lambda=0.95`, clip
`0.2`, two epochs, value weight `0.5`, and entropy weight `0.01`. The
controller switches from the `3e-4` BC rate to a fresh-moment `3e-5` RL rate
before Stage 2. A trust-region rollback halves that rate for the next update.
The critic is bounded to `[-1,1]`, uses clipped return targets and Smooth L1
loss, and cannot backpropagate through the actor's shared features. A batch
with zero total simulator return trains only the critic; PPO, entropy,
feasibility, and the `0.05` BC anchor cannot change its actor. Mixed-return
batches contrast successful and unsuccessful valid executions, and a singleton
positive advantage remains uncentered so sparse success is preserved.
PPO stores the bounded sampled action and its change-of-variables-corrected
log probability. Likelihood ratios are bounded before exponentiation, and every
actor-changing objective is checked with the standard nonnegative approximate
KL, `exp(log_ratio) - 1 - log_ratio`, using a default target of `0.03`; the
maximum absolute per-action log ratio remains a separate safety bound. This
avoids treating ordinary signed probability redistribution as policy collapse.
Every Stage 2 update records accepted PPO epochs, rollback status, approximate
KL, actor parameter-delta L2, and whether actor parameters actually changed.
Rollout summaries also report deterministic assist steps, assisted-episode
rate, and success among episodes that used zero deterministic assist steps.
The default `--execution-assistance train-only` permits those helpers only for
training collection and forcibly disables them during fixed-seed evaluation,
so reported evaluation success is controller-only. `on` is an assisted-system
ablation and `off` disables assistance in both phases.
An infeasible action is retried at smaller Cartesian scales. If all scales
fail, the unchanged observation is resampled up to three consecutive chunks
by default before the rollout is IK-truncated; each rejection remains negative
feasibility evidence for the controller.
Online actions are limited to 2 cm translation and 0.10 radians rotation per
simulator step before IK. IK uses a practical `5e-3` convergence tolerance and
up to 200 iterations. These defaults are configurable through
`--max-position-step`, `--max-rotation-step`, `--ik-tolerance`, and
`--ik-max-steps`.

Constraint-invalid plans never reach the controller. Non-finite actions receive
zero and are not sent to the environment. An IK failure is retried at `0.5`,
`0.25`, and `0.125` of the bounded delta. A hold-position command is not
misreported as recovery; exhausted retries reject that sampled action. Recovery
and truncation counts are reported per task, and a feasibility penalty targets
the exact fully rejected action rather than an executable chunk prefix.

## Reward separation

EAI reward and VLABench simulator reward are never added, averaged, or
substituted for each other. Each reward creates a policy-gradient loss only
inside its active domain scope. Dynamic activation controls graph execution;
it does not isolate trainable tensors. Stage 1 therefore applies PCGrad to the
shared Qwen/LoRA gradients, and Stage 2 freezes the shared Qwen/LoRA by default.
If `--no-stage2-freeze-shared-backbone` is selected, alternating optimizer
steps again create explicit cross-domain gradient coupling even though graph
activation remains correct.

For EAI, the reward is computed from the task `tl_goal`, predicted temporal
state, and applicable EAI world constraints. Its `0.5` teacher-forced term is
a separate loss, not part of the reward.

For VLABench, each controller chunk stores
`0.25 * delta_progress + 0.10 * delta_intention`, plus a transient graph-operation
potential when progress advances the active plan cursor. The terminal correction adds
success, efficiency, and initial-score terms so the stored rewards telescope
exactly to the final simulator formula:

```text
clip(0.60 * success
   + 0.25 * final_progress
   + 0.10 * final_intention
   + 0.05 * efficiency, 0, 1)
```

Planner decisions receive simulator return-to-go. Reference-plan similarity
remains an evaluation metric and is not blended into Stage 2 reward. The
`0.1` planner and `0.05` controller anchors are supervised losses, not reward
components.

## Data paths and canonical command

Run commands from the repository root. Defaults are relative to the source
tree, not a drive-specific `D:\datasets` directory:

```text
test_regr/VLABenchAgentInterface/data/planning
test_regr/VLABenchAgentInterface/data/control
test_regr/JointEmbodiedAgentInterface/checkpoints
```

Download VLABench processed data first with the standalone downloader:

```powershell
python -m test_regr.VLABenchAgentInterface.main download `
  --planning-dir test_regr\VLABenchAgentInterface\data\planning `
  --control-dir test_regr\VLABenchAgentInterface\data\control
```

The EAI, VLABench, and Joint planner defaults now use the same
`Qwen/Qwen3-VL-8B-Instruct` base model. On GPU2, download it once to a
persistent host path shared with the `vigorous_easley` container:

```bash
docker exec vigorous_easley /opt/dominows/venv/bin/hf download \
  Qwen/Qwen3-VL-8B-Instruct \
  --local-dir /home/auszok/models/Qwen/Qwen3-VL-8B-Instruct \
  --max-workers 4
```

The September 20 GPU2 download completed at that path. Pass
`--planner-model /home/auszok/models/Qwen/Qwen3-VL-8B-Instruct` to use
those files without fetching the model again. The three workflows share
the base model files, not a trained adapter or a checkpoint. Start a fresh
output directory when changing backbones; checkpoints from Qwen2.5-VL or
text-only Qwen3-8B cannot be resumed as Qwen3-VL-8B runs. The EAI standalone
CLI also requires `--baseline-model causal-lm --use-lora` to select this
backbone; its tiny-transformer debugging default is unchanged.

For a quick model-loading check on GPU2, run inside the container:

```bash
python -m test_regr.smoke_common_backbone --component joint \
  --model-path /home/auszok/models/Qwen/Qwen3-VL-8B-Instruct
```

Canonical joint training uses all EAI data, all ten VLABench tasks, five
Stage 1 epochs, a 20,000-step controller BC warm-up, three Stage 2 epochs, and
equal round-robin scheduling:

```powershell
python -m test_regr.JointEmbodiedAgentInterface.main train-agent --two-stage
```

On GPU2, add `--planner-model
/home/auszok/models/Qwen/Qwen3-VL-8B-Instruct` and choose a fresh
`--output` directory to load the downloaded files and avoid legacy
checkpoint collisions.

Dataset indexing, model initialization, Stage 1 rounds, Stage 2 domain turns,
and simulator rollouts emit flushed, newline-based progress. The messages
remain visible when both streams are redirected to a file and followed with
`tail -f`; they do not depend on terminal cursor control.
Each Stage 2 epoch reports VLABench `episodes`, `successes`, `success_rate`,
`valid_rate`, mean return, mean steps, IK failures/recoveries, IK truncation,
and execution completion separately for every task, in
addition to the aggregate metrics. This makes a success rate concentrated in
one task visible instead of presenting it as broad multi-task performance.
Control-video decoding uses a per-task LRU capped at eight TorchCodec decoders,
preventing full shuffled runs from exhausting the process file-descriptor
limit. The cap is configurable with `--video-decoder-cache-size`.

By default, training runs three fixed-seed simulator rollouts per task before
RL and after every Stage 2 epoch. Checkpoint ranking and acceptance use these
evaluation rollouts; the update-producing rollouts remain under
`vlabench_training` in the checkpoint metrics. Override the default with
`--stage2-eval-rollouts-per-task`, or set it to `0`
for a fast diagnostic run that intentionally uses training-rollout metrics.
The pre-RL evaluation is also a controller preflight: by default it requires at
least a `0.01` positive-return rate and no more than `0.50` IK truncation.
Failure writes `stage2-skipped` before any multi-hour RL epoch. Success and
task-coverage thresholds remain available but default to zero because the
fixed-seed baseline remains too small for a definitive statistical claim.
Joint Stage 2 also applies baseline-relative regression gates. By default an RL
checkpoint must match or exceed fixed-seed Stage 1 VLABench success
(`--stage2-max-baseline-success-regression 0.0`) and EAI success
(`--stage2-max-eai-success-regression 0.0`), while EAI recall may fall by at
most `0.02` (`--stage2-max-eai-recall-regression`). A regression aborts the
remaining epochs and restores the prior eligible checkpoint. The planner and
controller retain stronger Stage 1 anchors by default
(`--stage2-planner-anchor-weight 0.25`, `--stage2-controller-bc-weight 0.25`).
Configure these values explicitly when running controlled ablations.

Joint uses the same ordered 80/20 EAI split as the standalone EAI workflow and
scores the complete 88-example holdout by default. `paired_eai_eval.py` can
score any retained standalone or Joint checkpoint on those exact row identities
with `--selection canonical-validation`, including decoded actions and missed
goal facts for every failure.

Configure these thresholds with the `--stage2-preflight-*` options. Setting
evaluation rollouts to zero intentionally disables this gate.

Override the relative paths when necessary:

```powershell
python -m test_regr.JointEmbodiedAgentInterface.main train-agent --two-stage `
  --eai-data-path path\to\eai.json `
  --vlabench-planning-dir path\to\planning `
  --control-source path\to\control `
  --env-factory test_regr.VLABenchAgentInterface.environment:create_environment
```

The official VLABench Python 3.10 simulator environment is required for real
Stage 2 rollouts. Regression tests use a deterministic fake environment.

Component debugging commands remain available:

```powershell
python -m test_regr.EmbodiedAgentInterface.main --help
python -m test_regr.VLABenchAgentInterface.main train-planner --help
python -m test_regr.VLABenchAgentInterface.main train-controller --help
python -m test_regr.VLABenchAgentInterface.main train-agent --help
```

## Checkpoints and resume

Every epoch writes a resumable joint checkpoint, and every Stage 2 round
atomically refreshes `joint_stage2_progress.pt`. It identifies the immutable
shared backbone through its checked model configuration and stores its
trainable LoRA parameters once, both label heads, controller and value head,
both optimizer states, stage, epoch, round-robin cursor,
Python/NumPy/Torch/CUDA RNG states, both vocabularies and DFA configurations,
activation-profile version, model configuration, and the individual and
combined domain checksums. Frozen bitsandbytes NF4 base weights and their
loader-specific quantization buffers are reconstructed from the configured
backbone instead of being duplicated in every epoch checkpoint.
The progress checkpoint additionally records the next round, accumulated
training metrics, and any previously eligible best epoch. Resuming it does not
repeat completed simulator rollouts; if interruption occurs after the last
round but before evaluation, resume performs the missing evaluation and epoch
checkpoint.

An epoch is eligible to become `joint_stage2_best.pt` only when its aggregate
VLABench rollout success is at least `0.10`, at least three task families have
a success, and no more than `0.25` of rollouts terminate at IK. Configure these
acceptance gates with `--stage2-min-vlabench-success-rate`,
`--stage2-min-successful-tasks`, and `--stage2-max-ik-truncation-rate`.
Epoch checkpoints are always written even when they miss a gate. If every epoch misses them, training reports
`stage2-best-skipped` and does not label a weak epoch as the best model.
If fixed-seed VLABench evaluation loses the preflight task-signal or feasibility
gate, Stage 2 aborts remaining epochs and restores the last eligible Stage 2
checkpoint or the Stage 1/controller-warm-up checkpoint. That failed epoch is
kept for diagnosis but marked non-resumable by the checkpoint loader.

The controller-only warm-up additionally writes
`joint_controller_warmup.pt`. When resuming an existing Stage 1 checkpoint,
the warm-up runs before Stage 2. If the process stops later, resume the warm-up
checkpoint to avoid repeating those controller updates.
Joint checkpoint version 8 rejects older Stage 2 checkpoints, whose controller
updates include trajectories collected before the robot-frame rollout bridge.
It also rejects current Stage 2 epoch checkpoints that failed their fixed-seed
retention gate. Older Stage 1 and controller-warm-up checkpoints remain valid
resume sources.

Resume with:

```powershell
python -m test_regr.JointEmbodiedAgentInterface.main train-agent --two-stage `
  --resume test_regr\JointEmbodiedAgentInterface\checkpoints\joint_stage1_epoch_004.pt
```

After warm-up has completed, prefer:

```powershell
python -m test_regr.JointEmbodiedAgentInterface.main train-agent --two-stage `
  --resume test_regr\JointEmbodiedAgentInterface\checkpoints\joint_controller_warmup.pt
```

During Stage 2, prefer the latest round checkpoint after an interruption:

```powershell
python -m test_regr.JointEmbodiedAgentInterface.main train-agent --two-stage `
  --resume test_regr\JointEmbodiedAgentInterface\checkpoints\joint_stage2_progress.pt
```

Loading rejects a checkpoint when either domain definition, vocabulary, DFA,
activation profile, graph-decoder architecture, controller action
representation, or model configuration
differs. Checkpoints created before graph-decoder version 1 cannot be resumed
because their prefix-reprompt label heads have incompatible parameters.
Checkpoints created with the former unconstrained absolute-pose, unbounded
sample distribution, or non-plan-conditioned controller may
be resumed only from Stage 1. Loading resets that obsolete policy head and its
optimizer moments, then the configured controller warm-up retrains the local
chunk head and language/graph-operation embeddings. An old `joint_controller_warmup.pt` or Stage 2 checkpoint is
rejected because it has already crossed the migration boundary.
The physically scaled delta-BC objective is likewise compatibility-versioned:
an older Stage 1 or structurally compatible warm-up checkpoint resets and
re-warms the controller, while an older Stage 2 checkpoint is rejected rather
than silently retaining the absolute-coordinate objective.
Controller migration rejects `--controller-warmup-steps 0` because proceeding
with the reset, untrained action head would invalidate the simulator run.
Checkpoints with the correct language-conditioned actor but the former
unbounded critic are migrated at any stage by resetting only the critic and
controller optimizer moments; the learned action policy is preserved.
Standalone EAI and
VLABench checkpoints continue to work with their original CLIs but are not
joint checkpoints and cannot be resumed directly here. Activation is reset to
all concepts after restoration.

## Source code map

All non-test source files in this package are listed below.

| File | Functionality |
| --- | --- |
| `__init__.py` | Exposes the joint runtime, graph builder, shared planner, and both program classes. |
| `world_graph.py` | Builds the shared semantic spine, attaches sibling domain and generation graphs, compiles both DFAs, creates identity-based activation profiles, provides locked domain scopes, and computes joint checksums. |
| `models.py` | Loads one Qwen3-VL-8B/LoRA backbone, encodes each observation once, owns separate EAI/VLABench graph-token embeddings, recurrent decoders, label heads, and prompts, and provides teacher-forced and DFA-masked autoregressive domain APIs. |
| `program.py` | Implements equal Stage 1 round-robin updates, the controller-only BC warm-up, and equal Stage 2 EAI-REINFORCE/VLABench-REINFORCE-plus-PPO updates with domain-local activation and rewards. |
| `checkpoint.py` | Atomically saves and restores the complete joint state, RNGs, scheduling cursor, and compatibility metadata. |
| `main.py` | Defines the canonical `train-agent --two-stage` CLI, data/model construction, balanced checkpoint keys, exploration gate, and per-epoch resume files. |
