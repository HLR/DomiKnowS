# Unified EAI/VLABench two-stage agent

This package is the canonical workflow for training EmbodiedAgentInterface
(EAI) and VLABench together. It creates one DomiKnowS root graph, one
Qwen2.5-VL-3B backbone with one 4-bit LoRA adapter, two compact label heads,
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

Both planner turns update the same LoRA/backbone optimizer. Only the selected
domain label head receives gradients. The controller is updated only during
the VLABench turn. An epoch defaults to the smaller loader length; override it
with `--stage1-rounds-per-epoch`. Stage 1 checkpoint selection first maximizes
the minimum of EAI goal success and VLABench exact graph match, then their
mean, EAI recall, VLABench validity, and validation loss. The EAI exploration
gate is checked before Stage 2.

After the selected Stage 1 checkpoint is restored, the controller receives a
dedicated behavior-cloning warm-up (20,000 updates by default) without changing
either planner head or the shared LoRA. This is intentionally separate from
the balanced domain rounds so extra controller supervision does not increase
VLABench planner weight. The completed warm-up is saved as
`joint_controller_warmup.pt`; resume from that file skips both Stage 1 and the
already-completed warm-up. Configure or disable it with
`--controller-warmup-steps N` (use `0` to disable).

Stage 2 constructs one
`JointReinforcementProgram(ReinforcementProgram)` over the same root and the
exact same `JointQwenVLPlanner`:

1. EAI samples eight prefix-conditioned, DFA-masked trajectories. Its
   SimpleTL/final-state/world-constraint reward trains the planner with
   REINFORCE, plus a `0.5` teacher-forced EAI anchor.
2. VLABench performs eight simulator rollouts. Each replan samples four
   constrained graph plans. Simulator return-to-go trains the planner with
   REINFORCE plus a `0.1` supervised planner anchor. Controller actions use
   PPO/GAE with a `0.05` behavior-cloning anchor.

The controller uses the official LeRobot `task_index` for all 128 language
instructions from `meta/tasks.parquet`; it does not collapse distinct target
objects into a shared primitive-pattern ID. During rollout, the environment
instruction must resolve to exactly one of those IDs before controller
execution. The controller uses Normal distributions for six end-effector coordinates, a
Bernoulli gripper, learned log standard deviation, and a value head. Its actor
predicts bounded local xyz/Euler increments and cumulatively integrates them
around the last observed end-effector pose. The public actions remain absolute
coordinates, but a biased head therefore cannot make the robot repeatedly
walk toward one remote, unreachable pose. Translation and rotation use
separate exploration-noise scales. It runs four-action receding-horizon chunks. Each
`[x,y,z,roll,pitch,yaw,gripper]` action is converted with
`get_qpos_from_ee_pos`; the binary gripper becomes two `0.04` (open) or `0.0`
(closed) finger commands. PPO uses `gamma=0.99`, GAE `lambda=0.95`, clip
`0.2`, four epochs, value weight `0.5`, and entropy weight `0.01`.
The critic is bounded to `[-1,1]`, uses clipped return targets and Smooth L1
loss, and cannot backpropagate through the actor's shared features. Rollouts
with zero total simulator return still train the critic and `0.05` BC anchor,
but do not apply PPO or entropy gradients to failed sampled actions.
PPO stores the controller action and log probability before the deterministic
Cartesian safety transform; only the transformed command is sent to IK and the
simulator. Likelihood ratios are bounded before exponentiation, and repeated
PPO epochs stop after the mean per-action log-ratio leaves the trust region.
This keeps the importance ratio tied to the behavior policy and prevents one
stale trajectory from producing a catastrophic controller update.
Online actions are limited to 2 cm translation and 0.10 radians rotation per
simulator step before IK. IK uses a practical `1e-3` convergence tolerance and
up to 200 iterations. These defaults are configurable through
`--max-position-step`, `--max-rotation-step`, `--ik-tolerance`, and
`--ik-max-steps`.

Constraint-invalid plans never reach the controller. Failed inverse
kinematics and non-finite actions receive zero and are not sent to the
environment. A finite target that fails IK safely truncates the rollout at its
last executable state; it does not retroactively invalidate earlier actions or
erase their accumulated shaping reward.

## Reward separation

EAI reward and VLABench simulator reward are never added, averaged, or
substituted for each other. Each reward creates a policy-gradient loss only
inside its active domain scope. The alternating optimizer steps both update
the shared LoRA, which is the only cross-domain coupling.

For EAI, the reward is computed from the task `tl_goal`, predicted temporal
state, and applicable EAI world constraints. Its `0.5` teacher-forced term is
a separate loss, not part of the reward.

For VLABench, each controller chunk stores
`0.25 * delta_progress + 0.10 * delta_intention`. The terminal correction adds
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

Canonical joint training uses all EAI data, all ten VLABench tasks, five
Stage 1 epochs, a 20,000-step controller BC warm-up, three Stage 2 epochs, and
equal round-robin scheduling:

```powershell
python -m test_regr.JointEmbodiedAgentInterface.main train-agent --two-stage
```

Dataset indexing, model initialization, Stage 1 rounds, Stage 2 domain turns,
and simulator rollouts emit flushed, newline-based progress. The messages
remain visible when both streams are redirected to a file and followed with
`tail -f`; they do not depend on terminal cursor control.
Control-video decoding uses a per-task LRU capped at eight TorchCodec decoders,
preventing full shuffled runs from exhausting the process file-descriptor
limit. The cap is configurable with `--video-decoder-cache-size`.

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

Every epoch writes a resumable joint checkpoint. It identifies the immutable
shared backbone through its checked model configuration and stores its
trainable LoRA parameters once, both label heads, controller and value head,
both optimizer states, stage, epoch, round-robin cursor,
Python/NumPy/Torch/CUDA RNG states, both vocabularies and DFA configurations,
activation-profile version, model configuration, and the individual and
combined domain checksums. Frozen bitsandbytes NF4 base weights and their
loader-specific quantization buffers are reconstructed from the configured
backbone instead of being duplicated in every epoch checkpoint.

The controller-only warm-up additionally writes
`joint_controller_warmup.pt`. When resuming an existing Stage 1 checkpoint,
the warm-up runs before Stage 2. If the process stops later, resume the warm-up
checkpoint to avoid repeating those controller updates.

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

Loading rejects a checkpoint when either domain definition, vocabulary, DFA,
activation profile, graph-decoder architecture, controller action
representation, or model configuration
differs. Checkpoints created before graph-decoder version 1 cannot be resumed
because their prefix-reprompt label heads have incompatible parameters.
Checkpoints created with the former unconstrained absolute-pose or collapsed
skill-pattern-conditioned controller may
be resumed only from Stage 1. Loading resets that obsolete policy head and its
optimizer moments, then the configured controller warm-up retrains the local
chunk head and language-task embedding. An old `joint_controller_warmup.pt` or Stage 2 checkpoint is
rejected because it has already crossed the migration boundary.
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
| `models.py` | Loads one Qwen2.5-VL/LoRA backbone, encodes each observation once, owns separate EAI/VLABench graph-token embeddings, recurrent decoders, label heads, and prompts, and provides teacher-forced and DFA-masked autoregressive domain APIs. |
| `program.py` | Implements equal Stage 1 round-robin updates, the controller-only BC warm-up, and equal Stage 2 EAI-REINFORCE/VLABench-REINFORCE-plus-PPO updates with domain-local activation and rewards. |
| `checkpoint.py` | Atomically saves and restores the complete joint state, RNGs, scheduling cursor, and compatibility metadata. |
| `main.py` | Defines the canonical `train-agent --two-stage` CLI, data/model construction, balanced checkpoint keys, exploration gate, and per-epoch resume files. |
