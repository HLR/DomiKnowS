# VLABench graph-first hierarchical agent

For canonical joint EAI/VLABench training with one dynamically activated root
graph and shared Qwen2.5-VL/LoRA backbone, see
[`../JointEmbodiedAgentInterface/README.md`](../JointEmbodiedAgentInterface/README.md).

This package implements the same two-stage program structure used by
`test_regr/EmbodiedAgentInterface`: supervised learning through a DomiKnowS
`SolverPOIProgram`, followed by a shared-head `ReinforcementProgram`. The
second stage jointly updates a compact-label Qwen2.5-VL planner with simulator
REINFORCE and a continuous actor-critic controller with PPO.

`world_graph.py` is the start of the domain definition. It is not a schema
consumer. Skills, semantic roles, legal primitive-task patterns, transitions,
canonical plan validation, DomiKnowS concepts/relations/constraints, and the
domain checksum are defined there. The former `schema.py` module was removed
intentionally; importing it is a breaking error.

## Source code map

All non-test source files in this package are listed below.

| File | Functionality |
| --- | --- |
| [`__init__.py`](__init__.py) | Preserves package exports for `canonicalize_plan`, `validate_plan`, and the graph-owned `PlanVocabulary`, and exposes reward helpers. |
| [`world_graph.py`](world_graph.py) | Authoritative VLABench domain. Defines skills, argument roles, primitive-task automata, canonical plans, semantic DomiKnowS concepts and relations, hard logical constraints, plan materialization/verification, controller condition IDs, and a stable domain checksum. |
| [`graph.py`](graph.py) | Derives the compact planner token vocabulary, entity-pointer codecs, generation graph, and task-pattern DFA exclusively from a `VLABenchWorldGraphBundle`. Dataset examples may validate against this vocabulary but cannot add skills or roles. |
| [`dataset.py`](dataset.py) | Owns dataset identifiers; performs resumable Hugging Face downloads; loads planning and LeRobot control examples; creates numbered views and fixed history/action windows; bounds and releases TorchCodec decoder handles; and makes episode-level splits. |
| [`models.py`](models.py) | Implements one-pass Qwen2.5-VL context encoding followed by a compact recurrent graph-token decoder with teacher-forced sequence logits and DFA-masked autoregressive logits, plus 4-bit/LoRA loading. Implements the plan-conditioned controller's bounded transformed-Normal pose outputs, Bernoulli gripper, learned log standard deviation, and value head. |
| [`program.py`](program.py) | Builds Stage 1 `SolverPOIProgram` sensors and EOS-masked loss. Defines `VLABenchHierarchicalReinforcementProgram`, simulator collection, planner return-to-go REINFORCE with a supervised anchor, PPO/GAE with a behavior-cloning anchor, and hard plan rejection. |
| [`training.py`](training.py) | Builds the world-first constraint runtime, graph-label examples, component training/evaluation helpers, both program stages, and resumable joint checkpoints with RNG and graph-domain validation. |
| [`reward.py`](reward.py) | Implements diagnostic reference-plan scoring, semantic hard gating, DomiKnowS reward closures, and the authoritative final simulator rollout formula. Reference similarity is not used by Stage 2. |
| [`environment.py`](environment.py) | Lazily creates official VLABench environments, renders numbered graph-pointer observations, and converts 7D end-effector actions to joint commands with two finger controls. |
| [`agent.py`](agent.py) | Runs constrained compact planning and four-action receding-horizon control online, rejecting invalid plans and non-finite actions before environment execution. |
| [`main.py`](main.py) | CLI for dataset download/inspection, vocabulary export, component debugging, canonical two-stage training, evaluation, and rollout. |

## Data and simulator setup

Run commands from the repository root. Every path below is relative to
`test_regr/VLABenchAgentInterface`; no drive-specific dataset path is required.
The processed data is downloaded to:

```text
test_regr/VLABenchAgentInterface/data/planning
test_regr/VLABenchAgentInterface/data/control
```

Download the two processed repositories (the approximately 817 GB raw HDF5
repository is not required):

```powershell
uv run python -m test_regr.VLABenchAgentInterface.main download `
  --planning-dir test_regr\VLABenchAgentInterface\data\planning `
  --control-dir test_regr\VLABenchAgentInterface\data\control
```

On Linux:

```bash
unset HF_HUB_DISABLE_PROGRESS_BARS
python -m test_regr.VLABenchAgentInterface.main download \
  --planning-dir test_regr/VLABenchAgentInterface/data/planning \
  --control-dir test_regr/VLABenchAgentInterface/data/control
```

The downloader defaults to one worker, honors `Retry-After`, and resumes the
existing Hugging Face snapshot after HTTP 429 or transient failures. It routes
both the outer snapshot counter and the inner HTTP/Xet byte and reconstruction
bars through a newline-based terminal reporter, so progress remains visible on
redirected Linux servers without invoking any `tqdm` cursor, notebook, or async
renderer. If necessary, run `uv run hf auth login` and repeat the same command;
do not delete the partial data.

The processed sources are
[VLM planning episodes](https://huggingface.co/datasets/VLABench/vlm_evaluation_v1.0)
and [10-task LeRobot control data](https://huggingface.co/datasets/VLABench/vlabench_primitive_ft_lerobot_video).
Preserve their official splits locally.

Inspect either snapshot before training:

```powershell
uv run python -m test_regr.VLABenchAgentInterface.main inspect `
  --planning-dir test_regr\VLABenchAgentInterface\data\planning

uv run python -m test_regr.VLABenchAgentInterface.main inspect `
  --control-source test_regr\VLABenchAgentInterface\data\control --task add_condiment
```

For real rollouts, install
[OpenMOSS/VLABench](https://github.com/OpenMOSS/VLABench) in its supported
Python 3.10 environment and download the simulator assets:

```powershell
git clone https://github.com/OpenMOSS/VLABench.git test_regr\VLABenchAgentInterface\data\simulator\VLABench
Set-Location test_regr\VLABenchAgentInterface\data\simulator\VLABench
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts\download_assets.py
```

## Domain graph, vocabulary, and constraints

Construction is deliberately one-way:

```text
world_graph.py domain bundle
  -> graph.py compact vocabulary and token codecs
  -> DomiKnowS GenerationEncoder graph
  -> DFA masks used by training, sampling, and inference
```

The semantic graph contains plan, operation, entity, and transition concepts;
operation-to-entity grounding relations; skill subtypes; and named logical
constraints for skill uniqueness, required/forbidden roles, pointer validity,
legal task patterns, and adjacent transitions. A compact plan resembles:

```text
skill:pick role:target_entity_name entity:3
skill:place role:target_container_name entity:7 <eos>
```

The DFA masks illegal next labels after every sampled prefix. It rejects
unknown pointers, missing or forbidden roles, illegal transitions, incomplete
task patterns, trailing labels, and missing EOS. A second semantic check
materializes the decoded plan as DomiKnowS `DataNode`s. Dataset observations
only provide the current entity table; they never redefine domain skills or
signatures.

`build-vocab` exports the graph-derived vocabulary and both its vocabulary and
domain checksums:

```powershell
uv run python -m test_regr.VLABenchAgentInterface.main build-vocab `
  --planning-dir test_regr\VLABenchAgentInterface\data\planning `
  --output test_regr\VLABenchAgentInterface\checkpoints\vocab.json
```

## Two-stage joint learning

Stage 1 uses one planner head inside `SolverPOIProgram`. DomiKnowS reader,
edge, learner, and label sensors supply graph-derived target labels. Cross
entropy includes the first EOS but masks EOS padding after it. In the same
stage, the controller learns demonstrations by behavior cloning.
Qwen encodes each text-plus-vision observation once; a compact GRU conditions
on that vector and teacher-forces the entire graph-token target in one pass.
The old prefix-reprompt implementation performed a complete Qwen pass for
every target token and is intentionally checkpoint-incompatible.

Stage 2 constructs `VLABenchHierarchicalReinforcementProgram`, a
`ReinforcementProgram` subclass holding the identical planner-head object.
The planner samples genuine autoregressive graph-label trajectories and uses
simulator return-to-go REINFORCE. A `0.1` supervised exact-plan anchor prevents
catastrophic drift. Reference-plan similarity remains an evaluation metric
and is not blended into the Stage 2 reward.

The control loader preserves the official LeRobot `task_index` values for all
128 language instructions in `meta/tasks.parquet`. Distinct requested objects
therefore retain distinct controller conditions instead of being collapsed
into one primitive skill-pattern ID. Online execution resolves the environment
instruction through the same metadata and rejects unknown instructions.
The controller additionally receives the active graph skill and operation
position; demonstration phase supplies this context in Stage 1 and the
selected plan cursor supplies it online. The entity-context slot remains at
its padding value because demonstrations do not identify a stable
segmentation-to-graph pointer correspondence; this avoids selecting untrained
entity embedding rows online. Simulator
progress/intention advances that cursor when available; normalized episode
phase is the fallback when those signals stay flat, matching the
demonstration-window convention.
The controller samples six bounded local end-effector deltas from
tanh-transformed Normal distributions
and the gripper from a Bernoulli distribution. Its pose head predicts bounded
local xyz/Euler increments, cumulatively integrates the chunk around the last
observed pose, and exposes the resulting absolute end-effector targets to the
existing dataset and PPO interfaces. The official LeRobot state and action xyz
coordinates are relative to the robot base, while VLABench observations and IK
targets are world-frame. Online rollout subtracts the live robot-base position
before controller inference, applies safety/recovery in that learned frame, and
adds the base back exactly once at the IK boundary. Each rollout logs the frame
contract and measured world-frame base so a stale server process is visible.
Position and rotation have
separate physical exploration-noise scales. Behavior cloning compares these
wrapped pose deltas in the same physically scaled space instead of regressing
large absolute coordinates. PPO uses `gamma=0.99`, GAE
`lambda=0.95`, clip `0.2`, four PPO epochs, value weight `0.5`, and entropy
weight `0.01`; a `0.05` behavior-cloning anchor is retained. The bounded
critic uses clipped targets and Smooth L1 loss without changing shared actor
features. A batch with no positive task return trains only the detached critic;
it cannot change the actor through PPO, entropy, feasibility, or the supervised
anchor. Once a batch has task signal, PPO contrasts successful and unsuccessful
valid executions. A singleton positive advantage remains uncentered so sparse
success is not erased. The controller
stores the bounded policy sample and its change-of-variables-corrected behavior
log probability for PPO; an independent Cartesian envelope protects custom or
legacy controllers. Likelihood ratios are bounded before exponentiation. Every
actor-changing objective participates in the mean per-action log-ratio check,
including the final PPO pass. Crossing the trust region restores the complete
pre-update controller and clears stale optimizer moments. The controller
executes four actions before replanning. Each action
`[x,y,z,roll,pitch,yaw,gripper]` is converted with
`get_qpos_from_ee_pos`; the gripper becomes two `0.04` (open) or `0.0`
(closed) finger commands. The default safety envelope permits at most 2 cm of
translation and 0.10 radians of rotation per simulator action. IK uses a
`5e-3` convergence tolerance and at most 200 iterations; the hierarchical
program retries a failed target at `0.5`, `0.25`, and `0.125` scale. A
hold-position command is not counted as recovery. Exhausted retries reject the
current action chunk and provide a controller feasibility penalty on the exact
rejected action, not an earlier executable prefix. Recovered targets are not
counted as rejected policy actions. The policy
resamples from the unchanged observation up to three consecutive rejected
chunks by default; only then does it truncate the rollout, without erasing
reward accumulated by earlier valid actions. Override this bounded retry
budget with `--max-consecutive-ik-rejections`.

The canonical 24 GB GPU command runs both stages, samples all ten tasks
uniformly, configures four planner samples and eight simulator rollouts per
update, and writes an epoch checkpoint after Stage 1 and each RL epoch. It also
atomically refreshes `agent_rl_progress.pt` after every RL round:

```powershell
uv run python -m test_regr.VLABenchAgentInterface.main train-agent --two-stage `
  --planning-dir test_regr\VLABenchAgentInterface\data\planning `
  --control-source test_regr\VLABenchAgentInterface\data\control `
  --task all `
  --output test_regr\VLABenchAgentInterface\checkpoints\agent `
  --sft-epochs 3 --controller-warmup-steps 20000 --rl-epochs 3 `
  --rl-rounds-per-epoch 10 --rl-num-samples 4 --rollouts-per-update 8 `
  --eval-rollouts-per-task 3
```

Qwen defaults to a 512-wide graph decoder, 4-bit NF4 LoRA, and one backbone
pass per example. The controller freezes SigLIP features. Controller BC is
bounded by update count rather than twenty full passes over 459,675 windows;
set `--controller-warmup-steps 0` only when deliberately using the slower
`--controller-epochs` debugging path.
Simulator rollouts are sequential. Install the applicable CUDA, PEFT,
quantization, and video-decoding extras before a full run.
The ten RL rounds visit each primitive task once per epoch. Fixed-seed
simulator evaluation runs before RL and after every RL epoch; training-rollout
and evaluation metrics remain separate. Use at least
`--eval-rollouts-per-task 3` for the six-setting report, or `0` only for a
short diagnostic run. Setting `--rl-epochs 0` produces the supervised-only
VLABench setting and still writes `agent_stage1_evaluated.pt`.
The pre-RL evaluation is also a learning-signal and controller-feasibility
gate. By default, at least `0.01` of fixed-seed episodes must produce positive
task return and no more than `0.50` may truncate at IK. Failure writes
`reinforcement-skipped` and stops before a multi-hour RL run. Configure positive
return, success, task-coverage, and IK thresholds with
the `--rl-preflight-*` options. Setting evaluation rollouts to zero
intentionally disables this gate.

Every RL epoch is retained for diagnosis, but only an epoch with at least
`0.10` fixed-seed success, three successful task families, and no more than
`0.25` IK truncation can become `agent_rl_best.pt`. Configure those thresholds
with `--rl-min-success-rate`, `--rl-min-successful-tasks`, and
`--rl-max-ik-truncation-rate`. Training-rollout success cannot override a
failed fixed-seed evaluation.
If an epoch no longer meets the preflight learning-signal/feasibility gate,
training writes `reinforcement-aborted`, restores the last eligible RL epoch or
the evaluated supervised checkpoint, and does not spend later epochs extending
a collapsed policy. The rejected epoch checkpoint remains available for
diagnosis but is marked non-resumable by the loader.
TorchCodec decoders use a per-task LRU capped at eight open videos by default;
override it with `--video-decoder-cache-size` if the process has an unusually
low file-descriptor limit.

Standalone checkpoint version 5 contains trainable LoRA/graph-decoder state,
versioned controller semantics, controller, value head, both optimizer
states, stage/epoch, Python/NumPy/PyTorch RNG states, graph vocabulary, and the
world-domain checksum. Resume at either stage boundary with:

```powershell
uv run python -m test_regr.VLABenchAgentInterface.main train-agent --two-stage `
  --planning-dir test_regr\VLABenchAgentInterface\data\planning `
  --control-source test_regr\VLABenchAgentInterface\data\control `
  --output test_regr\VLABenchAgentInterface\checkpoints\agent `
  --resume test_regr\VLABenchAgentInterface\checkpoints\agent\agent_rl_epoch_002.pt
```

Resume is rejected before training if the graph-derived domain checksum,
vocabulary, or graph-decoder configuration differs. Checkpoints from the old
prefix-reprompt planner must restart Stage 1. Stage 1 resumes at boundaries; an
epoch reinforcement checkpoint restores the next RL epoch, while
`agent_rl_progress.pt` restores the next unfinished round in the current epoch.
Both forms restore optimizer and RNG states. A resumed partial RL epoch does
not repeat the fixed-seed baseline evaluation and carries forward any prior
eligible best epoch. Older supervised checkpoints remain valid: the controller
weights were learned in robot-frame coordinates and can use the corrected
online bridge without re-warm. Version-2 through version-4 RL checkpoints are
rejected because they contain optimizer/trajectory state collected under older
execution contracts; resume their supervised checkpoint instead.
Current-version RL epoch checkpoints that failed their fixed-seed retention
gate are also rejected as resume sources.
Controller migration cannot proceed when both `--controller-warmup-steps` and
`--controller-epochs` are zero.

For debugging, component commands remain available:

```powershell
uv run python -m test_regr.VLABenchAgentInterface.main train-planner `
  --planning-dir test_regr\VLABenchAgentInterface\data\planning `
  --output test_regr\VLABenchAgentInterface\checkpoints\planner

uv run python -m test_regr.VLABenchAgentInterface.main train-controller `
  --control-source test_regr\VLABenchAgentInterface\data\control `
  --task all --output test_regr\VLABenchAgentInterface\checkpoints\controller
```

## Reward usage: planner similarity is not combined with simulator reward

The reference-plan score and simulator reward have separate purposes in the
canonical `train-agent --two-stage` loop. They are never added together.

| Stage | Planner update | Controller update |
| --- | --- | --- |
| Stage 1 | Exact-match cross-entropy through `SolverPOIProgram` | Behavior cloning from demonstrations |
| Stage 2 | REINFORCE using simulator return-to-go, plus a `0.1` supervised planner loss anchor | PPO/GAE using simulator rewards, plus a `0.05` behavior-cloning loss anchor |
| Evaluation | Reference-plan similarity is reported as a diagnostic metric | Simulator success, progress, intention, and efficiency are reported |

The `0.1` planner anchor and `0.05` controller anchor are additional supervised
loss terms; they do not alter or blend the rewards. The separate
`train_planner_reinforcement_epoch` helper can train the planner with the
reference-plan score for component debugging, but the canonical two-stage
command does not call that helper.

### Simulator reward shared by both Stage 2 policies

Within a chunk, controller rewards are shaped as:

```text
r_t = 0.25 * delta(progress) + 0.10 * delta(intention)
```

`intention` uses VLABench's latched discrete 10 cm proximity milestone. The
upstream continuous helper is not used because its value decreases as the
recorded minimum distance improves inside the threshold, which would reverse
the shaping gradient.

At termination, success, successful efficiency, the initial-score correction,
and final clipping are added so stored rewards telescope exactly to:

```text
R = clip(
    0.60 * success
  + 0.25 * final_progress
  + 0.10 * final_intention
  + 0.05 * success * (1 - steps/max_steps),
  0, 1)
```

Invalid plans never reach the controller. Non-finite actions give both
policies zero simulator return. An unrecoverable IK target is never executed;
the action chunk is retained as negative feasibility evidence and resampled
from the last valid state. Reaching the bounded consecutive-rejection budget
terminates the rollout while retaining earlier shaped reward.

The controller receives these chunk rewards through PPO and GAE. Each selected
planner decision receives the simulator return-to-go from that decision onward
through REINFORCE. Constraint-invalid planner samples receive zero return and
never execute the controller. Joint-epoch metrics include per-task episode
count, successes, success rate, validity, return, and execution length so the
aggregate success rate cannot hide task concentration.

### Diagnostic reference-plan score

The separate planner evaluation score is:

```text
0.40 skill_match + 0.40 entity_match
+ 0.10 skill_with_entity_match + 0.10 exact_graph_match
```

It corrects the upstream aggregation that otherwise tops out at `0.8`. It is
used by `evaluate-planner` and the standalone component-debugging helper, but
it is not an RL reward in Stage 2.

## Evaluation, rollout, and tests

Planner evaluation remains available with `evaluate-planner`. Online rollout
accepts an environment factory and a joint checkpoint:

```powershell
python -m test_regr.VLABenchAgentInterface.main rollout `
  --env-factory test_regr.VLABenchAgentInterface.environment:create_environment `
  --env-kwargs '{"task":"add_condiment","episode":0}' `
  --instruction "Add the requested condiment." `
  --vocab test_regr\VLABenchAgentInterface\checkpoints\agent\vocab.json `
  --agent-checkpoint test_regr\VLABenchAgentInterface\checkpoints\agent\agent_rl_epoch_009.pt
```

At rollout and during Stage 2, RGB and simulator segmentation are rendered with
the same stable numeric entity pointers used by the planning dataset.

Run the package regression suite from the repository root:

```powershell
uv run pytest -q test_regr/VLABenchAgentInterface
```

It covers graph authority, schema-module removal, token round trips, checksums,
DFA and semantic adversaries, exact Stage 1 updates, shared program heads,
differentiable constrained sampling, PPO/GAE/action conversion, telescoping
rewards, invalid-plan gating, joint updates, and exact checkpoint restoration.

## Controller diagnostics after the V6 standalone run

The supplied V6 log reports **end-to-end success 0/30**, positive-return rate
0/30, 400 steps in every episode, 2,973 failed IK attempts and 1,285 recoveries.
Its `execution_complete_rate=1.0` measures executable rollouts; reaching the
step limit does not complete a task. The reinforcement preflight gate correctly
skipped PPO. These results do not exercise the cuDNN backward/replay fix.

The observation audit found a reproducible input-scale defect: the video-decoder
path cast byte frames to floats without dividing by 255. Live controller inputs
already used unit-range RGB. All image paths now use the same RGB conversion,
including PIL, numpy, inline tensors and TorchCodec frames. State and action
units are unchanged. This establishes a code defect, not proof that it explains
every zero-return episode in V6; the old log contains no input-range telemetry.

Behavior-cloning version **3** records this corrected contract. Resuming a V6
supervised agent checkpoint through `train-agent` triggers the existing
controller migration/warm-up flow. Keep the original checkpoint for comparison,
use a new output directory and log, and complete short diagnostics before any
long training run. An older reinforcement checkpoint cannot silently resume
under changed controller semantics. Diagnostic evaluation loads existing weights
without migration or optimizer updates, and reports the saved version alongside
the current preprocessing version; it is not a reproduction of V6's old offline
metric pipeline.

Run controller-only diagnostics from the repository root on the server. This
loads the controller checkpoint and control dataset without loading the Qwen
planner, retraining, or starting PPO:

```bash
python -m test_regr.VLABenchAgentInterface.main diagnose-controller \
  --checkpoint test_regr/VLABenchAgentInterface/checkpoints/agent_stage1.pt \
  --control-source test_regr/VLABenchAgentInterface/data/control \
  --task all --split validation --max-batches 32 \
  --output test_regr/VLABenchAgentInterface/checkpoints/v6_diagnostics
```

Use the original seed, full dataset and model architecture options. `--limit`
changes the indexed population and therefore the episode split; omit it for a
held-out comparison to the original run. The batch limit applies separately to
each task, avoiding a report that only covers the first task in a concatenated
loader. Output is `diagnostics.json`, with:

- Wrapped absolute pose MAE, delta MAE, normalized delta MAE and first-action
  normalized error for each of x/y/z/roll/pitch/yaw. Translation uses meters;
  rotation uses radians. Normalization uses the model's six step scales.
- Unclipped target deltas and the fraction exceeding each action limit. Clipping
  targets during training must not hide unreachable reference movements here.
- A hold-position baseline in the same normalized delta space.
- Gripper confusion matrix, per-class precision/recall, balanced accuracy,
  majority baseline and transition precision/recall. These count action slots
  in overlapping windows, not unique episode events. Unsupported ratios are
  `null`, not zero evidence. The legacy aggregate `pose_mae` field remains for
  comparison and still mixes units and unwrapped Euler coordinates.
- Actual model-input image range. The expected range is [0, 1].

Both standalone `train-agent` and Joint preflight now log dataset camera keys,
live camera names/indices, calibration arrays when available, per-episode EE
path length, command/observed gripper transitions and initial/minimum/final
distance to **each environment task target**. Distance is measured from world
EE position to the target entity's origin, not to an inferred grasp point.
Missing target geometry is explicitly unavailable. Evaluation JSON retains this
evidence in `episode_diagnostics`; it does not change the success/reward gate.

### Camera correspondence and held-out replay

Camera order alone is insufficient as an explanation: this controller adds
view embeddings before averaging, so the same set of images is permutation
invariant in evaluation. The embeddings do not bind camera identity to image
content. A missing wrist view or a different camera set can still matter.
The [upstream converter](https://github.com/OpenMOSS/VLABench/blob/main/scripts/convert_to_lerobot.py)
selects explicit front/wrist indices, while the
[live environment](https://github.com/OpenMOSS/VLABench/blob/main/VLABench/envs/dm_env.py)
returns all cameras in simulator order. The published dataset has a third
`second_image` feature. These sources alone do not verify its exact mapping on
the installed server, so the code does not guess a replacement mapping.

For a real paired replay, provide these three files/components:

1. A JSON manifest containing held-out `task`, `episode_index`, `offset`,
   `env_kwargs`, and any restoration metadata needed by the restorer. The
   command checks episode membership in the existing validation/test split.
2. A camera-map JSON object mapping **every actual dataset camera key** to its
   corresponding live MuJoCo camera name. Use the names reported by preflight;
   do not assume `rgb[:3]` includes the wrist camera.
3. A Python callable `module:function` with signature `restore(env, descriptor)`.
   It must restore the recorded scene configuration, robot joints, object poses,
   velocities and task state for that episode/offset after `env.reset()`.
   LeRobot's seven EE-state values alone cannot restore that full scene.

Append the following options to the diagnostic command once those recorded
assets and the restorer are available:

```bash
  --replay-manifest /path/to/heldout_replay.json \
  --camera-map /path/to/camera_map.json \
  --replay-restore recorded_scene:restore \
  --replay-steps 32 --execute-horizon 4
```

For each manifest entry, the harness independently restores two environments:
one executes recorded actions through the Cartesian/IK boundary; the other
feeds fresh simulator images and EE states back to the controller. Both use
the demonstration's task/phase context to isolate controller behavior from
planner errors. Both apply the default 2 cm/0.10 rad safety envelope; an IK
failure ends that diagnostic arm without recovery retries. Each trace records
the candidate, bounded command, observed state, next reference state and
per-axis tracking error. A terminal timestep is labeled terminal, without
claiming task success.

Before stepping, each arm checks the restored EE position (5 mm per axis),
orientation (0.05 rad per Euler axis), gripper state and every camera against
the same recorded frame. Paired PNGs are exported by camera slot, with live
name, dataset key, calibration and pixel MAE. Pixel comparison requires equal
shapes and mean RGB error at most 0.05 on [0, 1]; inspect the PNGs too, especially
for low-texture scenes. A mismatch stops that arm as
`restore_or_camera_mismatch`, rather than treating an unrelated reset as a
demonstration replay. Thresholds are exposed in the Python replay API. If no
replay assets are supplied, the report explicitly records replay as unavailable.

After verifying the mapping, `--controller-camera-names NAME1 NAME2 NAME3` on
standalone `train-agent` or Joint selects cameras by name in dataset slot order
and rejects missing/duplicate names. Defaults retain the existing first-three
selection and report it as unverified. Changing the camera set requires a new
controller evaluation; neither a name match nor a unit test proves live parity.

Local regression tests use synthetic images and a deterministic simulator double
to check normalization, named selection, wrapped/weighted errors, target
identity, gripper transitions, restored-scene rejection and feedback from fresh
observations. Real VLABench replay and any resulting success improvement require
the simulator, assets and recorded held-out scene state on the server.
