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
| [`dataset.py`](dataset.py) | Owns dataset identifiers; performs resumable Hugging Face downloads; loads planning and LeRobot control examples; creates numbered views and fixed history/action windows; and makes episode-level splits. |
| [`models.py`](models.py) | Implements the Qwen2.5-VL compact-label planner head with teacher-forced sequence logits and DFA-masked autoregressive logits, plus 4-bit/LoRA loading. Implements the controller's six Normal pose outputs, Bernoulli gripper, learned log standard deviation, and value head. |
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

The downloader defaults to one worker, honors `Retry-After`, and resumes the
existing Hugging Face snapshot after HTTP 429 or transient failures. If
necessary, run `uv run hf auth login` and repeat the same command; do not
delete the partial data.

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

Stage 2 constructs `VLABenchHierarchicalReinforcementProgram`, a
`ReinforcementProgram` subclass holding the identical planner-head object.
The planner samples genuine autoregressive graph-label trajectories and uses
simulator return-to-go REINFORCE. A `0.1` supervised exact-plan anchor prevents
catastrophic drift. Reference-plan similarity remains an evaluation metric
and is not blended into the Stage 2 reward.

The controller samples six end-effector coordinates from Normal distributions
and the gripper from a Bernoulli distribution. PPO uses `gamma=0.99`, GAE
`lambda=0.95`, clip `0.2`, four PPO epochs, value weight `0.5`, and entropy
weight `0.01`; a `0.05` behavior-cloning anchor is retained. The controller
executes four actions before replanning. Each action
`[x,y,z,roll,pitch,yaw,gripper]` is converted with
`get_qpos_from_ee_pos`; the gripper becomes two `0.04` (open) or `0.0`
(closed) finger commands.

The canonical 24 GB GPU command runs both stages, samples all ten tasks
uniformly, configures four planner samples and eight simulator rollouts per
update, and writes an epoch checkpoint after Stage 1 and each RL epoch:

```powershell
uv run python -m test_regr.VLABenchAgentInterface.main train-agent --two-stage `
  --planning-dir test_regr\VLABenchAgentInterface\data\planning `
  --control-source test_regr\VLABenchAgentInterface\data\control `
  --task all `
  --output test_regr\VLABenchAgentInterface\checkpoints\agent `
  --sft-epochs 3 --controller-epochs 20 --rl-epochs 10 `
  --rl-num-samples 4 --rollouts-per-update 8
```

Qwen defaults to 4-bit NF4 LoRA and the controller freezes SigLIP features.
Simulator rollouts are sequential. Install the applicable CUDA, PEFT,
quantization, and video-decoding extras before a full run.

Joint checkpoints contain planner, controller, value head, both optimizer
states, stage/epoch, Python/NumPy/PyTorch RNG states, graph vocabulary, and the
world-domain checksum. Resume at either stage boundary with:

```powershell
uv run python -m test_regr.VLABenchAgentInterface.main train-agent --two-stage `
  --planning-dir test_regr\VLABenchAgentInterface\data\planning `
  --control-source test_regr\VLABenchAgentInterface\data\control `
  --output test_regr\VLABenchAgentInterface\checkpoints\agent `
  --resume test_regr\VLABenchAgentInterface\checkpoints\agent\agent_rl_epoch_003.pt
```

Resume is rejected before training if either the graph-derived domain checksum
or vocabulary differs. Stage 1 resumes at joint boundaries; a reinforcement
checkpoint restores the next RL epoch and both optimizer/RNG states.

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

Invalid plans never reach the controller. Failed/non-finite end-effector
conversion invalidates the trajectory and gives both policies zero simulator
return.

The controller receives these chunk rewards through PPO and GAE. Each selected
planner decision receives the simulator return-to-go from that decision onward
through REINFORCE. Constraint-invalid planner samples receive zero return and
never execute the controller.

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
