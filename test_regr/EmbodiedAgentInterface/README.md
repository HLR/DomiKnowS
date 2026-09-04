# Embodied Agent Interface (EAI) DomiKnowS Framework

For canonical joint EAI/VLABench training with one dynamically activated root
graph and shared Qwen2.5-VL/LoRA backbone, see
[`../JointEmbodiedAgentInterface/README.md`](../JointEmbodiedAgentInterface/README.md).

This directory implements the DomiKnowS baseline and reinforcement learning framework for the **Embodied Agent Interface (EAI)** benchmark ([`Inevitablevalor/EmbodiedAgentInterface`](https://huggingface.co/datasets/Inevitablevalor/EmbodiedAgentInterface)).

The task requires an embodied agent to plan and generate multi-step action-object trajectories to achieve goal states specified by natural language instructions and temporal logic conditions across **BEHAVIOR** and **VirtualHome** environments.

---

## Key Features

- **Goal State Translation & Abstract World Simulator** (`reward.py`):
  - Full relational and state tracking: `inside(x, y)`, `ontop(x, y)`, `nextto(x, y)`, `under(x, y)`, `open(x)`, `closed(x)`, `on(x)`, `off(x)`, `clean(x)`, `soaked(x)`, `cooked(x)`, `frozen(x)`, `unfrozen(x)`, `sliced(x)`, `dusty(x)`, `stained(x)`.
  - Supports dual-hand BEHAVIOR interaction formats (`left_place_*`, `right_place_*`) and VirtualHome bracketed formats (`[ACTION] <obj> (id)`).
  - Evaluates final condition satisfaction: binary **$0/1$ Goal Success** (`gt_state_success`) and **Dense Goal Fact Recall** (`gt_state_recall`).
  - Achieves **100.0% goal satisfaction verification** across all 438 examples in the full benchmark dataset.

- **DomiKnowS Reinforcement Learning Integration** (`domiknows.reinforcement`):
  - Per-item reward closures (`make_eai_reward_function`) returning standard PyTorch reward tensors compatible with `ReinforcementProgram`.
  - Action token sequence decoder (`eai_action_decoder`) for policy gradient estimators (`reinforce`, `reinforce_with_baseline`).
  - A separate deterministic world/trajectory graph (`world_graph.py`) exposes namespaced action and state concepts for DomiKnowS constraints. Applicable constraints modulate task progress with `task_reward * ((1 - weight) + weight * constraint_score)`; they cannot independently reward a failed plan.

- **Two-Stage Training Pipeline (`--two-stage`)**:
  - **Stage 1**: Supervised Exact Match cross-entropy pretraining via `SolverPOIProgram`; EOS padding after the first sequence-ending EOS is excluded from the loss. Every epoch reports semantic exploration metrics and decoded examples. The best semantic epoch is retained as a compact trainable-parameter snapshot, restored after training, and saved to `<model-stem>.stage1.pth`; a later collapsed epoch cannot replace it.
  - **Stage 2**: Reinforcement learning fine-tuning via `ReinforcementProgram`, using final-state recall multiplied by ordered SimpleTL prefix progress, world constraints as a discount for violations, and a teacher-forced Stage 1 anchor to prevent catastrophic forgetting. The default is on-policy REINFORCE with 8 rollouts; the optional importance-weighted estimator records and detaches the rollout policy probability before applying proposal correction. Samples are true autoregressive rollouts conditioned on their own generated prefixes, not gold teacher-forced prefixes. During `no_grad` sampling, Qwen evaluates all rollouts as one batch and reuses their KV cache whenever the selected labels have equal native-token widths; unequal widths trigger one safe batched prefix/cache rebuild, never one forward per rollout. Differentiable rescoring remains teacher-forced and backpropagates one rollout microbatch at a time while accumulating the complete 8-rollout gradient before one optimizer step. The Stage 1 optimizer is released before Stage 2. RL starts only when the restored Stage 1 checkpoint meets the configured positive-reward, recall, and goal-success thresholds; otherwise the command retains Stage 1 and exits with status 2. Every RL epoch is evaluated semantically; the best epoch is retained, restored, and recorded in the final Stage 2 checkpoint instead of automatically saving the last epoch.

- **Model Backbones**:
  - **Tiny Transformer**: Lightweight autoregressive generator with BERT instruction encoder.
  - **Small LLM (Qwen)**: Causal-LM backbone support, including `Qwen/Qwen3-8B`, with optional PEFT / LoRA adaptation. EAI task fields are rendered as a structured user message through the tokenizer's native chat template with one assistant generation marker and Qwen3 thinking disabled. The assistant-side action prefix uses the same token IDs and next-label boundaries in Stage 1 teacher forcing, batched KV-cached RL rollout sampling, and differentiable rescoring. Checkpoint metadata records `prompt_format=qwen-chat-label-prefix-v1`; checkpoints trained with the older raw prompt are rejected and must be retrained. The default EAI label head uses fixed vectors from Qwen's native output embeddings plus a trainable low-rank residual, bias, and temperature; `--causal-label-head linear` preserves older linear-head architecture, but not old prompt semantics.

- **DFA Relational Constraints & Hybrid Inference**:
  - The DomiKnowS generation graph declares the first-token action rule, action/object successors, zero-argument actions, action/object compatibility, EOS closure, and maximum length. These marked logical constraints compile into the single `EAIProgramBundle.policy_dfa`; EAI adds no runtime policy overlays.
  - Object arguments are additionally guarded by the generic `domiknows.generation` contextual-DFA facility. Each VirtualHome task's PDDL `:objects` section supplies non-gold entity-type facts, the same facts are included in the model prompt, and semantic action→object transitions reject labels whose entity type is absent. Scene-navigation objects remain legal. A separate graph declaration permits `clean` only for tasks whose instruction, goal, or transition model contains a cleaning cue. Thus a book task cannot decode `clean bathtub_35`; neither rule reads that task's reference action trajectory. BEHAVIOR/iGibson filtering remains inactive because its action labels and PDDL use different taxonomies (for example, `hardback` versus `book`); enabling it requires an explicit ontology mapping rather than a gold-derived mask.
  - The same compiled DFA masks supervised evaluation, RL sampling, differentiable RL rescoring, and Qwen + HMM + DFA lookahead inference (`infer_qwen_hmm_dfa.py`).

Open [`eai-two-stage-flow.html`](eai-two-stage-flow.html) for an interactive view of both stages. Its concrete `book_demo` trajectory shows the exact Qwen prompt, SimpleTL goal, task-world inputs, gold labels, simulator snapshots, constraints, rewards, and outputs; each token step reveals only the inputs needed at that point and activates the components involved below.

See [`EAI_Operation.md`](EAI_Operation.md) for a detailed operational description of Stage 1 supervised learning, the Stage 1-to-Stage 2 handoff, and Stage 2 reinforcement learning.

---

## SimpleTL goals

A **SimpleTL goal** is the task's temporal-logic success specification. It states which facts or actions must hold and, when relevant, the order in which they must occur. For example:

```text
(exists x0. (GRAB(x0))) then exists x0. (READ(x0))
```

requires the agent to grab an object and later read it. Reading first and grabbing afterward does not fully satisfy the goal, even if the final state contains similar facts. The `tl_goal` field, rather than the reference action sequence, is the authority for reward and task-success evaluation. SimpleTL supports `and`, `or`, `not`, quantifiers, and temporal `then` expressions.

---

## Directory Structure

```
test_regr/EmbodiedAgentInterface/
├── dataset.py                # Dataset loader for BEHAVIOR, VirtualHome, and dummy splits
├── graph.py                  # DomiKnowS declarative graph and logical constraints
├── world_graph.py            # Independent world/trajectory schema and constraint verifier
├── modules.py                # Tiny Transformer & Qwen CausalLM autoregressive generators
├── reward.py                 # Goal state translator, 0/1 reward function, and RL decoder
├── train.py                  # CLI argument parser and training configurations
├── main.py                   # Main training, two-stage pipeline, evaluation, and scoring
├── test_reward.py            # Unit test suite verifying goal translation on all 438 examples
├── test_world_graph.py       # World schema, materialization, blending, and scale regressions
├── infer_qwen_hmm_dfa.py     # Qwen + HMM + DFA inference-only evaluation
├── eai_hmm_decoder_adapter.py# Adapter for HMM-DFA lookahead decoding
└── evaluate_settings.py      # Benchmark evaluation configurations
```

---

## Quickstart & Usage

### 1. Test Goal State Translation & Reward Function

Run the test suite verifying state extraction, goal satisfaction, and DomiKnowS reward compatibility on both dummy data and the full 438-example EAI dataset:

```powershell
uv run python test_regr/EmbodiedAgentInterface/test_reward.py
uv run python test_regr/EmbodiedAgentInterface/test_world_graph.py
```

### Declaring future world constraints

`build_program()` enables source-state action preconditions by default. Placement actions require an object in the appropriate hand, release/drop requires a held object, and pour requires some held source object. Place-inside actions additionally require an open destination when the simulator knows its open/closed status. Because the flat EAI action format names only a placement destination, an idempotent placement/release is also accepted when the referenced object is already spatially placed. These preconditions were audited against all 438 reference demonstrations.

The constraint reward averages only preconditions applicable to the materialized trajectory. These include action-argument existence in the task's PDDL world. Container openness is skipped when its initial status is unknown rather than treating missing information as `closed`. Inactive preconditions remain logically vacuous but do not inflate `world_constraint_score`.

Built-in constraints use a deterministic evaluator equivalent to their declared graph invariants during RL, avoiding per-sample solver construction. Constraints supplied by custom builders continue to materialize `DataNode`s and run `verifyResultsLC`. Per-example reward closures also cache repeated sampled trajectories.

Pass `world_constraint_builders=()` programmatically, or `--no-world-constraints` on the CLI, to recover the unblended task reward. Additional builders can be supplied to `build_program(world_constraint_builders=(...))` or directly to `build_eai_world_graph`. Aliases such as `switchon` resolve to the canonical state concept and do not create duplicate concepts. Action and state names are independent (`action__open` and `state__open`).

```python
from world_graph import build_eai_world_graph

def inspect_world_handles(bundle):
    # Handles available for relational/temporal constraints added later.
    open_state = bundle.states["open"]
    closed_state = bundle.states["closed"]
    open_action = bundle.actions["open"]
    first_argument = bundle.action_roles["arg1"]
    adjacent_steps = bundle.next_step
    source_step = bundle.action_roles["source_step"]
    result_step = bundle.action_roles["result_step"]
    current_step = bundle.step_roles["current"]
    following_step = bundle.step_roles["following"]

world = build_eai_world_graph(
    include_default_constraints=True,
    constraint_builders=(inspect_world_handles,),
)
```

Each trajectory contains ordered state steps, entity nodes (including `character` and an absent-argument sentinel), explicit action source/result/actor/argument links, complete unary groundings, and only the binary pairs actually tracked by the task or simulation. Action-event sub-concepts such as `precondition__placement_source_ready` carry deterministic truth values derived from the source snapshot, so declared precondition constraints and custom relational constraints can be verified with `/local/argmax`.

`bundle.state` and `bundle.action` are the actual `world_state` and `world_action` DomiKnowS concepts. Canonical sub-concepts are available through `bundle.states[name]` and `bundle.actions[name]`; for example, `state__open` is directly an `is_a` sub-concept of `world_state`, while `action__open` is directly an `is_a` sub-concept of `world_action`. Unary state instances use the absent-object sentinel in the common state `object` role, allowing unary and sparse binary groundings to share one valid concept hierarchy.

### 2. Two-Stage Training (Exact Match $\rightarrow$ Reinforcement Learning)

Train a two-stage model (Stage 1 Exact Match pretraining $\rightarrow$ Stage 2 dense-reward Reinforcement Learning fine-tuning):

```powershell
# Fast smoke test on dummy data
uv run python test_regr/EmbodiedAgentInterface/main.py --dummy --two-stage --epochs 2 --rl-epochs 2 --max-steps 8 --evaluate

# Training on a subset of the full EAI dataset
uv run python test_regr/EmbodiedAgentInterface/main.py --dataset all --limit 50 --two-stage --epochs 3 --rl-epochs 3 --max-steps 30 --evaluate

# Full dataset training
uv run python test_regr/EmbodiedAgentInterface/main.py --dataset all --two-stage --epochs 5 --rl-epochs 5 --max-steps 30 --evaluate
```

### 3. Small LLM Backbone Training (Qwen)

Run two-stage training with a Qwen causal language model backbone:

```powershell
# Dummy verification with Qwen
uv run python test_regr/EmbodiedAgentInterface/main.py --dummy --baseline-model causal-lm --llm-backbone-path Qwen/Qwen2.5-1.5B-Instruct --two-stage --epochs 1 --rl-epochs 1 --max-steps 4 --evaluate

# Qwen training on full EAI dataset with LoRA
uv run python test_regr/EmbodiedAgentInterface/main.py --dataset all --limit 50 --baseline-model causal-lm --llm-backbone-path Qwen/Qwen2.5-1.5B-Instruct --use-lora --lora-r 16 --two-stage --epochs 3 --rl-epochs 3 --max-steps 20 --evaluate

# Target one-H100 Qwen3-8B experiment
$env:CUDA_VISIBLE_DEVICES=3
uv run python test_regr/EmbodiedAgentInterface/main.py --dataset all --two-stage --epochs 5 --rl-epochs 5 --max-steps 30 --evaluate --baseline-model causal-lm --llm-backbone-path Qwen/Qwen3-8B --llm-device-map auto --use-lora --lora-r 8 --lora-alpha 16 --rl-num-samples 8 --device cuda:0 --model test_regr/EmbodiedAgentInterface/models/eai_qwen3_8b_lora.pth
```

#### Optional text-only VLABench warm-up

VLABench planning episodes can warm the same Qwen LoRA parameters before EAI
Stage 1. This is transfer learning, not dataset concatenation: VLABench retains
its graph-owned `skill:`, `arg:`, and `obj:` vocabulary, temporary label
adapter, generation graph, and compiled DFA. Prompts contain only the task
instruction and numbered entity table. Images, controller demonstrations,
VLABench simulator rewards, and VLABench Stage 2 are never loaded. The
temporary adapter and optimizer are released after the best auxiliary epoch is
restored; EAI then trains with its original vocabulary, adapter, graph, DFA,
simulator, SimpleTL reward, and official split.

```powershell
$env:CUDA_VISIBLE_DEVICES=3
uv run python -u test_regr/EmbodiedAgentInterface/main.py --dataset all --two-stage --epochs 5 --rl-epochs 5 --max-steps 30 --evaluate --baseline-model causal-lm --llm-backbone-path Qwen/Qwen3-8B --use-lora --lora-r 8 --lora-alpha 16 --device cuda:0 --vlabench-aux-epochs 2 --model test_regr/EmbodiedAgentInterface/models/eai_qwen3_8b_lora.pth
```

`--vlabench-aux-limit` limits locally loaded planning episodes and
`--vlabench-aux-lr` overrides the default EAI Stage 1 learning rate. The
EAI data layer downloads/resumes `VLABench/vlm_evaluation_v1.0` under
`test_regr/EmbodiedAgentInterface/data/vlabench_planning` the first time the
auxiliary phase is enabled. `--vlabench-aux-planning-dir` overrides that local
location; controller data is never downloaded. The
selected warm-up is saved as `<model-stem>.vlabench_aux.pth`; its epoch and
domain/vocabulary checksums are also recorded as optional provenance in later
EAI checkpoints. Setting auxiliary epochs to zero leaves the existing EAI
pipeline unchanged and performs no VLABench download.

### 4. Standalone Program Modes

- **Supervised Exact Match (`SolverPOIProgram`)**:
  ```powershell
  uv run python test_regr/EmbodiedAgentInterface/main.py --dataset all --program solver --epochs 3 --train --evaluate
  ```
- **Primal-Dual Constraint Program (`PrimalDualProgram`)**:
  ```powershell
  uv run python test_regr/EmbodiedAgentInterface/main.py --dataset all --program primal-dual --epochs 3 --train --evaluate
  ```
- **Standalone Reinforcement Learning (`ReinforcementProgram`)**:
  ```powershell
  uv run python test_regr/EmbodiedAgentInterface/main.py --dataset all --program reinforcement --epochs 3 --train --evaluate
  ```

### 5. Inference-Only Qwen + HMM + DFA Decoding

Run zero-shot / inference-only constrained decoding using a pretrained or distilled HMM together with compiled DFA constraints:

```powershell
uv run python test_regr/EmbodiedAgentInterface/infer_qwen_hmm_dfa.py --dataset all --limit 100 --eval-limit 100 --hmm models/eai_all_qwen25_ctrlg_hmm.npz --baseline-model causal-lm --llm-backbone-path Qwen/Qwen2.5-1.5B-Instruct --hmm-dfa-base hmm --output test_regr/EmbodiedAgentInterface/results/results_qwen_hmm_dfa.txt
```

---

## Evaluation Metrics

During evaluation, `sequence_score` reports:
- `examples`: Total evaluated test instances.
- `exact_sequence`: Fraction of trajectories with an exact action-object match to gold.
- `token_accuracy`: Per-step accuracy through the first gold EOS; trailing EOS padding is excluded.
- `nonempty_plan_rate`: Fraction of predictions containing at least one action token before EOS.
- `average_predicted_length`: Mean number of non-EOS generated labels.
- `positive_reward_rate`: Fraction of predictions with positive unmodulated task reward; this controls the Stage 1-to-Stage 2 gate.
- `dfa_valid`: Fraction of generated trajectories satisfying the compiled DomiKnowS declarative grammar constraints. Reported as `n/a` when DFA checking is disabled.
- `gt_state_success`: Binary $0/1$ final state satisfaction evaluated by the world state simulator.
- `gt_state_recall`: Mean fraction of goal condition facts satisfied by the generated trajectory.
- `world_constraint_score_applicable`: Mean satisfaction over examples with at least one applicable world constraint, or `n/a` when none apply.
- `world_constraints_applicable_per_example`: Mean number of distinct constraint definitions applicable to each generated trajectory.
- `world_constraints_declared`: Number of world-constraint definitions declared in the graph; most are intentionally non-applicable to any one trajectory.
- `rl_reward_score`: Dense or binary task reward, discounted by applicable constraint violations.

---

## Model Artifacts & Checkpoint Sizes

When training or running inference, you may encounter different model artifacts in `test_regr/EmbodiedAgentInterface/models/`:

### 1. PyTorch Neural Network Checkpoints (`*.pth`, e.g., `eai_action_sequence_baseline.pth`)
- **Size**: ~**6.17 GB** (when using `Qwen/Qwen2.5-1.5B-Instruct`) or a few **MB** (when using `--baseline-model tiny-transformer`).
- **Contents**: Full state dictionary of the deep neural network. For `Qwen2.5-1.5B-Instruct`, all **1.54 billion parameters** are saved in Float32 precision:
  $$1,543,785,472 \text{ parameters} \times 4 \text{ bytes} \approx 6,175,141,888 \text{ bytes} \approx 6.175 \text{ GB}$$
- **Usage**: Used to resume training or evaluate the learned neural policy with `load_trained_program(...)`.

### 2. Distilled HMM Matrices (`*.npz`, e.g., `eai_all_qwen25_ctrlg_hmm.npz`)
- **Size**: ~**44.6 KB**.
- **Contents**: Discrete statistical transition matrix $A \in \mathbb{R}^{K \times K}$, emission matrix $B \in \mathbb{R}^{K \times |V|}$, and initial state distribution $\pi \in \mathbb{R}^K$. It contains **no** neural network weights.
- **Usage**: Used exclusively by `infer_qwen_hmm_dfa.py` and `HMMDFADecoder` for lookahead future-constraint satisfaction scoring during constrained generation.

### Summary Comparison

| Artifact | File Type | Typical Size | Stored Content | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `eai_action_sequence_baseline.pth` | PyTorch Tensor Checkpoint (`.pth`) | **~6.17 GB** (Qwen-1.5B) / **~15 MB** (Tiny) | Full neural network parameter weights | Model training, two-stage fine-tuning, & neural policy inference |
| `eai_all_qwen25_ctrlg_hmm.npz` | Compressed NumPy Archive (`.npz`) | **~44.6 KB** | Discrete state transition & emission matrices ($A, B, \pi$) | HMM + DFA lookahead scoring during constrained decoding |

> **Note on Storage Optimization**:
> - If you train with `--baseline-model tiny-transformer` instead of `--baseline-model causal-lm`, the `.pth` file will be much smaller (a few megabytes for the tiny autoregressive head).
> - When training Qwen with LoRA (`--use-lora`), you can save only the LoRA adapter parameter delta instead of the full 6.17 GB base model.

---

## Command-Line Options Reference

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--dataset` | `str` | `"dummy"` | Dataset name (`"dummy"`, `"behavior"`, `"virtualhome"`, or `"all"`). |
| `--data-path` | `str` | `None` | Path to local parquet/csv/jsonl dataset file. |
| `--limit` | `int` | `None` | Cap number of examples loaded from dataset. |
| `--two-stage` | `flag` | `False` | Run Stage 1 Exact Match followed by Stage 2 Reinforcement Learning. |
| `--program` | `str` | `"solver"` | Program type: `"solver"`, `"primal-dual"`, or `"reinforcement"`. |
| `--epochs` | `int` | `5` | Number of epochs for Stage 1 / supervised training. |
| `--rl-epochs` | `int` | `3` | Number of epochs for Stage 2 Reinforcement Learning. |
| `--lr` | `float` | architecture-aware | Stage 1 learning rate: `1e-4` for causal LM/LoRA and `1e-3` for smaller baselines unless explicitly set. |
| `--rl-lr` | `float` | architecture-aware | Stage 2 learning rate: `1e-5` for causal LM/LoRA and `1e-4` for smaller baselines unless explicitly set. |
| `--rl-estimator` | `str` | `"reinforce"` | On-policy REINFORCE, or proposal-corrected `importance_weighted`. |
| `--rl-num-samples` | `int` | `8` | Rollouts per Stage 2 item; Stage 2 requires at least 4. |
| `--rl-rescore-microbatch` | `int` | `1` | Number of differentiable Qwen rollouts retained simultaneously; gradients still accumulate over all rollouts before stepping. |
| `--rl-reward-mode` | `str` | `"dense"` | Dense final-state recall times ordered temporal-prefix progress, or binary goal success. |
| `--rl-supervised-weight` | `float` | `0.5` | Weight of the teacher-forced Stage 1 anchor loss retained during RL. |
| `--rl-constraint-weight` | `float` | `0.25` | Maximum task-reward discount for world-constraint violations. |
| `--rl-constraint-aggregate` | `str` | `"mean"` | Constraint aggregation: `mean`, `min`, or `prod`. |
| `--no-world-constraints` | `flag` | `False` | Disable the default world and transition constraints and bypass reward blending. |
| `--baseline-model` | `str` | `"tiny-transformer"` | Model backbone: `"tiny-transformer"` or `"causal-lm"`. |
| `--llm-backbone-path`| `str` | `None` | Hugging Face model path or ID (e.g. `Qwen/Qwen2.5-1.5B-Instruct`). |
| `--causal-label-head` | `str` | `"pretrained-adapter"` | Native Qwen label-vector adapter, or legacy `linear`. |
| `--label-adapter-rank` | `int` | `64` | Low-rank residual size for the pretrained label adapter. |
| `--use-lora` | `flag` | `False` | Enable PEFT / LoRA adapters for Causal-LM backbone. |
| `--max-steps` | `int` | `8` | Maximum decoding/generation horizon per episode. |
| `--generation-constraints` | `str` | `"always"` | Apply the graph-compiled DFA during RL and evaluation (`always`), evaluation only (`eval`), or disable it as an explicit ablation (`off`). |
| `--use-dfa` | `flag` | `False` | Deprecated alias for `--generation-constraints always`. |
| `--stage1-checkpoint` | `path` | `<model-stem>.stage1.pth` | Explicit retained Stage 1 checkpoint path. |
| `--epoch-predictions` | `int` | `3` | Number of validation predictions printed after each Stage 1 epoch. |
| `--stage1-min-positive-reward-rate` | `float` | `0.25` | Minimum validation exploration rate required before RL. |
| `--stage1-min-goal-recall` | `float` | `0.10` | Minimum validation goal-fact recall required before RL. |
| `--stage1-min-goal-success-rate` | `float` | `0.05` | Minimum validation goal-success rate required before RL. |
| `--evaluate` | `flag` | `False` | Run evaluation after training. |

