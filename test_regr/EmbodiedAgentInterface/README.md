# Embodied Agent Interface (EAI) DomiKnowS Framework

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

- **Two-Stage Training Pipeline (`--two-stage`)**:
  - **Stage 1**: Supervised Exact Match cross-entropy pretraining via `SolverPOIProgram`.
  - **Stage 2**: Reinforcement learning fine-tuning via `ReinforcementProgram` guided by the binary $0/1$ goal satisfaction reward.

- **Model Backbones**:
  - **Tiny Transformer**: Lightweight autoregressive generator with BERT instruction encoder.
  - **Small LLM (Qwen)**: Causal-LM backbone support (`Qwen/Qwen2.5-1.5B-Instruct` or `Qwen/Qwen2.5-0.5B-Instruct`) with optional PEFT / LoRA adaptation.

- **DFA Relational Constraints & Hybrid Inference**:
  - Declarative DomiKnowS action-object relational grammar constraints compiled into Deterministic Finite Automata (DFA).
  - DFA-guided constrained autoregressive decoding and Qwen + HMM + DFA lookahead inference (`infer_qwen_hmm_dfa.py`).

---

## Directory Structure

```
test_regr/EmbodiedAgentInterface/
├── dataset.py                # Dataset loader for BEHAVIOR, VirtualHome, and dummy splits
├── graph.py                  # DomiKnowS declarative graph and logical constraints
├── modules.py                # Tiny Transformer & Qwen CausalLM autoregressive generators
├── reward.py                 # Goal state translator, 0/1 reward function, and RL decoder
├── train.py                  # CLI argument parser and training configurations
├── main.py                   # Main training, two-stage pipeline, evaluation, and scoring
├── test_reward.py            # Unit test suite verifying goal translation on all 438 examples
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
```

### 2. Two-Stage Training (Exact Match $\rightarrow$ Reinforcement Learning)

Train a two-stage model (Stage 1 Exact Match pretraining $\rightarrow$ Stage 2 Reinforcement Learning fine-tuning with 0/1 rewards):

```powershell
# Fast smoke test on dummy data
uv run python test_regr/EmbodiedAgentInterface/main.py --dummy --two-stage --epochs 2 --rl-epochs 2 --max-steps 8 --evaluate

# Training on a subset of the full EAI dataset
uv run python test_regr/EmbodiedAgentInterface/main.py --dataset all --limit 50 --two-stage --epochs 3 --rl-epochs 3 --max-steps 30 --evaluate

# Full dataset training
uv run python test_regr/EmbodiedAgentInterface/main.py --dataset all --two-stage --epochs 5 --rl-epochs 5 --max-steps 30 --evaluate
```

### 3. Small LLM Backbone Training (Qwen)

Run two-stage training with `Qwen/Qwen2.5-1.5B-Instruct` as the causal language model backbone:

```powershell
# Dummy verification with Qwen
uv run python test_regr/EmbodiedAgentInterface/main.py --dummy --baseline-model causal-lm --llm-backbone-path Qwen/Qwen2.5-1.5B-Instruct --two-stage --epochs 1 --rl-epochs 1 --max-steps 4 --evaluate

# Qwen training on full EAI dataset with LoRA
uv run python test_regr/EmbodiedAgentInterface/main.py --dataset all --limit 50 --baseline-model causal-lm --llm-backbone-path Qwen/Qwen2.5-1.5B-Instruct --use-lora --lora-r 16 --two-stage --epochs 3 --rl-epochs 3 --max-steps 20 --evaluate
```

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
uv run python test_regr/EmbodiedAgentInterface/infer_qwen_hmm_dfa.py --dataset all --limit 100 --eval-limit 100 --hmm models/eai_all_qwen25_ctrlg_hmm.npz --baseline-model causal-lm --llm-backbone-path Qwen/Qwen2.5-1.5B-Instruct --hmm-dfa-base hmm --output results_qwen_hmm_dfa.txt
```

---

## Evaluation Metrics

During evaluation, `sequence_score` reports:
- `examples`: Total evaluated test instances.
- `exact_sequence`: Fraction of trajectories with an exact action-object match to gold.
- `token_accuracy`: Per-step token prediction accuracy.
- `dfa_valid`: Fraction of generated trajectories satisfying the compiled DomiKnowS declarative grammar constraints.
- `gt_state_success`: Binary $0/1$ final state satisfaction evaluated by the world state simulator.
- `gt_state_recall`: Mean fraction of goal condition facts satisfied by the generated trajectory.

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
| `--epochs` | `int` | `10` | Number of epochs for Stage 1 / supervised training. |
| `--rl-epochs` | `int` | `10` | Number of epochs for Stage 2 Reinforcement Learning. |
| `--lr` | `float` | `1e-3` | Learning rate for Stage 1 training. |
| `--rl-lr` | `float` | `1e-4` | Learning rate for Stage 2 policy gradient optimization. |
| `--baseline-model` | `str` | `"tiny-transformer"` | Model backbone: `"tiny-transformer"` or `"causal-lm"`. |
| `--llm-backbone-path`| `str` | `None` | Hugging Face model path or ID (e.g. `Qwen/Qwen2.5-1.5B-Instruct`). |
| `--use-lora` | `flag` | `False` | Enable PEFT / LoRA adapters for Causal-LM backbone. |
| `--max-steps` | `int` | `8` | Maximum decoding/generation horizon per episode. |
| `--use-dfa` | `flag` | `False` | Enable DFA-constrained autoregressive greedy decoding. |
| `--evaluate` | `flag` | `False` | Run evaluation after training. |

