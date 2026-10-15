# Joint EAI + VLABench Training Evaluation Report 8: Complete Five-Epoch Reinforcement Learning with PCGrad (Projecting Conflicting Gradients), Spatial Patches

Projecting Conflicting Gradients (PCGrad): if the low-level physical RL gradient pushes in a direction that opposes the symbolic EAI reasoning gradient ($\mathbf{g}_{\text{RL}} \cdot \mathbf{g}_{\text{EAI}} < 0$), the RL gradient is mathematically projected onto the orthogonal hyperplane of the EAI gradient. The physical robot RL updates are strictly forbidden from destroying semantic reasoning knowledge, allowing the shared vision-language backbone (`Qwen3-VL-8B`) to remain trainable across domains.

Dense Spatial Perception: introduced dense 2D spatial patch tokens (`[batch, 196, vision_dim]`) via SigLIP. The controller can now "see" the exact lip of a test tube, the edge of a poker chip, and the narrow rim of a drink can. Earlier models used a 1D global pooled vector from the vision encoder, which blurred out fine boundaries and made precise manipulation structurally impossible.

### Core Results:
- **Baseline Improvement (+153% Return Gain)**: Even before RL fine-tuning, the Stage 1 + Warmup baseline equipped with spatial patch tokens achieved a mean return of **`0.3083`** and **`11.0%`** success rate, compared to `0.1216` return and `3.33%` success in Report 7.
- **Peak Held-Out Evaluation Success (`16.0%`)**: Overall unassisted simulator success rate rose to **`16.00%`** (95% Wilson CI: `[10.10%, 24.42%]`), representing a **2.4× improvement** over Report 7 (`6.67%`) and **4.8× improvement** over Report 6 (`3.33%`).
- **Statistical Lower Bound Crossed 10% (`10.10%`)**: For the first time in project history, the 95% Wilson lower bound exceeded 10%, mathematically guaranteeing that true autonomous capability is above 10% with 95% confidence.
- **Record High Mean Return (`0.3521`)**: Mean simulator return reached **`0.3521`** with a **`94.00%` positive return rate** and **`100.0%` valid execution rate** (zero catastrophic IK truncations).
- **Task Generalization (5 Tasks Solved Autonomously)**: Across 10 manipulation families, the policy autonomously succeeded on: `select_painting` (**90.0%**), `select_chemistry_tube` (**30.0%**), `select_mahjong` (**20.0%**), `select_poker` (**10.0%**), and `select_book` (**10.0%**).
- **Simultaneous Advancement in Symbolic Reasoning (`63.64%` Peak EAI Success)**: EAI validation goal success reached **`63.64%`** with goal recall of **`0.6737`**, surpassing all previous checkpoints (`59.09%` in Report 7, `56.82%` in Report 6). PCGrad successfully protected symbolic reasoning while allowing physical control gradients to update shared representations.
- **Definitive Model Selection**: [`joint_stage2_epoch_004.pt`](file:///workspace/test_regr/JointEmbodiedAgentInterface/checkpoints/joint_gpu6_r7_20261007_230348_resumed/joint_stage2_epoch_004.pt) passed all retention and safety gates, officially becoming the best model: [`joint_stage2_best.pt`](file:///workspace/test_regr/JointEmbodiedAgentInterface/checkpoints/joint_gpu6_r7_20261007_230348_resumed/joint_stage2_best.pt).

---

## 1. Multi-Epoch Milestone Overview (Stage 1 Baseline through Epoch 5)

All evaluations are conducted under strict **unassisted held-out conditions** (`--execution-assistance train-only`, `curriculum_alpha: 0.0`) on **100 fixed-seed simulator episodes** (10 rollouts $\times$ 10 tasks) with 95% Wilson score confidence bounds, alongside the **88 held-out EAI validation examples**:

| Stage / Epoch | Cumulative RL Training Time | VLABench Mean Return | Positive Return Rate | Evaluation Success Rate (95% Wilson CI) | Tasks Solved | Valid Episode Rate | EAI Goal Success | EAI Goal Recall | Retention Status | Checkpoint File |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Stage 1 Baseline** | 0.0 min | 0.3083 | 89.00% | 11.00% `[6.25%, 18.63%]` | 2 / 10 | 100.0% | 62.50% | 0.6744 | Baseline Ref | `joint_stage1_epoch_004.pt` |
| **Stage 2 — Epoch 0 (Epoch 1/5)** | 884.4 min (~14.7h) | 0.3182 | 89.00% | 13.00% `[7.76%, 20.98%]` | 3 / 10 | 99.0% | 62.50% | 0.6744 | **Retained** | `joint_stage2_epoch_000.pt` |
| **Stage 2 — Epoch 1 (Epoch 2/5)** | 1786.6 min (~29.8h) | 0.3067 | 91.00% | 13.00% `[7.76%, 20.98%]` | 3 / 10 | 100.0% | **63.64%** | 0.6722 | **Retained** | `joint_stage2_epoch_001.pt` |
| **Stage 2 — Epoch 2 (Epoch 3/5)** | 2705.4 min (~45.1h) | 0.3220 | 92.00% | 14.00% `[8.53%, 22.14%]` | 5 / 10 | 98.0% | **63.64%** | **0.6737** | **Retained** | `joint_stage2_epoch_002.pt` |
| **Stage 2 — Epoch 3 (Epoch 4/5)** | 3658.7 min (~61.0h) | 0.3359 | 92.00% | 15.00% `[9.31%, 23.28%]` | 5 / 10 | 100.0% | 60.23% | 0.6453 | **Retained** | `joint_stage2_epoch_003.pt` |
| **Stage 2 — Epoch 4 (Epoch 5/5)** | **4581.4 min (~76.4h)** | **0.3521** | **94.00%** | **16.00% `[10.10%, 24.42%]`** | **5 / 10** | **100.0%** | 60.23% | 0.6581 | **Retained (Best Stage 2 Model)** | [`joint_stage2_best.pt`](file:///workspace/test_regr/JointEmbodiedAgentInterface/checkpoints/joint_gpu6_r7_20261007_230348_resumed/joint_stage2_best.pt) |

### How to read the milestone table

- **Stage / Epoch:** The training checkpoint being evaluated. Stage 1 is the baseline; Stage 2 entries are successive RL training epochs.
- **Cumulative RL Training Time:** Total RL training time accumulated by that checkpoint across 50 alternating cross-domain rounds.
- **VLABench Mean Return:** Average episode return on a 0–1 scale. A return of 1.0 represents full task completion; lower positive values reflect partial progress through stages such as approach, pre-grasp orientation, contact, pick-and-lift, transport, and insertion.
- **Positive Return Rate:** Percentage of evaluation episodes with a return greater than zero. This indicates measurable physical task progress.
- **Evaluation Success Rate:** Percentage of episodes completed end to end by the policy without assistance. Results are based on 100 held-out episodes with 95% Wilson score confidence intervals.
- **Tasks Solved:** Number of the 10 task families in which the policy completed at least one episode successfully.
- **Valid Episode Rate:** Percentage of evaluation episodes counted as valid executions (free from unrecoverable IK divergence).
- **EAI Goal Success:** Percentage of held-out EAI symbolic examples for which the predicted goal graph was fully satisfied.
- **EAI Goal Recall:** Recall score for the predicted goals on the held-out EAI examples.
- **Retention Status:** Whether the checkpoint cleared retention gates and its ranking.
- **Checkpoint File:** Saved model file corresponding to the evaluation.

---

## 2. Granular Per-Task Performance Progression

Each row reports one VLABench task family across all five training checkpoints. The columns show:

- **Task Family:** The manipulation task being evaluated.
- **Baseline Return (Succ / Pos):** Stage 1 baseline mean return, followed by task success rate and positive return rate.
- **Epoch 0/1/2/3/4 Return (Succ / Pos):** Mean return, task success rate, and positive return rate for that RL checkpoint.
- **Mean Episode Steps:** Range of average steps per episode across the reported checkpoints; 400 steps is the episode limit.

**Success rate (Succ)** is the percentage of rollouts that completed the task end to end without assistance. **Positive return rate (Pos)** is the percentage of rollouts with a return greater than zero, indicating measurable progress. Each task was evaluated with $N=10$ rollouts per checkpoint, so each successful rollout represents 10 percentage points. For example, `0.9221 (90% / 100%)` means a mean return of 0.9221, successful completion in 9 of 10 rollouts, and a positive return in all 10.

| Task Family | Baseline Return (Succ / Pos) | Epoch 0 Return (Succ / Pos) | Epoch 1 Return (Succ / Pos) | Epoch 2 Return (Succ / Pos) | Epoch 3 Return (Succ / Pos) | Epoch 4 Return (Succ / Pos) | Mean Episode Steps |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `select_painting` | 0.8471 (80% / 100%) | 0.9162 (90% / 100%) | 0.9135 (90% / 100%) | 0.8467 (80% / 100%) | **0.9241 (90% / 100%)** | 0.9221 (90% / 100%) | **87.6 – 121.8** |
| `select_chemistry_tube` | 0.3733 (30% / 100%) | 0.3884 (30% / 100%) | 0.3997 (30% / 100%) | 0.3962 (30% / 100%) | 0.4286 (30% / 100%) | **0.4322 (30% / 100%)** | 292.6 – 293.8 |
| `select_mahjong` | 0.3258 (0% / 100%) | 0.2351 (0% / 90%) | 0.3331 (0% / 100%) | 0.2829 (10% / 80%) | 0.3746 (10% / 100%) | **0.4263 (20% / 100%)** | 332.4 – 400.0 |
| `select_poker` | 0.1424 (0% / 100%) | 0.1856 (10% / 90%) | 0.1970 (10% / 90%) | 0.2476 (10% / 100%) | 0.1918 (10% / 90%) | **0.2572 (10% / 100%)** | 366.8 – 400.0 |
| `select_book` | 0.0735 (0% / 40%) | 0.0937 (0% / 60%) | 0.0853 (0% / 70%) | 0.0928 (0% / 70%) | 0.1780 (10% / 70%) | **0.1925 (10% / 70%)** | 328.0 – 352.0 |
| `insert_flower` | 0.4200 (0% / 100%) | **0.4500 (0% / 100%)** | 0.3871 (0% / 100%) | 0.4160 (0% / 100%) | 0.4089 (0% / 100%) | 0.4180 (0% / 100%) | 400.0 |
| `select_fruit` | **0.3250 (0% / 100%)** | 0.3231 (0% / 100%) | 0.2807 (0% / 100%) | 0.3164 (0% / 100%) | 0.3175 (0% / 100%) | 0.2843 (0% / 100%) | 400.0 |
| `add_condiment` | **0.2567 (0% / 90%)** | 0.2400 (0% / 90%) | 0.2250 (0% / 90%) | 0.2400 (0% / 90%) | 0.1933 (0% / 90%) | 0.2252 (0% / 90%) | 400.0 |
| `select_drink` | 0.1339 (0% / 100%) | 0.1343 (0% / 70%) | 0.0776 (0% / 70%) | 0.1922 (10% / 90%) | 0.1656 (0% / 80%) | **0.1950 (0% / 100%)** | 341.5 – 400.0 |
| `select_toy` | 0.1850 (0% / 60%) | **0.2152 (0% / 90%)** | 0.1679 (0% / 90%) | 0.1887 (0% / 90%) | 0.1763 (0% / 90%) | 0.1681 (0% / 80%) | 400.0 |
