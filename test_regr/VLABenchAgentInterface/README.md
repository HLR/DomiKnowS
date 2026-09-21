# VLABench hierarchical agent

This package trains and evaluates the graph-first VLABench planner/controller.
Run commands from the repository root. Detailed design and diagnostics notes are in [Notes.md](Notes.md).

## Setup

Processed data lives under `test_regr/VLABenchAgentInterface/data/planning` and `test_regr/VLABenchAgentInterface/data/control`.

```bash
python -m test_regr.VLABenchAgentInterface.main download \
  --planning-dir test_regr/VLABenchAgentInterface/data/planning \
  --control-dir test_regr/VLABenchAgentInterface/data/control
```

For simulator runs, install the supported OpenMOSS/VLABench checkout and its assets in the server environment. Keep official dataset splits unchanged.

The planner now defaults to the same `Qwen/Qwen3-VL-8B-Instruct` base model
used by EAI and Joint. On GPU2 it is already downloaded at
`/home/auszok/models/Qwen/Qwen3-VL-8B-Instruct` (shared with the
`vigorous_easley` container). Pass that absolute path via `--planner-model`
for an offline load. Use a new output directory; checkpoints trained with
Qwen2.5-VL cannot be resumed with the new backbone. See the
[joint README](../JointEmbodiedAgentInterface/README.md#data-paths-and-canonical-command)
for the one-time `hf download` command.

## Bounded GPU2 audit

Run this from `/workspace` on GPU2. It pulls the current commit, runs one episode per adapter task, performs a short supervised warm-up, and skips PPO.

```bash
git pull --ff-only origin develop
RUN_TAG=$(date +%Y%m%d_%H%M%S)
VLA_LOG="test_regr/VLABenchAgentInterface/results/primitive_audit_${RUN_TAG}.log"
VLA_OUTPUT="test_regr/VLABenchAgentInterface/checkpoints/primitive_audit_${RUN_TAG}"
mkdir -p "$(dirname "$VLA_LOG")"
{
  echo "===== Git commit ====="
  git log -1 --oneline
  git rev-parse HEAD
  echo "===== VLABench primitive audit ====="
} >"$VLA_LOG" 2>&1
nohup env CUDA_VISIBLE_DEVICES=4 PYTORCH_ALLOC_CONF=expandable_segments:True \
  python -u -m test_regr.VLABenchAgentInterface.main train-agent \
  --two-stage \
  --planning-dir test_regr/VLABenchAgentInterface/data/planning \
  --control-source test_regr/VLABenchAgentInterface/data/control \
  --planner-model /home/auszok/models/Qwen/Qwen3-VL-8B-Instruct \
  --task all --limit 1 --max-steps 400 --output "$VLA_OUTPUT" --device cuda:0 \
  --sft-epochs 1 --controller-warmup-steps 8 --rl-epochs 0 \
  --rl-rounds-per-epoch 1 --rl-num-samples 1 --rollouts-per-update 1 \
  --eval-rollouts-per-task 1 \
  >>"$VLA_LOG" 2>&1 &
echo "VLABench audit PID=$! log=$VLA_LOG output=$VLA_OUTPUT"
tail -f "$VLA_LOG"
```

The audit is deliberately PPO-free. Require every task to report successful completion with progress 1.000, zero IK failures, and zero IK truncation before starting long RL.

## Single-task smoke

```bash
python -u -m test_regr.VLABenchAgentInterface.main train-agent \
  --two-stage \
  --planning-dir test_regr/VLABenchAgentInterface/data/planning \
  --control-source test_regr/VLABenchAgentInterface/data/control \
  --planner-model /home/auszok/models/Qwen/Qwen3-VL-8B-Instruct \
  --task add_condiment --limit 1 \
  --output test_regr/VLABenchAgentInterface/checkpoints/smoke_add_condiment \
  --device cuda:0 --sft-epochs 1 --controller-warmup-steps 8 \
  --rl-epochs 1 --rl-rounds-per-epoch 1 --rl-num-samples 1 \
  --rollouts-per-update 1 --eval-rollouts-per-task 1 \
  --rl-preflight-min-successful-tasks 999
```

Use a new output directory when camera selection, action frames, preprocessing, or reward logic changes. `execution_complete` means the controller remained executable; inspect `success_rate`, `positive_return_rate`, `return`, and `termination_reason` for task completion.

Full training evaluates three fixed-seed rollouts per task by default. PPO uses
an approximate-KL target of `0.03` (`--ppo-target-kl`) plus the independent
absolute log-ratio safety bound (`--ppo-max-log-ratio`). Checkpoint metrics under
`controller_update` report accepted PPO epochs, rollback state, approximate KL,
actor parameter-delta L2, and `actor_parameters_changed`; these fields distinguish
a high assisted rollout score from evidence that reinforcement actually updated
the controller. Use `--eval-rollouts-per-task` to increase evaluation replication
without changing the update-producing rollout count.
Rollout metrics additionally report `assist_steps`, `assisted_episode_rate`,
and `unassisted_success_rate` (successful episodes with zero deterministic
assist steps). This last field measures observed assist independence; it is not
a separate run with assist code forcibly disabled.

## Tests and diagnostics

```bash
uv run pytest -q test_regr/VLABenchAgentInterface

python -m test_regr.VLABenchAgentInterface.main diagnose-controller \
  --checkpoint test_regr/VLABenchAgentInterface/checkpoints/agent_stage1.pt \
  --control-source test_regr/VLABenchAgentInterface/data/control \
  --task all --split validation --max-batches 32 \
  --output test_regr/VLABenchAgentInterface/checkpoints/diagnostics
```

Camera parity requires a held-out replay manifest, camera map, and scene restorer. See [Notes.md](Notes.md) for the required options and interpretation.
