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

## Bounded GPU2 audit

Run this from `/workspace` on GPU2. It pulls the current commit, runs one episode per adapter task, performs a short supervised warm-up, and skips PPO.

```bash
git pull --ff-only
git log -1 --oneline
VLA_LOG=test_regr/VLABenchAgentInterface/results/primitive_audit_$(git rev-parse --short HEAD).log
VLA_OUTPUT=test_regr/VLABenchAgentInterface/checkpoints/primitive_audit_$(git rev-parse --short HEAD)
mkdir -p "$(dirname "$VLA_LOG")"
{
  echo "===== Git commit ====="
  git log -1 --oneline
  echo "===== VLABench primitive audit ====="
} >"$VLA_LOG" 2>&1
nohup env CUDA_VISIBLE_DEVICES=4 PYTORCH_ALLOC_CONF=expandable_segments:True \
  python -u -m test_regr.VLABenchAgentInterface.main train-agent \
  --two-stage \
  --planning-dir test_regr/VLABenchAgentInterface/data/planning \
  --control-source test_regr/VLABenchAgentInterface/data/control \
  --task all --limit 1 --max-steps 80 --output "$VLA_OUTPUT" --device cuda:0 \
  --sft-epochs 1 --controller-warmup-steps 8 --rl-epochs 1 \
  --rl-rounds-per-epoch 1 --rl-num-samples 1 --rollouts-per-update 1 \
  --eval-rollouts-per-task 1 --rl-preflight-min-successful-tasks 999 \
  >>"$VLA_LOG" 2>&1 &
echo "VLABench audit PID=$! log=$VLA_LOG output=$VLA_OUTPUT"
tail -f "$VLA_LOG"
```

The `999` threshold is deliberate: this is a diagnostic audit, not an RL run. Require successful task completion and no IK truncation before long training.

## Single-task smoke

```bash
python -u -m test_regr.VLABenchAgentInterface.main train-agent \
  --two-stage \
  --planning-dir test_regr/VLABenchAgentInterface/data/planning \
  --control-source test_regr/VLABenchAgentInterface/data/control \
  --task add_condiment --limit 1 \
  --output test_regr/VLABenchAgentInterface/checkpoints/smoke_add_condiment \
  --device cuda:0 --sft-epochs 1 --controller-warmup-steps 8 \
  --rl-epochs 1 --rl-rounds-per-epoch 1 --rl-num-samples 1 \
  --rollouts-per-update 1 --eval-rollouts-per-task 1 \
  --rl-preflight-min-successful-tasks 999
```

Use a new output directory when camera selection, action frames, preprocessing, or reward logic changes. `execution_complete` means the controller remained executable; inspect `success_rate`, `positive_return_rate`, `return`, and `termination_reason` for task completion.

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


