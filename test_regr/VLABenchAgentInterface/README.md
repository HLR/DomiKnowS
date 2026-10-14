# VLABench hierarchical agent

This package trains and evaluates the graph-first VLABench planner/controller.

## Table 1 planning metrics

`evaluate-planner` reports `planning_metrics` alongside the existing reward and
validity diagnostics. Evaluate the complete supplied planning collection with:

```bash
python -m test_regr.VLABenchAgentInterface.main evaluate-planner \
  --planning-dir test_regr/VLABenchAgentInterface/data/planning \
  --resume-adapter path/to/planner_adapter --split all
```

The default split remains `test`; `all` evaluates the supplied examples without
holding out a subset. `--limit` still limits the loaded collection. Training
rewards, validity gates, and checkpoint selection use their existing behavior.

All new scores use a 0–100 scale. SR is multiset skill recall, PR is multiset
recall of entity/container arguments with their roles, SPR is multiset recall of
skill-and-argument tuples, and PM is the fraction of matching ground-truth
subtask dependency-graph nodes. PM is partial credit, not binary whole-plan exact
match. Recall does not penalize extra predicted actions. Benchmark subtask
patterns are separate from the expanded training vocabulary.

`score = 0.4*SR + 0.4*PR + 0.1*SPR + 0.1*PM` includes all four declared terms.
`legacy_score = 0.4*SR + 0.4*PR` reproduces the upstream helper's omitted SPR/PM
terms: perfect plans score 100 and 80, respectively. These are the official
component track described in CLAMP Table 1 / Appendix A2.5, rather than the
InternVL local F1 track or a reproduction of the Qwen VL-main pipeline.

`planning_metrics.overall` contains pooled per-example means. `dimensions`
contains means and example counts for M&T, Spatial, ComSense, Semantic, PhysLaw,
and Complex. `Macro` and `legacy_Macro` average the six dimension totals equally,
regardless of dimension size. Unevaluated dimensions have `null` scores; both
Macro values are `null` until every dimension is represented. Empty evaluation
also has `null` overall means. Unknown tasks count toward overall means and
`unknown_task_count`, but not dimension means. Malformed/empty predictions count
as zero; readable predictions are scored independently of grounding and world
validity. Invalid reference plans/dependencies raise errors.

Task mappings and sequence-independent task defaults are bundled in
`planning_benchmark.json`, pinned to OpenMOSS/VLABench revision
`cf588fe60c0c7282174fe979f5913170cfe69017`. Tasks come from episode directories;
explicit output dependency metadata overrides environment metadata, which
overrides the benchmark task default. Unknown tasks default to Sequential.
The report includes `source_revision` so mapping/scoring provenance is visible.
Source: [official scoring helper](https://github.com/OpenMOSS/VLABench/blob/cf588fe60c0c7282174fe979f5913170cfe69017/VLABench/evaluation/utils.py),
[dimension mapping](https://github.com/OpenMOSS/VLABench/blob/cf588fe60c0c7282174fe979f5913170cfe69017/VLABench/configs/evaluation/dim2task.json),
and [independent tasks](https://github.com/OpenMOSS/VLABench/blob/cf588fe60c0c7282174fe979f5913170cfe69017/VLABench/configs/evaluation/seq_independent_task.json).
Run commands from the repository root. Detailed design and diagnostics notes are in [Notes.md](Notes.md).

## Measuring a planning baseline

For this package, use the supervised planner before RL as the baseline. Evaluate
its saved adapter on the held-out test split. For example, in PowerShell:

```powershell
$plannerAdapter = "path/to/checkpoint-directory/planner_stage1"
python -m test_regr.VLABenchAgentInterface.main evaluate-planner `
  --planning-dir test_regr/VLABenchAgentInterface/data/planning `
  --resume-adapter "$plannerAdapter" `
  --vocab "$plannerAdapter/vocab.json" `
  --split test --seed 42
```

Replace the adapter path with the directory saved by `train-planner`, containing
the adapter, `graph_decoder.pt`, and `vocab.json`. Use the same `--planner-model`,
`--planner-decoder-hidden-dim`, `--max-entities`, and `--max-operations` settings
as training. Compare later checkpoints on exactly the same examples, split seed,
images, generation settings, and scoring formula. Keep evaluation examples out
of training. Use `--split all` only when the supplied collection is entirely the
evaluation set; `--limit` selects a reduced collection before splitting.

Read these fields in the JSON output:

- `planning_metrics.dimensions`: six dimension scores and example counts.
- `planning_metrics.Macro`: the unweighted six-dimension average, or `null` if
  any dimension is absent.
- `planning_metrics.overall`: pooled SR, PR, SPR, PM, full score, and legacy score.
- `valid`: the separate constraint-validity diagnostic, on a 0–1 scale.

This measures the supervised, DFA-constrained graph planner. The unconstrained
frozen-VLM baseline in CLAMP Table 1 requires a different generation path:
direct VLM plan generation without the DFA or graph decoder, followed by the
same metrics evaluator on the same examples. The current CLI always generates
through the DFA-constrained graph decoder. Omitting `--resume-adapter`
initializes that decoder randomly and does not measure the pretrained VLM's
planning ability. A direct, unconstrained VLM evaluation path is not currently
implemented.

## Setup

### Reproducible native and graph planner comparison

The separate `planning_experiment` entry point supports frozen native Qwen
generation (`raw`), native-token JSON DFA decoding (`dfa`), native-token
HMM lookahead (`hmm`), and planner-only evaluation of version-5 standalone
checkpoints (`graph`). It does not construct a controller or update planner
weights. The native model is loaded without LoRA or the graph decoder.

```bash
python -m test_regr.VLABenchAgentInterface.planning_experiment inventory \
  --planning-dir /path/to/planning --output /path/to/run/manifests
python -m test_regr.VLABenchAgentInterface.planning_experiment evaluate \
  --mode raw --planning-dir /path/to/planning \
  --manifest /path/to/run/manifests/test_manifest.json \
  --model /path/to/Qwen3-VL-8B-Instruct --output /path/to/run/raw
python -m test_regr.VLABenchAgentInterface.planning_experiment evaluate \
  --mode graph --planning-dir /path/to/planning \
  --manifest /path/to/run/manifests/test_manifest.json \
  --model /path/to/Qwen3-VL-8B-Instruct \
  --checkpoint /path/to/agent_stage1.pt --output /path/to/run/supervised
```

Use `--mode dfa` for native constrained generation, or `--mode hmm --hmm
/path/to/best.pt --weight 0.1` for the HMM variant. The native DFA projects
canonical compact JSON through byte-level BPE tokens and enforces graph-defined
skills, required argument roles, complete subtask patterns, an operation bound,
scene ID bounds, and EOS after a complete plan. These are language constraints;
they do not establish simulator goal reachability. Graph checkpoint decoding
retains its existing constraints.

Outputs include durable `episodes.jsonl`, settings, model/checkpoint provenance,
and `summary.json`. Resume with the same command and output directory; changed
settings or duplicate episode records are rejected. Runtime/model failures stop
the stage; malformed generated plans remain zero-scored evaluation records.
Dataset manifests are reconstructed from the supplied collection and seed;
without the original training manifest they do not prove historical split
identity. The inventory records this limitation.

Independent experiments use scene configuration only for prompt entity tables.
The legacy dataset loader also appends reference-plan arguments to those tables;
that augmentation is excluded here to prevent gold IDs appearing in model inputs.
The legacy adapter evaluation command above retains its existing input behavior
and should be treated as a diagnostic rather than an independent baseline.
Historical graph checkpoints may have been trained with those augmented tables;
this limitation is recorded without changing their saved weights or decoding.

`run_planning_experiment` orchestrates smoke tests, the three baseline runs,
teacher corpus generation, HMM fitting, validation selection, and final HMM
evaluation sequentially. Pass `--output`, `--model`, `--planning-dir`,
`--supervised`, and `--reinforcement`; `--smoke-only` runs the initial model
checks. Select the physical GPU using `CUDA_VISIBLE_DEVICES` before launching.
The HMM smoke test follows fitting, since it requires a trained HMM checkpoint.
Teacher sampling uses training and validation prompts, with no gold-plan targets
or test prompts, and preserves malformed continuations. Sampling batches of eight
are resumable with fixed batch seeds. HMM fitting uses 128 states, native token
emissions, Adam at 0.001, batches of 16, at most ten epochs, and validation-NLL
patience two. Validation chooses among lookahead weights 0.1, 0.5, and 1.0;
test examples never participate in weight selection. Corpus data, HMM checkpoints,
and experiment outputs belong in ignored `outputs/` directories.

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

Full training evaluates three fixed-seed rollouts per task by default. Under
the default `--execution-assistance train-only`, deterministic expert-motion
and object-attachment helpers remain available while collecting training data
but are forcibly disabled during fixed-seed evaluation. Use `on` only for an
explicit assisted-system ablation; `off` disables the helpers everywhere.

The controller uses `3e-4` for supervised BC and switches to a fresh-moment
`3e-5` PPO regime (`--controller-rl-learning-rate`) before Stage 2. PPO uses
two epochs by default, an approximate-KL target of `0.03`
(`--ppo-target-kl`), plus the independent
absolute log-ratio safety bound (`--ppo-max-log-ratio`). Checkpoint metrics under
`controller_update` report accepted PPO epochs, rollback state, approximate KL,
actor parameter-delta L2, and `actor_parameters_changed`; these fields distinguish
a high assisted rollout score from evidence that reinforcement actually updated
the controller. A trust-region rollback restores the pre-update weights, clears
stale optimizer moments, and halves the PPO learning rate down to
`--ppo-min-learning-rate`; `learning_rates` records the effective value. Use
`--eval-rollouts-per-task` to increase evaluation replication without changing
the update-producing rollout count.
Rollout metrics additionally report `assist_steps`, `assisted_episode_rate`,
and `unassisted_success_rate` (successful episodes with zero deterministic
assist steps). With the default fixed-seed evaluation policy these are a direct
unassisted intervention, not merely observational assist accounting.

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
