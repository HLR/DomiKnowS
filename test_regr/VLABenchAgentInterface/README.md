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
The report includes `source_revision` for the pinned upstream rules and
`dimension_mapping_revision` for task coverage. The bundled
`dim2task_extensions` adds billiards tasks to M&T, Spatial, ComSense, and Semantic,
and thermal expansion, reflection, speed of sound, and magnetism QA to PhysLaw.
Revision `vlabench-planning-coverage-v2` covers all 450 episodes of the seed-42
test split: M&T 113, Spatial 53, ComSense 84, Semantic 84, PhysLaw 60, Complex 56.
The upstream mapping remains preserved separately in `dim2task`.

Earlier reports classified only 373 episodes. Their dimension means and macro
scores cannot be compared directly with the extended mapping; rescore saved
per-episode predictions under the same mapping for both runs. Aggregate-only
reports cannot recover the missing dimension contributions. This correction
changes no per-episode or pooled overall scores and requires no retraining.
Genuinely unknown tasks remain explicitly counted and excluded from dimensions.
Source: [official scoring helper](https://github.com/OpenMOSS/VLABench/blob/cf588fe60c0c7282174fe979f5913170cfe69017/VLABench/evaluation/utils.py),
[dimension mapping](https://github.com/OpenMOSS/VLABench/blob/cf588fe60c0c7282174fe979f5913170cfe69017/VLABench/configs/evaluation/dim2task.json),
and [independent tasks](https://github.com/OpenMOSS/VLABench/blob/cf588fe60c0c7282174fe979f5913170cfe69017/VLABench/configs/evaluation/seq_independent_task.json).
Run commands from the repository root. Detailed design and diagnostics notes are in [Notes.md](Notes.md).

## Standard VLABench VLM-planning protocol

The results above use this repository's own setup (seed-42 test split, scene
entity names in the prompt, extended task mapping, optionally corrected labels).
They are not comparable with published VLM rows. To report standard numbers run:

```bash
python -m test_regr.VLABenchAgentInterface.main evaluate-planner --protocol official \
  --planning-dir /path/to/vlm_evaluation_v1.0 --planner-model /path/to/Qwen3-VL-8B-Instruct \
  --few-shot-num 0 --official-output-dir results/official_qwen3vl_0shot
```

`--protocol official` follows OpenMOSS/VLABench `scripts/evaluate_vlm.py` at the
revision pinned in `planning_benchmark.json`:

- **Data:** every episode of every task (45 tasks x 100 = 4,500), taken from a
  fresh clone of `VLABench/vlm_evaluation_v1.0` with its original labels. There is
  no train/validation/test split. The summary records a `dataset_fingerprint` of
  episode ids, instructions and reference files; compare it between runs.
- **Input:** the unlabeled image, the number-labeled image (`input_mask.png`) and
  the instruction under the unmodified `official_vlm_prompt_en.txt` (SHA-256 checked
  at load). Entity names are never given. `--few-shot-num` examples are drawn at
  random from any task, as upstream does, seeded per episode by `--seed`;
  `--with-cot` adds the step-by-step line.
- **Decoding and parsing:** greedy, no grammar or DFA. The answer is the first
  `json` fence of the reply; anything else is a `format_error` scoring zero. Skill
  names are compared exactly. Raw replies are kept in `outputs.jsonl`, and a rerun
  with the same settings resumes where it stopped.
- **Scoring:** dependencies depend only on the task; the headline is upstream's
  `total_score` (`0.4*SR + 0.4*PR`, our `legacy_score`; a perfect plan scores 80),
  averaged per dimension over the episodes. `official_total_score_by_dimension` in
  `summary.json` is the comparable table. `dimension_mean_of_total_score` is
  not an upstream quantity.
- **Mapping:** upstream's `dim2task` lists 41 task names and its script cannot place
  the other 8 tasks of the released data (800 episodes: the four billiards variants and
  thermal-expansion, reflection, speed-of-sound and magnetism QA). They are left out
  of the official table, counted in `unmapped_examples`, and included in the
  `extended_mapping` section.
- `--limit` or `--tasks` mark the run `standard: false`.
- `--constraints grammar` adds the DomiKnowS grammar DFA (integer ids below 64, benchmark subtask
  patterns, at most 16 operations) and no scene information. `--press-question` adds a first yes/no
  question to the model and restricts the plan to one `press` on "Yes"; its wording was chosen on
  validation episodes and names the benchmark's question-task topics, so report it as a variant. In
  the measured run it recognised 127 of 600 question tasks, mostly tasks outside upstream's
  dim2task, and never chose the right button, so the official score barely moved.
- `official_hmm.py` samples the frozen model on the benchmark inputs without labels, fits an HMM on the
  canonical plan tokens (`json.dumps(plan, indent=4)`, integer ids) and picks a lookahead weight on the
  450 validation episodes of the seed-42 split. The lookahead is action-level, after CLAMP's budget-aware
  reachability (`domiknows/generation/applications/action_lookahead.py`, `native_lookahead.py`): the
  grammar DFA stays the only authority on which tokens are legal, and the HMM adds
  `weight * (log HMM mass - best log HMM mass)` to the legal tokens that decide an action (skill name,
  object id, continue or stop), with the mass taken over the complete valid plans that token leaves
  open. Formatting tokens are neutral. The token-level product with the DFA (`HMMDFAStaticLookahead`)
  is not used here: it ran out of memory at about 93 GB. Fit the HMM with a large learning rate
  (`--states 2048 --learning-rate 0.3`): 128 states at 1e-3 stalled at 2.9 nats per token and made the
  lookahead prefer one-step plans, 2048 states at 0.3 reaches 0.18. Measured result (weight 0.5, chosen
  on validation): 38.02 against 38.24 for the grammar DFA alone over all 4,500 episodes (38.62 against
  38.78 on the held-out test split). The lookahead changed 873 of 265,192 greedy tokens, so treat it as
  a negative result: the HMM does not see the image or the instruction. `lookahead_score="conditional"` is the Ctrl-G form CLAMP uses (the HMM's chance that a valid plan can still be
  completed); on validation it was worse than the joint score (37.93 at weight 0.5, 33.6 at weight 5, against 38.18
  without the HMM) because it pushes plans to stop early and loses the long Complex plans.
  Validation results for the other additions (450 episodes, six-dimension mean; grammar DFA alone 38.18, joint HMM
  weight 0.5 38.37): `--scene-ids` changes no plan (the model already picks only visible ids); per-instance
  adaptation (`official_adapt.py`, 16 samples, one EM step) scores 38.03 at best (weight 0.5, lam 0.1) and falls as
  lam and weight grow (down to 37.38); `--pattern-first` (the model names the subtask patterns from the official
  prompt's list and the grammar is limited to them; 39 of 450 replies were unusable) scores 38.29.

Evaluations from `evaluate-planner` without `--protocol official` also report an
`official_mapping` block (upstream-mapped tasks only, upstream dimensions) to help
orient them, but they remain internal-split results.

Fine-tuned graph planners under this protocol (names-free inputs, all 4,500
episodes) are not supported yet; report them as a separate held-out track.

## Measuring a planning baseline

For an unconstrained frozen-VLM planning baseline, generate directly with the
pretrained model's native language head. No LoRA adapter, graph decoder, DFA,
HMM, or instruction-dependent logits mask is used. For example, in PowerShell:

```powershell
python -m test_regr.VLABenchAgentInterface.main evaluate-planner `
  --generation-mode frozen-vlm `
  --planner-model "path/to/Qwen3-VL-8B-Instruct" `
  --planning-dir test_regr/VLABenchAgentInterface/data/planning `
  --split test --seed 42 --max-new-tokens 768
```

The model must be available locally. Generation is greedy, with thinking disabled
and a 768-token default continuation cap. Weights are frozen; 4-bit loading is
enabled by default (`--no-load-in-4bit` disables it). Inputs use the instruction,
segmented images when available (otherwise ordinary images), and scene-only
entity IDs. Reference plans and goal/condition fields do not augment the prompt's
object table. Missing scene configurations raise an error. Malformed/empty
generations receive zero planning metrics and remain in the denominator; GPU or
model errors raise rather than becoming fabricated scores.

For a trained graph-planner baseline before RL, evaluate its saved supervised
adapter on the same held-out split:

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

`--generation-mode graph` is the compatibility default and requires
`--resume-adapter`; evaluation rejects a randomly initialized graph decoder.
`--generation-mode frozen-vlm` rejects `--resume-adapter` and `--vocab`.
Both paths use `evaluate_planner()` and the same Table 1 metric accumulator.
`planning_metrics` scores readable predictions independently of constraint
validity. In native mode, `native_valid` reports benchmark JSON/schema validity;
`valid` and `reward` retain the legacy graph-schema diagnostic (without graph
materialization). Report Table 1 scores from `planning_metrics`, rather than
interpreting reward or validity as simulator success. The native prompt and
quantization settings must also be disclosed in comparisons with CLAMP's article.
The separate `planning_experiment --mode raw` path below additionally supports
manifest-based resumable per-example evidence.

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
/path/to/best.pt --weight 0.1` for the HMM variant. The default native
`--grammar-profile benchmark` uses the pinned official subtask patterns and the
benchmark argument protocol: notably `push` requires `target_container_name`.
Its default bound is 16 operations, allowing the dataset's nine- and ten-operation
reference plans. IDs cover the entire supplied scene table; native generation
does not inherit the graph decoder's 64-pointer vocabulary limit. Graph
checkpoint decoding retains its existing schema, vocabulary, and eight-operation
bound. `--grammar-profile legacy --max-operations 8 --object-constraints none`
selects the old compact language and prompt; new output directories are still
required because the scene-only loader now excludes goal/condition fields.

The native token DFA consumes actual byte-level BPE bytes, including tokens
crossing JSON boundaries. It accepts compact or spaced/indented JSON, both
`name`/`params` member orders, and ASCII Unicode escapes within string literals.
Whitespace inside names, split integers, duplicate/extra keys, noninteger IDs,
trailing commas, incomplete subtasks, and premature EOS remain invalid.

`--object-constraints instruction` (default) restricts `pick` candidates using
explicit scene object names in a supported source clause, or instruction nouns
and scene classes. Explicit destination container names restrict `place`,
`insert`, `pour`, and `push`. Multiple matching objects remain candidates.
Unsupported wording, negation, spatial/ordinal descriptions, and source relations
cause abstention; unstated semantic knowledge is not guessed. These rules use
only the instruction and scene component identities/classes, never reference
actions, environment conditions, target fields, or goal labels. Use
`--object-constraints none` for the grammar-only ablation. Neither configuration
enforces simulator reachability or solves arbitrary natural-language grounding.

Each native record includes the candidate restrictions and their evidence.
`native_valid` checks the benchmark language independently; `grounded_valid`
also checks instruction candidates. Existing `valid`/`validation_errors` retain
the graph schema diagnostic, which may reject benchmark-valid `push` arguments.
Use the same new prompt/profile and manifest when comparing raw, DFA, and
instruction-grounded DFA. Existing compact-grammar results and HMM validation
weights are historical; the old HMM weight must be reselected on validation
before claiming a new HMM result. DFA/HMM cache identities include profile,
candidate sets, operation budget, tokenizer, model, and language implementation.

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


## Density QA label repair

The upstream density generator used the selected material's sorted density rank
as its button index, despite assigning materials to buttons in shuffled order.
`repair_density_labels.py` changes that comparison to material identity and
migrates saved labels from the instruction, public material ordering, and scene
positions. It does not introduce answer facts into decoder inputs.

Audit first (the generator argument is optional for a dataset-only audit):

```bash
python -m test_regr.VLABenchAgentInterface.repair_density_labels \
  --planning-dir test_regr/VLABenchAgentInterface/data/planning \
  --generator /opt/dominows/VLABench/VLABench/tasks/hierarchical_tasks/primitive/physical_series/density_qa.py
```

To apply, add `--apply --backup-dir /path/outside/dataset/new-backup-directory`.
The backup directory must not already exist. The migration saves every original
file and a manifest of original/repaired SHA-256 hashes before writing. It
updates both `conditions.press_button.target_button` and the corresponding
`skill_sequence` target, preserving numeric or named entity encoding. Ambiguous
associations, unknown materials, conflicting instructions, or inconsistent old
labels prevent the entire batch from being applied. Images and instructions
remain unchanged. Repeating the audit after repair should report zero changes.

Repair revision `density-material-button-v1` corrected 74 of the 100 density
episodes on gpu2 (2026-10-18), including nine test episodes. All 100 passed the
post-repair audit; all images/instructions retained their original hashes.
The actual patched generator passed all 18 permutations of three material
positions and target choices. These are dataset/source validation results;
no model benchmark was rerun. Historical scorecards retain their original
labels and must be rescored from saved predictions against the repaired dataset
before comparisons. Source patching affects newly loaded generator code.


## Physics QA observation modes

All six PhysLaw QA task aliases declare a single `press` operation. Evaluation
uses `--observation-mode visual` by default in standalone and joint CLIs. Visual
mode leaves the property-based button choice to the VLM, enforces the existing
structural LCs, and scores every episode against its stored reference answer.
Missing numeric reflectance does not remove reflection episodes or assign them
a failure automatically. Symbolic property grounding is reported separately.

For QA episodes, visual mode ignores physical-property metadata, observation
sidecars, selection annotations, declared goal/target annotations, and reference
entity-table augmentation. It retains input images, scene entity identifiers,
the question, and the known QA action contract. This is a visual property-choice
profile within the existing scene/entity interface, not a claim that all metadata
has been removed from the entire benchmark pipeline.

Use `--observation-mode metadata-assisted` to enable the qualitative material/
shape adapters, explicitly annotated selection facts, or reflection observations.
This is a separate input contract and should use a separate output report.
The generic selection engine supports min/max, thresholds, and zero-based ranks,
including ties and unknown-score diagnostics. Enforcement uses the compiled LCs.
Native `--generation-mode frozen-vlm` evaluation uses visual mode exclusively.

Example for the original static-image checkpoint benchmark:

For the full 450-episode test command, parameter explanations, and the four
visual/metadata and original/corrected-label combinations, see the
[joint evaluation README](../JointEmbodiedAgentInterface/README.md#full-vlabench-planning-test).

```bash
python -m test_regr.JointEmbodiedAgentInterface.eval_planning \
  --checkpoint test_regr/JointEmbodiedAgentInterface/checkpoints/joint_stage2_best.pt \
  --split test --observation-mode visual \
  --allow-planning-schema-migration \
  --output test_regr/JointEmbodiedAgentInterface/results/planning_eval_test_visual.json
```

For a separate metadata-assisted graph run, change the mode to
`metadata-assisted` and choose a different output filename. Reports identify the
requested `observation_mode` and effective `detailed_metrics.observation_profiles`.
Visual QA episodes use `visual_qa_v1`, with unavailable symbolic property checks;
this status describes verification coverage, not answer accuracy. See
`VLABENCH_LOGICAL_CONSTRAINTS_AND_COMPONENTS.md` for observation schemas,
source-export repairs, and validation evidence.

Planning now defaults to 12 operations and supports optional push destinations.
The migration flag above is required when warming the earlier champion into
this audited schema; exact resume otherwise rejects schema changes. New
checkpoints load without it. Input-grounded Complex goals use the shared LCs.
Use `--no-ground-complex` for the representation-only ablation, and
`--decision-trace decisions.jsonl` for read-only token diagnostics. See
`test_regr.JointEmbodiedAgentInterface.finetune_planning` for balanced, resumable
planner-head adaptation with entity-order augmentation and validation selection.

### Original and corrected scoring labels

`eval_planning` uses the labels stored in `--planning-dir` by default. The server's
main planning dataset has been restored to its original density labels. Add
`--updated-data` (alias `--updated-labels`) to compute corrected density reference
plans in memory. This changes scoring references only: images, instructions,
entity tables, decoder facts, and disk files remain unchanged. The option works
with both `--observation-mode visual` and `--observation-mode metadata-assisted`.
Reports record `updated_data` and `reference_labels` (`as-stored` or
`density-material-button-v1`). As-stored labels are only original when the supplied
dataset itself contains original labels.

For an explicit on-disk migration, audit first, then apply with a new backup:

```bash
python -m test_regr.VLABenchAgentInterface.repair_density_labels \
  --planning-dir DATA --apply --backup-dir NEW_BACKUP
```

Return that dataset to its exact original bytes using the same backup:

```bash
python -m test_regr.VLABenchAgentInterface.repair_density_labels \
  --planning-dir DATA --restore --apply --backup-dir NEW_BACKUP
```

Omit `--apply` to preview either operation. Restoration validates every backup
and current-file checksum before writing, is idempotent, refuses unrelated edits,
and leaves the upstream generator fix intact. Existing historical reports retain
their original label protocol; restoring data does not retroactively change them.
