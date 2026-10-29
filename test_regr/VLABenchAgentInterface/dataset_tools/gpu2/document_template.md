# VLABench datasets on GPU2

Updated: {{TIME}}. Changes are on gpu2.ihmc.us, container vigorous_easley. The Windows checkout was not changed.

## Correction and existing experiments

The existing 4,500-example planning collection is the official VLM evaluation collection. Its local 80/10/10 split is an internal supervised experiment; the entire collection is not an untouched official test set.

The shared GPU2 checkout now rejects that collection for standalone planner, standalone agent, task-classifier, and joint-agent training. Renamed copies with the same canonical initial-scene fingerprint, even after rewriting instructions or goals are rejected too. Training roots require independent provenance, a completed generation audit, images, and unique scenes. Untracked or evaluation-exposed checkpoint resumes are rejected. Historical output directories cannot be relabelled clean.

Joint training defaults to data/planning_train. Evaluation retains data/planning. The installed fixed simulator evaluation scenes are reserved as well (simulator_evaluation_manifest.json). Original evaluation reference files were preserved; their SHA-256 inventory and scene fingerprints are in official_evaluation_manifest.json under the evidence directory below. These are server-snapshot hashes, not a claim of equivalence to the latest Hub revision.

The following running snapshots were preserved and labelled using dataset_protocol.json beside their runs and inside their checkpoint directories:

- /home/auszok/experiments/joint_ce_only_fixed_gpu2_20261028_3bc2b88d — container PID 267427, physical GPU 1 at audit.
- /home/auszok/experiments/ce_dfa_grammar_budgetfixed_gpu2_20261029_33d8d319 — container PID 415205, physical GPU 3 at audit.

Their weights remain evaluation-exposed. Changing datasets cannot undo that history. Corrected training must start from the original pretrained backbone, use a new output directory, and avoid resuming those checkpoints. Existing scores remain historical internal-split results.

## Official releases

| Release | Intended role | Observed size |
|---|---|---|
| [Primitive demonstrations](https://huggingface.co/datasets/VLABench/vlabench_primitive_ft_lerobot_video) | Controller/VLA training | 5,000 episodes; 575,101 frames; 128 instruction IDs |
| [Composite demonstrations](https://huggingface.co/datasets/VLABench/vlabench_composite_ft_lerobot_video) | Multi-step controller/VLA training | 5,977 episodes; 2,539,771 action rows; 167 instruction IDs |
| [VLM evaluation](https://huggingface.co/datasets/VLABench/vlm_evaluation_v1.0) | Static planner evaluation | 4,500 examples |
| [Simulator evaluation tracks](https://github.com/OpenMOSS/VLABench#evaluate) | Interactive execution evaluation | Fixed episodes from the pinned upstream installation |

Instruction IDs are language-task variations, not skill classes. The Hub composite viewer reports a different row total; the action parquet files were counted directly for the numbers above.

Sources: [official release collection](https://huggingface.co/collections/VLABench/vlabench10), [collection tools](https://github.com/OpenMOSS/VLABench#data-collection), [evaluation guidance](https://github.com/OpenMOSS/VLABench#evaluate).

## Server paths and pinned download

Dataset root: /home/auszok/DomiKnowS/test_regr/VLABenchAgentInterface/data, also mounted under /workspace/test_regr/VLABenchAgentInterface/data.

| Directory | Role |
|---|---|
| planning | Official static evaluation; prohibited for training |
| control | Existing official primitive demonstrations |
| control_composite | Official composite demonstrations |
| planning_train | Independent reset images and current-schema expert plan labels |
| planning_extended | Independent labels requiring additional skills/schema/budget; excluded from default training |

Composite revision: 8ceba7df14015ba00c4572f6df648750364caf18. Pinned remote size: 20,028,960,407 bytes across 22,997 files. Transfer state: **{{DOWNLOAD}}**. Composite training is blocked until download_complete.json exists; it is created after every pinned file passes completeness/size validation.

## Actual coverage

The installed SkillLib has 19 methods: pick, place, open_door, close_door, open_drawer, press, pull, push, pour, lift, flip, open_laptop, moveto, move_offset, reset, close_gripper, open_gripper, wait, and step_trajectory.

| Upstream method | Original static reference labels | Fresh default planner labels | Native expert call labels, unexecuted |
|---|---:|---:|---:|
{{METHOD_COVERAGE}}

Native counts include examples excluded from default training. They do not demonstrate execution success or per-frame controller supervision.

This is 18 manipulation/control operations plus an execution helper, not 19 task classes. The shared GPU2 graph schema still has nine skills, including the insert abstraction. This dataset repair does not silently migrate old weights or install the separately developed expanded-domain code.

The original official static reference plans contain:

| Graph action | Occurrences across the full collection |
|---|---:|
{{ORIGINAL_COUNTS}}

These are eight graph labels, seven literal upstream methods. There are no open_door, close_door, or open_laptop reference-plan labels.

The pinned composite release contains:

| Task family | Episodes, counted from unique source HDF5 paths |
|---|---:|
{{COMPOSITE_FAMILIES}}

set_study_table is present. cool_drink, take_out_cool_drink, heat_food, and a general door/drawer suite are absent from this release. Task membership can suggest motions; it does not prove per-frame skill supervision or success by our policy. The parquet release contains continuous actions and instruction IDs without explicit skill boundaries. No all-19 labelled-coverage guarantee was found.

## Independent planner corpus

Fresh simulator resets use upstream revision cf588fe60c0c7282174fe979f5913170cfe69017, with source-file hashes and random seeds recorded. Official test scenes and evaluator outputs are not used to construct training examples. Expert target names and plans are supervision, not planner inputs.

| Corpus | Completed examples | Graph plan labels |
|---|---:|---|
{{GENERATED_COUNTS}}

This is a **pilot planner corpus**, not a balanced replacement for thousands of demonstrations. Expert plans have not been physically executed or certified successful; these are not validated controller trajectories.

Each episode keeps its native expert sequence under provenance/expert_sequence.json. The graph projection uses audited compact planner macro contracts and omits executor helpers and intermediate motion lifts. All examples admitted to the default training root are checked against the existing task-tier logical constraints under the current default observation policy; this is not measured simulator success. press can use the expert task's explicit target-button label; pour uses its explicit destination. Flower lowering can be represented by the existing insert abstraction. These transformations create training labels; they do not infer targets from hidden metadata names. Unsupported articulation, unresolved roles, excessive budgets, and unvalidated composite projections are kept outside the default training root. The initial book_rearrange and get_coffee projections failed current LCs and were moved to planning_extended; projection_corrections.json records the changes.

Known upstream gaps and failed seeds are retained in generation_first_pass.json, generation_audit.json, and supplemental/generation_audit.json. In particular, cool_drink.get_expert_skill_sequence is empty in the installed revision: the stock trajectory command cannot collect it as-is. Some seeds fail MuJoCo stability checks and are excluded. Missing coverage needs a working expert or teleoperation, measured terminal success, and explicit segment boundaries.

## Composite loader changes

The adapter reads observation.state.eef as the seven-dimensional end-effector state, rather than joint coordinates. It loads parquet action files only from data/, excluding metadata tables.

For composite data, select --task all. Instruction IDs 0..166 are preserved and the controller task embedding is sized to 167. Unknown skill conditioning remains at padding; the loader does not divide composite episodes into guessed equal skill segments. Primitive behavior is retained.

The releases use different camera names/resolutions. They are selectable sources and have not been blindly merged into a supposedly skill-segmented dataset. A composite-trained checkpoint needs its matching instruction vocabulary and task-embedding size.

## Reproduction and launch requirements

Run inside vigorous_easley from /workspace with /opt/dominows/venv/bin/python. Preparation uses CPU/software rendering and selects no CUDA devices for scene generation.

Evidence/scripts directory: **/home/auszok/experiments/vlabench_dataset_fix_20261029**.

The preparation code and this document template are versioned under test_regr/VLABenchAgentInterface/dataset_tools/. To restore the working tool copies from Git, first run inside the container:

    python -m test_regr.VLABenchAgentInterface.dataset_tools.install_gpu2_tools

Datasets, generated corpora, manifests, logs and checkpoints remain server-only.

```bash
python /home/auszok/experiments/vlabench_dataset_fix_20261029/download_composite.py
python /home/auszok/experiments/vlabench_dataset_fix_20261029/generate_planning.py
python /home/auszok/experiments/vlabench_dataset_fix_20261029/generate_supplemental.py
python /home/auszok/experiments/vlabench_dataset_fix_20261029/finalize_datasets.py
```

Downloads resume from the pinned revision. Finalization verifies disjointness and freezes the corpus only after generation workers finish. Do not train from a growing corpus.

For a new clean joint training launch, retain the intended model/EAI/optimizer options and supply:

```text
--vlabench-planning-dir /workspace/test_regr/VLABenchAgentInterface/data/planning_train
--control-source /workspace/test_regr/VLABenchAgentInterface/data/control_composite
--task all
--output /home/auszok/experiments/NEW_CLEAN_RUN/checkpoints
```

This selects the composite controller source; it does not combine it with primitive data. Do not resume historical evaluation-exposed weights. Before expensive training, expand the pilot corpus and verify skill balance, operation budgets, schema support, and label validity.

Evaluate planning on the full official static collection with independently trained weights. Evaluate robot execution using fixed simulator tracks and report measured success, progress and assistance settings. Internal generated-data splits remain separate diagnostics.

## Evidence and limitations

- dataset_inventory.json: current counts, roles and transfer state.
- official_evaluation_manifest.json: frozen reference hashes and scene identities.
- composite_audit.json: pinned action columns, frame/episode counts and source task paths.
- generation*.json and supplemental/generation_audit.json: label provenance and collection failures.
- historical_runs.json: evaluation-exposure labels for existing runs.
- validation.log: 22 focused regression tests passed.
- training_lc_validation.json: all 110 admitted compact plans pass the current task-tier LCs.
- composite_window_smoke.json: a real three-camera composite episode decodes into the expected controller tensors.
- backups/: original copies of changed server source files.

No improved benchmark score, all-19 trained support, or successful new articulation execution is claimed. The running historical jobs were not restarted. Fresh training and simulator validation remain separate work.
