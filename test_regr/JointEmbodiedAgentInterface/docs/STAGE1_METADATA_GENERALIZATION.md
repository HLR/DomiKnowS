# Four-epoch metadata Stage 1 campaign

This is an **internal, seen-task experiment** using the historical 4,500 official
VLABench episodes, split with seed 42 into 3,600 training, 450 validation and 450
test episodes. It is not a clean official evaluation. Full-metadata results are
a privileged-information upper bound. Original data files remain unchanged.

## Training and artifacts

Start with Qwen3-VL-8B-Instruct and fresh Joint heads. Retain joint EAI
supervision, controller behavior cloning, batch size 8, 384-pixel controller
images, 4-bit planner loading, seed 42 and the existing EAI warm-up. Disable
Stage 2 and extra controller warm-up. Use unmasked CE plus allowed-label mass
regularization, rather than masked-only supervision:

```powershell
python -m test_regr.JointEmbodiedAgentInterface.main train-agent `
  --ce-dfa-mass-only --ce-dfa-mass-weight 0.5 `
  --knowledge metadata --observation-mode metadata-assisted --stage1-epochs 4 `
  --allow-internal-evaluation-training --scene-verified-training-references `
  --save-initialization --batch-size 8 --image-size 384 --seed 42 --load-in-4bit `
  --vision-model google/siglip-base-patch16-384 `
  --vlabench-planning-dir test_regr/VLABenchAgentInterface/data/planning `
  --control-source test_regr/VLABenchAgentInterface/data/control --output OUTPUT
```

`--allow-internal-evaluation-training` accepts only the canonical historical
evaluation corpus with explicit provenance; the default clean-data guard remains
enabled. `--save-initialization` writes `joint_initialization.pt` before EAI
warm-up, controller updates or Stage 1 updates. This is an **untrained Joint
initialization**, not a native Base Qwen baseline. Keep it and all four
`joint_stage1_epoch_000.pt` through `003.pt` snapshots. Best selection uses
validation only; all five snapshots enter the fixed campaign.

The campaign archives split identities, original and scene-verified density
targets and hashes, every changed reference, rendered prompts and image hashes,
observational/context fields, active graph rules, model hashes, source commit,
loss configuration and checkpoint lineage. Targets never bind generation DFAs.
The preflight walks every corrected reference through the observation-bound
metadata DFA and fails on a conflict. Passing this check demonstrates consistency
with these constraints, not that every reference is semantically correct.
Earlier non-density reference-quality flags remain unresolved; no automatic
scene-condition replacement is authorized by this campaign.

## External dataset review

| Dataset | Actions and observations | Independent goals and requirements | Campaign decision |
|---|---|---|---|
| LIBERO Spatial/Object | Native continuous Panda OSC arm/gripper actions; RGB, segmentation, robot state and scene geometry. The bridge supports abstract pick/place only. | BDDL predicates and official initialization states; MuJoCo/robosuite/BDDL in an isolated pinned Python runtime with EGL for preparation. | Twenty tasks, all 100 evaluation states (0–4 per task); another 100 calibration states (5–9). |
| RLBench | Configurable arm and gripper action modes, RGB/depth/masks and low-dimensional state. Diverse tasks include articulation and tools beyond pick/place. | Task-specific native success conditions; CoppeliaSim/PyRep and rendering infrastructure. No BDDL-compatible goal or shared object-ID bridge is assumed. | Feasibility review only. No task substitution or scores in this campaign. |

Primary sources: [LIBERO repository](https://github.com/Lifelong-Robot-Learning/LIBERO),
[LIBERO environment wrapper](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/envs/env_wrapper.py),
[RLBench installation and action modes](https://github.com/stepjam/RLBench),
[RLBench task conditions](https://github.com/stepjam/RLBench/blob/master/rlbench/backend/conditions.py).

Eligibility is determined from task specifications before model evaluation.
Unsupported goal predicates, region semantics or skills fail explicitly. Native
`In(object, basket_contain_region)` is represented abstractly as containment in
that region's physical parent basket; the original native goal is preserved for
physical scoring. Other goal regions are not silently approximated.

## Frozen external inputs and shared rules

The external public adapter exposes instruction, raw and numbered RGB, anonymous
entity IDs for **every** object/fixture, and initial physical-parent relations.
IDs follow an all-entity ordering independent of goals. Images preserve that
ordering. It excludes native task names, BDDL goals, reference plans,
`obj_of_interest` and raw region names such as `target_object_region`. Object
types are observational bindings for affordance rules. Both constraint conditions
use byte-identical prompts and images, verified by hashes.

The `shared-verified` whitelist reuses five existing graph LCs: one skill per
operation, operation signature validity, entity-pointer validity, no
self-placement, and no picking static fixtures. Tests include legitimate
re-picking and moving earlier destinations. Benchmark-specific task templates,
repeated-destination prohibitions, destinations-stay-put and physical-property QA
are excluded. LIBERO tasks never alias VLABench task identities.

`overlap_report.py` inventories declared rules and eight semantic families across
EAI, VLABench, LIBERO and RLBench. It distinguishes observational grounding from
declaration and records applicable candidate-transition events and actual rule
rejections on generated prefixes. Counts use explicit rule/event/episode
denominators, not shared spelling as evidence of semantic equivalence. No
unproved EAI/VLABench literal-rule equivalence is claimed.

## Fixed evaluation and executor

For initialization and epochs 1–4:

* VLABench test450: `full-metadata`, `shared-verified`, `none`; six existing
  dimension scores, Macro, four-term Progress Score, exact match and valid plans.
* LIBERO100: `shared-verified`, `none`; independently held abstract BDDL goal
  satisfaction, plan validity/violations and physical execution. No fabricated
  VLABench dimensions or Macro.
* EAI88: canonical native validation and independent goal success/recall.

`eval_generalization.py` accepts a checkpoint, frozen external manifest and
`--constraint-policy {full-metadata,shared-verified,none}`. Omit policy to run the
fixed per-dataset conditions. Split and checkpoint/manifest hashes prevent
changed-input resumes. Predictions, rendered input records, token decisions and
per-rule traces remain in a separate artifact tree.

The goal-blind executor consumes predicted operations and entity IDs. It uses
Panda OSC geometry-based manipulation, bounded translation/axis-angle commands,
explicit gripper polarity, grasp lift checks, fixed waypoint limits and a
1,000-step horizon. It never substitutes targets using goals, teleports objects,
or queries success to steer manipulation. The outer evaluator checks native
LIBERO success only after execution. Each condition uses identical frozen resets.

Calibration uses reference plans only on states5–9, never evaluation states0–4.
Require at least90/100 native successes and an unchanged executor source hash.
Keep failed calibration attempts; do not exclude difficult tasks. Report Wilson
95% intervals for execution, plus paired successes/failures. Describe the result
as **planner transfer through a fixed executor**, not learned-controller transfer.

`generalization/run_campaign.py` waits for the five atomic snapshot files, runs
model evaluation on free GPUs, executes the plans only after the calibration
gate, and produces `campaign_results.json`/`.md`. It stops after these snapshots.
The Google report receives a separate section, preserving its existing tables.
