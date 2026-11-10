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
roles are observational bindings for affordance rules. Both constraint conditions
use byte-identical prompts and images, verified by hashes.

The `shared-verified` whitelist reuses five existing graph LCs: one skill per
operation, operation signature validity, entity-pointer validity, no
self-placement, and no picking static fixtures. Tests include legitimate
re-picking and moving earlier destinations. Benchmark-specific task templates,
repeated-destination prohibitions, destinations-stay-put and physical-property QA
are excluded. LIBERO tasks never alias VLABench task identities.

Fixture facts use explicit LIBERO movable/fixture declarations and private
semantic IDs with unchanged pointer indices. They never use name substrings
(for example, `table_salt` is movable). The VLABench shared adapter currently
has no verified fixture-role binding: this rule is declared but ungrounded there
and imposes no fixture exclusion. Reports separate groundable episode counts
from applicable transition events. Full-metadata retains the native bindings.

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
Require at least 90/100 native successes and an unchanged executor source hash.
Keep failed calibration attempts; do not exclude difficult tasks. Report Wilson
95% intervals for execution, plus paired successes/failures. Describe the result
as **planner transfer through a fixed executor**, not learned-controller transfer.

`generalization/run_campaign.py` waits for the five atomic snapshot files, runs
model evaluation on free GPUs, executes the plans only after the calibration
gate, and produces `campaign_results.json`/`.md`. It stops after these snapshots.
The Google report receives a separate section, preserving its existing tables.

## Next test: RLBench follow-up

RLBench is now configured as a **separate follow-up evaluation**, rather than
being added to or changing this running LIBERO campaign. The design is in
[`rlbench_next_test.json`](../generalization/configs/rlbench_next_test.json).
It reuses Joint initialization and epochs 1–4, with `shared-verified` and `none`,
without retraining or selecting checkpoints using external results.

The initial suite has **two native task classes and ten task-variation cases**:
`put_rubbish_in_bin` variation0 and `put_groceries_in_cupboard` variations0–8.
Their task sources declare detection-in-destination success; groceries also
requires an empty gripper. The exact scene destination mapping still needs
independent review. These are candidates for the existing abstract pick/place
vocabulary, not a claim that the LIBERO executor runs in RLBench.

Reserve ten evaluation resets and ten calibration resets per case: 100 episodes
in each split. The design assigns deterministic distinct seeds; these are **reset
blueprints**, not official initialization states. The RLBench adapter must freeze
actual simulator/robot states and rendered inputs before any model scoring.

```powershell
python test_regr/JointEmbodiedAgentInterface/generalization/rlbench_next_test.py `
  --output OUTPUT/rlbench_next_test_plan.json
```

The generator rejects overlapping reset splits, duplicate cases, unsupported
skills, changed frozen designs and incorrect episode counts. Optionally pass
`--rlbench-root` to verify the pinned checkout and archive native task-source
hashes. Source revision: `02720bba4c73fe02eb75df946b8791b806028a9d`.

The isolated runtime and separate goal-blind Panda execution bridge are now
implemented; see [RLBench bridge commands and protocol](RLBENCH_BRIDGE.md).
The reserved 100-reset calibration now passes: 93/100 native goal successes,
including 91 fully completed successful plans. Use the matching `source_v33`
controller and calibration summary; RLBench model evaluation has not been
automatically launched. Audit reset reproducibility, native grasp
attachment behavior, quaternion/frame/gripper conventions, wrong-object plans,
timeouts and native success detection. Require at least 90/100 independent
calibration successes with the same bridge/settings used for evaluation. Do not
use task waypoints, success sensors, goal-selected graspable-object lists or
demonstrations to select model/executor targets. The public entity inventory must
cover all physical scene objects independently of task goals, with anonymous
IDs; native names can contain `target` or `distractor`. Exclude task-specific
low-dimensional state, which can expose selected targets or success poses.

Drawer opening, target-pose lifting, selected rack-slot insertion and counted
stacking are explicitly excluded pending verified skill/region/goal mappings.
Do not silently approximate these as generic placement or remove difficult
evaluation cases after calibration. Report native execution success with Wilson
intervals, plan validity, rule violations and paired predictions/traces. Abstract
goal scoring requires its own adjudicated mapping; do not invent BDDL goals or
VLABench dimensions/Macro for RLBench. Report declared, groundable, applicable
and triggered constraints using this adapter's actual coverage. The LIBERO
96/100 calibration result does not qualify an RLBench executor.

Primary task sources:
[rubbish](https://github.com/stepjam/RLBench/blob/02720bba4c73fe02eb75df946b8791b806028a9d/rlbench/tasks/put_rubbish_in_bin.py),
[groceries](https://github.com/stepjam/RLBench/blob/02720bba4c73fe02eb75df946b8791b806028a9d/rlbench/tasks/put_groceries_in_cupboard.py),
[native conditions](https://github.com/stepjam/RLBench/blob/02720bba4c73fe02eb75df946b8791b806028a9d/rlbench/backend/conditions.py).

### Completed RLBench snapshot evaluation (10 November 2026)

The explicitly requested follow-up evaluated all five snapshots on the same
100 frozen evaluation episodes under both policies (1,000 executions). The
archived `source_v33` controller was unchanged; its reset verifier was corrected
to audit unused helper dummies separately, then independently revalidated at
91/100 native and completed successes on reserved calibration resets. Physical
bodies, scoring sensors, roots and joints retained their strict reset checks.
The working-tree contact-aware replacement remains unqualified and was not used.

| Snapshot | No DFA native success | Shared-verified native success | No DFA valid plans | Shared-verified valid plans |
|---|---:|---:|---:|---:|
| Joint initialization (untrained heads) | 0/100 | 0/100 | 0/100 | 0/100 |
| Epoch 1 | 0/100 | 0/100 | 92/100 | 100/100 |
| Epoch 2 | 0/100 | 0/100 | 95/100 | 99/100 |
| Epoch 3 | 0/100 | 0/100 | 92/100 | 100/100 |
| Epoch 4 | 0/100 | 1/100 | 90/100 | 97/100 |

The 95% Wilson intervals are 0.0–3.7% for 0/100 and 0.2–5.4% for 1/100.
Completed successful plans equal native successes in every condition. All
generation error counts are zero. Shared rules improve structural validity
without reliable goal transfer: predictions repeatedly choose destination
entity 1, while the cupboard is entity 9. This is a metadata-assisted subset of
two task classes and ten variations, not the full RLBench benchmark. Dynamic-role
metadata can identify the only movable rubbish object. Joint initialization is
not a native Base Qwen baseline.

All conditions have 100 unique episode IDs, matched input hashes across policies,
and verified prediction hashes; all 1,200 frozen data-file hashes are unchanged.
Source, checkpoint, runtime hashes, physical GPU UUIDs, PIDs, commands, logs,
paired failures and per-case scores are archived on `gpu2.ihmc.us` at
`/home/auszok/experiments/rlbench_model_evaluation_v33_20261110_attempt2/`:
`results.json`, `RESULTS.md`, `COMMANDS.md`, `manifest.json` and
`execution_manifest.json`. Native scoring used CPU software rendering. Section 13
of the Google report contains the separate results table and paired examples;
its 16 earlier tables were verified unchanged. No additional training or executor
promotion followed this evaluation.
