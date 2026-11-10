# RLBench simulator bridge and fixed executor

RLBench runs in its own CoppeliaSim/PyRep environment, separately from the active
Stage 1 and LIBERO campaign. This is **planner transfer through a fixed executor**,
not learned-controller transfer. No model evaluation is qualified until its own
100-reset executor calibration reaches at least 90 native successes.

**Current status: aggregate calibration gate passed.** The fixed 100 reserved
resets reached **93/100 native goal successes** (95% Wilson interval 86.3–96.6%).
**91/100 plans fully completed with native success**; two additional episodes met
the native goal but exhausted the horizon during retreat. Both aggregate counts
exceed the required 90/100 gate. No RLBench model evaluation has been launched.
Tuna remains the weakest case at 5/10 native successes.

## Runtime and commands

The [Dockerfile](../generalization/rlbench_runtime.Dockerfile) pins RLBench
`02720bba4c73fe02eb75df946b8791b806028a9d`, PyRep
`8f420be8064b1970aae18a9cfbc978dfb15747ef`, CoppeliaSim 4.1 and Python 3.8 package
versions. It records the simulator archive digest. A private Xvfb/Mesa display
uses CPU software rendering, with no GPU allocation. The bootstrap starts Xvfb
without Coppelia's library path, then starts the simulator with its required
libraries. Use a separate output tree for every protocol revision.

```bash
BRIDGE=test_regr/JointEmbodiedAgentInterface/generalization
docker build -f "$BRIDGE/rlbench_runtime.Dockerfile" -t joint-rlbench:fixed "$BRIDGE"

# Mount the repository read-only and an experiment directory writable.
docker run --rm --cpus=6 --memory=12g \
  -v "$PWD:/repo:ro" -v /absolute/experiment:/experiment \
  joint-rlbench:fixed /repo/$BRIDGE/rlbench_campaign.py prepare \
  --output /experiment/frozen

docker run --rm --cpus=6 --memory=12g \
  -v "$PWD:/repo:ro" -v /absolute/experiment:/experiment \
  joint-rlbench:fixed /repo/$BRIDGE/rlbench_campaign.py calibrate \
  --manifest /experiment/frozen/manifest.json --output /experiment/calibration

# Prediction JSONL: exactly one episode_id + prediction per evaluation reset.
docker run --rm --cpus=6 --memory=12g \
  -v "$PWD:/repo:ro" -v /absolute/experiment:/experiment \
  joint-rlbench:fixed /repo/$BRIDGE/rlbench_campaign.py evaluate \
  --manifest /experiment/frozen/manifest.json \
  --calibration /experiment/calibration/summary.json \
  --predictions /experiment/predictions.jsonl --output /experiment/execution

python "$BRIDGE/rlbench_report.py" --results /absolute/experiment/calibration \
  --output /absolute/experiment/calibration_report
```

The physical evaluator consumes predictions; it does not generate plans or train
a model. [The planning evaluator](../eval_rlbench.py) runs in the existing
PyTorch/GPU environment, separately from the simulator. It supports
`--constraint-policy shared-verified` and `none`; omitting policy runs both:

```bash
python -m test_regr.JointEmbodiedAgentInterface.eval_rlbench \
  --checkpoint /absolute/campaign/checkpoints/joint_stage1_epoch_000.pt \
  --episode-manifest /absolute/experiment/frozen/manifest.json \
  --calibration /absolute/experiment/calibration/summary.json \
  --experiment-output /absolute/experiment/planning/epoch1 \
  --planning-dir /absolute/VLABench/planning \
  --planner-model /absolute/models/Qwen3-VL-8B-Instruct \
  --vision-model google/siglip-base-patch16-384 --image-size 384 384
```

It checks calibration before loading a model, reads only public inputs, preserves
pointer IDs, saves actual rendered-input hashes and token/rule decisions, and
verifies matched inputs across constraint conditions. Simulator evaluation then
uses each condition's `predictions.jsonl`. Native goals are scored there, not
manufactured in the planner evaluator. Storage paths are loader references only;
images reach the model as pixels and paths must never be serialized into prompt
text. The historical LIBERO queue remains unchanged. No RLBench model tests are
automatically launched.

## Frozen resets and input separation

The suite contains two task classes and ten task/variation cases: rubbish 0 and
groceries 0–8. Each has ten evaluation seeds and ten different calibration seeds.
All 200 resets are retained. These are deterministic native resets, not official
RLBench initialization-state files.

`prepare` archives native states, then canonicalizes every reset through the same
restore API before freezing state and rendered images. This removes native
finger-coordinate overshoots beyond joint limits without filtering episodes.
Original native states are retained as
`native_state_before_canonicalization.json`. Task-object poses, robot joint
coordinates and colors are restored before execution. Quaternion sign changes
are treated as the same rotation. Reset checks require task/base-pose and joint
coordinate errors below 1e-5; passive robot-link pose deviations are separately
bounded at 0.002 in metres/quaternion components and recorded per episode.

The public input is an explicit allowlist: instruction, raw/numbered front RGB,
anonymous IDs for the complete physical task-entity inventory, dynamic/static
roles and target-independent object positions. Robot and nonphysical helpers are
excluded. Native entity names, variation/task IDs, success sensors, custom
low-dimensional task state, demonstrations, waypoints and task-provided
graspable-object lists never enter that input. Numbered images use actual
segmentation handles and the frozen entity table. Private references and native
success conditions remain separate. Input/state files are individually hashed.

**Metadata caveat:** the native rubbish task makes its distractors static while
the rubbish is dynamic. Publishing native dynamic/static roles can identify its
only movable object. Label this metadata-assisted transfer; it is not an
instruction-and-images-only official baseline.

## Goal-blind manipulation

The executor accepts only pick/place operations and integer entity pointers.
Invalid pointers, unsupported actions, static picks, empty plans, failed grasps
and timeouts remain failures. Deterministic retries keep the same predicted
object, operate within the same 1,000-step budget, and never reset its pose.
Approaches use observed widths and heights, with tilted-wrist, opposite-heading
and lateral-grasp fallbacks. Lateral clearance comes from the rotated physical
hand envelope and the observed supporting surface. Finger symmetry does not
imply equivalent arm reachability for a 180-degree wrist flip. Failed pinches release and settle before recomputing approach
geometry, avoiding retries against transient falling-object poses. Recovery
lifts are reserved for hands close to the body; a clear pregrasp pose no longer
triggers an unnecessary extra lift on every rejected approach. Bodies wider than the finger opening use native proximal top grasping rather
than a straddling descent. The tip stays 2 mm above the observed top surface,
leaving the body below the finger ends while it remains within the native
proximity proposal. The actual native detector may reject this pose; attachment
is never forced and retries remain restricted to the same predicted body. This
also applies to short, wide bodies: a flat can can otherwise
remain physically wedged after native attachment has been released. This depends on the explicitly authorized
simulated-attachment protocol; every attachment still requires the native gate
for the predicted body. An earlier all-object shallow-grasp variant was rejected and retained in the
artifact archive. The current top clearance applies specifically to oversized
bodies.

Panda motion uses native IK/configuration paths, world coordinates and xyzw
quaternions; gripper 1 means open and 0 means closed. Native simulator steps are
counted explicitly. No object is teleported during execution. Reset restoration
is confined to the episode boundary; it does not use PyRep's per-joint
`disable_dynamics=True`, which would advance physics between joint updates.

The user explicitly selected **native proximity-gated grasp attachment** after
physics-only probes failed. `try_grasp` calls PyRep's native mechanism for the
predicted physical object only, requires its detector check and verifies that
attachment preserves world pose. It never enumerates registered task targets.
Opening releases the native attachment. Physical pinching can still prevent a
clean deposit, which is retained as a failure when the native goal is false.
This is standard simulated attachment, not a
claim of a physics-only grasp or a learned grasp policy.

Receptacle approaches come from physical meshes and gravity. Side-entry
orientation minimizes the hand/payload vertical envelope; floor/ceiling rays
check fit, and the release point keeps the payload inside the opening. Its
height is near the lowest feasible hand/payload placement, with the existing
floor clearance plus 1 cm, rather than the middle of the opening. This reduces
unsupported dropping and increases reach at distant shelf positions. The hand
reorients in a clear area before insertion. Ordinary rotation uses the direct native path. A completed
trajectory with no meaningful tracking improvement for 20 physics ticks activates
local recovery, using orientation waypoints of at most 0.2 radians at a fixed
Cartesian anchor. Recovery starts at the actual stalled pose. These motions use
the same global step budget. Neither success-sensor coordinates
nor expert waypoints supply control targets. Unloaded transit uses collision-
checked planning. During contact motion, the predicted body is exempted from kinematic
self-contact checks. The planner first retains all neighboring obstacles. Only
a failed strict path search enables a second attempt allowing contact with
movable neighbors whose observed bounding-box gap is at most 0.06 m. Static fixtures, walls, the table and robot
collision checks remain active. This permits physical contact with nearby
clutter instead of rejecting every such grasp before simulation. Every flag is
restored before physical motion; respondability is never changed. Only the
predicted body is eligible for native attachment. Per-motion contact-exemption
IDs are archived in each episode result. Tracking
failures still fail. This policy is part of the calibrated bridge, not a
collision-free execution guarantee.

Only the outer evaluator calls native success after execution. Calibration
references are scoring-private and are used exclusively on the 100 calibration
resets. They identify native goal objects but never pass detector coordinates to
the executor. Physical destination mappings are restricted to reviewed bin and
cupboard entities. Other skills, selected rack regions and counted stacking need
separate mappings rather than silent substitutions.

## Qualification and artifacts

`evaluate` requires exactly 100 unique predictions and a 100-reset calibration
score of at least 90%. Manifest, runtime, executor, bridge, runner, horizon and
grasp-protocol settings must match. Changed predictions, source or settings reject
resume. Every case remains in the denominator. Failures preserve last requested
poses, phases, attempted object IDs, step counts and native goal outcomes.

Reports include native success and Wilson 95% intervals, complete-plan native
success, per-case counts and failure phases. They do not manufacture VLABench
dimensions/Macro or abstract BDDL scores. The LIBERO 96/100 result cannot qualify
this executor. Simulator source, model-independent rule opportunities and actual
rule triggers must be reported separately from native physical conditions.

Server artifacts for the current implementation are under
`gpu2.ihmc.us:/home/auszok/experiments/rlbench_followup_20261109/`.
`frozen/` preserves preliminary resets; `frozen_v2/` contains canonical resets.
Calibration attempts and logs are retained in separate numbered directories.
Reset identities and images are fixed; native IK/OMPL sampling and physical
execution are not claimed to be bitwise repeatable. Executor repair uses only
the reserved calibration states, never model evaluation outcomes.
Only a full passing calibration with matching code/settings qualifies model
execution; successful single-scene probes do not.

## Constraint coverage

The shared profile reuses the five semantically verified rule families from
`shared_constraints.py`, without mapping RLBench tasks to VLABench task IDs.
Coverage is over 200 frozen states (100 calibration, 100 evaluation); actual
applicable events and rejected alternatives are counted during model decoding,
and remain pending until that evaluation runs.

| Family | Declared / groundable evidence | Applied by |
|---|---|---|
| Grammar, operation schema, pointers | Shared rules; anonymous inventory in 200/200 states | Shared DFA and plan-validity diagnostics |
| Action preconditions | Predicted holding state and static/dynamic roles | Fixed executor and plan-validity diagnostics; not a claim of full precondition coverage by the shared DFA |
| Object affordances | Declared static/dynamic roles in 200/200 states | Shared static-pick exclusion; no inferred material or graspability labels |
| Scene relations | Initial positions, physical meshes, opening geometry | Goal-blind manipulation; no task-specific relation DFA |
| Task sequences | Native success conditions held privately | Scoring only; no benchmark answer templates during generation |
| Physical-property rules | No verified cross-dataset density/material mapping | Excluded from this profile |

The native predicate and execution success rate must be reported separately from
DFA validity. A valid schema or a rule rejection is not evidence that the robot
completed the instruction.

## Original 88/100 calibration evidence

Original source: `source_v20/`; results: `calibration_native_v20/`; human-readable
report: `calibration_report_selected/REPORT.md`, under the server artifact root
above. This historical run did not meet the gate. Its container exited with
status 1 because the qualification threshold was not reached, rather than a
simulator crash. It used CPU software rendering and no physical GPU.

| Native task / variation | Calibration success |
|---|---:|
| Rubbish / 0 | 10/10 |
| Groceries / 0 (crackers) | 9/10 |
| Groceries / 1 (chocolate jello) | 10/10 |
| Groceries / 2 (strawberry jello) | 7/10 |
| Groceries / 3 (soup) | 10/10 |
| Groceries / 4 (tuna) | 6/10 |
| Groceries / 5 (spam) | 9/10 |
| Groceries / 6 (coffee) | 9/10 |
| Groceries / 7 (mustard) | 9/10 |
| Groceries / 8 (sugar) | 9/10 |
| **Total** | **88/100** |

Failure cases remain in the denominator. The remaining defects include blocked
or unreachable grasp approaches, insertion/reorientation tracking failures and
completed deposits that do not satisfy native goals. Passing schema checks or
lowering the 90% threshold must not substitute for repairing these failures.
Exploratory executor configurations used calibration states only; incomplete,
rejected attempts are not reported as completed 100-episode calibrations.

Separate native semantic negative controls execute wrong-object pick/place plans
on a calibration state. Candidate `v24` completed its wrong-object manipulation
and native goal success remained false. The selected `v20` negative-control run
hit the 1,000-step horizon and also scored false; its empty plan scored false.
These checks use no external model predictions and do not qualify the executor.

## Qualified 93/100 follow-up calibration

The selected immutable controller is `source_v33/`, with full results in
`calibration_native_v33/` and the generated report in
`calibration_report_v33/REPORT.md`. Use
`calibration_native_v33/summary.json` as the matching calibration gate. The
container `rlbench-calibration-native-v33-20261109` exited **0**; host PID was
**3570487**, using CPU software rendering and no GPU. The exact launch command,
image digest and source hashes are archived in `calibration_native_v33_job.json`.
The original 88/100 artifacts and all rejected trials remain preserved.

The effective changes are:

- Try strict collision avoidance first; allow only local movable contact after
  that path search fails. Static/structural checks and physical response remain
  enabled, and only the predicted object can be attached.
- Recover rejected low hovers using the observed clutter height and hand bounds;
  retain opposite-wrist and lateral same-object grasp alternatives.
- Recover failed direct wrist rotations through bounded local orientation
  waypoints. The original global 1,000-step horizon remains unchanged.
- Use a 25 mm tolerance only at clear staging waypoints. Final placement keeps
  its 8 mm tolerance. After rotation and staging, freeze the actually realized
  orientation and recompute the held-object offset before commanding placement;
  stale geometry is no longer mixed with a different nominal orientation.
- Propose top clearance for bodies near the gripper opening limit and release
  near the lowest feasible physical support height. Native contact detection
  remains authoritative, and seven native-goal failures remain in the total.

| Native task / variation | Original native success | Qualified native success |
|---|---:|---:|
| Rubbish / 0 | 10/10 | 9/10 |
| Groceries / 0 (crackers) | 9/10 | 10/10 |
| Groceries / 1 (chocolate jello) | 10/10 | 10/10 |
| Groceries / 2 (strawberry jello) | 7/10 | 10/10 |
| Groceries / 3 (soup) | 10/10 | 10/10 |
| Groceries / 4 (tuna) | 6/10 | 5/10 |
| Groceries / 5 (spam) | 9/10 | 10/10 |
| Groceries / 6 (coffee) | 9/10 | 9/10 |
| Groceries / 7 (mustard) | 9/10 | 10/10 |
| Groceries / 8 (sugar) | 8/10 | 10/10 |
| **Total** | **88/100** | **93/100** |

The aggregate gate passes; this does not mean every variation reaches 90%.
Tuna regressed from 6/10 to 5/10 and remains a material executor limitation.
The 100 evaluation resets were not used for controller tuning or model scoring.
All 200 frozen state/input hashes and the local/native controller source hashes
were verified after the completed run.

**Validation:** 51 focused tests passed, including prompt/reference separation,
ID preservation, reset semantics, collision-flag restoration, strict-first
contact behavior, local rotation/horizon handling, physical support clearance
and recomputation after realized staging-pose drift. Three native wrong-object
controls all scored false; two completed their manipulations and one exhausted
the horizon. Their empty-plan controls also scored false. Checked planning calls
changed neither task-object poses nor simulation time and preserved physical
response flags. See `semantic_negative_controls_v33.json`.

`STATUS.json` records the selected passing source, manifest, calibration and
report. This qualifies the fixed executor for the planned transfer evaluation;
it supplies no trained-planner or learned-controller transfer result.

## Contact-aware repair candidate

The later contact-aware grasp/placement candidate was not promoted. Its two full
calibrations scored 78/100 and 77/100 native success, with Tuna 5/10 in both;
the stricter two-run promotion gates failed. `source_v33` remains selected and
unchanged. See [implementation and gates](RLBENCH_CONTACT_REPAIR.md) and
[complete calibration evidence](RLBENCH_CONTACT_CALIBRATION_20261109.md).
