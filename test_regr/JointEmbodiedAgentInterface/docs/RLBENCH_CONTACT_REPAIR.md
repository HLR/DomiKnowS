# Contact-aware RLBench executor repair

The qualified `source_v33` controller and its 93/100 native successes (91/100
completed successful plans) remain archived. Its Tuna score is 5/10. This repair
must pass two fresh, complete calibrations before replacing that controller.

## Movement policy

The policy uses predicted entity IDs and physical geometry, never task names to
select a manipulation strategy. Wide, thin bodies trigger a search using the
measured open-finger gap, rim offsets, radial wrist headings, both tilt directions,
and table clearance. The specialized flat-face policy requires an observed
short-axis support face; an upright carton is not classified from dimensions
alone. Each grasp is checked jointly with its predicted destination.
Ordinary objects retain the earlier grasp proposals; every object now requires
native detector acceptance before attachment and lifting. A bounded contact seek
may move at most 10 mm in 1 mm increments, subject to support clearance.

A persistent, separate native simulator process receives physical shapes and
joints, the scene-loading identity, and the predicted object/destination IDs.
It receives no goal sensors, reference plans, or task waypoints. Hypothetical
restoration, attachment and FK occur only there. The live bridge verifies unchanged
physical state, attachment identities, and simulation time around every request.
Worker timeouts discard the process and socket, preventing stale replies from
being reused for later episodes.

Placement candidates combine face-stable object rotations, shelf depth, release
height and the measured grasp transform. Native paths check robot/fixture
collisions and joint limits; explicit spatial mesh checks cover the complete hand
and carried payload against scene structures (including the table), payload/wrist
collisions, and the fully open release pose. Only the payload must be contained
between the shelf planes: a palm remaining outside the opening is not rejected
for extending beyond those planes. All 16 side-entry rolls are considered.
The gripper hierarchy includes its base. After live staging, the measured pose/transform is checked again before
insertion. Failed searches report infeasibility, rather than executing an unchecked
orientation. Native IK/planning remains stochastic and can reject feasible poses.

Release verification checks native detachment, open-finger separation after
physical withdrawal, observed
physical support, and motion over a 15-tick window. Thresholds are 2 mm translation,
2 degrees rotation, and 12 mm support gap; thin shelf objects must settle within
15 degrees of their flat face. Up to 60 verification ticks, all motion and gripper
steps count against the unchanged 1,000-step horizon. Native terminal task success
is evaluated separately; a successful movement event does not imply task success.

## Calibration and promotion

Only the existing calibration resets 10–19 may be used for engineering. Evaluation
resets 0–9 and model predictions are excluded. All original artifacts are retained
under `/home/auszok/experiments/rlbench_followup_20261109/`; candidate sources,
diagnostics, job descriptors and results are under `contact_repair_20261109/`.
Runs use CPU software rendering, no GPU, and leave active training untouched.

Each of two fresh 100-episode processes must reach **93 native successes**, **91
completed successful plans**, and **9/10 native successes in every task/variation**,
including Tuna. Results cannot be assembled from selected episodes across runs.
All executable RLBench modules, the shared protocol dependency, control settings,
runtime, frozen data, process identity and exit status are recorded. Resume and
model-evaluation gates reject changed fingerprints or missing two-run promotion
evidence. A failed candidate leaves `source_v33` selected.

Run the promotion check after both complete:

```bash
python rlbench_qualification.py --runs CALIBRATION_RUN1 CALIBRATION_RUN2 \
  --manifest /home/auszok/experiments/rlbench_followup_20261109/frozen_v2/manifest.json \
  --output promotion.json
```

No model evaluation or additional training is launched by this repair campaign.

## Follow-up repair, 2026-11-10

The failed v4 runs remain archived. The follow-up lives separately at
`/home/auszok/experiments/rlbench_repair_v2_20261110/`; its focused probes are
engineering diagnostics, not qualification runs. `source_v33` remains selected.

The pinned PyRep mesh binding overwrites the native normals pointer with triangle
indices, reads beyond the index buffer, and attempts to free an integer index.
The local `rlbench_geometry.native_mesh` adapter copies the vertices/triangles and
frees the three original buffers exactly once; the shared runtime is unchanged.
Mesh topology is cached per reset, with world coordinates recomputed from measured
poses. Physical support uses this mesh rather than visual bounding boxes, which
can extend below a rotated cylinder's actual support face.

Round bodies now enter contact-aware search even when they are tall or narrower
than the gripper. Thin objects interleave level side contacts from multiple
headings with tilted rim and top approaches.
A rigid relative-pose screen checks the native proximity detector and actual open
fingers in the isolated simulator before expensive arm IK; it restores the shadow
body on acceptance, rejection, and error. The reached FK pose is checked again,
and native live detector acceptance remains mandatory before attachment.

The live controller reuses candidate joint paths, then revalidates them against
the measured scene (including native whole-arm collision checks) before executing
each segment. A failed measured closed grasp is rejected before lifting and
can request another candidate for the same entity. Placement is recomputed from
the measured held transform. Open-finger withdrawal continues commanding opening;
a detached body is allowed 30 physical settling ticks before withdrawal, followed
by up to 120 support-verification ticks. Motion, opening, dwell and verification
all count against the original 1,000-step execution horizon. Support, motion,
face-angle and promotion thresholds are unchanged.

Focused native probes have completed Coffee reset 10 and Soup reset 10 with native
success. Tuna approach, lift and release failures remain under investigation;
these isolated successes do not establish the required two-run reliability.

### Contact feedback and release corrections

The follow-up also fixes native joint-limit interpretation: PyRep returns
`[minimum, range]`, so the upper limit is their sum. The first lift preserves the
measured post-contact hand orientation and travels vertically, rather than
reorienting toward the nominal pre-contact target while the payload is still on
its support. Contact targets are expressed in the predicted body's measured
frame; bounded replanning follows translations and rotations during approach.
Stopping a stale approach cancels only its RML trajectory and commands measured
joint targets, without setting live poses or advancing hidden physics steps.

Finger closure previously ran at 0.2 m/s, equivalent to 10 mm per 50 ms simulator
tick. Contact feedback now closes at 0.02 m/s and stops at the first measured
contact/near-contact instead of continuing to squeeze the body. Native detector
acceptance is checked again after closure. Finger motors stop on completion,
timeout, and horizon exhaustion.

Release checks now inspect the **realized inserted body**, not just the pose
predicted before insertion. They verify its support face, entire support
footprint, opening clearance, and the full vertical settling envelope. Candidate
settling envelopes retain a 2 mm finger-clearance margin. A single bounded outward
correction can replan the same held body for the same predicted destination.

The explicitly authorized native proximity-gated attachment remains active while
fingers open at 0.04 m/s. After a second check of the **actual open fingers** and
settling clearance, native detachment must preserve world pose; detachment occurs
before withdrawal. This changes native attachment timing and is included in the
calibration fingerprint. It neither disables payload dynamics nor teleports the
payload. Native success predicates remain evaluator-only.

One physical regrasp recovery may move the same predicted body to a clear site on
its observed support surface and reacquire it. It uses no goal-selected object
list or task-name strategy; canonical entity IDs and prediction formats are
unchanged. Opening, contact following, staging, regrasping, correction, settling,
and withdrawal all share the original **1,000 physical-step horizon**.

**139 focused tests pass.** The earlier complete ten-reset engineering probes
still failed qualification (p16: 4/10 native, 4/10 completed; p28: 4/10 native,
3/10 completed). Later isolated probes completed the difficult Tuna reset 13,
Coffee reset 10, and Tuna reset 16. These results use different development
snapshots and must not be combined into a qualification score. The frozen p33
Tuna preflight finished at 7/10 native successes and 6/10 completed successful
plans. Subsequent p34 reserved-reset probes completed resets 12 and 18; these
isolated results are not qualification evidence. Placement staging now checks
the onward grasp and placement feasibility before selecting a staging site.
Promotion still requires
two fresh complete 100-episode runs meeting every original replacement gate.

The separately requested model evaluation uses the unchanged archived `source_v33`
controller, not this unqualified replacement. Its reset verifier audits unused
native helper dummies separately: physical bodies, scoring sensors, roots, and
joints retain the original strict reset checks. The changed verifier requires a
fresh complete calibration before physical model evaluation. Frozen reset files
are unchanged, and evaluation resets are excluded from controller repair decisions.

## Validation status

**60 focused tests pass, but the candidate failed physical qualification.** Both
fresh runs completed all 100 frozen episodes from immutable `source_final_v4`.

| Metric | Run 1 | Run 2 | Required in each run |
|---|---:|---:|---:|
| Native success | 78/100 | 77/100 | 93/100 |
| Completed successful plans | 69/100 | 64/100 | 91/100 |
| Tuna | 5/10 | 5/10 | 9/10 |
| Coffee | 1/10 | 1/10 | 9/10 |

The repair is **not promoted**. `source_v33` remains selected with its archived
93/100 native successes and 91/100 completed successful plans. Tuna did not
improve reliably, and other cases regressed. Native path search, acquisition,
withdrawal contact, and stability failures remain; passing unit tests and a
successful isolated placement do not establish physical reliability.

Full case scores, commands, PIDs, confidence intervals and per-episode failures:
[calibration report](RLBENCH_CONTACT_CALIBRATION_20261109.md). Evidence and logs are
archived under `contact_repair_20261109/`, including `promotion_v4.json` and
`calibration_final_v4_run{1,2}/`. Both native jobs exited 1 after writing complete
summaries because the qualification guard failed; they were not interrupted.
Source_v33 hashes, candidate source hashes and all frozen dataset files were
verified unchanged. No model evaluation or training campaign followed.

Reserved diagnostic trials completed with native success for cracker reset 10
(312 steps) and Tuna reset 19 (419 steps). These are preflight diagnostics, not
full-calibration scores. Earlier candidate sources and incomplete attempts are
retained; they do not count as successful validation. The earlier complete v3
candidate runs are retained separately and cannot replace the final v4 evidence.

Focused tests cover detector/reference separation, rejected grasps without lifts,
pointer preservation, stable-axis transforms, both wrist directions, gripper-base
classification, wrist collisions, joint limits, rolling/rotation/support failures,
native detachment, live-state isolation, and strict promotion coverage. Native
calibration results are recorded separately; unit tests do not qualify an executor.
