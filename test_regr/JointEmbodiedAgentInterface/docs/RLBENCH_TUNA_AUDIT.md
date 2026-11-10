# Tuna calibration audit

The selected `source_v33` executor remains unchanged and qualified at 93/100
native successes (91 completed successful plans). Its tuna variation scored
5/10. This audit investigates controller behavior with correct reference plans;
it does not measure a trained planner or replace the calibration score.

## Original failure breakdown

| Reserved reset | Original outcome | Failure phase |
|---|---|---|
| 10 | Motion routine completed; native goal false | Deposit/retreat |
| 11 | 1,000-step horizon exhausted without a successful pick | Pregrasp |
| 16 | Motion routine completed; native goal false | Deposit/retreat |
| 17 | Motion routine completed; native goal false | Deposit/retreat |
| 19 | Motion routine completed; native goal false | Deposit/retreat |

A `place` event with `success: true` means the motion routine returned; the
independent native task condition still determines benchmark success.

## Instrumented diagnostic replays

The replays used the same reserved episode identities and stored poses, with RGB
capture and passive phase observers. Native paths/contact dynamics are not
bitwise deterministic. Results below are diagnostic observations, not a new
5/10 denominator or a replacement benchmark score.

| Reset | Replay result | Observation |
|---|---|---|
| 10 | Native goal true | Can stayed inside through retreat and settling |
| 16 | Native goal true | Can stayed inside through retreat and settling |
| 17 | Native goal false | Insertion path could not be created |
| 19 | Native goal false | Goal true after release, then can rolled outside and fell onto table |
| 11 | Native goal false | Grasp approach/standoff consumed the 1,000-step budget |

Reset 19 is a concrete release-stability counterexample. Immediately after
release, its centre was `(0.5904, 0.0687, 1.3316)` metres and native success was
true. After retreat it was already outside the physical cupboard bounding box;
after settling it was `(0.2039, 0.0804, 0.7707)` metres and native success was
false. Native attachment had been released. The can's roughly 1.331 m shelf
centre height is consistent with resting on its round edge, rather than its
flat face (roughly 1.306 m). The sideways entry rotation makes rolling possible.
This observation does not establish that every original deposit failure had
exactly the same cause.

## Grasp geometry and sensor mismatch

The physical primary can is 80 mm across and 30 mm thick. The fully open Panda
finger gap is about 80 mm, leaving very little release clearance. World-axis
bounds of the rotated can are larger and must not be mistaken for its diameter.

The recorded vertical top proposal is rejected by the actual native gripper
gate. Its closest can material is around 49 mm along the sensor axis. Tilted
retries later obtain native attachment, after contact can already have disturbed
the can. The 2 mm top offset is therefore a proposal, not a guaranteed valid
grip. A synthetic unit-test sensor does not establish the actual Panda detection
boundary; native traces are authoritative. Attachment is never forced when the
gate is false.

## Geometry-only oblique-placement feasibility probe

A separate, unqualified prototype aligned the can's short physical axis toward
a 45-degree oblique placement, rather than unconditionally turning it sideways.
It used mesh geometry and the predicted object/container IDs; native goal sensor
positions were not used for control.

| Reset | Prototype result | Final short-axis tilt from vertical |
|---|---|---:|
| 19 | Completed; native goal true | 33.4 degrees |
| 10 | Horizon exhausted; native goal false | 44.9 degrees |
| 16 | Completed; native goal true | 36.7 degrees |

This is mixed feasibility evidence, not a validated repair: two of three goals
were met, one timed out, and the final cans were tilted rather than proven to be
flat and stably settled. The prototype was not promoted into the qualified
executor, and no new full-calibration or model result is claimed.

## Repair direction

The implementation and its separate promotion criteria are documented in
[RLBENCH_CONTACT_REPAIR.md](RLBENCH_CONTACT_REPAIR.md). The historical measurements
below remain the evidence for `source_v33`; they are not replacement scores.
The follow-up candidate completed two full calibrations but retained Tuna 5/10
in both and failed promotion; `source_v33` remains selected. See the
[full follow-up results](RLBENCH_CONTACT_CALIBRATION_20261109.md).

Use the actual gripper gate for surface/contact acquisition rather than a fixed
above-top height. For wide, thin bodies, evaluate edge/rim grasps that permit the
open fingers to clear the object. Select a face-stable placement orientation
subject to the whole hand/payload envelope, wrist reach and fixture clearance;
a single fixed oblique angle is insufficient. Verify separation and settling
using physical object motion/support observations, without feeding native task
goals back into control. Test all ten reserved tuna states and then the complete
100-episode calibration before replacing `source_v33`.

## Artifacts

Server root: `/home/auszok/experiments/rlbench_followup_20261109/`.

- `tuna_release_audit_v33/episodes.jsonl`: phase-by-phase state and native scores.
- `tuna_release_audit_v33/reset*/`: native RGB captures and individual traces.
- `audit_tuna_release_phases.log`, `tuna_audit_job.json`: replay command/log.
- `audit_initial_can_geometry.log`, `audit_release_clearance.log`: dimensions.
- `audit_native_gate_range.log`: actual native gate acceptance/rejection.
- `tuna_oblique45_probe.json`, `probe_tuna_oblique45.py`,
  `probe_tuna_oblique45.log`: unqualified oblique feasibility experiment.

Only reserved calibration states were exercised. Training jobs, model-evaluation
states, reference files and the selected qualified controller were unchanged.
