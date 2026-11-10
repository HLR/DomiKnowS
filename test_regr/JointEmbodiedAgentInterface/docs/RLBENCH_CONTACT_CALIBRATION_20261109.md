# Contact-aware RLBench calibration

Fixed-executor calibration on all 100 reserved resets, not model generalization accuracy.

**Promotion: FAILED; source_v33 remains selected**.

Both final runs completed all 100 frozen episodes. Exit status 1 after a complete summary is the intentional failed-qualification guard, not an interrupted run.

| Metric | Run 1 | Run 2 | Required in each run |
|---|---:|---:|---:|
| Native success | 78/100 | 77/100 | 93/100 |
| Completed successful plans | 69/100 | 64/100 | 91/100 |
| Tuna | 5/10 | 5/10 | 9/10 |

| Case | Run 1 native | Run 2 native |
|---|---:|---:|
| Rubbish | 9/10 | 9/10 |
| Crackers | 9/10 | 10/10 |
| Chocolate | 10/10 | 7/10 |
| Strawberry | 9/10 | 8/10 |
| Soup | 8/10 | 10/10 |
| Tuna | 5/10 | 5/10 |
| Spam | 9/10 | 9/10 |
| Coffee | 1/10 | 1/10 |
| Mustard | 9/10 | 10/10 |
| Sugar | 9/10 | 8/10 |

## Runtime and provenance

Baseline source hashes unchanged: **True**. Candidate source matches frozen hashes: **True**.
Local executable sources match the evaluated snapshot: **True**. All frozen data files unchanged: **True**.
Manifest SHA-256: `566fef1e372d1cb8de951fb8ce11355dabe26433630905afb76821e1681d76ed`.
60 focused tests passed; `git diff --check` passed. Native goal checks remain outside the controller.
No GPU was used. Existing training jobs and trained checkpoints were unchanged. No model evaluation was launched.
The original source_v33 result remains archived: 93/100 native success, 91/100 completed successful plans, Tuna 5/10.

### Run 1

Physical GPU: none (CPU software rendering). PID: `4158607`. Exit status: `1`.
Native success Wilson 95% interval: `[0.6892946528499969, 0.849988254963442]`.
Frozen source manifest: `/home/auszok/experiments/rlbench_followup_20261109/contact_repair_20261109/final_v4_source_manifest.json`.
Log: `/home/auszok/experiments/rlbench_followup_20261109/contact_repair_20261109/calibration_final_v4_run1.log`.

Command:

```bash
docker run --name rlbench-contact-final-v4-run1-20261109 --cpus=6 --memory=12g -v /home/auszok/experiments/rlbench_followup_20261109:/home/auszok/experiments/rlbench_followup_20261109 sha256:5fff7d4968af50c6278328c2086aa6747f9fbbf0523765ae9c389dee83b55a58 /home/auszok/experiments/rlbench_followup_20261109/contact_repair_20261109/source_final_v4/rlbench_campaign.py calibrate --manifest /home/auszok/experiments/rlbench_followup_20261109/frozen_v2/manifest.json --output /home/auszok/experiments/rlbench_followup_20261109/contact_repair_20261109/calibration_final_v4_run1
```

Failures:

- `put_rubbish_in_bin/variation00/reset17`: native=False, steps=229; released object failed physical support/stability verification
- `put_groceries_in_cupboard/variation00/reset15`: native=False, steps=135; planning worker: RuntimeError: no stable shelf placement: ['wrist_orientation: geometry-only Panda path failed: Could not create path.', 'insertion: geometry-only Panda path failed: Could not create path.', 'insertion: geometry-only Panda path failed: Could not create path.', 'insertion: geometry-only Panda path failed: Could not create path.']
- `put_groceries_in_cupboard/variation02/reset10`: native=False, steps=251; planning worker: ValueError: payload/structure collision: cupboard
- `put_groceries_in_cupboard/variation03/reset10`: native=True, steps=401; released object failed physical support/stability verification
- `put_groceries_in_cupboard/variation03/reset11`: native=False, steps=343; planning worker: ValueError: payload/structure collision: cupboard
- `put_groceries_in_cupboard/variation03/reset12`: native=True, steps=451; released object failed physical support/stability verification
- `put_groceries_in_cupboard/variation03/reset13`: native=True, steps=393; released object failed physical support/stability verification
- `put_groceries_in_cupboard/variation03/reset14`: native=True, steps=437; released object failed physical support/stability verification
- `put_groceries_in_cupboard/variation03/reset15`: native=True, steps=372; released object failed physical support/stability verification
- `put_groceries_in_cupboard/variation03/reset16`: native=True, steps=406; released object failed physical support/stability verification
- `put_groceries_in_cupboard/variation03/reset17`: native=False, steps=297; planning worker: ValueError: payload/structure collision: cupboard
- `put_groceries_in_cupboard/variation03/reset18`: native=True, steps=379; released object failed physical support/stability verification
- `put_groceries_in_cupboard/variation03/reset19`: native=True, steps=447; released object failed physical support/stability verification
- `put_groceries_in_cupboard/variation04/reset10`: native=False, steps=368; released object remained in contact with fingers after withdrawal
- `put_groceries_in_cupboard/variation04/reset13`: native=False, steps=0; planning worker: TimeoutError: geometry search budget exhausted; feasibility unverified
- `put_groceries_in_cupboard/variation04/reset14`: native=False, steps=214; planning worker: RuntimeError: no stable shelf placement: ['preplace: geometry-only Panda path failed: Could not create path.', 'preplace: geometry-only Panda path failed: Could not create path.', 'preplace: geometry-only Panda path failed: Could not create path.', 'preplace: geometry-only Panda path failed: Could not create path.']
- `put_groceries_in_cupboard/variation04/reset18`: native=False, steps=342; released object failed physical support/stability verification
- `put_groceries_in_cupboard/variation04/reset19`: native=False, steps=469; grasp failed to lift the predicted object
- `put_groceries_in_cupboard/variation05/reset15`: native=False, steps=835; planning worker: ValueError: release finger clearance
- `put_groceries_in_cupboard/variation06/reset10`: native=False, steps=336; planning worker: ValueError: release finger clearance
- `put_groceries_in_cupboard/variation06/reset11`: native=False, steps=136; planning worker: TimeoutError: geometry search budget exhausted; feasibility unverified
- `put_groceries_in_cupboard/variation06/reset12`: native=False, steps=250; planning worker: ValueError: release finger clearance
- `put_groceries_in_cupboard/variation06/reset13`: native=False, steps=249; planning worker: ValueError: release finger clearance
- `put_groceries_in_cupboard/variation06/reset14`: native=False, steps=211; planning worker: ValueError: release finger clearance
- `put_groceries_in_cupboard/variation06/reset15`: native=False, steps=140; planning worker: RuntimeError: no stable shelf placement: ['wrist_orientation: payload/structure collision: diningTable', 'insertion: geometry-only Panda path failed: Could not create path.', 'insertion: geometry-only Panda path failed: Could not create path.', 'insertion: geometry-only Panda path failed: Could not create path.']
- `put_groceries_in_cupboard/variation06/reset16`: native=True, steps=642; released object failed physical support/stability verification
- `put_groceries_in_cupboard/variation06/reset17`: native=False, steps=135; planning worker: TimeoutError: geometry search budget exhausted; feasibility unverified
- `put_groceries_in_cupboard/variation06/reset18`: native=False, steps=236; planning worker: ValueError: release finger clearance
- `put_groceries_in_cupboard/variation06/reset19`: native=False, steps=111; planning worker: TimeoutError: geometry search budget exhausted; feasibility unverified
- `put_groceries_in_cupboard/variation07/reset10`: native=False, steps=1000; executor horizon exhausted
- `put_groceries_in_cupboard/variation08/reset17`: native=False, steps=1000; executor horizon exhausted
### Run 2

Physical GPU: none (CPU software rendering). PID: `4159239`. Exit status: `1`.
Native success Wilson 95% interval: `[0.6784543514317231, 0.841568452531236]`.
Frozen source manifest: `/home/auszok/experiments/rlbench_followup_20261109/contact_repair_20261109/final_v4_source_manifest.json`.
Log: `/home/auszok/experiments/rlbench_followup_20261109/contact_repair_20261109/calibration_final_v4_run2.log`.

Command:

```bash
docker run --name rlbench-contact-final-v4-run2-20261109 --cpus=6 --memory=12g -v /home/auszok/experiments/rlbench_followup_20261109:/home/auszok/experiments/rlbench_followup_20261109 sha256:5fff7d4968af50c6278328c2086aa6747f9fbbf0523765ae9c389dee83b55a58 /home/auszok/experiments/rlbench_followup_20261109/contact_repair_20261109/source_final_v4/rlbench_campaign.py calibrate --manifest /home/auszok/experiments/rlbench_followup_20261109/frozen_v2/manifest.json --output /home/auszok/experiments/rlbench_followup_20261109/contact_repair_20261109/calibration_final_v4_run2
```

Failures:

- `put_rubbish_in_bin/variation00/reset18`: native=False, steps=221; released object failed physical support/stability verification
- `put_groceries_in_cupboard/variation01/reset11`: native=False, steps=240; geometry-only Panda path failed: Could not create path.
- `put_groceries_in_cupboard/variation01/reset16`: native=False, steps=220; planning worker: ValueError: payload/structure collision: cupboard
- `put_groceries_in_cupboard/variation01/reset19`: native=False, steps=243; planning worker: ValueError: payload/structure collision: cupboard
- `put_groceries_in_cupboard/variation02/reset12`: native=False, steps=331; planning worker: ValueError: payload/structure collision: cupboard
- `put_groceries_in_cupboard/variation02/reset15`: native=False, steps=262; planning worker: ValueError: payload/structure collision: cupboard
- `put_groceries_in_cupboard/variation03/reset10`: native=True, steps=457; released object failed physical support/stability verification
- `put_groceries_in_cupboard/variation03/reset11`: native=True, steps=475; released object failed physical support/stability verification
- `put_groceries_in_cupboard/variation03/reset12`: native=True, steps=452; released object failed physical support/stability verification
- `put_groceries_in_cupboard/variation03/reset13`: native=True, steps=435; released object failed physical support/stability verification
- `put_groceries_in_cupboard/variation03/reset14`: native=True, steps=519; released object failed physical support/stability verification
- `put_groceries_in_cupboard/variation03/reset15`: native=True, steps=373; released object failed physical support/stability verification
- `put_groceries_in_cupboard/variation03/reset16`: native=True, steps=466; released object failed physical support/stability verification
- `put_groceries_in_cupboard/variation03/reset17`: native=True, steps=368; released object failed physical support/stability verification
- `put_groceries_in_cupboard/variation03/reset18`: native=True, steps=376; released object failed physical support/stability verification
- `put_groceries_in_cupboard/variation03/reset19`: native=True, steps=496; released object failed physical support/stability verification
- `put_groceries_in_cupboard/variation04/reset11`: native=False, steps=498; grasp failed to lift the predicted object
- `put_groceries_in_cupboard/variation04/reset12`: native=False, steps=0; planning worker: TimeoutError: geometry search budget exhausted; feasibility unverified
- `put_groceries_in_cupboard/variation04/reset13`: native=False, steps=0; planning worker: TimeoutError: geometry search budget exhausted; feasibility unverified
- `put_groceries_in_cupboard/variation04/reset14`: native=False, steps=0; planning worker: TimeoutError: geometry search budget exhausted; feasibility unverified
- `put_groceries_in_cupboard/variation04/reset15`: native=False, steps=300; planning worker: ValueError: payload/structure collision: cupboard
- `put_groceries_in_cupboard/variation05/reset19`: native=False, steps=255; planning worker: ValueError: payload/structure collision: cupboard
- `put_groceries_in_cupboard/variation06/reset10`: native=False, steps=333; planning worker: ValueError: release finger clearance
- `put_groceries_in_cupboard/variation06/reset11`: native=False, steps=136; planning worker: TimeoutError: geometry search budget exhausted; feasibility unverified
- `put_groceries_in_cupboard/variation06/reset12`: native=False, steps=236; planning worker: ValueError: release finger clearance
- `put_groceries_in_cupboard/variation06/reset13`: native=False, steps=259; planning worker: ValueError: release finger clearance
- `put_groceries_in_cupboard/variation06/reset14`: native=False, steps=210; planning worker: ValueError: release finger clearance
- `put_groceries_in_cupboard/variation06/reset15`: native=False, steps=144; planning worker: RuntimeError: no stable shelf placement: ['insertion: geometry-only Panda path failed: Could not create path.', 'insertion: geometry-only Panda path failed: Could not create path.', 'insertion: geometry-only Panda path failed: Could not create path.', 'insertion: geometry-only Panda path failed: Could not create path.']
- `put_groceries_in_cupboard/variation06/reset16`: native=True, steps=516; released object failed physical support/stability verification
- `put_groceries_in_cupboard/variation06/reset17`: native=False, steps=135; planning worker: TimeoutError: geometry search budget exhausted; feasibility unverified
- `put_groceries_in_cupboard/variation06/reset18`: native=False, steps=235; planning worker: ValueError: release finger clearance
- `put_groceries_in_cupboard/variation06/reset19`: native=False, steps=113; planning worker: RuntimeError: no stable shelf placement: ['insertion: geometry-only Panda path failed: Could not create path.', 'insertion: geometry-only Panda path failed: Could not create path.', 'insertion: geometry-only Panda path failed: Could not create path.', 'insertion: geometry-only Panda path failed: Could not create path.']
- `put_groceries_in_cupboard/variation07/reset10`: native=True, steps=1000; executor horizon exhausted
- `put_groceries_in_cupboard/variation07/reset19`: native=True, steps=403; released object failed physical support/stability verification
- `put_groceries_in_cupboard/variation08/reset17`: native=False, steps=1000; executor horizon exhausted
- `put_groceries_in_cupboard/variation08/reset18`: native=False, steps=230; planning worker: ValueError: payload/structure collision: cupboard

## Preserved development evidence

Earlier candidates and aborted attempts remain archived. The v3 full runs used an overconservative global shelf-plane test; they are not combined with v4 or used as promotion evidence.

- v3 run 1: 65/100 native, 54/100 completed native success.
- v3 run 2: 65/100 native, 56/100 completed native success.

Reserved preflight diagnostics for the final movement policy: cracker reset10 completed with native success in 312 steps; Tuna reset19 completed with native success in 419 steps. These do not substitute for either full calibration.

No evaluation resets or model predictions were used to repair the executor. Native IK/contact behavior is stochastic; two runs do not establish universal manipulation reliability.
