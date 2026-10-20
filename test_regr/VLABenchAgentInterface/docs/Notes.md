# VLABench implementation notes

`world_graph.py` is the authoritative domain definition: skills, roles, legal task patterns, plan validation, and task contracts. The adapter currently covers ten primitive tasks. Audit those before composite tasks.

Stage 1 trains the graph-token planner and controller by supervised learning. Stage 2 uses simulator return-to-go for planner REINFORCE and PPO/GAE for the controller, with supervised anchors. Reference-plan similarity is diagnostic, not the Stage 2 reward.

## Rollout interpretation

Task success comes from the simulator task predicate. Reaching the step limit or remaining executable is not success. Read per-task `success_rate`, `positive_return_rate`, `return`, `termination_reason`, and `ik_truncation_rate` together. Distance and intention are shaping signals.

The controller predicts local end-effector deltas in robot coordinates. The adapter converts them to world-frame IK targets, limits translation to 2 cm and rotation to 0.10 radians, and retries failed IK targets at smaller scales. The gripper becomes two binary finger commands.

`add_condiment` follows the official `pick -> lift -> move above container -> pour` sequence. The live adapter aligns the expert grasp orientation, keeps fingers open until the grasp point, closes gradually while holding the captured arm joints, and checks that the condiment rises with the arm. Contact without object lift ends as `grasp_lost`; the controller must not pour with an empty gripper.

## Camera and replay

The default camera report records dataset keys, live names and indices, calibration, RGB shape/range, and `mapping_status`. The default alias mapping is configured but unverified. A real check needs:

1. A held-out replay manifest with task, episode, offset, and scene metadata.
2. A JSON map covering every dataset camera key.
3. A `module:function` restorer that reconstructs scene, robot, objects, and task state after reset.

Append these options to the diagnostic command when available:

```bash
--replay-manifest /path/to/heldout_replay.json \
--camera-map /path/to/camera_map.json \
--replay-restore recorded_scene:restore \
--replay-steps 32 --execute-horizon 4
```

The replay reports `paired_frame_verified: true` only when all camera slots and EE state pass comparison. A camera-name match alone does not prove parity.

## Reward and RL gate

Controller shaping is `0.25 * delta(progress) + 0.10 * delta(intention)`. The final score is clipped to:

```text
0.60 * success + 0.25 * final_progress + 0.10 * final_intention
+ 0.05 * success * (1 - steps / max_steps)
```

The smoke/audit override `--rl-preflight-min-successful-tasks 999` records diagnostics while preventing PPO. The simulator preflight prints a Markdown table with `Task`, `Success`, `Progress`, and `IK` columns; progress is the average authoritative final task-progress signal, falling back to normalized distance progress when the upstream signal remains zero. Do not treat an audit with that override as evidence that reinforcement learning is ready.


## September 17 rollout investigation

The September 15 Joint run's rejected evaluation contains physics failures for all
three `insert_flower` episodes, one `select_fruit` episode, and one `select_toy`
episode. These were previously returned with `termination_reason=unknown` and no
diagnostics. Two failed drink episodes never reached the grasp point (minimum
distance roughly 0.49 and 0.52 m). The IK-truncated toy episode already had its
target over 2 m away at the first observation. These observations do not establish
that a successful grasp was subsequently lost.

The adapter now isolates upstream mutable task/robot configuration during each
construction and seeds Composer's independent RNG from the evaluation-seeded
NumPy stream. `simulator_seed` in environment kwargs can override that seed.
Old fixed-seed reports did not control this simulator RNG; re-evaluate both the
baseline and candidate before interpreting their difference as an RL effect.
A repeated GPU4 toy reset after this change produced exactly identical positions.

Placement searches now try the closest predicate-valid point to the official
waypoint first, rather than starting 80 cm below it. Physics exceptions retain
the last valid diagnostics and are labeled `physics_failure`; missing distance
progress no longer crashes aggregate reporting.

Camera alias selection is still configuration, not paired-frame verification.
The existing held-out replay checks require the original scene restorer and
paired dataset/live images. Do not change `unverified` to `verified` based solely
on matching camera names, and do not interpret a reset reproducibility check as
camera parity or end-to-end task success.

## September 18 execution-completeness investigation

`execution_complete_rate` counts episodes that stay valid and are not IK
truncated. The GPU6 joint run (`full_gpu6_20260918_014559`) lost it on
`insert_flower`, `select_toy`, `select_fruit`, `add_condiment`, and
`select_drink`. Re-running the run's controller checkpoint with a fixed task
pattern instead of the Qwen planner reproduced every failure family on seeded
scenes; the causes were adapter bugs, not simulator randomness:

- `_quat_multiply` had a sign error in its y component. Every attachment
  re-teleport composes `ee * (ee^-1 * object)`; the product was not even unit
  norm, so the "identity" teleport at the latch step rotated the carried object
  by tens of degrees into the closed fingers or the table, and MuJoCo raised
  `BADQACC` a few steps later. This was the physics failure behind all four
  attachment tasks.
- Attachment offsets were latched while the fingers were still closing. The
  fingers then closed into the teleported object and kicked it away every
  step. Offsets are now latched only once both finger joints stop moving, and a
  deferred condiment latch completes during the pour sequence.
- A carried object that had slipped out of the pads free-fell for a whole
  control step and was slammed back by the next teleport. Teleported entities
  now have their free-joint velocity reset and their weight cancelled with
  per-body applied forces (`xfrc_applied`; runtime `body_gravcomp` is ignored
  because the compiled `ngravcomp` is zero) until `place`/`insert` release
  them. With the object at rest between teleports the thin flower stem also
  stays physically inside the finger pads.
- `insert_flower` used the policy's grasp orientation. Resting on the flower
  head, the wrist tilted until IK failed and the flower was pushed off the
  table (IK truncation in every fixed-seed evaluation). The pick now uses the
  official top-down grasp with the gripper x-axis along the stem (grasp site to
  the `weight` geom), and the insertion follows the expert lift/rotate/move/
  lower sequence with the attachment offset stored in the gripper frame so the
  stem hangs below the fingertips. The stem droops a few degrees in the asset,
  so the midpoint between the flower origin and the stem end is centred over
  the vase axis, and the descent backs off whenever the carried flower touches
  the vase instead of pushing the rigidly attached flower through the rim. The
  previous 55 cm descent target and the snap-into-vase from 20 cm away are
  gone.
- The grasp latch now waits for the wrist to reach the commanded grasp
  orientation (closing mid-rotation pinched beside the book spine), and the
  thin-target picks (`select_book`, `select_drink`, `insert_flower`) keep
  approaching until the fingertips are within 1.2 cm of the keypoint or the
  approach stops making progress.
- Grasp/insertion poses with `pitch=±pi/2` (`select_book`, `select_drink`,
  flower insertion) are stepped by quaternion slerp and written directly into
  the bounded action; component-wise Euler clipping at that singularity
  produced erratic intermediate poses, IK rejections, and a can pushed 8 cm
  deeper into the fridge.
- A `PhysicsError` raised while composer settles a freshly built scene now
  retries construction like a failed constructor instead of returning an
  invalid zero-step episode.

Seeded re-runs of the run's controller checkpoint (8 `insert_flower`, 6
`select_toy`, 6 `select_fruit`, 8 `select_drink`, 3 `select_book` seeds plus
3 seeds of every other task, 46 episodes) complete every episode after these
changes: no physics failures and no IK truncation, where the same seeds
previously lost 2/8 `insert_flower`, 4/6 `select_toy` and 1/8 `select_drink`
episodes. Success on those seeds moved from 2/8 to 8/8 (`insert_flower`),
1/6 to 3/6 (`select_toy`), 3/8 to 5/8 (`select_drink`) and 1/3 to 3/3
(`select_book`); the other tasks stay at their previous success. Remaining
limit: VLABench's Franka has no joint limits, so the incremental IK can still
wind the wrist over a long rollout; a few `select_toy`, `select_drink` and
`add_condiment` scenes therefore end at the step limit without success.

## September 19 follow-up: remaining limits

The remaining step-limit episodes of the seeded re-runs were traced one by
one. Several more adapter defects and one upstream VLABench bug surfaced; the
rest are scene limits that the official expert cannot solve either.

- `ee_action_to_env_action` now solves the Franka IK with a damped
  least-squares solver that adds a bounded null-space step toward the default
  posture (`posture_regularized_qpos` in `environment.py`; joint ranges from
  the real robot, `joint7` treated as periodic). VLABench's Franka MJCF has
  its joint ranges commented out, so the plain `dm_control` IK wound the
  wrist a little further on every one of the hundreds of incremental solves
  until the arm folded into itself. The solver falls back to the upstream IK
  for other robots.
- Upstream `AboveCondition.is_met` writes the platform height into the live
  `data.xpos` view of the poured entity (`point_to_check[-1] = ...`). The
  simulation itself is unaffected, but every pose read until the next
  kinematics pass (the condiment grasp target, the latched attachment offset)
  saw a bottle origin at counter height: the grasp aimed too low and the
  attachment teleported the bottle into the counter (`add_condiment` physics
  failures). `_refresh_kinematics` runs `physics.forward()` after every
  `env.step`, inside `_task_signals`, and after `_task_success`, which also
  removes the one-substep lag of composer's legacy stepping.
- The upstream accessors `get_xpos`/`get_xqaut` return live views into
  MuJoCo's state. Anything stored across steps has to be copied
  (`_live_task_entity_bounds`, `_live_task_entity_position`); a stored view
  silently follows the object.
- `add_condiment` bottles are grasped halfway between their `bottom_site` and
  `top_site` instead of at the origin, which sits about 1 cm above the
  counter. Wide bottles (ketchup, 3.1 cm radius) cannot take the fingertips
  at their axis because the palm starts 2.2 cm behind the fingertips and
  shoved the bottle over, so the grasp point is pulled back to the finger
  reach and the pads close on the near half of the body; the close waits
  until the fingertips actually overlap the body. The final approach first
  lines up on the approach axis at a stand-off point just outside the bottle:
  leaving the prepare point up to 8 cm early turned the last 10 cm into a
  diagonal sweep that knocked the bottle over with a finger pad.
- `select_toy` figures whose head is wider than the 8 cm gripper opening
  (Jessie's hat, seed 300006) are pushed over by the top-down approach; the
  fingers then close on nothing and the attachment latched a toppled 24 cm
  figure lying 20 cm below the hand. Lowering it into a 17 cm gift box drove
  it through the walls, launched the box and either exploded the simulation
  or dragged the arm after the box into IK failures. Carried objects are now
  re-seated when the grasp keypoint is nowhere near the fingertips or when the
  carried object does not fit the container as held: they hang upright in
  their resting orientation, centred under the hand with their top 1 cm below
  the fingertips, switching only once the hand is high enough for the upright
  object to clear its former support and the container rim. The place
  approach also rises to the approach height before translating, checks the
  object's collision extent against the container's key-site box before
  descending (dropping from above if it still does not fit), and stops chasing
  a container that has moved more than 20 cm from where the place started.
- The hand pose held after a forced release is frozen instead of re-targeting
  the drifting live pose; the latter sank the hand into the container within a
  couple of hundred steps.

Scene limits left as they are (valid episodes ending at the step limit):

- `add_condiment` seed 400003 places the bottle 0.7 m in front of the base at
  counter height; the horizontal grasp pose is unreachable from every IK seed
  (the elbow is straight and the shoulder past its range), so the approach
  stalls 10 cm short with hundreds of IK rejections.
- `select_drink` seed 400002 puts the can 9 cm from the fridge's right side
  wall; the Franka palm is 21 cm wide along the finger axis (collision geom in
  the `end_effector` frame: y in [-0.106, 0.102]), so the horizontal grasp
  wedges the palm against the wall 7 cm short of the can. Offsetting the
  grasp cannot help because the fingers only open 1.4 cm wider than the can,
  and tilting the palm does not shrink its footprint enough.

Seeded re-runs of the run's controller checkpoint after these changes (the
same 46 episodes as above plus 3 `add_condiment` seeds of the `select_book`
batch and 3 more `add_condiment` seeds, 52 episodes): every episode completes
(no physics failure, no IK truncation) and 50/52 succeed. `select_toy` moved
from 3/6 to 6/6 and `add_condiment` from 1/3 to 3/3 on the original seeds
(8/9 with the six extra seeds); `insert_flower` 8/8,
`select_fruit` 6/6, `select_book` 3/3, `select_drink` 7/8 and the three-seed
tasks 3/3 are unchanged. The two remaining step-limit episodes are the scene
limits listed above.
