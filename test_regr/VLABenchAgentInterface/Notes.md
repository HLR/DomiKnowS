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

The smoke/audit override `--rl-preflight-min-successful-tasks 999` records diagnostics while preventing PPO. Do not treat an audit with that override as evidence that reinforcement learning is ready.
