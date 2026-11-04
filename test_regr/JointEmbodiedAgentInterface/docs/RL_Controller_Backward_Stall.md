# Controller PPO backward stall and restart requirements

On 4 November 2026, the shared-backbone RL run on gpu2 (physical GPU 3,
host PID 1290080) stopped reporting progress during round 5 of epoch 3.
Repeated Python/native stack samples locate it in controller PPO `loss.backward()`:
the autograd worker spins in the CUDA driver inside cuBLAS SGEMM. The planner
update had already completed. GPU utilization was zero; a separate small CUDA
backward test on the same GPU succeeded. This establishes a process-local stall,
but does not establish the exact driver-level trigger.

Live frame inspection found PPO epoch index 0 and transition index 290: the old
code had retained 291 independent controller graphs, stacked all losses, then
launched one backward pass. No controller update completed for that round.
The original run remains unchanged until explicitly stopped and restarted.

## Implemented mitigation

`VLABenchAgentInterface/simulation/program.py` now backpropagates each transition
immediately with full-batch normalization and accumulates its parameter gradients.
Backward uses the calling thread through a scoped
`torch.autograd.set_multithreading_enabled(False)` context. Only detached scalar
losses remain in the batch list. This bounds the retained transition graphs to one
and avoids the autograd-worker CUDA path seen in the stalled process.

Parameters remain fixed throughout a PPO pass. There is still one optimizer step
per accepted epoch, with the existing normalization, clipping, KL/rollback and
non-finite checks. A rejected batch clears accumulated gradients. The optional
supervised anchor is differentiated once, after the aggregate checks. Progress
messages distinguish forward, backward, completed transitions and optimizer steps;
an explicit update-complete message ends the watchdog interval.

This reduces a concrete scheduling/memory risk. It is not proof that every possible
CUDA hang is eliminated, nor a change to rewards or training-success gates.

## Safe restart

1. Use a fresh process with this fix, preserving the historical run's compatible
   model, vocabulary, constraint declarations and original launch settings. Editing
   files cannot repair a CUDA call already running in the old interpreter.
2. Resume `joint_stage2_progress.pt`: epoch index 2, `next_round=4` (four rounds
   completed in epoch 3). It preserves the last committed optimizer/model/RNG state.
   The interrupted fifth round is replayed. Do not restart Stage 1 or change labels.
3. Wrap the training command with the independent parent watchdog below, using a
   new log path. The default 300-second deadline applies only between controller PPO
   progress messages; long simulator rollouts do not arm it. On a stall it terminates
   only its own training child group, records exit 124 and preserves checkpoints.
   It does not automatically retry or skip a failed round.

```bash
python -m test_regr.JointEmbodiedAgentInterface.utils.controller_watchdog \
  --log /path/to/new_run/train.log --stall-seconds 300 -- \
  python -u -m test_regr.JointEmbodiedAgentInterface.main train-agent \
  --resume /path/to/joint_stage2_progress.pt <the preserved training options>
```

The inner command is a template; reuse the archived `run.sh` arguments and working
directory/environment. Select a currently free physical GPU on gpu2.ihmc.us.
The watchdog writes `<log>.watchdog.json` with the command, PID, exit codes and
timeout reason, and refuses to overwrite an existing log.

## Validation evidence

CPU regression cases verify the analytic full-batch gradient for 17 and 291
transitions, at most one live replay graph, autograd-setting restoration, partial
gradient cleanup, and the existing trust-region rollback behavior. Parent-watchdog
tests exercise a stalled child and normal work outside a controller update.

An isolated GPU-4 test strictly loaded the real controller from the recovery
checkpoint and ran 291 synthetic two-frame/four-view 480p transitions through four
PPO passes, including full SigLIP forward and controller backward: 60.09 seconds,
1,087,057,920 peak allocated CUDA bytes, finite loss and a measured parameter update.
This tests the update path, not the interrupted simulator observations or execution
success. Artifacts are under the original experiment's
`investigation_20261104/fix_validation/` directory.
