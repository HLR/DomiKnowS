# Retention failure versus diagnostic continuation

The anonymous CE-only RL run stopped after epoch 1 because EAI validation success
fell from 52/88 to 51/88. Its zero-regression retention threshold rejected that
change. VLABench retained task signal (66% positive-return rollouts, 2% success,
4% IK truncation); the old abort message incorrectly blamed VLABench.

The default `--stage2-retention-policy abort` is unchanged. Failed epochs are saved
for diagnosis and remain ineligible for best-checkpoint selection. Logs now name
the actual EAI-success, EAI-recall, VLABench-signal or VLABench-regression failure.

For an explicitly diagnostic experiment that must complete its epoch budget, use
`--stage2-retention-policy continue`. This continues the current weights and
optimizers after a failed epoch, without rollback or threshold changes. Each epoch
is still evaluated, and a failed epoch cannot become `joint_stage2_best.pt`.
Subsequent round-progress checkpoints inherit the last evaluation's failed
retention status until a new evaluation passes; they cannot bypass the rejection
by being loaded as normal evaluation checkpoints.

Training resume with this explicit policy can load a failed-retention checkpoint
only when both optimizer objects are supplied. Ordinary evaluation and default
training resume continue to reject it. All schema, vocabulary, model, input,
parameter-ownership and provenance checks remain active.

Resume `joint_stage2_epoch_000.pt` with a total budget of four epochs to run epochs
2–4 without repeating epoch 1. Preserve the seed, data, anonymous no-DFA prompt,
shared-parameter policy and optimizer settings. Record the continuation policy and
the old checkpoint hash in the new run's provenance. This is a labeled internal
diagnostic experiment, not acceptance of a failed epoch or a clean benchmark claim.
