# Per-operation binding: matched planner-CE training

This implements item 4 of the Object-transfer repair proposal. Both independent
arms start from the source-validation-selected anonymous-ID grounding continuation
epoch-2 checkpoint, including its required grounding sidecar. Each receives four
source-only VLABench + EAI epochs, seed 42, batch 8, AdamW/PCGrad, planner LR 2e-5,
grounding LR 1e-3, 4-bit Qwen loading, the same ID permutations and fixed controller.
The historical source split is 3,600/450/450; this retains the initialization's
official-evaluation-corpus exposure. No target-domain scenes enter training.

Both arms share the new operation-binding mechanism; **only the VLABench graph-label
CE denominator differs**. `masked` uses corrected DFA-masked CE; `full` uses the
complete planner label vocabulary, including illegal alternatives. Both retain
the same allowed-mass penalty (0.5) and operation-grounding auxiliary loss (1.0).
EAI's objective is matched. CE means the planner's graph labels, not Qwen's LM
token vocabulary. Validation losses are compared only within each arm, never
across the two different denominators. An unchanged initialization is evaluated
separately, so comparisons with it include the whole continuation recipe.

The grounding head scores each source/destination pointer at the causal hidden
boundary following its emitted argument label. Teacher-forced row t can see
only labels before t. In inference, the hook sees only the generated prefix.
The target pointer supervises the current operation's auxiliary loss; target
roles are never copied into generation inputs or substituted for predictions.
Unknown/unobserved pointer targets are skipped while syntax supervision remains.
This replaces the episode-level union of all source/destination targets.

Predicted primary hints use frozen paired SigLIP instruction-phrase retrieval.
Source priors use the existing RGB feature cache (two crop scales, averaged
observed views); all feature rows and hints follow the same ID permutation.
Unparseable phrases produce unknown hints. No reference plan supplies phrases.
The trainable operation head starts from the selected head's exact weights and
adds a finite, learned pointer-logit residual, bounded by gain 4. Its gate starts
at zero, preserving initial planner logits; it does not hard-force an argmax or
remove candidate labels. Shared LoRA, label heads, operation head and gate train;
the image encoder and physical controller remain frozen.

`train.py` explicitly supplies these components to the retained audited trainer
in its isolated process. Its source preparation, batching, PCGrad, reference
corrections and ID transforms are reused. Model-independent audit proofs are
reused only with identical native compiler/auditor dependencies; every current
per-path input hash is still verified. GPU models are loaded only after CPU
preflight. A second availability check rejects a GPU claimed during setup.

Initialization and all four epoch snapshots are retained in each arm, with head
sidecars and full lineage. Selection uses canonical + fixed-permutation source
validation and canonical EAI validation, never external outcomes. Checkpoints
require `planner_binding.agent.load_binding`; the ordinary grounding evaluator
must not be used. The paired image/text representation stays frozen: this is
planner/operation-head training, not hard-negative SigLIP fine-tuning.

Before any optimizer update, freeze 100 fresh LIBERO episodes using official
states 40--44 for the same 20 tasks. Verify availability and scene/state geometry
disjointness from states 0--39. Existing test 0--4, calibration 5--9 and separate
target-training/validation 10--39 remain reserved. Exact original calibration
records and native executor/reset hashes qualify reuse; the derived certificate
records this explicitly and does not claim a new calibration run. Stop rather
than filter scenes on a split/correctness failure.

After training, evaluate the shared initialization and each arm's source-val
best on the frozen fresh 100 episodes under no DFA and the declared executor
mask. Match all rendered inputs and priors; open hidden references after
generation. Perform fresh native executions with the retained fixed executor,
1,000-step horizon, and evaluator-only native goals. EAI goal retention is
evaluated separately. Report native success, completed successes, Object/Spatial
counts, abstract goals, validity, paired task-cluster intervals and errors. The
inspected 23/57 pilot is not a matched numerical baseline for the new states.
Stop after these two arms and their fixed evaluations; do not tune on outcomes.

```text
python -m pytest test_regr/JointEmbodiedAgentInterface/transfer/anonymous_grounding/planner_binding -q
python -m test_regr.JointEmbodiedAgentInterface.transfer.anonymous_grounding.planner_binding.deploy --root /home/auszok/experiments/joint_planner_binding_ce_20261113
```

The deployer targets the separately frozen scene-preparation root, preserves
prior failed setup artifacts, and launches an isolated source manifest. Host
jobs record exact commands, GPU UUIDs, PIDs, logs and exit codes. Resume refuses
changed settings, observations, tokenizer priors, sidecars or source hashes.

## Pre-update resource-contention recovery

`recover.py` adopts the existing masked container and queues a separate full-CE
retry only when the recorded failure is an OOM during backward before its first
optimizer update. It requires a completely unused GPU, keeps the frozen source,
full configuration and all failed artifacts, and writes fresh per-arm status.
The recovery's read/job aliases let the original campaign evaluate the successful
retry without moving or replacing the failed files. A failure after training
updates requires explicit checkpoint recovery instead of restarting silently.
## All-epoch diagnostic evaluation

`epoch_campaign.py --root <evaluation-root>` evaluates the four saved masked
epochs under no DFA, shared verified rules, and VLABench-only full metadata.
It leaves training, the original `best.json`, and the matched-arm campaign
unchanged. Each no-DFA job also evaluates the canonical 88 EAI validation cases.
LIBERO uses the campaign's fresh 100 episodes at states 40–44 and the retained
calibrated physical executor; it does not use the older inspected states 0–4.
The extra executor-capability overlay is not part of the shared-rule rows.

The evaluation root freezes checkpoint and head hashes, its isolated source,
input manifests, reused frozen-encoder features, and public instruction priors.
Only completely unused GPUs are allocated. Commands, GPU UUIDs, worker PIDs,
logs, and exit status are recorded in `jobs/`; waiting jobs are in `queues/`.
`results.json`, `table_results.csv`, and `RESULTS.md` are produced after coverage,
prediction/execution, reference, and cross-condition prompt checks pass. These
external learning curves are diagnostic and must not select an epoch or tune
training against the external test episodes.

The targeted `epoch_recovery.py` helper preserves the archived epoch-3 failure,
replays EAI with exact-search computational caps of 100,000 and at most
1,000,000, and fails if feasibility remains unknown. It never relaxes a hard
constraint. Prefix checks compare every generated label and semantic result,
normalizing only the display order of serialized fact sets. Separate retry
outputs and additive source manifests preserve the original provenance.

## Matched initialization and executor-policy verification

`epoch_evaluation` accepts either the existing `--epoch 1` interface or
`--snapshot initialization|epoch-1|epoch-2|epoch-3|epoch-4`. Initialization must
be the saved pre-continuation checkpoint, with `initialization_before_updates`
metadata and its own hashed grounding sidecar. Its existing source-training
lineage is retained; it is not native Base Qwen. `--datasets libero` and
`--skip-eai` support isolated external inference. The `executor` policy is
LIBERO-only and intersects the unchanged shared whitelist with the existing
pick/place capability contract. It never substitutes a predicted or gold role
assignment for the canonical object pointers.

```text
python -m test_regr.JointEmbodiedAgentInterface.transfer.anonymous_grounding.planner_binding.epoch_evaluation --epoch-config <config.json> --snapshot initialization --checkpoint <joint_initialization.pt> --constraint-policy none --experiment-output <output> <model-options>
python -m test_regr.JointEmbodiedAgentInterface.transfer.anonymous_grounding.planner_binding.epoch_evaluation --epoch-config <config.json> --epoch 1 --checkpoint <joint_stage1_epoch_000.pt> --constraint-policy executor --datasets libero --skip-eai --experiment-output <output> <model-options>
python -m test_regr.JointEmbodiedAgentInterface.transfer.anonymous_grounding.planner_binding.verification_control --root <verification-root>
python -m test_regr.JointEmbodiedAgentInterface.transfer.anonymous_grounding.planner_binding.verification_campaign --root <verification-root>
```

The verification root freezes its preregistration before scoring: Epoch 1 is
primary, states 30–34 are correctness validation, and states 45–49 are new-scene
confirmation. Existing states 40–44 form the matched descriptive control.
`verification_freeze` checks state availability, all reserved-scene geometry,
settled scenes and decoded image hashes. Native manifests remain immutable;
metadata-only qualification certificates are written to host-owned `cohorts/`.
The exact calibration episodes, qualified executor and bridge must match.

`verification_control` requires archived Epoch-1 predictions, scores, input
hashes and EAI semantic outcomes to reproduce before interpreting controls.
Its host-owned `control_reproduction/` certificates avoid modifying model
container outputs. `verification_campaign` runs validation before admitting
confirmation, then independently executes all six successful planning jobs.
Unknown semantic feasibility is an explicit error; larger exact-search budgets
never relax constraints. Valid wrong-object alternatives are tested explicitly.

Per-token observations distinguish rejected alternatives from raw-top
interventions and record allowed mass and logit margins. Counterfactual traces
stop at their first rejected prefix; actual decoding supplies post-divergence
predictions. Confirmation reporting retains all 100 episodes per condition and
uses 10,000 paired task-cluster bootstrap samples, seed 42, keeping five scenes
per sampled task. The two primary contrasts use 97.5% intervals; secondary
comparisons use descriptive 95% intervals. No test-based checkpoint selection,
weight tuning, retraining or executor repair is launched by these helpers.

`verification_finalize --root <verification-root>` waits for fixed scoring and
runs CPU publication checks in `verification_diagnostics`. The publication gate
requires identical rendered inputs across checkpoints, records factual per-token
rule attribution and preserves native success when abstract scoring disagrees.
Those disagreements are archived separately. Google report cells remain pending
until both the scoring results and this publication gate are complete.
