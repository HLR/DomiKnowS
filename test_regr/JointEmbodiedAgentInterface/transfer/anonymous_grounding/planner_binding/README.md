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
