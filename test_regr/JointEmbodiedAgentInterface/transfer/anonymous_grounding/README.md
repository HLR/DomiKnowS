# Anonymous IDs and supervised visual grounding

This is a separately identified **combined, source-only experiment**, independent
of the semantic-prompt adapter and the unfinished matched-continuation campaign.
It starts from the full-metadata Stage 1 `joint_stage1_epoch_003.pt` snapshot and
continues for four epochs, seed 42, accumulated batch size 8. Historical
official-corpus exposure remains explicitly labeled an internal experiment.

## Inputs and supervision

- Preserve the frozen 3,600/450/450 VLABench identities. Only training and
  validation images are prepared. Retain canonical EAI training/validation and
  its existing symbolic input protocol. Anonymous numbered entities apply to
  the VLABench visual planner; no target dataset supplies training examples.
- Recognize complete numeric tags using an exact glyph atlas transcribed from
  **two source training images**. Full-word boundaries reject single digits
  exposed by occluded multi-digit tags. Conservative colored-occlusion recovery
  accepts a complete digit only when it cannot belong to any valid multi-digit
  ID. Duplicate or ambiguous bindings remain unknown. References do not choose
  bindings, visibility, crops, or objects.
- Start every reannotated image from the original clean RGB, eliminating both
  old tags and ID-coded instance colors. Retain the original photographed scene;
  no simulator reconstruction or missing-property fabrication is needed.
  Permute image tags, canonical table order, metadata pointers, output pointers,
  cached entity features, and role targets together.
  Mix in canonical numbering once per four epochs; two-object scenes use a
  balanced identity/swap schedule instead of repeating their only nonidentity
  permutation. Validation's second condition remains a fixed nonidentity map.
- Use frozen SigLIP features from 64- and 192-pixel label-anchor crops, at most
  two observed views per entity, plus normalized observed image coordinates.
  Load the exact visual-encoder weights from the starting checkpoint. These are
  anchor crops, not oracle object boxes or hidden simulator geometry.
- An index-free shared scorer with permutation-equivariant set attention predicts
  primary source and destination roles from instruction and observed features.
  Its auxiliary multi-positive softmax objective has weight 1.0. Empty roles
  train a null option; unobservable positive roles are not relabeled as negatives.
- Preserve corrected masked CE plus allowed-mass weight 0.5. Mask loss terms for
  pointer targets whose image-to-ID binding is unknown; syntax supervision and
  EAI remain. Record observability and skipped-pointer counts, not just losses.
  Predicted roles enter the neural prompt; constraints bind the original
  observation context. Gold assignments never replace predictions.

Planner LoRA/label heads and the new grounding head train with AdamW and domain
minibatch PCGrad. Planner learning rate is 2e-5; the new head uses 1e-3. The
physical controller is fixed; there is no controller warm-up or Stage 2 RL.
Checkpoint selection uses source validation only. Shared-backbone and grounding
gradients, parameter changes, input hashes, and controller immutability are
checked. Validation grounding accuracy includes unobservable positive targets as
failures; do not interpret partial teacher-forced loss as complete-plan success.
Validation uses canonical numbering and one fixed permutation of each of the
same 450 scenes, with the joint objective averaged across these two conditions.
The 900 conditions are not 900 independent scenes.

## Run and artifacts

```powershell
python -m test_regr.JointEmbodiedAgentInterface.transfer.anonymous_grounding.deploy `
  --host gpu2 --output /home/auszok/experiments/NEW_UNIQUE_EXPERIMENT
```

The launcher freezes source/configuration hashes, prepares all original-image
observations on CPU, and then selects a completely unused GPU. It preserves
other jobs and services. All four epoch snapshots and first-batch/100-batch
progress snapshots are retained with their grounding-head sidecars. Compatible
within-epoch resume is exposed by `anonymous_grounding.train --resume PATH` with
the frozen configuration, source manifest, feature directory and output.

The Docker environment binds `VLABENCH_OFFICIAL_EVALUATION_ROOT` to the canonical
corpus independently of the frozen source checkout, and `HF_HOME` to the mounted
local model cache. This preserves the default clean-data guard; the historical
corpus still requires the explicit internal-experiment permission and matching
checkpoint lineage. A failed startup without snapshots can reuse completed
observation and feature caches in a fresh directory:

```powershell
python -m test_regr.JointEmbodiedAgentInterface.transfer.anonymous_grounding.restart_host `
  --previous /home/auszok/experiments/FAILED_STARTUP `
  --output /home/auszok/experiments/NEW_UNIQUE_EXPERIMENT
```

The restart freezes new source/configuration hashes, preserves prior artifacts,
and retains the original initialization and four-epoch source-only recipe.
It may stop the experiment's owned startup container only while no training
snapshots exist. Reference-path and complete allowed-token-set permutation
audits use eight CPU processes before CUDA initialization. Per-episode proofs
are saved under `checkpoints/audits/`, with phase, completed count, cached count,
and estimated remaining time in `checkpoints/audits/status.json`. Compatible
proofs can resume without repeating completed checks; changed inputs, code,
checkpoint, or split provenance invalidate their reuse. The checks are required
before optimizer updates, including all four training permutations and the fixed
validation permutation.

Use `anonymous_grounding.agent.load_snapshot` and `generate` for inference, with
observation-derived features from the checkpoint's frozen visual encoder.
Standard checkpoint loading rejects this extension to prevent accidentally
evaluating the planner without its head or predicted-role prompt. Sidecar,
encoder, source, checkpoint, dataset and resume fingerprints must match.

No LIBERO/RLBench test scoring or target adaptation launches automatically. Four
source epochs and their fixed validation are the scope of this experiment.

## Explicit selected-snapshot evaluation

After training, `deploy_evaluation` separately freezes the validation-selected
checkpoint and head, evaluation-only source, 450 VLABench test identities,
88 EAI validation examples, and the archived 100 LIBERO evaluation episodes.
It retains the training runtime's `main.py` and checkpoint loader, preventing
newer local prompt interfaces from mixing with older frozen dependencies.
No retraining or target-domain adaptation runs.

```powershell
python -m test_regr.JointEmbodiedAgentInterface.transfer.anonymous_grounding.deploy_evaluation `
  --training-run /home/auszok/experiments/COMPLETED_TRAINING_RUN `
  --output /home/auszok/experiments/NEW_EVALUATION_DIRECTORY
```

VLABench uses original RGB and canonical reannotation with the training-only
glyph atlas. LIBERO's original raw and numbered planner images stay unchanged;
unique complete public numeric stamps supply crop anchors. Missing and ambiguous
bindings remain unknown. The external single camera is scaled to 480 pixels for
the frozen 64/192-pixel crop encoder, populating only camera slot zero. Names,
object types, BDDL goals, reference plans and success sensors do not supply
grounding features. The predicted head roles alone condition the planner.

The evaluator binds constraints to the original canonical observation context
and verifies the actual neural inputs. It compares no DFA and shared verified
rules on both datasets, plus full metadata on VLABench only. Corrected density
references remain scorer-only. Frozen rendered prompts, images, token decisions,
predictions, role predictions, rule traces and original/corrected scores are
saved. Constraint conditions must have identical rendered-input hashes.
LIBERO execution uses the unchanged Section 12 executor with its 96/100
calibration; physical success remains planner transfer through a fixed executor.
VLABench input reannotation and the new role prompt distinguish this combined
experiment from the earlier baseline; scores alone cannot attribute a gain to
one component. New Google report tables must label pending results explicitly.

## Object-transfer diagnostics

`object_fixes/` adds an explicit raw-instruction grounding query, a declared
pick/place executor contract over the native shared-rule decoder, and paired
SigLIP source/destination phrase retrieval over RGB object regions. Its six
frozen conditions reproduce the selected-snapshot baseline before testing the
three changes and two explicit combinations on all 100 retained LIBERO episodes.
See `object_fixes/README.md` for interfaces, leakage boundaries, tests and launch
commands. This is a diagnostic regression without training updates; it records
competing-object scores but does not claim hard-negative fine-tuning.
