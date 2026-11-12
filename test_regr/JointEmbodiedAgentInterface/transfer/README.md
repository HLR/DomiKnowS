# Sequential planner transfer

Each method is an independent experiment from the frozen full-metadata epoch-4
checkpoint. The six-method order is semantic prompt, ID augmentation, explicit
grounding, shared-format training, target adaptation, and grounded replanning.
Existing native executors and test/calibration resets remain unchanged.

## Current rollout

The shared integrity checks and first method are implemented. Later methods must
not be launched until the preceding method has complete planning and native
evaluation, a validated report, and a qualification record. A score improvement
is not required for technical qualification. Missing qualification is an error,
not permission to skip an experiment.

`prepare` freezes configuration, source, checkpoint, datasets, archived baseline
predictions, and retained calibration fingerprints. `evaluate` reproduces the
baseline exactly before accepting semantic-prompt results. `campaign` launches
isolated owned containers sequentially and records commands, physical GPUs,
UUIDs, PIDs, logs and exit codes. `report` requires every episode and checks that
native execution consumed the generated predictions. No partial success table
is published. These modules run with `python -m
test_regr.JointEmbodiedAgentInterface.transfer.<module>` in the project runtime.
The host launcher also supports direct execution without model dependencies.

```powershell
python -m test_regr.JointEmbodiedAgentInterface.transfer.deploy `
  --host gpu2 --output /home/auszok/experiments/NEW_UNIQUE_CAMPAIGN
```

The deployment preserves prior attempts and overlays a frozen copy of the
transfer code onto the archived evaluation source. It explicitly selects the
retained RLBench v33 executor and matching reset-verifier revalidation, not the
unqualified contact-aware replacement in the current checkout. Unused physical
GPUs are preferred; no other process is stopped or evicted. The optional
`--idle-service-gpu 5` allocation additionally permits the inspected Celery
service's idle GPU capacity only with empty active/reserved/scheduled task lists,
no unrelated GPU processes, under 4 GB resident memory, and at least 80 GB
available. The service stays running.

## Prompt method

The semantic adapter adds generic pick/place definitions and a task-family
description derived only from the instruction's verbs. It does not resolve gold
entity IDs, alias benchmark task names, or activate task-specific constraints.
No-DFA and shared-rule conditions receive identical rendered prompts and images.
Baseline and semantic prompts are intentionally different and separately hashed.

The evaluation reference remains private and is opened only after generation.
Raw and numbered images, canonical pointers, scene relations, positions, and
dynamic-role declarations are preserved. The metadata-assisted protocol and the
source checkpoint's historical evaluation-corpus exposure must remain explicit.

## Target-data allocation

`data.py` defines LIBERO states 10–29 for training and 30–39 for validation, and
RLBench resets 20–39 and 40–49 respectively. Blueprints cannot pass the frozen
scene integrity gate. Actual state and input fingerprints must be checked against
both reserved test and executor-calibration scenes. Cross-split duplicates cause
a stop, never selective replacement of episodes. All 20 LIBERO tasks have 50
official initialization states, so the proposed index ranges are available.

## Validation

```powershell
python -m pytest test_regr/JointEmbodiedAgentInterface/transfer/test_protocol.py -q
```

The tests cover prompt/reference separation, pointer preservation, changed-image
rejection, resume mismatches, exact archived-baseline checks, sequential rollout,
training/validation allocation, scene-overlap rejection, and CLI compatibility.

## Entity-ID sensitivity phase

`launch_bindings` freezes an independent method-2 source and verifies the original
numeric image markers for all 100 episodes per dataset. Restored segmentation
centers can differ from stored annotation centers. Recovery accepts exact local
stamps or a unique complete numeric stamp in the public observation; the entire
original numbered image must still reconstruct pixel for pixel. Ambiguity,
corruption, or incomplete coverage stops the phase without filtering episodes.

After validation, `launch_sensitivity --root CAMPAIGN` freezes a new model source,
waits for both complete binding gates, and prepares five deterministic ID
permutations. It runs the unchanged checkpoint under `none` and `shared-verified`
rules. Object tables, metadata bindings, explicit ID mentions, image markers and
predicted pointers are transformed together; scoring uses inverse-mapped plans
and the original frozen references. Both conditions have identical inputs within
each trial. Archived baseline reproduction, model hashes and original manifests
must match before interpreting the diagnostic. The five trials reuse 100 physical
episodes per dataset and are not 500 independent test scenes.

This launcher uses completely unused GPUs only and preserves resident services.
It stops after the diagnostic. The matched four-epoch augmented and unaugmented
continuations require separate source-training image verification first; neither
training arm is silently replaced by a prompt-only augmentation.

## Matched continuation implementation and current gate

`launch_continuation`, `continuation_campaign`, `source_views`, and `train` provide
the independent four-epoch control and augmentation arms. They retain the fixed
epoch-4 initialization, source split and in-memory reference correction, use
batch-8 domain means with the existing PCGrad/AdamW recipe, and freeze the physical
controller. Shared-gradient and parameter-change checks guard the first update;
checkpoint selection uses the full canonical source validation objective only.

The common regenerated source-view path is **not qualified**. Native rendering
successfully restores explicitly named backgrounds (whose serialization omits
`use_default_scene`) and uses zero warm-up steps to avoid automatic reset loops
when saved scenes already satisfy their native goal. However, archived mirror
configurations omit required reflectance values and cannot be faithfully
regenerated. The campaign stops before training, preserves every failed attempt,
and does not fill these properties from references or remove affected episodes.
Use the existing source images with a verified complete tag-relabeling bridge, or
recover authoritative missing simulator properties, before enabling this path.
