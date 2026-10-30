# Research implementation validation, 2026-10-29

This receipt covers implementation checks and expert-collection diagnostics.
It does not report learned-controller success or improved benchmark performance.
Historical evaluation scores have not been recomputed.

## Source and runtime

Validation used isolated GPU2 snapshots, preserving the primary checkout and its
existing training processes. The source base was
`6a30f42cd6281160a7907683e26757b27f56b0fb`; each snapshot includes a manifest of
the exact working source bytes. The final implementation manifest SHA-256 is
`6f476b0cc5b729b47b9b3c99d17b10fcc364cdf519af6ca9dc8bc7e8e8a97a27`.
This receipt was added after that snapshot.

GPU2 artifact root:
`/home/auszok/experiments/vlabench_research_20261029`.
The container is `vigorous_easley`, with Python at
`/opt/dominows/venv/bin/python`. Validation used physical H100 GPU 4,
`CUDA_VISIBLE_DEVICES=4`, EGL device 0, and one OpenMP/MKL thread.

## Validation scope

The focused suite includes research constraints, model-dependent LC gradients,
research integration and experiment budgets, native decoding, planner/controller
pipeline and checkpoint checks, articulated goals, poker rules, syntax/visibility,
bounded reachability, frozen baselines, dataset roles, and the joint EAI/VLABench
pipeline. All fifteen research formulas have removal/recompilation cases checking
the intended acceptance and loss/gradient change.

The real-physics tests cover isolated MuJoCo path checking, measured joint reads,
actual expert-control capture including terminal controls, and released-object
support on a flat surface. `VLABENCH_ARTICULATION_SIM_TESTS=1` also enables the
live VLABench laptop-joint adapter test. These checks do not execute a newly trained
controller.

Final validation PID: `599237`. Its log and completion record are
`verified_validation.log` and `verified_validation_run.json` under the artifact root.
Result: **388 passed**, no skips, in 197.71 seconds.
Existing warnings concern an escaped DFA docstring and positional `exactL` syntax.

## Measured data and experiment gates

`measured_step_corrected` (PID `591297`, physical GPU 4) produced one verified pick,
one place, and one press segment. It exposed a flat-table containment-reader error,
subsequently repaired using measured upward contact and released-object checks.

`articulation_expert_check` (PID `597271`, physical GPU 4) completed independent
study-table and cool-drink scenes. Neither met measured task success, and neither
contributes training segments. This is evidence that their collection experts are
not yet validated; a returned helper flag or symbolic plan cannot override it.
Its `run.json`, `collection.log`, `collection_gate.json`, scene configs, and control
archives remain on GPU2. Later collections also retain hashed `episode.json`
macro-outcome audits, including segments excluded from training.

`audited_study_expert` (PID `599397`, physical GPU 4) verified that audit contract
and all seven stored file hashes. Its pick/pull postconditions passed, but placement
failed the independent check despite a positive native helper flag. Subsequent
laptop grasp/opening and measured task success also failed. No segments from this
failed task entered the training manifest.

The pilot's twenty successful segments per skill and the full collection's five
hundred per skill have not been met. No matched planner/controller training,
three-seed RL comparison, or held-out trained-checkpoint evaluation was run.
Research observation corpora, validated experts, split coverage, and measured
demonstration counts must pass the documented gates before those runs start.

Use [RESEARCH_CONSTRAINTS.md](RESEARCH_CONSTRAINTS.md) for the data contracts,
main-command options, loss mapping, experiment runner, and ablation procedure.
