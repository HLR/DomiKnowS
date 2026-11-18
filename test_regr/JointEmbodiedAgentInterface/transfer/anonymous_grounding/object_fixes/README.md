# Object-transfer repairs

All planning restrictions originate in DomiKnowS syntax and are compiled.
`executor_graph.py` declares the pick/place grammar, termination and operation
limits, plus pointer, movability and self-placement LCs. `contract.py` supplies
primitive observations to the generic compiled monitor; it does not define a
separate procedural legality policy. Saved settings fingerprint both policy
declarations and compilers and reject resumes from the historical Python mask.

Historical executor-mask scores in `RESULTS.md` and Google report Sections 12.3
and 12.5 were generated with the earlier Python implementation. Preserve their
artifacts, but do not label them as results of this compiled replacement.

Validation on 17 November 2026 passed 76 checks and replayed 2,100/2,100 archived
token masks exactly across the validation Epoch-1 and confirmation initialization
and Epoch-1 executor trajectories (300 episodes). No greedy choice changed.
This is compiled-mask/logit replay, not fresh neural or simulator evaluation.
Its isolated CPU artifacts are in
`/home/auszok/experiments/joint_executor_lc_validation_20261117`.

Reproduce the read-only archive check with a fresh output directory:

```text
python -m test_regr.JointEmbodiedAgentInterface.transfer.anonymous_grounding.object_fixes.replay_compiled --archive /home/auszok/experiments/joint_matched_transfer_verification_20261115 --output <fresh-output>
```

These are independent inference diagnostics on the frozen selected anonymous-ID +
grounding snapshot. They do not retrain weights, change references, or replace a
qualified controller. They use the existing 100 LIBERO episodes (50 Object, 50
Spatial); these already-inspected scenes are diagnostic regression data.

1. `raw` supplies the learned grounding head only the public natural-language
   instruction, matching training. The planner still receives the original scene
   relations and predicted-role hints. The optional `grounding_instruction` agent
   interface consumes the raw query carried by external public contexts by
   default. Contexts without a separate query retain their existing behavior;
   legacy controls explicitly request the augmented instruction for reproduction.
2. `contract` intersects the native shared-rule decoder with the fixed executor's
   declared pick/place action signatures, pointer range, movable-object role,
   holding-state preconditions and complete-operation termination. It excludes
   unsupported pull/pour, without goal-selected object masks, task aliases, or
   answer templates. It respects the decoder's remaining-token budget.
3. `aligned` extracts separate source/destination phrases from the public
   instruction and scores observed objects with frozen, paired SigLIP text/image
   representations. RGB GrabCut around public numeric-tag draw anchors proposes
   an object region; a 192-pixel contextual crop is also scored. This never uses
   simulator segmentation, canonical object names, category lookup, or BDDL
   goals. All visible candidates compete; scores, runner-up IDs and margins are
   saved. This implements explicit visual discrimination, **not additional
   hard-negative training**. Spatial relational phrases may remain difficult.

`raw-contract` and `aligned-contract` test the explicit combinations. They do not
inherit other methods. Baseline predictions, actual rendered-input hashes and
abstract scores must reproduce the archive before the five interventions start.
Private goals are first opened after each generation returns. The capability
mask must leave rendered inputs unchanged for each matched grounder condition.
Each new condition gets a fresh native physical execution; the baseline's native
execution archive is identified as reused. The fixed 1,000-step horizon remains.

Run focused unit/native-decoder tests with:

```text
python -m pytest test_regr/JointEmbodiedAgentInterface/transfer/anonymous_grounding -q
```

Freeze and launch on completely unused GPUs on gpu2 with:

```text
python -m test_regr.JointEmbodiedAgentInterface.transfer.anonymous_grounding.object_fixes.deploy --output /home/auszok/experiments/NEW_UNIQUE_ROOT
```

The launcher replays and verifies the entire archived evaluation source manifest,
overlays this folder and the raw-query context/agent interfaces, runs tests
in the retained runtime image, then gates the campaign on baseline reproduction.
It records checkpoint/head/source/input hashes, GPU UUIDs, PIDs, commands, logs,
exit statuses, prompts, token decisions, rule rejections and paired predictions.
`results.json` reports native success first, suite retention, validity, grounding,
paired changes and task-cluster bootstrap intervals. No hyperparameters or
checkpoints are selected from these results, and no further training is launched.

The completed 13 November 2026 diagnostic comparison, validation evidence and
remote artifact paths are recorded in [RESULTS.md](RESULTS.md).
