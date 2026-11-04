# Official inputs, uncertainty-aware grounding and reference review

The native official evaluator keeps the pinned upstream prompt, the original
reference files, upstream scoring and frozen VLM. Its zero-shot inputs are the
original image, numbered image and instruction. Scene configurations, canonical
simulator names, target references and task IDs are not grounding inputs.
The trained Joint planner and its internal split remain a separate experiment.

## Image-derived preferences

When image perception is enabled through the official evaluator CLI, the default
is now **soft** grounding. The auxiliary frozen model's candidate objects supply
a finite preference at object-ID decisions. Visibility is positive evidence,
never proof that an unreported object is absent. Grammar and causal masks remain
hard; perception cannot forbid a legal token, resurrect an illegal token, change
the requested skill, or force the planner to follow an incorrect candidate.
The preference is bounded and does not accumulate with multi-token ID length.

```bash
python -m test_regr.VLABenchAgentInterface.main evaluate-planner \
  --protocol official --knowledge official \
  --planner-model /path/to/frozen/model --planning-dir /path/to/official/data \
  --official-perception-policy soft --official-perception-weight 0.5 \
  --official-output-dir /path/to/new_soft_run
```

`--official-perception-policy legacy` reproduces the old hard visibility/candidate
interpretation. Direct generator API calls also retain their legacy default;
pass `perception_policy="soft"` explicitly. Plain official greedy generation
without perception remains unchanged. These are labeled decoding extensions,
not changes to the upstream prompt or proof of higher benchmark performance.

Optional `--perception-verifier-model /path/to/another/frozen/model` checks the
same two images and instruction with a different frozen model. Only agreeing
categories and card faces are used; bounding-box unions make ordinal ordering
conservative. Missing facts, conflicting card readings, overlapping order bounds
and unsupported requests abstain. Agreement is still uncalibrated evidence and
feeds soft preferences, not hard exclusions. No few-shot answers are passed to
either perception model, even when the planner uses demonstrations.

The default preference weight is a fixed conservative constant, not a calibrated
probability. Non-default confidence thresholds or preference weights require
`--perception-calibration-data` naming independently sourced training scenes;
the dataset guard rejects official-evaluation overlap. Do not fit settings on
official test labels. Explicit hard research grounding policies require legacy
mode; they cannot silently reintroduce hard exclusions into soft mode.

Run metadata records policy, weight, model/config fingerprints and verifier.
Outputs record the observed candidates and preference-change counts. Resume
rejects changed settings, so use a new output directory for a new configuration.
This change does not retrofit or restart existing Joint training runs.

The Joint CLI also accepts `--knowledge official --official-perception-policy soft`
for future training/evaluation. Joint retains its legacy default for compatibility;
select soft explicitly and use a new experiment when changing training policy.
Its compact-label decoder applies the same finite preference at object decisions,
both when sampling and when replaying RL trajectories. Checkpoints save the policy
and refuse training resumes that change it; evaluation marks the override. Decision
traces distinguish finite `log_bias` from hard DFA rejection. The native official
CLI defaults to soft only when auxiliary perception is enabled.

## Review references offline

First create a review manifest **outside** the benchmark directory:

```bash
python -m test_regr.VLABenchAgentInterface.official.reference_adjudication audit \
  --planning-dir /path/to/official/data --output /path/to/review.json
```

This checks simple explicit scene success targets against stored plans and
independently checks density questions against the existing qualitative material
model. Conflicts are flags, not replacement answers: scene conditions and even
instructions can also be wrong. Every entry starts as `needs_review`, with a
null replacement reference. No benchmark files are changed.

For an independently adjudicated entry, fill its `reference` operation list,
set `status` to `approved`, and record a nonempty reviewer, reason and evidence
list in `review`. Keep its original-reference and image/instruction hashes.
Unresolved entries remain pending or are marked rejected; never approve them by
copying a model prediction. Review actual numbered images and independent task
success semantics. The audit may inspect scene metadata for label review only.

Then rescore the **same** saved official replies:

```bash
python -m test_regr.VLABenchAgentInterface.official.reference_adjudication score \
  --planning-dir /path/to/official/data \
  --outputs /path/to/original_run/outputs.jsonl --manifest /path/to/review.json \
  --output-dir /path/to/adjudicated_scores
```

The command verifies the original dataset fingerprint, reviewed input/reference
hashes, unique episode coverage and approval evidence. It writes only
`adjudicated_scores.jsonl` and `adjudicated_summary.json`, marked `standard=false`
and `scoring_only=true`. The supplementary summary contains both original-label
and adjudicated-label scores computed from identical saved replies. It never
writes the original `summary.json`, `final_scores.jsonl`, predictions or dataset.
Changed review/prediction inputs require a new supplementary output directory.

Reference adjudication is not a grounding feature: its manifest is not accepted
by the generation API and cannot alter the target episode's prompt. Keep the
original official score for published comparisons and report adjudicated scores
as a separate evaluation variant. This workflow does not automatically approve
the currently flagged cases or claim that all reference errors have been found.
