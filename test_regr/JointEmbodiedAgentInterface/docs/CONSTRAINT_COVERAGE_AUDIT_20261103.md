# Why constraints do not improve anonymous CE: coverage audit

3 November 2026. Selected anonymous CE-only epoch 3; frozen seed-42 internal test, 450 episodes. Training used the official evaluation corpus's internal split, so these are seen-task diagnostics. This audit uses existing predictions and frozen perception replies. No model generation, retraining, dataset edits or active-training changes were performed.

## Main finding

The current reliable constraints target different errors from those the model makes. The three causal rules are `causal__anti_self_placement`, `causal__distinct_subtask_destinations`, and `causal__destinations_stay_put`. None rejects an incorrect free plan in this test. The only rejection without perception is the grammar's operation cap on a repetitive book plan. The perception rules can distinguish objects, but currently exclude many reference-compatible choices and offer almost no safe correction opportunities.

The existing measured results are Progress 88.0151 without DFA, 87.9978 with official knowledge/perception off, and 80.5065 with official knowledge/perception on. Exact match is 72.00%, 72.00%, and 62.22%, respectively. The perception-off policy changes one plan; the full-perception policy changes 95. Offline replay independently reproduces exactly those changed episode IDs as the policies' rejected free plans.

## What the 126 reference mismatches contain

There are 324 exact matches and 126 non-matches. Of the latter, 115 (91.27%) have the **same skill sequence and operation count** as the reference but different target-entity assignments. None of these 115 changes a `target_container_name` binding. Six are permutations of the same complete operation multiset, including two cooking cases that already receive full Progress; these may include acceptable alternatives and should not automatically be treated as execution failures.

The other 11 have extra operations: eight Texas Hold'em episodes, two book-rearrangement episodes, and one cooking episode. Eight non-exact plans receive Progress 100: the metric can fully credit coverage/dependency structure despite extra operations or a different order. Strict exact match and Progress therefore measure different things. Neither alone certifies execution success.

The highest mismatch task counts are cook_dishes 11, texas_holdem 11, density_qa 9, reflection_qa 9, select_poker 9, thermal_expansion_qa 8 and magnetism_qa 8. These are counts against the archived references, not independently verified failure counts.

## Constraint coverage and reference feasibility

Each free plan and its scene-verified-density reference was replayed with the actual Joint runtime, vocabulary, operation limit, and contextual DFA. The five settings change only supplied visibility/candidate evidence. All 450 original planner-input hashes match the perception-off archive; all 450 reconstructed enriched-input hashes match the full-perception archive.

| Policy | Wrong free plans rejected /126 | Wrong rejected, reference allowed | References rejected /450 |
|---|---:|---:|---:|
| Perception off | 1 | 1 | 0 |
| Full perception | 50 | 2 | 93 |
| Visibility only | 47 | 1 | 83 |
| Candidates only | 4 | 2 | 11 |
| Empty visibility treated as unknown | 36 | 2 | 68 |

“Wrong” means non-exact relative to the archived reference. The middle column is a necessary correction opportunity, **not proof decoding will recover the reference**. Every row's middle column includes the same book operation-limit case, which actually reduces Progress. Thus the full perception and candidate-only policies each have just **one additional semantic correction opportunity**. Visibility alone has none. The two single-rule reference-exclusion counts overlap on one episode; do not add them to obtain the full-policy union.

Full perception rejects 45 of the 324 already-exact free plans. Visibility alone rejects 37, and candidates alone rejects nine. First-failure rule traces are saved per episode; attribution is to the immediate rejecting rule rather than a guessed cause of future completion pruning.

## Representative cases

**A genuinely useful grounded restriction:** `M&T/add_condiment/example43`, instruction “Add salt to the dish.” Free decoding picks object 6 (sugar); the reference picks object 4 (salt). Perception supplies the pick candidate `{4}`, so `official__instruction_candidates_bound_targets` rejects object 6 while allowing the reference. Actual official decoding corrects the plan, increasing Progress from 65 to 100. This is evidence that grounded constraints can help when their observations are correct.

**A harmful restriction:** `PhysicsLaw/reflection_qa/example41`. Free decoding and the reference select object 2, but perception's visible set is `{0,4}`. The DFA prohibits object 2 and forces object 4. The recorded allowed probability mass is approximately 0.00358. The problem is an observation interpreted as an exhaustive support set, not an error in choosing the largest unmasked logit.

**A redundant structural check:** `M&T/select_poker/example49`, instruction asks for the queen of hearts. Free decoding picks object 4 (4 of hearts) while the reference selects object 6 (queen of hearts). Both use a valid pick/lift sequence and satisfy the causal checks. Structural consistency cannot identify the requested card without reliable image-to-tag identity grounding.

**An operation cap that does not solve the goal:** `Complex/book_rearrange/example73`. The free plan repeats several pick/push/pull subtasks and contains 18 operations. The constrained plan stops at 12 operations instead of choosing the correct nine-operation reference. It omits a required book and lowers Progress from 93.33 to 85.56. A length limit is not an instruction-grounded uniqueness/order/coverage rule.

## Reference consistency: separate from rule feasibility

Being accepted by a DFA does not establish that a reference is correct. A conservative, read-only check compared target entities from simple explicit scene success conditions with the reference targets in 367 episodes. It flagged 18 disagreements. Nine are expected disagreements with the deliberately unrepaired on-disk density conditions; the corrected density references are intentionally different and these are **not nine new defects**.

The remaining **nine non-density cases** are reference/condition conflicts: five select_fruit_spatial, two select_drink_semantic, one select_drink_common_sense and one select_drink. In five, free decoding is scored non-exact while selecting the entity named by the scene condition. In four, free decoding exactly matches the reference but both omit the condition's target. These are review flags rather than automatic label repairs: some instructions and scene conditions also disagree, and a full execution-based adjudication was not performed.

One visually checked example is `Semantic/select_drink_semantic/example11`. The instruction asks for a healthy breakfast drink; the scene condition requires removing juice_seen, and the numbered image confirms juice is object 2 and Red Bull is object 3. Free decoding picks juice (2); the stored reference picks Red Bull (3). This is not a global one-based/zero-based indexing error. Another case, `M&T/select_drink/example78`, has a three-way disagreement: instruction says spirit, scene condition says cola, reference selects jumpstart. Scene conditions cannot automatically replace every label.

All nine repaired density test episodes match the free model's **original** reference exactly and fail the corrected target. The snapshot was trained using the original labels, while evaluation repairs those nine targets in memory. This is a training/evaluation supervision inconsistency, not evidence that the model solved the corrected physical question.

The scored results table remains unchanged. No additional label changes or oracle target facts were supplied to generation or the audited DFAs. Scene-derived conditions are used only for offline consistency review.

## What to fix next

1. **Adjudicate labels before rewarding agreement with them.** Review the nine new non-density conflicts against instruction, numbered images, independent task success semantics and source generation. Audit the training split too. Keep original and adjudicated references versioned; do not infer a new answer from the reference itself. Repair density supervision consistently before a future clean training campaign.
2. **Keep uncertainty out of hard exclusion sets.** An empty detector result should be unknown; this alone is insufficient because 68 references are still excluded afterward. Calibrate instruction-candidate and visibility precision on separate train/validation scenes. Missing detections are not verified absence. Use uncertain candidates as soft preferences or abstain; allow hard masks only for facts with independently validated support. Do not tune to these test references.
3. **Target entity grounding, not more generic syntax.** Prioritize card rank/suit reading, explicit object identity, image-tag association, and instruction-relative spatial/rank relationships. The salt case demonstrates the relevant correction mechanism. Do not inject canonical simulator names, hidden scene facts or reference-derived candidates into prompts.
4. **Scope composite constraints to independently grounded goals.** Coverage, uniqueness and order could address some book/poker errors, but only where the instruction actually requires them and the corresponding object facts are observed. A global no-repeat rule could invalidate legitimate plans. Alternative valid cooking orders should remain legal.
5. **Require evidence before another expensive experiment.** On a separate validated split, measure rejected-error/reference-allowed coverage, reference exclusion, rule-trigger reasons and abstention. Then run paired decoding with the same checkpoint and frozen inputs, reporting improvements and regressions, exact match, Progress and independent goal/execution success. The current constraints have little safe coverage; retraining alone would not repair this mismatch.

## Evidence

Remote root: `/home/auszok/experiments/constraint_coverage_audit_20261103/`. Files: `audit.py`, `manifest.json`, `audit.log`, `episode_audit.jsonl`, `summary.json`, `scene_target_audit.json`, `scene_target_checks.jsonl`, and this report. The local artifact directory also includes the visually inspected numbered drink image. Input prediction/report/decision file hashes remained unchanged after replay. Ongoing RL and masked-CE training were untouched.
