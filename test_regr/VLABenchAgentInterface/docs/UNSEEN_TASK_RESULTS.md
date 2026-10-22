# Unseen-task test: planning without, and with, declared task knowledge

Question: if whole tasks never appear in training, how much of the planner's score on them comes from the
declared task knowledge (task concepts and logical constraints) rather than from what the model learned?

## Setup

- **Held-out tasks (700 episodes, 100 each):** `cook_dishes`, `take_chemistry_experiment` (Complex),
  `friction_qa`, `thermal_expansion_qa` (PhysLaw) and the `select_book` family: `select_book`,
  `select_book_semantic`, `select_book_spatial` (M&T, Semantic, Spatial). Whole families are held out so no
  near-duplicate remains in training.
- **Training:** Joint planner, Stage 1 only, 5 epochs of 350 rounds on the other 3,800 episodes (`--holdout-tasks default`).
  The planner loss is cross entropy plus the DFA loss (allowed mass on the DFA path for EAI and VLABench,
  primal-dual constraint loss for VLABench). The held-out tasks have no training episodes, no task concepts and
  no task constraints; their episodes carry no declared sequence, targets or name-keyed facts. The controller was
  trained on all control demonstrations, which do not contain the planning labels.
- **One model, not two.** Training does not depend on the knowledge tier (the Stage 1 losses never read task facts),
  so the planned task-tier and scene-tier training runs would have produced the same weights. The tiers differ
  only at evaluation. Checkpoint: epoch 3 of 0-4 (best validation key).
- **Evaluation:** the same checkpoint on (a) the 700 held-out episodes and (b) the 380-episode test split of the
  seen tasks, seed 42. `heldout_declared` switches the held-out tasks' constraints back on at inference
  (`--declare-heldout-knowledge`), with no retraining. No generation failures anywhere.

Reports: `test_regr/JointEmbodiedAgentInterface/results/unseen_task/<name>.json`.

## Results (planning progress score, 0-100)

| Evaluation | Episodes | Overall | SR | PR | SPR | PM |
| --- | --- | --- | --- | --- | --- | --- |
| held-out, no knowledge for them (task level) | 700 | 28.7 | 54.3 | 12.0 | 11.1 | 10.1 |
| held-out, generic level | 700 | 28.0 | 52.8 | 12.0 | 11.1 | 10.1 |
| **held-out, knowledge declared at inference** | 700 | **57.9** | 85.6 | 40.5 | 46.7 | 28.0 |
| seen tasks, task level | 380 | 59.2 | 85.1 | 43.0 | 48.0 | 32.1 |
| seen tasks, generic level | 380 | 40.7 | 58.9 | 32.5 | 29.4 | 11.6 |

Per-dimension score (episodes in brackets). Macro is not defined on the held-out set (no ComSense episodes).

| Evaluation | M&T | Spatial | ComSense | Semantic | PhysLaw | Complex |
| --- | --- | --- | --- | --- | --- | --- |
| held-out, no knowledge | 25.4 (100) | 20.0 (100) | - | 20.4 (100) | 61.3 (200) | 6.1 (200) |
| held-out, generic | 25.2 | 20.0 | - | 20.1 | 61.3 | 4.2 |
| held-out, declared | 51.6 | 20.0 | - | 20.4 | 61.3 | 95.3 |
| seen, task level (Macro 58.6) | 69.9 (82) | 43.9 (52) | 61.6 (82) | 53.4 (81) | 51.7 (41) | 71.3 (42) |
| seen, generic level (Macro 38.5) | 47.8 | 35.2 | 47.9 | 41.4 | 50.7 | 8.2 |

## Reading

- **Declaring the rules for an unseen task is enough for a large part of it.** With no retraining, writing the
  constraints of the held-out tasks lifts their overall score from 28.7 to 57.9. The Complex tasks go from 6.1 to
  95.3, higher than the seen Complex tasks (71.3). The declared sequence and obligations most likely decide the
  plan there, but I did not check this plan by plan.
  `select_book` (M&T) goes from 25.4 to 51.6.
- **No change where nothing was declared or the rule cannot help.** `select_book_semantic` and `select_book_spatial` have
  no declared rules in any run (Semantic and Spatial stay at about 20). The two QA tasks stay at 61.3 with or without
  their rules. I did not investigate why; one possibility is that the answer (which button to press) depends on
  property information the visual input does not give, which a sequence rule cannot supply.
- **Without any task knowledge, unseen tasks are poor** (28.7 overall; PR 12.0), and the generic rules add little
  (a 0.6-point difference from a single run).
- **On seen tasks the same pattern holds**, so the task rules matter even where the model was trained: Macro 38.5
  generic to 58.6 with the task rules, with Complex 8.2 to 71.3.

## The planner still depends on the grammar

No-DFA decoding of the same checkpoint on the seen test split:

| Decoding | Macro | SR | PR |
| --- | --- | --- | --- |
| free (no grammar) | 0.0 | 0.0 | 0.0 |
| free, end label withheld for one step | 7.4 | 21.6 | 0.0 |

The Stage 1 DFA loss did not make the planner grammar-independent. The allowed-mass term fell from 3.9 to 1.74
during training, which corresponds to a geometric-mean probability of about 18% on grammar-valid tokens per step
(exp(-1.74)); most of the mass is still outside the grammar. The model without grammar emits the end label first (Macro 0.0) and, with that blocked, never
produces a correct argument (PR 0.0). The earlier Stage 1 plus Stage 2 champion scored 26.6 in the same
end-withheld setting, so this Stage-1-only model is not better than the champion here; the two are not directly
comparable (different training stages and test splits).

## Caveats

- Single seed and a single checkpoint. The model is Stage 1 only, so its absolute scores are well below the
  champion's (Stage 1 plus Stage 2); only the comparisons within this table are meaningful.
- The seen-task test split (380) is a seeded split of the 3,800 seen episodes and is not the champion's 450-episode
  test split.
- The held-out set mixes dimensions unevenly (Complex and PhysLaw have 200 episodes each), so the overall score
  weights those more than the per-dimension scores would.
- Training stopped after Stage 1; Stage 2 (reinforcement) was not run.
