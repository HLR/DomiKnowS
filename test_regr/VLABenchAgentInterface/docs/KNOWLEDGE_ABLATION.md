# Knowledge ablation of the champion Joint planner

What each knowledge tier of the world graph contributes when the same checkpoint is evaluated on the same
450 test episodes. Task knowledge is expressed as graph concepts and logical constraints
(`knowledge/`, see the README source code map); `--knowledge` switches which constraint groups are active.

| Level | Active knowledge | Source of the task |
| --- | --- | --- |
| `grammar` | plan grammar DFA only: the token syntax of a plan, the nine skills, any of the 64 entity ids, the end token, the operation budget and the language of allowed skill patterns; no constraint group | none |
| `generic` | + skill schema, subtask patterns, pointer, anti-self-placement and repeated-pick rules | none |
| `scene` | + scene-grounded rules (buttons, fixtures, containers, open-before-place) | none |
| `learned` | + the per-task LCs of `task` (skill-sequence, completion, obligations) and the grounded physics/complex families | predicted from the instruction by a classifier fitted on the training split |
| `task` | the same rules as `learned` | the episode's task name |
| `metadata` | + QA property observations (`--observation-mode metadata-assisted`) | the episode's task name |

`grammar` is the floor of this graph planner: its grammar DFA is the decoding grammar, so no lower row exists.
The frozen VLM in the standard protocol is a different system and is not comparable with these numbers.

## Setup

- Checkpoint `joint_stage2_best.pt` (Stage 2, epoch 4), 450 test episodes, seed 42, original stored labels,
  no generation failures at any level. Code: commit `4c04f1e1` (the `learned` and `grammar` rows were run on later commits that only add those levels), run on gpu2 (one GPU per level).
- Command (per level; `metadata` uses `--observation-mode metadata-assisted`):

```bash
python -u -m test_regr.JointEmbodiedAgentInterface.eval_planning \
  --checkpoint test_regr/JointEmbodiedAgentInterface/checkpoints/joint_stage2_best.pt \
  --planning-dir test_regr/VLABenchAgentInterface/data/planning \
  --split test --seed 42 --device cuda:0 --allow-planning-schema-migration \
  --observation-mode visual --knowledge <grammar|generic|scene|task> \
  --output test_regr/JointEmbodiedAgentInterface/results/knowledge_ablation/<level>.json
```

The `learned` run adds `--knowledge learned --task-classifier task_classifier.pt` (see below). Raw reports: `test_regr/JointEmbodiedAgentInterface/results/knowledge_ablation/<level>.json`.

## Results (planning progress score, 0-100)

| Level | Macro | Overall | SR | PR | SPR | PM |
| --- | --- | --- | --- | --- | --- | --- |
| grammar | 60.9 | 64.3 | 89.4 | 51.1 | 55.2 | 25.5 |
| generic | 66.9 | 70.8 | 89.4 | 62.4 | 63.0 | 37.7 |
| scene | 67.9 | 71.8 | 89.4 | 63.7 | 65.5 | 39.6 |
| learned | 77.1 | 78.0 | 98.2 | 66.6 | 70.2 | 50.8 |
| task | 77.1 | 78.0 | 98.2 | 66.6 | 70.2 | 50.8 |
| metadata | 81.5 | 81.5 | 98.2 | 72.4 | 75.9 | 56.6 |

Per-dimension score:

| Level | M&T | Spatial | ComSense | Semantic | PhysLaw | Complex |
| --- | --- | --- | --- | --- | --- | --- |
| grammar | 74.7 | 63.2 | 75.8 | 75.8 | 37.7 | 37.9 |
| generic | 85.0 | 71.2 | 81.6 | 82.3 | 37.7 | 43.6 |
| scene | 85.4 | 72.2 | 81.7 | 85.5 | 37.7 | 44.6 |
| learned | 78.9 | 72.2 | 81.7 | 85.5 | 57.0 | 87.4 |
| task | 78.9 | 72.2 | 81.7 | 85.5 | 57.0 | 87.4 |
| metadata | 78.9 | 72.2 | 81.7 | 85.5 | 83.0 | 87.4 |

## Learned task concept

`learned` replaces the episode's task name by a classifier over the instruction (hashed word and bigram bag,
linear softmax head, `knowledge/task_classifier.py`). It is fitted on the training split only (3,600 episodes,
45 task names, the episode ids are stored with the weights and evaluation refuses an overlapping split).
Task-name accuracy is 95.1% on the 450 test episodes (94.2% on validation). At this level the declared
sequence, task targets and the facts grounded from the task name are rebuilt from the prediction.

```bash
python -m test_regr.VLABenchAgentInterface.main fit-task-classifier   --planning-dir test_regr/VLABenchAgentInterface/data/planning --out task_classifier.pt
python -u -m test_regr.JointEmbodiedAgentInterface.eval_planning ... --knowledge learned   --task-classifier task_classifier.pt
```

The `learned` scores equal the `task` scores to every digit. Of the 212 test episodes whose task has declared
knowledge, the rebuilt requirements differ from the task-level ones in 66: 60 are QA episodes where the same
sequence is now supplied by the declared LCs instead of the dataset (same constraint), and 6 are
`insert_bloom_flower` episodes predicted as `insert_flower`. Those 6 changed no score; I did not check
whether their generated plans are identical. The misclassifications otherwise fall on task variants
(`*_semantic`, `*_common_sense`) that have no declared rules. So on the seen tasks the task name adds
nothing the instruction does not already give; the unseen-task test (Phase D) is what separates the two.

## Reading

- **The generic rules are worth +6.0 Macro over the grammar alone** (60.9 to 66.9): parameter recall 51 to 62 and
  graph match 26 to 38, with gains in every dimension except PhysLaw. Skill recall is unchanged (89.4), so the gain is in
  arguments and graph structure, not in which skills are chosen.
- **Scene knowledge is worth about 1 point of Macro** (66.9 to 67.9); only Semantic moves (+3.2).
- **The task tier is the large step, +9.2 Macro**, and it comes from two dimensions: Complex 44.6 to 87.4
  (the obligations and sequence LCs) and PhysLaw 37.7 to 57.0 (grounded QA families). Skill recall rises
  from 89 to 98.
- **Metadata adds +4.3 Macro**, all of it PhysLaw (57.0 to 83.0), the dimension whose answer needs measured
  property values.
- **Task knowledge hurts M&T** (85.4 to 78.9): skill recall reaches 100 but parameter recall falls from 83 to 65.
  The cause was not investigated; the declared M&T sequence may force a skill order that the model's
  parameters were not trained to follow.
- Spatial, ComSense and Semantic are unchanged by the task tier, so for those dimensions the model's own
  predictions plus generic rules already match what the task LCs enforce.

## Caveats

- The checkpoint was trained with task-tier inputs, so the `generic` and `scene` rows mix the removal of
  knowledge with a train/test mismatch. They are a lower bound for a model trained at that level; Phase D
  (retrain per level, hold out whole tasks) removes the confound.
- Single seed, one checkpoint; differences of about 1 point (generic vs scene) are not established.
- `task` gives Macro 77.1 here against the 75.1 recorded in `README_checkpoint.md`; that earlier figure came
  from an older code revision. This run is the reference for the knowledge comparison.
- The `metadata` level uses a different observation mode, so it is not a pure LC-group toggle.
