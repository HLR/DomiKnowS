# Where constraints get their information

A logical constraint decides what is allowed, but it needs observations to do that. This document lists where each
constraint group gets them, and which groups need an extra prompt to the LLM. The last section covers
[where the constraints are used](#where-the-constraints-are-used), during decoding and elsewhere.

For what each rule does, see [VLABenchLogicaConstraints.md](VLABenchLogicaConstraints.md). For how to select the
groups, see [Knowledge_QuickStart.md](Knowledge_QuickStart.md).

Links with a line number point to the code as it was when this document was written, so search for the linked name
if a line has moved.

## The sources

| Source | What it provides | Implementation | Extra LLM prompt | Privileged |
|---|---|---|---|---|
| [The plan being written](#the-plan-being-written) | skills, ids and earlier operations in the plan so far | [`semantics.py`](../semantics.py#L262) | no | no |
| [Declared tables](#declared-tables) | skill schemas, subtask patterns, task sequences, property orders, recipes | [`world_domain.py`](../world_domain.py), [`knowledge/tables.py`](../knowledge/tables.py) | no | no |
| [Episode configuration](#episode-configuration) | entity names and classes from `env_config` | [`_extract_entities`](../dataset.py#L525), [`scene_objects`](../object_grounding.py#L19) | no | yes: the standard protocol doesn't show it |
| [Task identity](#task-identity-oracle-or-predicted) | the episode's task name (`oracle`), or a word-and-bigram classifier's prediction from the instruction (`predicted`) | [`task_requirements`](../task_policy.py#L94), [`predict_tasks`](../knowledge/task_classifier.py#L96) | no | `oracle` only |
| [Instruction text](#instruction-text) | clauses and phrases read by Python rules | [`instruction_object_constraints`](../object_grounding.py#L128), [`selection_request`](../official/selection_observations.py#L40), [`complex_observations`](../complex_grounding.py#L24) | no | no |
| [Simulator metadata](#simulator-metadata) | each answer option's material and shape from `env_config`, read only with `--observation-mode metadata-assisted`; reflectance from `property_observations.json` when an episode has one | [`ground_planning_facts`](../dataset.py#L544), [`physics_planning_facts`](../physics_grounding.py#L145) | no | yes |
| [Research input](#research-input) | `research_observations.json`, when the episode has one (the planning dataset in this repository has none) | [`dataset.py`](../dataset.py#L643) | no | depends on its contents |
| [Model question flags](#model-question-flags) | the model's answers to `--press-question`, `--scene-ids` and `--pattern-first` | [`says_yes`](../official/official_protocol.py#L565), [`read_entity_count`](../official/official_protocol.py#L576), [`choose_patterns`](../official/official_protocol.py#L589) | yes | no |
| [Image perception](#image-perception) | visible tags, object kinds, boxes, card faces, door states and role hypotheses | [`perception_prompt`](../research/perception.py#L49), [`observe_episode`](../official/perception_observations.py#L159), [`enrich_examples`](../research/perception.py#L390) | yes | no |

### The plan being written

This is the partial plan the decoder has produced so far in the current episode. No outside information is involved.

The model generates its plan one token at a time. Before each token, the DFA allows only the tokens that can still
lead to a valid plan. To decide that, it keeps a small state that records what the finished operations did, and updates
it after each one ([`advance`](../semantics.py#L296) on the internal paths,
[`advance_state`](../official/knowledge_dfa.py#L105) in the official decoder).

[`PlanSemanticState`](../semantics.py#L97) includes, for example:
- the object currently held and the doors opened so far;
- the objects picked so far;
- the destinations used in the current subtask;
- every destination used so far.

For example, suppose the model has written `pick 2, place 3`:

| Recorded so far | Value |
|---|---|
| Objects picked | 2 |
| Destinations in this subtask | 3 |
| All destinations so far | 3 |

If the model now tries `pick 3`, `destinations_stay_put` refuses it, because 3 is already a destination. The rule
compares ids and nothing else: it doesn't know what object 3 is, and it needs no image, scene or task. That is why the
causal rules can run under the standard protocol.

### Declared tables

These are fixed, hand-written lists of domain knowledge in the source code. They are the same for every episode, and
nothing is read from the model, the images or the dataset to fill them. When a graph is built, Python loops over the
tables and generates logical constraints from them, so a table holds the facts and the generated constraint enforces
them.

| Table | What it declares | Example |
|---|---|---|
| [`SKILL_ARGUMENTS`](../world_domain.py#L135) | which arguments each skill takes | `pick` takes `target_entity_name`, `place` takes `target_container_name` |
| [`SUBTASK_PATTERNS`](../world_domain.py#L35) | the skill sequences that form a valid subtask | `pick`, `pour`, `place` |
| [`PRIMITIVE_TASK_PATTERNS`](../knowledge/tables.py#L13), [`QA_TASK_PATTERNS`](../knowledge/tables.py#L33) | the skill sequence of a task | `add_condiment` is `pick`, `pour` |
| [`COMPLEX_TASK_SCHEMAS`](../knowledge/tables.py#L59) | required skills and minimum length of a complex task | `cook_dishes` needs `pick` and `place`, and at least 4 operations |
| [`PHYSICS_PROPERTIES`](../knowledge/tables.py#L80) | how materials rank on a physical property | for `density_qa`: wood, rubber, water, glass, stone, metal |
| [`RECIPES`](../knowledge/tables.py#L111) | the ingredients of a named dish | `tomato_and_egg_stir_fry` needs tomato and egg |
| [`COMPLEX_SKILL_TEMPLATES`](../knowledge/tables.py#L145) | the skill cycle repeated for each subject | `cook_dishes` repeats `pick`, `place` |

A table is general knowledge, not an observation of the current episode. `PHYSICS_PROPERTIES` says metal is denser than
wood, but which button belongs to a metal object has to come from another source.

### Episode configuration

Each episode has a saved scene description, `env_config/env_config.json`, used to set up that episode's simulator scene.
Unlike a table, it differs from episode to episode. The constraints use it to learn what the objects are.

**Which file an episode uses:** nothing has to choose one, because each episode folder contains its own file. The model
never identifies a scene from the image. [`load_planning_examples`](../dataset.py#L597) walks the episode folders and
[takes the config inside each one](../dataset.py#L627).

```text
CommenSence/add_condiment_common_sense/example0/
├── env_config/env_config.json   this episode's description
├── input/                       images and instruction
└── output/                      reference plan
```

**How many there are:** the answer depends on what counts as different.

| What is counted | Number |
|---|---|
| Configuration files (45 tasks × 100 episodes) | 4,500 |
| Different files, byte for byte | 4,500 |
| Different scenes, ignoring the goal | 4,399 |
| Different entity tables (the same names in the same order) | 2,134 |
| Different background scenes | 6 (`bedroom_0`, `kitchen_2`, `lab_1`, `living_room_0`, `studyroom_0`, `wall_0`) |

Entity tables have 5 to 17 entries, with a median of 7.

**The entity table** comes from the `components` and `scene` entries ([`_extract_entities`](../dataset.py#L639)). Each distinct
name gets the next id, in the order it appears. Object classes come from the same components
([`scene_objects`](../object_grounding.py#L19)). For example, `add_condiment_common_sense/example0` gives:

| Id | Name | Class |
|---|---|---|
| 0 | `counter` | `Counter` |
| 1 | `stove` | `Stove` |
| 2 | `pan_seen` | `FlatContainer` |
| 3 | `dishes` | `FlatContainer` |
| 4 | `bbq_sauce` | `CommonGraspedEntity` |
| 5 | `sugar` | `CommonGraspedEntity` |
| 6 | `sugar_tag` | `NameTag` |
| 7 | `ketchup` | `CommonGraspedEntity` |
| 8 | `kitchen_2` | none: it is the scene, not a component |

**Ids and image tags:** an entity id appears to be the number tag on the labeled image (`input/input_mask.png`), but no
file states this. No file maps ids to tags, and the code only numbers names in the order they appear in the configuration.
Under `--protocol official` the table isn't used, and the model reads the tags from the image. The evidence for the match
is:
- **Two episodes checked by eye:** in `add_condiment_common_sense/example0`, tag 6 is the label reading "sugar"
  (`sugar_tag`) and tag 4 is the bottle the plan picks (`bbq_sauce`). In `select_drink_common_sense/example36`, tag 3 is
  the Red Bull can (`redbull`).
- **Reference plans:** all 9,070 integer ids in the plans fall inside their episode's entity table.
- **Goal conditions:** 4,362 of the 4,500 plans target an object named in the episode's goal. Of the other 138,
  `texas_holdem` (100) has an empty goal, and in the `select_drink` tasks (38) the plan and the goal name different
  objects. In `select_drink_common_sense/example36` the image still matches the table, so that conflict looks like one
  inside the dataset. Only that one case was checked by eye.

The scene entry (the last id) had no tag in either image checked, and a tag can be hidden or cut off in some of the four
views.

Most rules read an object's type from its name, using simple word checks:
- **Button:** the name [contains "button"](../semantics.py#L275).
- **Static fixture:** the name [contains a word](../semantics.py#L41) such as "table" or "counter", so `pick 0` is
  refused here.
- **Enclosed container:** the name [contains a declared keyword](../semantics.py#L30).

The instruction candidates use the class instead, through lists such as [`_CONTAINERS`](../object_grounding.py#L35).

**What isn't used:** the same file also has a `conditions` entry, which states the goal. In this episode it names
`bbq_sauce` as the thing to pour. The loader passes only `components`, `scene` and `entities` on, so that answer doesn't
reach the entity table, and [`ground_planning_facts`](../dataset.py#L544) doesn't treat conditions as observations.

**Why it is privileged:** the names and classes say what each object is, without looking at the image. Under the
standard protocol the planner sees only the two images and the instruction, so it has to work this out from pixels.
That is why `official` and `--protocol official` use image perception instead, and why levels from `generic` upward can't
run under that protocol.

### Task identity: oracle or predicted

The task rules need to know which task the episode is, such as `cook_dishes`. `--task-identity` selects where that comes
from. Neither option uses the LLM, and the setting matters only at the `task` level.

**`oracle`** (the default) reads the true task name. The name is taken from the episode's folder path by
[`planning_metadata`](../planning_metrics.py#L46), which picks the folder whose name is a known task. No prediction is
involved, so it is always right. It is privileged because the standard protocol doesn't give the planner the task.

**`predicted`** works out the task from the instruction text alone, using a small classifier that isn't an LLM. It
ignores the images and the scene.

1. **Fitting** ([`fit_task_classifier`](../knowledge/task_classifier.py#L65)): the
   [`fit-task-classifier`](../main.py#L1115) command fits it once on the training split, using each episode's instruction
   and true task name.
2. **Words and bigrams** ([`features`](../knowledge/task_classifier.py#L21)): the instruction is lowercased and split
   into words and pairs of neighbouring words (bigrams). "Pick the apple" gives `pick`, `the`, `apple`, `pick the` and
   `the apple`.
3. **Hashing:** each word or bigram is hashed into one of 8,192 buckets, so there is no vocabulary to store. Different
   words can occasionally share a bucket. The bucket counts are log-scaled and normalized to give the input vector.
4. **Prediction** ([`probabilities`](../knowledge/task_classifier.py#L39)): a single linear layer gives each task a
   score, and softmax turns the scores into probabilities.
5. **Choice** ([`predict`](../knowledge/task_classifier.py#L43)): the most likely task is used. The evaluation commands
   set no confidence threshold, so there is always an answer, even when the classifier is unsure.

After prediction, [`with_predicted_task`](../dataset.py#L572) drops the episode's task name, declared skill sequence and
task targets, then rebuilds the task's observations from the prediction. [`assert_disjoint`](../knowledge/task_classifier.py#L59)
refuses to evaluate on any episode the classifier was fitted on.

Task-name accuracy is 95.1% on the 450 test episodes and 94.2% on validation, as recorded in
[KNOWLEDGE_ABLATION.md](KNOWLEDGE_ABLATION.md). A wrong prediction isn't flagged: the planner is held to the rules of the
wrong task.

### Instruction text

This is the sentence in the episode's `input/instruction.txt` ([`_read_instruction`](../dataset.py#L512)), such as "Add 1st
condiment from bottom to the dish." The model sees it too, so it isn't privileged.

Python rules read it with regular expressions and word matching. They don't understand meaning, so "something with a hint
of sweetness and smoke" tells them nothing. When a rule isn't sure, it abstains and restricts nothing.

| Parser | What it looks for | Used by |
|---|---|---|
| [`instruction_clauses`](../object_grounding.py#L108) | an action verb such as "pick" or "add", then the source and destination clauses | instruction candidates |
| [`selection_request`](../official/selection_observations.py#L40) | a group, a position and an order ("1st condiment from bottom", "the largest poker"), or a card face ("poker 2 of clubs") | selection rules |
| [`instruction_phrases`](../knowledge/physics_lcs.py#L158) | fixed phrases from [`PHYSICS_PHRASES`](../knowledge/tables.py#L91), such as "medium" or "fastest" | physics rules |
| [`complex_observations`](../complex_grounding.py#L24) | recipe names, reagents and the painting named, plus phrases such as "latest to earliest" | complex-task rules |

The first two parsers run on any task. Across the 4,500 instructions they found:

| Parser | Result | Episodes |
|---|---|---|
| `instruction_clauses` | found a clause | 1,503 |
| | abstained: unsupported action wording | 1,967 |
| | abstained: spatial or ordinal word | 668 |
| | abstained: relation in the source clause | 301 |
| | abstained: negation or exclusion | 61 |
| `selection_request` | spatial order | 292 |
| | largest or smallest card | 100 |
| | named card face | 100 |
| | no request | 4,008 |

Two limits apply to these counts:
- **A found clause isn't yet a restriction.** The clause must also name an object. In "perhaps add something that could
  help", the clause "something that could help" matches nothing, so no object is ruled out.
- **Abstaining is sometimes accidental.** The check only looks for words, so "doesn't taste salty enough right now" counts
  as spatial because of "right". That loses a restriction, but it can't reject a correct plan.

The instruction text only finds what is asked for. To restrict anything, the request needs objects to match against:
entity names on the internal paths, or perceived kinds, boxes and card faces under official knowledge.

### Simulator metadata

These are values the simulator used to build the scene, which the model isn't given as text. The physics rules for the QA
tasks need them to judge an answer, such as which object is densest. They are read only with
`--observation-mode metadata-assisted`, and only for the six QA tasks.

| | `visual` | `metadata-assisted` |
|---|---|---|
| Implementation | [`visual_qa_planning_facts`](../physics_grounding.py#L43) | [`physics_planning_facts`](../physics_grounding.py#L145) |
| What it reads | only which option buttons exist | the buttons, plus each option's material and shape |
| Effect on the rule | properties stay unknown, so no option is rejected | the rule rejects the options that don't fit |

**Where the values come from:** despite the name of the optional file, five of the six tasks read `env_config.json`:
1. [`option_objects`](../physics_grounding.py#L73) binds each `Button` to a `RandomGeom` object, either nested under it or
   the clearly nearest one.
2. [`option_attributes`](../physics_grounding.py#L28) reads the object's `material`, and its shape from `geom_type` or
   its name.

**Example:** `speed_of_sound_qa/example0` asks which object "has sound traveling moderately".

| Button | Bound object | Material |
|---|---|---|
| `button0` | stone box | stone |
| `button1` | wood box | wood |
| `button2` | metal sphere | metal |

"moderately" gives the phrase `medium`. [`PHYSICS_PROPERTIES`](../knowledge/tables.py#L80) ranks sound speed as wood, then
stone, then metal, so only `button0` is allowed. [`PhysicsMonitor`](../knowledge/physics_lcs.py#L188) applies this rule
when the plan presses a button, and here it allows the same button as the reference plan.

**Reflection is the exception:** reflectance isn't saved in the scene, so `reflection_qa` needs an
`input/property_observations.json` file ([`reflection_observations`](../physics_grounding.py#L96); the format is in
[VLABENCH_LOGICAL_CONSTRAINTS_AND_COMPONENTS.md](VLABENCH_LOGICAL_CONSTRAINTS_AND_COMPONENTS.md)). The planning dataset in
this repository has no such files, so reflectance stays unknown and all three options remain allowed in all 100 episodes.

**How the rules compare with the reference answers** across the 100 episodes of each task:

| Task | Reference answer accepted | Reference answer rejected | Options the rule leaves |
|---|---|---|---|
| `friction_qa` | 100 | 0 | 1 |
| `magnetism_qa` | 100 | 0 | 1 |
| `speed_of_sound_qa` | 100 | 0 | 1 |
| `thermal_expansion_qa` | 100 | 0 | 1 |
| `density_qa` | 26 | 74 | 1 |
| `reflection_qa` | unknown: no values | 0 | 3 |

**`density_qa` has wrong labels in the dataset, not a wrong rule:** in 74 of 100 episodes as stored here, the rule allows
only one option and it isn't the reference answer. For example, `density_qa/example1` asks for the smallest density among
metal, stone and wood. The rule allows wood, but the stored plan and goal both choose metal. The README traces this to a
[bug in the upstream density generator](../README.md#density-qa-label-repair), which used a material's density rank as its button index even
though materials are assigned to buttons in shuffled order. So a score against the stored labels counts these correct
answers as wrong. In `eval_planning`, [`--updated-data`](../README.md#original-and-corrected-scoring-labels) scores against the corrected labels without
changing the files or the decoder's inputs.

**Why it is privileged:** a material read from the configuration is exact, while the planner under the standard protocol has
to judge it from pixels. That is why the [quick start](Knowledge_QuickStart.md#observation-modes---observation-mode) rejects
this mode with `--protocol official`.

### Research input

This is an optional file, `input/research_observations.json`, that supplies ready-made observations for an episode. It
serves the `research_v1` profile, whose rules cover things like freshness, spatial relations and event order. The planning
dataset in this repository has no such files, so this source matters only for datasets prepared for those experiments.

[The loader](../dataset.py#L643) stores it as `planning_facts.research`, and [`ResearchMonitor`](../research/monitor.py#L20)
reads it. The monitor is [attached only when `research_v1` is on](../task_policy.py#L157), so under the default `baseline`
profile the file has no effect on the planning rules.

The file holds two kinds of content: observations, and the requirements checked against them.

| Field | Kind | What it holds |
|---|---|---|
| `evidence` | observation | stamped records, such as an object's pose, each with its step and dependencies ([`Evidence`](../research/evidence.py#L25)) |
| `grounding` | observation | action and role requests, and the hypotheses about which objects fill those roles |
| `events`, `executed_operations` | observation | measured events, and the operations they confirm |
| `freshness` | requirement | evidence that must be current before an action |
| `spatial`, `spatial_effects` | requirement | relations between evidence records, and how a planned action moves them |
| `temporal`, `duration_upper_steps` | requirement | event order and deadlines, and the time bound for each action |
| `sensors`, `sensor_profile`, `perception` | description | how the evidence was collected |

Any other field makes the loader [raise an error](../dataset.py#L646), which keeps supervision out of the file. For example,
`{"freshness": [{"action": "pick", "evidence": "pose:apple"}]}` requires a current `pose:apple` record before a `pick`. The
fuller format is in [RESEARCH_CONSTRAINTS.md](RESEARCH_CONSTRAINTS.md#observation-contract).

Two more details affect how to read this source:
- **Who wrote it decides whether it is privileged.** The file may come from a sensor, a simulator or a model, and each
  record names its `source`. The perception command in
  [IMAGE_PERCEPTION.md](IMAGE_PERCEPTION.md#gpu2-single-pair-preparation) can write one, in which case the LLM prompt
  happened earlier and isn't made during the run.
- **Image perception overrides it.** With perception on, [`merge_context`](../research/perception.py#L359) keeps only the
  `freshness`, `spatial`, `temporal` and `duration_upper_steps` requirements, and replaces everything else with the
  perception result.

## By constraint group

| Group | Constraints | Declared in | Source of observations | Extra LLM prompt |
|---|---|---|---|---|
| Grammar | the 8 grammar rules | [`create_plan_grammar_graph`](../graph.py#L145) | the plan being written, declared tables | no |
| Causal | `anti_self_placement`, `distinct_subtask_destinations`, `destinations_stay_put` | [`build_causal_graph`](../official/causal_graph.py#L34), [`_declare_causal_rules`](../world_graph.py#L169) | the plan being written: ids are only compared with earlier ids | no |
| Generic | schema, pattern and argument rules, `no_redundant_repeated_picks` | [`_declare_generic_rules`](../world_graph.py#L132) | the plan being written, declared tables, entity table; reuse permissions come from the planning facts | no |
| Scene | `operation_pointer_is_valid`, button rules, `pick_excludes_static_fixtures`, `container_destination_consistency`, `open_before_enclosed_placement` | [`_declare_scene_rules`](../world_graph.py#L209) | entity names from the episode configuration (for example, a [button](../semantics.py#L275) is a name containing "button"), planning facts | no |
| Scene | `observation_supported_references` | [`_declare_scene_rules`](../world_graph.py#L209) | a supplied visible set ([`visible_entity_ids`](../visibility.py#L10)), which normally comes from image perception | only through perception |
| Task | `task__*`, grounded families | [`declare_task_concepts`](../knowledge/task_lcs.py#L126) | task identity, declared tables, planning facts | no |
| Task: physics | physics selection rules | [`build_physics_knowledge`](../knowledge/physics_lcs.py#L154) | option buttons from the configuration ([`visual_qa_planning_facts`](../physics_grounding.py#L43)) and instruction phrases; property values only with `metadata-assisted` | no |
| Task: complex | complex-task rules | [`build_complex_knowledge`](../knowledge/complex_lcs.py#L186) | instruction text and entity names, bound to roles by [`complex_observations`](../complex_grounding.py#L24) | no |
| Decoding | `press_answer_single_press` | [`build_decoding_graph`](../official/decoding_graph.py#L52) | answer from [`says_yes`](../official/official_protocol.py#L565) | yes |
| Decoding | `scene_ids_bound_targets` | [`build_decoding_graph`](../official/decoding_graph.py#L52) | answer from [`read_entity_count`](../official/official_protocol.py#L576) | yes |
| Decoding | `chosen_patterns_continue`, `chosen_patterns_complete` | [`build_decoding_graph`](../official/decoding_graph.py#L52) | answer from [`choose_patterns`](../official/official_protocol.py#L589) | yes |
| Decoding | `instruction_candidates_bound_targets` | [`build_decoding_graph`](../official/decoding_graph.py#L52) | image perception and instruction text ([`perceived_observations`](../official/perception_observations.py#L180)) in the official decoder; see the note below for the internal decoder | depends on the path |
| Decoding | `observation_supported_references` | [`build_decoding_graph`](../official/decoding_graph.py#L52) | a complete visible set the caller supplies; perception does not ground it, because correct targets can lack a visible tag | no |
| Selection | `ordinal_rank_1..8`, `chosen_has_requested_property` | [`build_selection_graph`](../official/selection_graph.py#L61) | instruction text gives the request ([`selection_request`](../official/selection_observations.py#L40)); image perception gives the group, order and card faces ([`selection_constraints`](../official/selection_observations.py#L115)) | yes |
| Official | `official__scene_ids_bound_targets` | [`_declare_official_rules`](../world_graph.py#L187) | image perception's largest visible tag | yes |
| Official | `official__instruction_candidates_bound_targets` | [`_declare_official_rules`](../world_graph.py#L187) | image perception, instruction text and the selection rules | yes |
| Research | `research__grounding_*` | [`declare_research`](../knowledge/research_lcs.py#L5) | image perception's role hypotheses, or research input | through perception |
| Research | `research__fresh_*`, `research__spatial_*`, `research__temporal_*` | [`declare_research`](../knowledge/research_lcs.py#L5) | evidence records from [`compile_evidence`](../research/perception.py#L237) or research input, read by [`ResearchMonitor`](../research/monitor.py#L20) | through perception |
| Research | `research__geometry_*` | [`declare_research`](../knowledge/research_lcs.py#L5) | a certificate from the environment's geometry backend ([`geometry_admission`](../research/runtime.py#L278)) | no |

The decoding and selection rules are compiled for plan decoding by [`KnowledgePlanDFA`](../official/knowledge_dfa.py#L30).
The physics and complex rules reach the decoder through [`knowledge_monitors`](../task_policy.py#L64).

**Internal frozen-VLM decoder:** `instruction_candidates_bound_targets` makes no extra prompt on this path by default.
[`grounding_for_example`](../native_planning.py#L603) matches the instruction against scene component names and
classes in Python.

## The extra LLM prompts

### Model question flags

Each flag adds one short question per episode, before the plan. The answer is an observation for a constraint in
[`official/decoding_graph.py`](../official/decoding_graph.py#L59). The flags are off by default and work only with
`--protocol official`. All three run in [`make_grammar_dfa_generator`](../official/official_protocol.py#L432), and their
options are declared in [`main.py`](../main.py#L1603).

| Flag | Question sent to the model | Reply parsed by | Activates |
|---|---|---|---|
| `--press-question` | [Does the instruction ask about the objects' properties or behaviour](../official/official_protocol.py#L363), "answered by choosing one object and pressing its button"? Yes or No | [`says_yes`](../official/official_protocol.py#L565) | `press_answer_single_press` |
| `--scene-ids` | ["What is the largest number written on any tag?"](../official/official_protocol.py#L373) | [`parse_entity_count`](../official/official_protocol.py#L406) | `scene_ids_bound_targets` |
| `--pattern-first` | [Which of the listed sub-skill patterns solve the instruction?](../official/official_protocol.py#L381) A JSON list | [`parse_patterns`](../official/official_protocol.py#L387) | `chosen_patterns_continue`, `chosen_patterns_complete` |

For how the flags behave, why they stay off and what they scored, see
[Flags that ask the model first](Knowledge_QuickStart.md#flags-that-ask-the-model-first---protocol-official) in the quick
start. [When the information is missing](#when-the-information-is-missing) below covers unusable answers.

The press question's wording names the benchmark's property topics, so it encodes benchmark knowledge. The other two name
no task or topic. The `--scene-ids` answer is only an upper limit on ids and doesn't establish that an object is visible,
which is a separate observation from image perception.

### Image perception

This is one larger prompt per episode, sent with both images and the instruction
([`perception_prompt`](../research/perception.py#L49)). The model acts as a visual observer and doesn't plan.
[`apply_research_arguments`](../research/config.py#L114) decides whether it runs. The full reference is
[IMAGE_PERCEPTION.md](IMAGE_PERCEPTION.md).

The model must reply with one JSON object. This one is made up to show the shape:

```json
{
  "objects": [
    {"id": 4, "confidence": 0.95, "bbox": [0.25, 0.35, 0.28, 0.42], "category": "bottle", "card": null,
     "description": "a pink bottle", "states": {"visible": true}}
  ],
  "requests": [{"action": "pick", "role": "object"}],
  "hypotheses": [{"id": "h0", "bindings": {"object": 4}, "weight": 1.0}],
  "complete": false
}
```

The reply becomes observations in three steps:
1. **Strict parsing** ([`parse_response`](../research/perception.py#L93)): anything but valid JSON with exactly these
   four fields is rejected as a whole. Nothing is partly recovered.
2. **Filtering** ([`compile_evidence`](../research/perception.py#L237)): an object counts only if its confidence reaches
   [`--perception-confidence`](../research/config.py#L86) (0.85 by default) and it reports `visible: true`. A hypothesis that binds any other object is dropped, and the grounding is
   marked incomplete. The confidence is the model's own report and isn't calibrated.
3. **Reading the instruction** ([`perceived_observations`](../official/perception_observations.py#L180)): the instruction
   parsers described in [Instruction text](#instruction-text) match the request against what was perceived.

Each part of the reply feeds different constraints:

| Part of the reply | Observation | Used by |
|---|---|---|
| Confident object ids | largest visible tag | `scene_ids_bound_targets`, `official__scene_ids_bound_targets` |
| `category` | kind of each tag, such as "tomato" | `instruction_candidates_bound_targets`, and the group in the selection rules |
| `bbox` | order of the tags in the image | `ordinal_rank_1..8` |
| `card` | card face, the reading source in use ([`card_observations.py`](../official/card_observations.py)) | `chosen_has_requested_property`, and value ranks |
| `hypotheses`, `requests`, `complete` | which objects fill the instruction's roles | `research__grounding_*`, only with a `--grounding-policy` |
| `states` | open or closed doors | existing door rules on the internal paths; the official decoder keeps no world state |

It can't supply everything. Image boxes give no real-world distance, contact or containment, and the prompt doesn't report
material or mass, so it doesn't feed the physics rules.

**Measured effect:** on the 4,500 official episodes, perception supplied no facts for 856 episodes (19%). Visibility, kinds
and selection together were about neutral (−0.04, with an interval spanning zero), and the selection rules changed only 5
plans. Adding a grounding policy lowered the score, by about 1 point for `joint` and about 5 points for `robust`. These
runs used hard limits and predate the `soft` policy. The full table is in
[IMAGE_PERCEPTION.md](IMAGE_PERCEPTION.md#constraint-ablation-on-the-official-episodes).

| When it runs | |
|---|---|
| `--knowledge official` | automatically |
| [`--constraint-profile research_v1`](../research/config.py#L69) | automatically |
| [`--research-perception image-vlm`](../research/config.py#L83) | when you ask for it, on the paths that allow it |
| `--research-perception off` | never; the official and selection constraints then have nothing to act on |

There are two call paths:
- **Official decoder:** [`observe_episode`](../official/perception_observations.py#L159), called once per episode from
  the generator.
- **Internal paths:** [`enrich_examples`](../research/perception.py#L390), called once per example when the dataset is
  loaded.

Other details:
- **Which model:** the planner's own model by default. Under `--protocol official` it [shares the loaded model](../main.py#L1058),
  and [`--perception-model`](../research/config.py#L85) can name a different one.
- **Second call:** [`--perception-verifier-model`](../main.py#L1629) asks a different model the same prompt. Only what
  both models agree on is used ([`verified_observation`](../official/grounding_policy.py#L30)).
- **Cache:** [`--perception-cache`](../main.py#L1622) reuses saved replies, so a repeated run may not call the model
  again.
- **Prompt content:** besides its fixed instructions and the skill list, the prompt carries only the images and the
  instruction. No entity names, task or simulator state enter it.
- **Metadata facts:** on the internal paths, [`merge_context`](../research/perception.py#L359) replaces the example's
  planning facts with the perception result, so metadata grounded physics and target facts are not mixed with it.

## When the information is missing

Constraints are written so that a missing observation doesn't reject anything. An `ifL` whose condition has no
observation never fires.

| Situation | Result | Implementation |
|---|---|---|
| A question flag is off | its constraint never fires | [`official/decoding_graph.py`](../official/decoding_graph.py#L59) |
| The press answer doesn't start with "yes" | it counts as No, so the plan isn't limited to one press | [`says_yes`](../official/official_protocol.py#L565) |
| `--scene-ids` gets no number | ids keep the grammar's cap | [`parse_entity_count`](../official/official_protocol.py#L406) |
| `--pattern-first` gets no usable pattern | no pattern limit | [`parse_patterns`](../official/official_protocol.py#L387) |
| Perception output is malformed or unavailable | no facts are supplied, so the perception constraints don't fire | [`observe_episode`](../official/perception_observations.py#L159) |
| An object is low-confidence or a selection is unclear | the selection rules abstain instead of restricting | [`official/selection_observations.py`](../official/selection_observations.py) |
| A property value is unknown under `visual` | the property rule never rejects an option | [`visual_qa_planning_facts`](../physics_grounding.py#L43) |

## Hard limit or preference

Perception doesn't always produce a hard limit. `--official-perception-policy` decides:

| Policy | Effect | Default on |
|---|---|---|
| `soft` | the visible tags (so the largest-tag bound) and candidates are cleared, so those constraints don't fire; the candidates only add a bounded score preference ([`soft_observation`](../official/grounding_policy.py#L15)) | [VLABench `evaluate-planner`](../main.py#L1625) |
| `legacy` | the largest visible tag and the candidates reach the constraints as hard limits | commands that take the option from [`research/config.py`](../research/config.py#L94) |

The perception prompt runs under both policies. `soft` also requires official knowledge without hard research grounding.

## Which sources each knowledge level uses

The levels are defined in [`knowledge/__init__.py`](../knowledge/__init__.py).

| `--knowledge` | Sources | Extra LLM prompt by default |
|---|---|---|
| `grammar` | the plan, declared tables | none |
| `causal` | the plan, declared tables | none |
| `generic` | + entity table | none |
| `scene` | + episode configuration and planning facts | none |
| `task` | + task identity and physics/complex observations | none |
| `official` | the plan, declared tables, image perception, instruction text | one perception call per episode |

These defaults assume a standard `baseline` profile. `research_v1` turns perception on at any level, and the question
flags add their own calls.

## Where the constraints are used

Limiting the decoder through the DFA is the main use, but not the only one. It depends on which graph declares the
constraint:

| Constraints | DFA decoding | Other uses |
|---|---|---|
| Official standalone graphs: [`causal_graph.py`](../official/causal_graph.py#L34), [`decoding_graph.py`](../official/decoding_graph.py#L52), [`selection_graph.py`](../official/selection_graph.py#L61) | yes | none: `--protocol official` uses a frozen model, so nothing is trained on them |
| World graph, including its `official__*` rules (the `--knowledge` levels on the internal paths) | yes | plan checking, reward, training loss |
| Research rules (`research__*`, under `research_v1`) | yes | research loss, execution checks |

So a `--protocol official` run uses constraints only to limit what the decoder can write. The upstream VLABench scorer
doesn't read them, so its score shows only their effect on the generated plans.

The other uses are:

| Use | What it does | When | Implementation |
|---|---|---|---|
| Plan checking | builds a DataNode from a finished plan and runs `verifyResultsLC` on every active constraint, separately from the DFA | agent plan decisions, plan rejection during Joint episode collection, structured candidate ranking, dataset audits | [`verify_plan_constraints`](../world_runtime.py#L240), called from [`agent.py`](../simulation/agent.py#L166), [`simulation/program.py`](../simulation/program.py#L1435), [`structured_inference.py`](../../JointEmbodiedAgentInterface/utils/structured_inference.py#L280) and [`audit_planner_lcs.py`](../dataset_tools/gpu2/audit_planner_lcs.py#L17) |
| Reward | a plan scoring below 1.0 on those constraints gets a total reward of 0 | [training with the plan reward](../training.py#L226) | [`score_vlabench_plan`](../reward.py#L355) |
| Primal-dual training loss | penalises the probabilities the model gives to constraint violations | Joint Stage 1, with [`--stage1-dfa-loss on`](../../JointEmbodiedAgentInterface/main.py#L1224) (the default) | [`PrimalDualProgram`](../../JointEmbodiedAgentInterface/program.py#L256) |
| Masked cross entropy | trains on the labels the DFA allows at each step, so it uses the compiled DFA instead of reading the constraints | Joint Stage 1, same flag | [`masked_cross_entropy`](../../JointEmbodiedAgentInterface/program.py#L324) |
| Research loss | penalises probability on labels the compiled policy doesn't admit | `research_v1` training | [`ProjectedPrimalDual`](../research/losses.py#L9) |
| Execution checks | re-check an operation against fresh observations, and the motion before `env.step`, so they can stop a rollout after planning | simulator rollouts | [`admit_with_refresh`](../simulation/agent.py#L279), [`geometry_admission`](../simulation/agent.py#L297) |

Perception can also act without any constraint firing: under the `soft` policy it only changes scores. See
[Hard limit or preference](#hard-limit-or-preference).
