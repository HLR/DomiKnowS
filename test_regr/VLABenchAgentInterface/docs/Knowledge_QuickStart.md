# Knowledge quick start

How to choose what the planner knows (`--knowledge`) and how it is evaluated (`--protocol`). Each rule is a
DomiKnowS logical constraint. The details are in [VLABenchLogicaConstraints.md](VLABenchLogicaConstraints.md),
and the measured effect of each level is in [KNOWLEDGE_ABLATION.md](KNOWLEDGE_ABLATION.md). Where each constraint
gets its information, including which ones need an extra LLM prompt, is in
[Constraint_Observation_Sources.md](Constraint_Observation_Sources.md).

## Three separate settings

| Flag | What it decides | Values |
|---|---|---|
| `--knowledge` | Which constraint groups the decoder enforces (the knowledge) | `grammar`, `causal`, `generic`, `scene`, `task`, `official` |
| `--observation-mode` | Where the physical-property observations of the QA tasks come from | `visual` (default), `metadata-assisted` |
| `--protocol` (VLABench `evaluate-planner` only) | What is evaluated and how it is scored | `internal` (default), `official` |

These flags are independent: `--knowledge official` doesn't select the official protocol, and
`--protocol official` doesn't select the official knowledge. `--observation-mode` changes only where the
observations come from, never which rules are active.

## Knowledge levels

The first five levels are cumulative: each one keeps everything below it.

| Level | Active constraints | Needs |
|---|---|---|
| `grammar` | Plan grammar only: plan ends, skill patterns and arguments, token and operation budgets | nothing |
| `causal` | + causal rules over the ids the plan writes: no self-placement, distinct subtask destinations, a destination is never picked again | nothing |
| `generic` | + entity-type rules: static fixtures are never picked, press needs a button, destinations must be containers, no redundant repeated picks | entity table |
| `scene` | + scene-grounded rules: plans refer only to entities the observation supports | entity table, scene |
| `task` | + task rules: skill sequences, obligations, physics and complex-task knowledge | task identity |
| `official` | `causal` + the `official` group (largest perceived tag, instruction candidates) | image perception (turned on automatically) |

`official` sits outside the chain. Its observations come only from what the benchmark gives:
- the episode's two images;
- its instruction.

It uses no entity table, scene fact or task. It turns image perception (`--research-perception image-vlm`) on, so
training and evaluation see the same observations. Combining it with `--research-perception off` is an error.

The default when `--knowledge` is omitted is `task`, or the value of the environment variable `VLABENCH_KNOWLEDGE`.

### Related flags

- **`--task-identity oracle|predicted`** (at the `task` level) sets where the task comes from:
  - `oracle` is the episode's task name. It is privileged, and it is the default.
  - `predicted` comes from a classifier fitted on the training split, so it also needs `--task-classifier`.
- **Legacy aliases:**
  - `learned` is `task` with `--task-identity predicted`;
  - `metadata` is `task` with `--task-identity oracle`.
- **`--constraints none|grammar|causal`** (only with `--protocol official`) is an older alias of `--knowledge`. Leave
  it out, or give it the same value as `--knowledge`; any other value is a conflict error.

## Observation modes (`--observation-mode`)

The task-level physics rules of the QA tasks choose among the answer options by a property, such as which one is
heaviest or which one sinks. A rule can reject an option only if that property has been observed. The
observation mode decides whether those observations exist and where they come from.

| | `visual` (default) | `metadata-assisted` |
|---|---|---|
| Source of property observations | none: properties stay unknown, so the property rule never rejects an option and the VLM chooses from the images; only the option identities are used (the press must target one of the option buttons) | each option's material and shape from the episode's `env_config` (simulator metadata), plus `property_observations.json` for reflectance when an episode has one; the property rule rejects the losing options |
| Privileged | no | yes: it reads simulator state the model never sees |
| Required level | any | `--knowledge task` (the physics rules live in the task group) |
| Affects | QA (physics) tasks only; other tasks are grounded the same way in both modes | the same |

Restrictions:
- **Which commands accept it:** VLABench `evaluate-planner` and the standalone
  `python -m test_regr.JointEmbodiedAgentInterface.eval_planning`. The Joint `main evaluate-planning` command and
  all training commands don't accept it, and always use `visual`.
- **Rejected combinations:**
  - `metadata-assisted` below the task level, including `--knowledge official` (the physics rules aren't active
    there);
  - `metadata-assisted` with `--protocol official`, which uses the dataset's images and instruction only;
  - `metadata-assisted` with `--generation-mode frozen-vlm`, which has no constraint decoder.
- **Reports:** an evaluation report records the observation mode next to the knowledge level, so the two
  settings can't be confused.

Use `metadata-assisted` only as an upper bound: it shows what the task rules can do with perfect property
observations. It isn't a fair comparison with `visual` runs or with published results.

## Protocols (`evaluate-planner`)

| | `--protocol internal` (default) | `--protocol official` |
|---|---|---|
| Episodes | this repository's seeded split (`--split`) | all 4,500 benchmark episodes |
| Model | trained adapter / checkpoint | frozen base VLM (`--resume-adapter` and `--vocab` are rejected) |
| Inputs | entity table, scene, task (as the level allows) | the two images and the instruction only |
| Scoring | internal metrics | upstream VLABench prompt and scorer |
| Allowed `--knowledge` | any level | `grammar`, `causal` or `official` |
| Comparable to published numbers | no | yes |

Under `--protocol official`, the knowledge must come from `--knowledge` (or its `--constraints` alias). If both are
omitted, the decoder runs with no constraints at all, not even the grammar.

## Flags that ask the model first (`--protocol official`)

Three optional flags make the model answer a short question before it generates the plan. They are off by default,
and each one needs `--protocol official` with `--knowledge grammar`, `causal` or `official`. The questions' wording and
parsers are in [Constraint_Observation_Sources.md](Constraint_Observation_Sources.md#model-question-flags).

| Flag | What the model is asked | What it is shown | Constraint it activates | Effect | If the answer is unusable |
|---|---|---|---|---|---|
| `--press-question` | Is this a choose-one-and-press question? | the episode's input and the question | `press_answer_single_press` | a Yes limits the plan to one press | anything that doesn't start with "yes" counts as No |
| `--scene-ids` | What is the largest tag number? | the question and the number-labeled image | `scene_ids_bound_targets` | object ids stay within that number | no number leaves the limit at the grammar's id cap |
| `--pattern-first` | Which subtask patterns does the plan use? | the episode's input and the question | `chosen_patterns_continue`, `chosen_patterns_complete` | the plan is made of the named patterns | no usable pattern means no pattern limit |

How they work:
1. **Each flag adds one model call per episode.** With all three off, none of these calls is made.
2. **The flag supplies an observation, not a rule.** The constraints in `official/decoding_graph.py` are always
   compiled, but each one fires only when its observation exists, so without the flag it never restricts anything.
3. **The answer limits the decoder, not the prompt.** The plan is generated from the same episode input as a normal
   run, and the answer only changes which tokens the DFA allows.
4. **`--pattern-first` is skipped after a Yes** to the press question, because the plan is already one press.

The answers are the model's own guesses, but the DFA enforces them as hard limits. A wrong Yes, for example, leaves
one press as the only possible plan.

Leave these flags off for a standard run, because they make it a variant of the protocol. The press question's
wording was also chosen on validation episodes and names the benchmark's question-task topics, so report any run that
uses it as a variant. They have added little so far:
- **`--press-question`:** recognised 127 of 600 question tasks and never chose the right button, so the official
  score barely moved.
- **`--scene-ids`:** changed no plan on the 450 validation episodes (38.18 for the grammar alone).
- **`--pattern-first`:** scored 38.29 on the same episodes, with 39 of 450 replies unusable.

The full results are in the [README](../README.md).

## Constraints in the prompt, no DFA (`--prompt-constraints`)

`--prompt-constraints [grammar|causal|official]` is a baseline: the frozen VLM reads a level's constraints in plain
English and generates freely. With no value, the level is `official`. Nothing enforces the rules, so the model can
ignore them, and its replies are scored as written.

```bash
python -m test_regr.VLABenchAgentInterface.main evaluate-planner --planning-dir test_regr/VLABenchAgentInterface/data/planning --protocol official --prompt-constraints
```

```bash
python -m test_regr.VLABenchAgentInterface.main evaluate-planner --planning-dir test_regr/VLABenchAgentInterface/data/planning --split test --no-dfa-official --prompt-constraints
```

Without `--protocol official` or `--no-dfa-official`, the frozen prompt lists scene object names such as
`stone_cylinder`, so the scores include that [knowledge leak](../../JointEmbodiedAgentInterface/docs/KNOWLEDGE_LEAKAGE.md).
`--no-dfa-official` replaces the names with numbered objects and keeps this repository's split and scoring. The number
of listed objects still comes from the scene configuration. It runs on the frozen VLM and can't be combined with
`--protocol official`, which shows no object list at all.

| Level | Rules stated in the prompt |
|---|---|
| `grammar` | end of plan, operation limit, skill parameters, subtask patterns |
| `causal` | + no self-placement, distinct subtask destinations, a destination is never picked again |
| `official` | + ids at most the largest visible tag, ids match what the instruction names, ordinal choice, requested property |

- **No DFA or perception:** the run adds no model calls, so flags that enforce or observe rules are rejected:
  `--knowledge`, `--constraints`, `--press-question`, `--scene-ids`, `--pattern-first`, `--hmm` and the formula and
  perception options. Without `--protocol official`, `--generation-mode` defaults to `frozen-vlm`, and `graph` is
  rejected.
- **Operation limit:** under `--protocol official` the text states 16, the limit that protocol's DFA uses, and
  `--max-operations` doesn't change it. With `--generation-mode frozen-vlm` the text states `--max-operations`.
- **What is left out:** the press, scene-id and chosen-pattern constraints, because they act only on the answers from
  the [flags that ask the model first](#flags-that-ask-the-model-first---protocol-official).
- **Official protocol:** the rules follow the unchanged official prompt, so the report is marked `standard=false` and
  isn't comparable with published numbers.
- **Reports:** the report records the level, the exact text and its hash. Under `--protocol official`, resuming an
  output directory with different text is an error.

The text is in [`official/prompt_constraints.py`](../official/prompt_constraints.py). Each rule names the logical
constraints it words, and a test checks those names against the declared graphs.

## Which flags do I need?

**I want to train and evaluate with the official knowledge.** Pass `--knowledge official` to both training and
evaluation. You don't need `--protocol official`: the internal protocol evaluates the trained model with the
official constraints.

```bash
python -m test_regr.JointEmbodiedAgentInterface.main train-agent ... --knowledge official
```

```bash
python -m test_regr.JointEmbodiedAgentInterface.main evaluate-planning --checkpoint <ckpt> --planning-dir test_regr/VLABenchAgentInterface/data/planning --split test --knowledge official --output out/official.json
```

The standalone `python -m test_regr.JointEmbodiedAgentInterface.eval_planning` takes the same flags.

**I want numbers comparable to the published VLABench results.** Pass both flags:

```bash
python -m test_regr.VLABenchAgentInterface.main evaluate-planner --planning-dir test_regr/VLABenchAgentInterface/data/planning --protocol official --knowledge official
```

**I want the task rules with perfect property observations (an upper bound).** Use the standalone evaluator with
the task level:

```bash
python -m test_regr.JointEmbodiedAgentInterface.eval_planning --checkpoint <ckpt> --planning-dir test_regr/VLABenchAgentInterface/data/planning --split test --knowledge task --task-identity oracle --observation-mode metadata-assisted --output out/metadata.json
```

The legacy alias `--knowledge metadata` means `task` with `--task-identity oracle`. It doesn't set the observation
mode, so you still need `--observation-mode metadata-assisted`.

**I want to compare knowledge levels on one checkpoint (an ablation).** Evaluate the same checkpoint once per level,
each time with an explicit `--knowledge`. An explicit level changes only the decoding constraints, not the trained
weights.

## Checkpoints and levels

A Joint checkpoint records the constraint configuration it was trained with.
- **Without `--knowledge`,** evaluation uses the default level (`task`) and checks that it matches the checkpoint.
  A checkpoint trained with `--knowledge official` therefore fails this check.
- **With an explicit `--knowledge`,** evaluation runs as a deliberate ablation, and that check is skipped.

So always pass `--knowledge` when you evaluate, and use the same level as training unless you mean to run an
ablation.

## In code

```python
from test_regr.VLABenchAgentInterface.knowledge import set_default_level, group_active

set_default_level("official")          # graphs built afterwards use this level
group_active("official", "causal")     # True
group_active("official", "task")       # False
```

`set_knowledge_level(bundle, level)` in `world_graph.py` switches an already-built world graph between levels.
