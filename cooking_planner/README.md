# Cooking Planner Graph-HMM Demo

This task demonstrates a declarative DomiKnowS planning graph.  The graph in
`graph.py` is the human-facing domain definition: concepts, subconcepts,
relations, domain facts, enum action labels, and logical constraints are all
declared together.

The execution layer is derived from that graph:

- a hard planning DFA checks fridge/table/phase/count/required-action rules;
- a `DomiKnowSAwareHMM` learns acceptable compact action plans from
  graph-declared reference plans;
- a `GraphHMMGenerationHead` shows the same graph-HMM structure as a
  PMD-compatible Torch module.

Run the offline demo:

```powershell
uv run --project Tasks/cooking_planner python Tasks/cooking_planner/run_demo.py --dish cookie --show-invalid
```

Supported dishes are `cookie`, `omelette`, and `salad`.

## Human Declaration

The graph declares:

- planning concepts: `plan`, `step`, `dish`, `action`, `plan_phase`;
- action subconcepts: `open_fridge`, `take_eggs`, `close_fridge`,
  `put_on_table`, `mix_dough`, `bake_cookies`, `serve`, `done`, and others;
- dish subconcepts: `cookie`, `omelette`, `salad`;
- phase subconcepts: `start`, `fridge_open`, `after_fridge`, `table_ready`,
  `prep`, `cook`, `served`, `done_phase`;
- domain fact relations such as `dish_requires_action`,
  `reference_plan_step`, `phase_transition`, and `action_count_limit`;
- logical constraints such as terminal-action closure, max non-terminal
  length, max fridge opens, dish requirements, and basic dependencies.

No Python `ACTIONS`, `DISH_SPECS`, or phase tables configure the domain outside
the graph.  Small local helpers in `graph.py` only reduce repeated DomiKnowS
syntax; the domain facts they create are still graph concepts and relations.

## Generated Execution Layer

`domiknows.generation.applications.planning` reads the graph and derives:

- the ordered action vocabulary from `planned_action`;
- task requirements from `dish_requires_action`;
- reference plans from `reference_plan_step`;
- HMM masks from `phase_transition`;
- a hard planning DFA from phase transitions, count limits, terminal action,
  and required actions.

The DFA is the hard verifier.  The HMM learns and scores acceptable plan
dynamics; it does not replace the hard graph-derived verifier.

## What The HMM Learns

The graph declares the action labels, hidden plan phases, allowed phase
transitions, and hard constraints.  The graph-HMM learns probabilities inside
that graph-shaped space:

```text
P(action_t | hidden_phase_t, G, C)
P(hidden_phase_t+1 | hidden_phase_t, G, C)
```

So it learns how acceptable cooking plans usually flow, such as moving from
`fridge_open` through food-taking actions into `after_fridge`, `table_ready`,
`prep`, `cook`, and `served`.  Impossible graph paths remain impossible; they
do not receive a small learned probability.

The DFA remains the hard verifier.  The HMM scores and reranks candidate plans,
and Viterbi decoding explains the most likely hidden phase path for a plan.
`GraphHMMGenerationHead` is the PMD-compatible trainable Torch version of this
same graph-HMM idea.

```text
DomiKnowS graph + DFA = hard validity
Graph-HMM = learned plan-flow likelihood
GraphHMMGenerationHead = PMD-compatible Torch learner
```
