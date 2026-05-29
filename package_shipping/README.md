# Package Shipping Planner Graph-HMM Demo

This task is a second declarative planning domain used to verify that
`domiknows.generation.applications.planning` is not cooking-specific.

The human-facing graph in `graph.py` declares package-shipping concepts,
subconcepts, action labels, phase transitions, task requirements, reference
plans, count limits, and logical constraints.  The execution layer derives the
planning DFA and graph-HMM masks from those graph constructs.

Run the offline demo:

```powershell
uv run --project Tasks/package_shipping python Tasks/package_shipping/run_demo.py --task ship_fragile_vase --show-invalid
```

Supported tasks are `ship_book`, `ship_fragile_vase`, and `return_item`.

## Human Declaration

The graph declares:

- planning concepts: `plan`, `step`, `shipping_task`, `action`, `plan_phase`;
- actions such as `choose_box`, `wrap_item`, `add_padding`, `insert_item`,
  `print_label`, `print_return_label`, `seal_box`, `drop_off`,
  `request_pickup`, and `done`;
- task concepts: `ship_book`, `ship_fragile_vase`, `return_item`;
- phase concepts: `start`, `box_ready`, `item_protected`, `item_inserted`,
  `labeled`, `sealed`, `shipped`, `done_phase`;
- domain fact relations such as `task_requires_action`,
  `reference_plan_step`, `phase_transition`, and `action_count_limit`;
- logical constraints for terminal closure, max plan length, sealing at most
  once, fragile padding, return labels, and delivery after sealing.

## Execution Layer

`planning_bundle_from_graph(...)` is called with custom schema names so the
domain can use `shipping_task`, `planned_shipping_task`, and
`task_requires_action` instead of cooking-specific names.

The DFA is the hard verifier.  The HMM learns and scores compact action
dynamics from the graph-declared reference plans.

## What The HMM Learns

The graph declares the action labels, hidden plan phases, allowed phase
transitions, and hard constraints.  The graph-HMM learns probabilities inside
that graph-shaped space:

```text
P(action_t | hidden_phase_t, G, C)
P(hidden_phase_t+1 | hidden_phase_t, G, C)
```

So it learns how acceptable shipping plans usually flow, such as moving from
`box_ready` through protection or insertion, into `labeled`, `sealed`,
`shipped`, and `done_phase`.  Impossible graph paths remain impossible; they do
not receive a small learned probability.

The DFA remains the hard verifier.  The HMM scores and reranks candidate plans,
and Viterbi decoding explains the most likely hidden phase path for a plan.
`GraphHMMGenerationHead` is the PMD-compatible trainable Torch version of this
same graph-HMM idea.

```text
DomiKnowS graph + DFA = hard validity
Graph-HMM = learned plan-flow likelihood
GraphHMMGenerationHead = PMD-compatible Torch learner
```
