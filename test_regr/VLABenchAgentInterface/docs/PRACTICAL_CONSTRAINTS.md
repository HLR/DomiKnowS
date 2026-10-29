# Articulated goals and benchmark-compatible poker

## Domain and checkpoint compatibility

The default internal domain is schema version 4: eleven skills, adding
`close_door(target_entity_name)` and `open_laptop(target_entity_name)`. Opening a
door, closing it and opening a laptop have standalone patterns. Closing establishes
`closed` and removes `open`, including door aliases; it preserves held objects.
The controller has twelve skill-conditioning rows including its padding row.

The general native planner API and planning CLI default to `grammar_profile="world"`.
This profile supports the eleven skills and standalone articulation. Like the
benchmark JSON interface, native `push` explicitly names its destination; the
internal symbolic domain also permits omission of that optional role. Explicit
`benchmark` and `legacy` profiles retain their original nine-skill languages, and
the official protocol continues to select `benchmark` explicitly.

The vocabulary/domain checksum changes. Old nine-skill checkpoints are rejected
by the expanded runtime with a retraining message; weights are not silently
remapped. Existing checkpoint serialization formats are unchanged. New checkpoints
and new controller training are required before deploying the added skills.

## Articulation inputs and logical goals

At knowledge level `task`, `cool_drink` requires the requested drink inside the
requested fridge and the fridge closed. Both `set_study_table` and its local alias
`manage_study_table` require the selected book outside its original container and
the laptop joint inside the benchmark's inclusive `[1.2, 1.5]` interval.
These tasks use final-state requirements, not mandatory extra actions when their
goals are already satisfied.

```python
context = {
    "entities": ("book", "shelf", "table", "laptop"),
    "task": "set_study_table",
    "task_target_entity": "book",
    "planning_facts": {
        "initial_state": {"positive": [["inside", "book", "shelf"]]},
        "articulation_task": {"source_container": "shelf", "laptop": "laptop"},
        "articulation_targets": [
            {"entity": "laptop", "joint": "hinge", "lower": 1.2, "upper": 1.5}
        ],
        "joint_observations": [{"entity": "laptop", "joint": "hinge", "value": 0.3}],
    },
}
policy = runtime.dfa_for("vlabench", context)
```

Use the actual named joint from the observed entity; `hinge` is illustrative.
For cooling, supply `task_target_entity` and `task_target_container`, or bind
`articulation_task.subject` and `.container`. The study task also accepts
`articulation_task.subject`. Missing task/entity/joint bindings remain unknown.
Capabilities use the existing `entity_properties` observations: `openable` for
opening and `closeable` for closing. Strict reachability requires these and other
applicable capabilities to be known.

`articulation.py` declares four relayed logical rules for drink containment,
fridge closure, book removal and joint targets. They feed the existing
`grounded__goal_completion` family. `open_laptop` predicts its declared target
contract; this prediction is not a measured joint angle or a simulator success.

The live agent calls `refreshed_articulation_context` before each replan. It replaces
joint/door observations with actual simulator measurements and discards stale
predicted range/open/closed facts. A missing sensor yields unknown evidence.
The authoritative success check remains the simulator task evaluator.

## Poker inputs and logical selection

Supply `planning_facts.poker_selection` for `texas_holdem` or
`texas_holdem_explore`. It contains an ordered list of all scene cards, a
destination, and optionally `order_known` (default true). Use explicit entries
with unknown faces for cards whose identity is unavailable; do not omit them or
decode their identities from metadata names.

```python
selection = {
    "cards": [
        {"entity": "card0", "rank": "ace", "suit": "spades", "face_observed": True},
        # Remaining scene cards in the exposed original order (five to seven total).
        {"entity": "face_down_card", "face_observed": False},
    ],
    "destination": "mat",
    "order_known": True,
}
# This incomplete example remains unknown until all required card faces are observed.
```

Ranks accept integers 2..14 or upstream names (`2`..`10`, `jack`, `queen`, `king`,
`ace`). Suits are `spades`, `clubs`, `diamonds`, `hearts`. Entity ids and names are
resolved against the supplied inventory. Hidden winning-card lists are not accepted.
Perception supplies the complete ordered inventory; this adapter cannot detect an
object that perception never reports. Missing faces/order/destination remain
unknown rather than being guessed.

The adapter enumerates five-card candidates and observes suit equality, rank
adjacency and multiplicities. Ten compiled logical classification queries feed
eleven relayed goal rules that choose the highest category and earliest tied
candidate. Python does not select the winner. Required-card containment is
checked at completion; extra destination cards are permitted by the upstream
condition. A high-card result requires one card, a pair two, trips three, and so on.

Semantics are pinned to OpenMOSS/VLABench
`cf588fe60c0c7282174fe979f5913170cfe69017`, `poker_utils.py`. Its first-candidate
tie behavior and ace-low-straight-flush-as-royal-flush quirk are intentionally
preserved. These rules do not implement standard kicker tie-breaking.
The ten classification queries are auxiliary computations, like relay
applies/decided views, not ten mutually incompatible hard plan demands.

## Validation and evaluation limits

Run `test_articulated_goals.py` and `test_poker_knowledge.py` under pytest. The
poker fixture records 32 differential cases produced by the pinned upstream
helper, including its source SHA-256; independent fixtures cover every category,
ace-low cases, order-dependent ties and variable-sized target groups.

Optional simulator measurement checks require installed VLABench assets and
`VLABENCH_ARTICULATION_SIM_TESTS=1`. They check measured joint-goal satisfaction,
not trained physical execution of the new controller skills. No benchmark scores
or historical ablation results are updated by this feature.
