# Action-budget goal reachability

`WorldActionReachability` searches complete operations using the request-bound
world policy. Every candidate operation is stepped through its compiled grammar
and logical constraints; the existing semantics applies preconditions and effects.
Acceptance uses the same EOS and completion constraints as plan verification.
The search retains object identity, state facts, pattern progress, milestones,
and the history needed by the active rules. It does not infer goals from reference
plans or success annotations.

```python
from test_regr.VLABenchAgentInterface.training import build_constraint_runtime

runtime = build_constraint_runtime(max_entities=3, max_operations=3)
context = {
    "entities": ("apple", "bowl", "cup"),
    "task_skill_sequence": ["pick", "place"],
    "planning_facts": {
        "initial_state": {"held": {"right": None}},
        "capacities": {"right": 1},
        "entity_properties": {"apple": {"graspable": True}},
        "destinations": {"apple": "bowl"},
        "compatibility": [
            {"subject": "apple", "destination": "bowl", "allowed": True}
        ],
        "goals": {"positive": [["at", "apple", "bowl"]]},
    },
}
policy = runtime.dfa_for("vlabench", context)
search = policy.action_reachability(strict=True, max_state_expansions=10000)
assert search.can_reach(remaining_actions=2) is True
assert search.can_reach(remaining_actions=1) is False

# A loaded FrozenNativePlanner can use this exact episode policy:
raw, tokens = planner.generate(
    example, mode="constrained", max_operations=3,
    reachability_policy=policy,
)
```

The example's entities must match the bound policy, and its native action budget
must not exceed the compiled world's bound. `ReachablePlanDFA` also accepts a bound
policy directly for native decoding outside `FrozenNativePlanner`. Native syntax,
instruction candidates, chosen patterns and enabled causal rules all participate
in its search. An argument token survives only if a compatible complete action
has a proven feasible continuation. World state advances at action boundaries;
formatting consumes no actions. An empty plan is allowed only when the world
policy accepts it and native completion requirements hold.

Queries return `True` for a witness, `False` for exhausted possibilities, and
`None` for missing required grounding. Strict mode excludes unknown applicable
requirements of active grounded LC families. `strict=False` preserves the world's
existing permissive treatment of unknown observations; it is not a proof of fully
grounded executability. Without supplied goals, completion means only the active
policy's declared completion requirements, not independently established task
success. Goals and effects must be supplied from the permitted episode inputs.

The expansion cap raises `SemanticSearchLimitExceeded`; it never returns or
caches an approximate impossibility result. Native generation raises
`NativePlanFailure` when there is no proven continuation, distinguishing missing
observations in its message. Search caches belong to one bound episode/policy;
rebind after changing facts or logical constraints. Search can be expensive with
many objects and long horizons; the explicit cap bounds state expansions.

This feature is Boolean action reachability. Existing HMM guidance still ranks
tokens inside the hard feasible set. It does not add CLAMP's separate exclusion
of zero-HMM-mass continuations. The existing native token horizon remains separate
from the action horizon. Official-protocol runs have no bound symbolic world/goal
inputs: entity names, goals, destinations, capacities and properties are metadata or
task knowledge, which that protocol does not expose. Image perception does not supply
them either; it reports visible tags, predicted object kinds and role hypotheses. So
official decoding keeps only the comma-level reachability of its grammar, causal and
perception constraints (`CausalPlanDFA.reachable`), not this world search.

Validation: `python -m pytest test_regr/VLABenchAgentInterface/test/test_action_reachability.py -q`
