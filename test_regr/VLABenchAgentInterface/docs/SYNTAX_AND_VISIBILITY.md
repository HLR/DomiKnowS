# Syntax and initial-observation visibility

The world graph's `operation_schema_is_valid` formula enforces declared skill
roles, required/forbidden arguments, argument value usability and rejection of
undeclared argument keys. Canonical plan parsing rejects duplicate JSON members
and non-string skill names. Symbolic token encoding validates each operation
before serialization, so an extra argument cannot disappear during encoding.
Existing canonical wrappers and skill-name normalization remain supported.
Native decoding additionally enforces the exact executor JSON format, integer
object ids, punctuation, argument keys, and complete action boundaries. Its grammar
handles tokens spanning multiple JSON boundaries and the supported formatting
and JSON escape variants.

Visibility is a separate logical constraint, `observation_supported_references`.
Every explicitly referenced subject or destination, including optional roles,
must belong to the supplied initial-observation support set. It is active at the
world graph's `scene` and `task` levels and is also declared in the native
`decoding_graph`. Scene inventories, instruction candidates and largest-tag bounds
remain distinct observations; they never create visibility support.

```python
context = {
    "entities": ("apple", "bowl", "cup"),
    "visible_entities": [0, 1],  # Equivalent: ["apple", "bowl"].
    # Other task/planning facts can be included as before.
}
policy = runtime.dfa_for("vlabench", context)
root = materialize_plan(plan, context["entities"], runtime.world_bundle, policy=policy)
result = verify_plan_constraints(root, runtime.world_bundle)

# Standalone native decoding, without requiring symbolic reachability:
raw, tokens = planner.generate(example, mode="constrained", visible_entities=[0, 1])

# Compose visibility with bounded symbolic goal reachability:
raw, tokens = planner.generate(
    example, mode="constrained", max_operations=runtime.max_operations,
    reachability_policy=policy,
)
```

`PlanningExample.visible_entities` is preserved by `as_reward_item()` and consumed
by native generation. Native generation also accepts `visible_entities` directly;
its explicit non-None argument takes precedence over the example field. When both
native evidence and a bound world policy are supplied, both logical programs must
admit a reference. Names are resolved against canonical ids; absent, ambiguous,
boolean or floating-point references in the support set raise `ValueError`.

Missing evidence (`None`) retains the previous behavior and is reported as
`visibility_report.status = "unavailable"` on materialized plans. An explicitly
empty collection establishes an empty support set and rejects all entity
references. This distinction prevents missing perception data from being described
as proof that an inventory entity is visible. The support set stays fixed while
actions are generated, following CLAMP's initial-observation semantics.

For simulator perception with explicitly bound instance segmentation:

```python
from test_regr.VLABenchAgentInterface.visibility import visibility_from_segmentation

visible_ids = visibility_from_segmentation(
    initial_view_label_masks,              # Iterable of 2D integer masks.
    {12: "apple", 77: "bowl", 99: "cup"},  # Segmentation label -> canonical entity.
    context["entities"],
)
context["visible_entities"] = visible_ids
```

The adapter unions pixel-supported ids across initial views. Unobserved bindings
and unbound background/robot labels do not contribute support. For real images,
a perception system must supply canonical id/name bindings; this feature does not
infer identity from tag order, object metadata or a reference plan. Existing
official-protocol runs keep their current behavior unless explicit support is
passed to `KnowledgePlanDFA`.

Run `python -m pytest test_regr/VLABenchAgentInterface/test/test_syntax_visibility.py -q`.
