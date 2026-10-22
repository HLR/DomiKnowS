# EAI logical constraints

Inventory checked on **2026-10-22**, following [VLABenchLogicaConstraints.md](../../VLABenchAgentInterface/docs/VLABenchLogicaConstraints.md). This covers generation, world preconditions, explicit requirements, abstract action effects, task relevance and SimpleTL satisfaction. EAI has its own activation settings; VLABench ablation levels do not automatically apply to it.

## Counting and baseline

One logical constraint is one complete formula. `ifL(A, andL(B, C))` counts as **one**; nested operators are not additional constraints. Concepts, relations, observations, DFA states and evaluations are not declarations. A grounded instance applies one existing formula to a token pair, action or requirement. Instance counts depend on the example.

The enabled default world graph has **62 constraints: 54 action preconditions + 8 grounded families**, over **67 action types and 37 state predicates**. `build_eai_world_graph()` alone defaults to zero constraints; pass `include_default_constraints=True`, as the ordinary program does unless custom builders are supplied.

For a concrete configured generation baseline, use vocabulary `(<eos>, open, close, clean, sleep, cup, door)`, objects `(cup, door)`, actions `(open, close, clean, sleep)`, and compatibility pairs `open→door`, `close→door`, `clean→cup`. With the default sequence options and no required/forbidden entries, this declares **6 constraints**, giving **68 generation/world constraints** together. The bare `create_generation_graph()` without action/object arguments declares just its two boundary/length rules.

<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
<colgroup><col style="width: 27%;"><col style="width: 18%;"><col style="width: 55%;"></colgroup>
<thead><tr><th style="text-align: left; padding: 0.5em; overflow-wrap: anywhere;">Declaration group</th><th style="text-align: left; padding: 0.5em; overflow-wrap: anywhere;">Count per build</th><th style="text-align: left; padding: 0.5em; overflow-wrap: anywhere;">Operators and use</th></tr></thead>
<tbody>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Configured generation</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">6 + R + F</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Four ifL, one andL, one atMostAL. R/F count in-vocabulary required/forbidden entries; they add atLeastAL/atMostAL rules.</td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">World preconditions</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">54</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">ifL, expanded over applicable action concepts.</td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Grounded world requirements</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">8</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">ifL, one per family; each required fact is a separate observation node.</td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Abstract effects</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">26</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">ifL in a separate transition graph used for simulation; these do not add world-score penalties.</td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Cleaning evidence</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">equivalenceL in a separate task-feature graph.</td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">SimpleTL operator</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1 per (kind, arity, count)</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">equivalenceL in separately cached goal graphs; actual shapes depend on parsed goals/entity domains.</td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Legacy scalar projection</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">5 copied templates</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Compatibility copies of the same five precondition formulas; no distinct knowledge or additional active world rules.</td></tr>
</tbody>
</table>

Effects, task features and goal templates are separate computational graphs, so their counts are not added to the 68 generation/world total. There is no fixed process-wide count: optional rules, custom builders and requested goal templates vary. Vocabulary labels, serialization and checkpoint parameter shapes are preserved.

## Reading the formulas

Names outside quotes are concepts/operators; names in quotes bind variables. `right_place_inside("operation")` classifies the action node assigned to the variable `operation`; it does not refer to a token literally named operation. Reusing the variable means the same node.

World action aliases below mean `world.actions[name]`; `observed(name)` means `world.facts[name]("operation")`. Actual concept names use `action__` and `observed__` prefixes. For generation, `ctx = generation.context`, `end = generation.vocabulary.eos_token`; `ctx.token_value` resolves a surface label to its enum concept. `first_token` and `second_token` select endpoints of the ordering relation. Marked successor formulas compile as immediate-successor rules.

Primitive observations supply entity membership, signed state facts, hand contents, matching relation tuples and numeric measurements. Formulas determine the logical combinations. Unknown facts remain unknown and are reported. Rule names are labels, not policy dispatch. Recompilation reads formula edits/removal/activation; previously compiled snapshots retain their prior behavior.

## Generation families

<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
<colgroup><col style="width: 23%;"><col style="width: 8%;"><col style="width: 47%;"><col style="width: 22%;"></colgroup>
<thead><tr><th style="text-align: left; padding: 0.5em; overflow-wrap: anywhere;">Family</th><th style="text-align: left; padding: 0.5em; overflow-wrap: anywhere;">Count</th><th style="text-align: left; padding: 0.5em; overflow-wrap: anywhere;">Activation and example</th><th style="text-align: left; padding: 0.5em; overflow-wrap: anywhere;">Defined at</th></tr></thead>
<tbody>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">End-label closure</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Always. After end, another action is rejected; repeated end labels remain legal.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../graph.py#L35"><code>graph.py:35</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(ctx.is_before_rel(&quot;edge&quot;),
    ifL(ctx.token_value(end, &quot;first&quot;, path=(&quot;edge&quot;, ctx.first_token)),
        ctx.token_value(end, &quot;second&quot;, path=(&quot;edge&quot;, ctx.second_token))),
    name=&quot;end_labels_are_closed&quot;)</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Token limit</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Always. max_steps=8 permits at most seven non-end labels.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../graph.py#L43"><code>graph.py:43</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>atMostAL(notL(ctx.token_value(end, &quot;position&quot;)), max(0, max_steps - 1),
         name=&quot;plan_token_limit&quot;)</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Required label</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">R</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Supplied in-vocabulary entries. Example: require two open labels.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../graph.py#L53"><code>graph.py:53</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>atLeastAL(ctx.token_value(&quot;open&quot;, &quot;position&quot;), 2, name=&quot;required_token__open&quot;)</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Forbidden label</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">F</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Supplied in-vocabulary entries. Example: prohibit clean.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../graph.py#L62"><code>graph.py:62</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>atMostAL(ctx.token_value(&quot;clean&quot;, &quot;position&quot;), 0, name=&quot;forbidden_token__clean&quot;)</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Start with an action</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">0 or 1</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">enforce_action_object; nonempty action set. An object cannot start the plan.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../graph.py#L104"><code>graph.py:104</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(ctx.sequence_start(&quot;start&quot;),
    orL(*(ctx.token_value(action, &quot;first&quot;, path=(&quot;start&quot;, ctx.sequence_start_token))
          for action in actions)), name=&quot;plan_starts_with_action&quot;)</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Action/argument sequence and compatibility</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">0 or 1</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">enforce_action_object. Object-taking action→object; object→action/end; sleep/standup→action/end. Supplied compatibility narrows objects when enforce_action_object_constraints is enabled. All branches form one complete conjunction.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../graph.py#L84"><code>graph.py:84</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code># Complete example for actions=(open, sleep), objects=(door,).
andL(
    ifL(ctx.is_before_rel(&quot;argument_edge&quot;),
        ifL(ctx.token_value(&quot;open&quot;, &quot;action&quot;, path=(&quot;argument_edge&quot;, ctx.first_token)),
            ctx.token_value(&quot;door&quot;, &quot;object&quot;, path=(&quot;argument_edge&quot;, ctx.second_token)))),
    ifL(ctx.is_before_rel(&quot;object_edge&quot;),
        ifL(ctx.token_value(&quot;door&quot;, &quot;object&quot;, path=(&quot;object_edge&quot;, ctx.first_token)),
            orL(*(ctx.token_value(label, &quot;next&quot;, path=(&quot;object_edge&quot;, ctx.second_token))
                  for label in (&quot;open&quot;, &quot;sleep&quot;, end))))),
    ifL(ctx.is_before_rel(&quot;zero_argument_edge&quot;),
        ifL(ctx.token_value(&quot;sleep&quot;, &quot;action&quot;, path=(&quot;zero_argument_edge&quot;, ctx.first_token)),
            orL(*(ctx.token_value(label, &quot;next&quot;, path=(&quot;zero_argument_edge&quot;, ctx.second_token))
                  for label in (&quot;open&quot;, &quot;sleep&quot;, end))))),
    ifL(ctx.is_before_rel(&quot;compatibility_edge&quot;),
        ifL(ctx.token_value(&quot;open&quot;, &quot;action&quot;, path=(&quot;compatibility_edge&quot;, ctx.first_token)),
            ctx.token_value(&quot;door&quot;, &quot;object&quot;, path=(&quot;compatibility_edge&quot;, ctx.second_token)))),
    name=&quot;action_argument_sequence&quot;)</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Task entity availability</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">0 or 1</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Context binding when object labels and guarded actions exist. bathtub_35 requires bathtub in generation_entity_types after a guarded action. Navigation landmarks are excluded from the guard. Missing entity context is allowed.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../graph.py#L197"><code>graph.py:197</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code># Example: guarded action open and object door.
ifL(ctx.is_before_rel(&quot;task_entity_edge&quot;),
    ifL(ctx.token_value(&quot;open&quot;, &quot;action&quot;, path=(&quot;task_entity_edge&quot;, ctx.first_token)),
        ifL(ctx.token_value(&quot;door&quot;, &quot;object&quot;, path=(&quot;task_entity_edge&quot;, ctx.second_token)),
            entity_available(&quot;task_entity_object&quot;))),
    name=&quot;generated_object_exists_in_task_world&quot;, active=False)</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Cleaning relevance guard</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">0 or 1</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Context binding when clean is present. An empty semantic_action_permissions set forbids clean.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../graph.py#L226"><code>graph.py:226</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(ctx.token_value(&quot;clean&quot;, &quot;clean_action&quot;), task_action_permitted(&quot;clean_action&quot;),
    name=&quot;generated_semantic_action_is_task_relevant&quot;, active=False)</code></pre></td></tr>
</tbody>
</table>

The contextual concepts alias `eai_task_entity_available` and `eai_task_action_permitted`. Their ordinary solver activation is false because no ordinary sensor fills them; the task-bound DFA enforces these same formulas from context. Markers select observation bindings. The baseline does not add an unconditional “an end label occurs” rule; termination/acceptance are runtime behavior. A verified explicitly complete planning goal can permit an empty continuation in the grounded adapter.

## World preconditions

<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
<colgroup><col style="width: 23%;"><col style="width: 8%;"><col style="width: 47%;"><col style="width: 22%;"></colgroup>
<thead><tr><th style="text-align: left; padding: 0.5em; overflow-wrap: anywhere;">Family</th><th style="text-align: left; padding: 0.5em; overflow-wrap: anywhere;">Count</th><th style="text-align: left; padding: 0.5em; overflow-wrap: anywhere;">Activation and example</th><th style="text-align: left; padding: 0.5em; overflow-wrap: anywhere;">Defined at</th></tr></thead>
<tbody>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Arguments belong to task entities</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">26</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Defaults enabled. clean(bathtub_35) fails against only book/door types. Arg2 is checked if supplied. The action schema permits at most two arguments.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../world_graph.py#L162"><code>world_graph.py:162</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(andL(world.actions[&quot;clean&quot;](&quot;operation&quot;), observed(&quot;task_entities_supplied&quot;)),
    andL(
                observed(&quot;first_argument_supplied&quot;),
                observed(&quot;first_argument_in_task&quot;),
                ifL(observed(&quot;second_argument_supplied&quot;), observed(&quot;second_argument_in_task&quot;)),
            ),
    name=&quot;action_precondition__clean__argument_available&quot;)</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Placement source</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">17</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Defaults enabled. Known right-hand state must hold a source, or a matching placement must already exist at the destination (idempotence).</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../world_graph.py#L170"><code>world_graph.py:170</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(andL(world.actions[&quot;right_place_inside&quot;](&quot;operation&quot;), observed(&quot;source_state_known&quot;)),
    orL(
                observed(&quot;selected_hand_occupied&quot;), observed(&quot;matching_placement_exists&quot;)
            ),
    name=&quot;action_precondition__right_place_inside__source_holding&quot;)</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Release source</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">5</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Defaults enabled. Require a held source or an existing placement relation for this object.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../world_graph.py#L176"><code>world_graph.py:176</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(andL(world.actions[&quot;right_release&quot;](&quot;operation&quot;), observed(&quot;source_state_known&quot;)),
    orL(
                observed(&quot;selected_hand_occupied&quot;), observed(&quot;matching_release_relation_exists&quot;)
            ),
    name=&quot;action_precondition__right_release__source_holding&quot;)</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Pour source</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Defaults enabled. A known source state with both hands empty cannot pour.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../world_graph.py#L182"><code>world_graph.py:182</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(andL(world.actions[&quot;pour&quot;](&quot;operation&quot;), observed(&quot;source_state_known&quot;)),
    orL(observed(&quot;left_hand_occupied&quot;), observed(&quot;right_hand_occupied&quot;)),
    name=&quot;action_precondition__pour__source_holding&quot;)</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Container access</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">5</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Defaults enabled. Open/not-closed permits insertion; closed/not-open blocks it, including contradictory observations. Unknown access is inapplicable.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../world_graph.py#L185"><code>world_graph.py:185</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(andL(
                world.actions[&quot;right_place_inside&quot;](&quot;operation&quot;),
                observed(&quot;destination_supplied&quot;),
                orL(
                    observed(&quot;destination_observed_open&quot;),
                    observed(&quot;destination_observed_not_closed&quot;),
                    observed(&quot;destination_observed_closed&quot;),
                    observed(&quot;destination_observed_not_open&quot;),
                ),
            ),
    andL(
                orL(
                    observed(&quot;destination_observed_open&quot;),
                    observed(&quot;destination_observed_not_closed&quot;),
                ),
                notL(observed(&quot;destination_observed_closed&quot;)),
                notL(observed(&quot;destination_observed_not_open&quot;)),
            ),
    name=&quot;action_precondition__right_place_inside__destination_open_if_known&quot;)</code></pre></td></tr>
</tbody>
</table>

Expansion over action concepts is defined at <a href="../world_graph.py#L98"><code>world_graph.py:98</code></a>. Both fast evaluation and DataNode verification compile actual formulas and read their implication guards for applicability. Compatibility scalar fields are projections of copies of these same declarations, and do not decide default enforcement.

With no planning_facts, legacy timeline snapshots are authoritative source states: no hold tuple means an empty hand. With explicit planning facts, source_state_known instead records whether the selected hand was observed; an unobserved hand is not fabricated as empty. Signed destination access remains guarded in both paths.

## Explicitly grounded requirements

Each of the eight families declares **one constraint**, active when defaults are enabled. Each supplied obligation becomes its own requirement node, so one true capability cannot hide another false precondition. `family_requirement` aliases `world.grounded_constraints.families[family]`; `observation_known` and `fact_holds` alias its known/holds concepts. Input aliases such as hand_known refer to `world.grounded_constraints.inputs[name]`. Unknown facts are guarded and remain visible in grounding reports.

<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
<colgroup><col style="width: 22%;"><col style="width: 8%;"><col style="width: 48%;"><col style="width: 22%;"></colgroup>
<thead><tr><th style="text-align: left; padding: 0.5em; overflow-wrap: anywhere;">Family</th><th style="text-align: left; padding: 0.5em; overflow-wrap: anywhere;">Count</th><th style="text-align: left; padding: 0.5em; overflow-wrap: anywhere;">Knowledge/example</th><th style="text-align: left; padding: 0.5em; overflow-wrap: anywhere;">Defined at</th></tr></thead>
<tbody>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">destination</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Use the explicitly requested destination: apple→bowl rejects apple→cooler.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../../../domiknows/generation/grounded.py#L102"><code>grounded.py:102</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(andL(destination_requirement(&quot;requirement&quot;), observation_known(&quot;requirement&quot;)),
    fact_holds(&quot;requirement&quot;), name=&quot;grounded__destination&quot;)</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">holding</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Hold the required subject in the observed hand; idempotent operations need no new grasp.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../../../domiknows/generation/grounded.py#L90"><code>grounded.py:90</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(andL(holding_requirement(&quot;requirement&quot;),
         orL(idempotent(&quot;requirement&quot;), hand_known(&quot;requirement&quot;))),
    orL(idempotent(&quot;requirement&quot;), andL(held_exists(&quot;requirement&quot;),
        orL(subject_unspecified(&quot;requirement&quot;), held_matches_subject(&quot;requirement&quot;)))),
    name=&quot;grounded__holding&quot;)</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">capacity</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Zero capacity prevents grasping. Positive capacity also requires that known occupancy does not conflict with this subject.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../../../domiknows/generation/grounded.py#L96"><code>grounded.py:96</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(andL(capacity_requirement(&quot;requirement&quot;), capacity_known(&quot;requirement&quot;),
         orL(capacity_zero(&quot;requirement&quot;), hand_known(&quot;requirement&quot;))),
    andL(capacity_positive(&quot;requirement&quot;),
         notL(andL(hand_known(&quot;requirement&quot;), held_exists(&quot;requirement&quot;),
                   notL(held_matches_subject(&quot;requirement&quot;))))),
    name=&quot;grounded__capacity&quot;)</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">affordance</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Every supplied capability, precondition and measured selection must hold; openable=False blocks open.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../../../domiknows/generation/grounded.py#L102"><code>grounded.py:102</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(andL(affordance_requirement(&quot;requirement&quot;), observation_known(&quot;requirement&quot;)),
    fact_holds(&quot;requirement&quot;), name=&quot;grounded__affordance&quot;)</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">compatibility</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Respect supplied subject/destination compatibility and container requirements.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../../../domiknows/generation/grounded.py#L102"><code>grounded.py:102</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(andL(compatibility_requirement(&quot;requirement&quot;), observation_known(&quot;requirement&quot;)),
    fact_holds(&quot;requirement&quot;), name=&quot;grounded__compatibility&quot;)</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">milestone_order</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Satisfy declared predecessor milestones before the dependent operation.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../../../domiknows/generation/grounded.py#L102"><code>grounded.py:102</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(andL(milestone_order_requirement(&quot;requirement&quot;), observation_known(&quot;requirement&quot;)),
    fact_holds(&quot;requirement&quot;), name=&quot;grounded__milestone_order&quot;)</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">goal_completion</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">At completion, establish every supplied signed goal and required milestone.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../../../domiknows/generation/grounded.py#L102"><code>grounded.py:102</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(andL(goal_completion_requirement(&quot;requirement&quot;), observation_known(&quot;requirement&quot;)),
    fact_holds(&quot;requirement&quot;), name=&quot;grounded__goal_completion&quot;)</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">preservation</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Preserve protected entities and supplied invariants. Observation does not itself modify an entity.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../../../domiknows/generation/grounded.py#L102"><code>grounded.py:102</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(andL(preservation_requirement(&quot;requirement&quot;), observation_known(&quot;requirement&quot;)),
    fact_holds(&quot;requirement&quot;), name=&quot;grounded__preservation&quot;)</code></pre></td></tr>
</tbody>
</table>

The formulas are shared with VLABench in `domiknows/generation/grounded.py`. `grounded_facts.py` parses requests, measures facts and supplies independent observations. Immediate requirements use operation events; goal requirements use completion events. A family score averages applicable observation groundings; configured mean/min/product then combines applicable family scores. Hard grounded decoding requires every checked formula to hold. Unknown/vacuous groundings do not inflate the applicable count. Per-requirement percentages can differ from the old aggregate Boolean while preserving hard validity.

## Abstract action effects

These **26 constraints** describe the established abstract simulator, including its legacy conventions. They compute state transitions rather than add 26 penalties to world compliance. They run whenever simulation runs, independently of --no-world-constraints. In the formulas below, `input_fact(name) = effects.inputs[name]("transition")` and `result_fact(name) = effects.results[name]("transition")`. Positive result literals establish bound facts; negative literals remove them. Unmentioned facts persist. Previous-content/location bindings exclude the newly established tuple, so an idempotent update does not assert and negate the same fact. Conflicting declared writes fail explicitly.

<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
<colgroup><col style="width: 25%;"><col style="width: 7%;"><col style="width: 46%;"><col style="width: 22%;"></colgroup>
<thead><tr><th style="text-align: left; padding: 0.5em; overflow-wrap: anywhere;">Rule family</th><th style="text-align: left; padding: 0.5em; overflow-wrap: anywhere;">Count</th><th style="text-align: left; padding: 0.5em; overflow-wrap: anywhere;">Knowledge/example</th><th style="text-align: left; padding: 0.5em; overflow-wrap: anywhere;">Defined at</th></tr></thead>
<tbody>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">grasp_effects</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Grasp apple: replace selected-hand contents and record matching requested negative spatial relations.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../utils/state_effects.py#L123"><code>state_effects.py:123</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(
            andL(input_fact(&quot;grasp_action&quot;), input_fact(&quot;argument_supplied&quot;)),
            andL(
                notL(result_fact(&quot;selected_hand_contents&quot;)),
                result_fact(&quot;held_argument&quot;),
                result_fact(&quot;negative_spatial_goal&quot;),
                notL(result_fact(&quot;positive_spatial_goal&quot;)),
            ),
            name=&quot;grasp_effects&quot;,
        )</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">placement_effects</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Establish the placement relation, clear its negation and release the source hold.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../utils/state_effects.py#L133"><code>state_effects.py:133</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(
            andL(input_fact(&quot;placement_action&quot;), input_fact(&quot;argument_supplied&quot;), input_fact(&quot;source_exists&quot;)),
            andL(
                result_fact(&quot;placement&quot;),
                notL(result_fact(&quot;negative_placement&quot;)),
                notL(result_fact(&quot;source_hand_contents&quot;)),
            ),
            name=&quot;placement_effects&quot;,
        )</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">placement_replaces_location</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">A non-nextto placement replaces previous spatial locations of this source.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../utils/state_effects.py#L138"><code>state_effects.py:138</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(
            andL(andL(input_fact(&quot;placement_action&quot;), input_fact(&quot;argument_supplied&quot;), input_fact(&quot;source_exists&quot;)), notL(input_fact(&quot;placement_nextto&quot;))),
            notL(result_fact(&quot;previous_placement&quot;)),
            name=&quot;placement_replaces_location&quot;,
        )</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">placement_on_top_touches</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">On-top placement also establishes touching. A room_floor destination normalizes to onfloor.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../utils/state_effects.py#L143"><code>state_effects.py:143</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(
            andL(andL(input_fact(&quot;placement_action&quot;), input_fact(&quot;argument_supplied&quot;), input_fact(&quot;source_exists&quot;)), input_fact(&quot;placement_ontop&quot;)),
            result_fact(&quot;touching&quot;),
            name=&quot;placement_on_top_touches&quot;,
        )</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">compound_placement_nextto</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">The compound nextto_ontop action also establishes nextto.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../utils/state_effects.py#L148"><code>state_effects.py:148</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(
            andL(andL(input_fact(&quot;placement_action&quot;), input_fact(&quot;argument_supplied&quot;), input_fact(&quot;source_exists&quot;)), input_fact(&quot;compound_nextto&quot;)),
            result_fact(&quot;nextto&quot;),
            name=&quot;compound_placement_nextto&quot;,
        )</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">legacy_unbound_placement</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Legacy reward may create relation(object,destination) with no known source. Observed-state decoding removes this placeholder.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../utils/state_effects.py#L158"><code>state_effects.py:158</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(
            andL(
                input_fact(&quot;placement_action&quot;),
                input_fact(&quot;argument_supplied&quot;),
                notL(input_fact(&quot;source_exists&quot;)),
            ),
            result_fact(&quot;placement&quot;),
            name=&quot;legacy_unbound_placement&quot;,
        )</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">release_effects</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Establish released and matching legacy onfloor goals; clear holds of the released source.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../utils/state_effects.py#L163"><code>state_effects.py:163</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(
            andL(input_fact(&quot;release_action&quot;), input_fact(&quot;source_exists&quot;)),
            andL(result_fact(&quot;released&quot;), result_fact(&quot;floor_goal&quot;), notL(result_fact(&quot;source_hand_contents&quot;))),
            name=&quot;release_effects&quot;,
        )</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">cleaning_effects</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Establish clean/not_dusty/not_stained; remove dusty/stained.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../utils/state_effects.py#L176"><code>state_effects.py:176</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(
            andL(input_fact(&quot;clean_action&quot;), input_fact(&quot;argument_supplied&quot;)),
            andL(
                result_fact(&quot;clean&quot;),
                result_fact(&quot;not_dusty&quot;),
                result_fact(&quot;not_stained&quot;),
                notL(result_fact(&quot;dusty&quot;)),
                notL(result_fact(&quot;stained&quot;)),
            ),
            name=&quot;cleaning_effects&quot;,
        )</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">washing_effects</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Wash/rinse also establish washed and rinsed.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../utils/state_effects.py#L181"><code>state_effects.py:181</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(
            andL(input_fact(&quot;wash_action&quot;), input_fact(&quot;argument_supplied&quot;)),
            andL(result_fact(&quot;washed&quot;), result_fact(&quot;rinsed&quot;)),
            name=&quot;washing_effects&quot;,
        )</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">opening_effects</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Establish open/not_closed; remove closed/not_open.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../utils/state_effects.py#L188"><code>state_effects.py:188</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(
            andL(input_fact(&quot;open_action&quot;), input_fact(&quot;argument_supplied&quot;)),
            andL(
                result_fact(&quot;open&quot;), result_fact(&quot;not_closed&quot;), notL(result_fact(&quot;closed&quot;)), notL(result_fact(&quot;not_open&quot;))
            ),
            name=&quot;opening_effects&quot;,
        )</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">closing_effects</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Establish closed/not_open; remove open/not_closed.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../utils/state_effects.py#L195"><code>state_effects.py:195</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(
            andL(input_fact(&quot;close_action&quot;), input_fact(&quot;argument_supplied&quot;)),
            andL(
                result_fact(&quot;closed&quot;), result_fact(&quot;not_open&quot;), notL(result_fact(&quot;open&quot;)), notL(result_fact(&quot;not_closed&quot;))
            ),
            name=&quot;closing_effects&quot;,
        )</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">switch_on_effects</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Establish on; remove off/not_on.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../utils/state_effects.py#L200"><code>state_effects.py:200</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(
            andL(input_fact(&quot;turn_on_action&quot;), input_fact(&quot;argument_supplied&quot;)),
            andL(result_fact(&quot;on&quot;), notL(result_fact(&quot;off&quot;)), notL(result_fact(&quot;not_on&quot;))),
            name=&quot;switch_on_effects&quot;,
        )</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">switch_off_effects</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Establish off/not_on; remove on.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../utils/state_effects.py#L205"><code>state_effects.py:205</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(
            andL(input_fact(&quot;turn_off_action&quot;), input_fact(&quot;argument_supplied&quot;)),
            andL(result_fact(&quot;off&quot;), result_fact(&quot;not_on&quot;), notL(result_fact(&quot;on&quot;))),
            name=&quot;switch_off_effects&quot;,
        )</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">freezing_effects</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Establish frozen; remove not_frozen.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../utils/state_effects.py#L222"><code>state_effects.py:222</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(
            andL(input_fact(&quot;freeze_action&quot;), input_fact(&quot;argument_supplied&quot;)),
            andL(result_fact(&quot;frozen&quot;), notL(result_fact(&quot;not_frozen&quot;))),
            name=&quot;freezing_effects&quot;,
        )</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">unfreezing_effects</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Establish not_frozen; remove frozen.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../utils/state_effects.py#L227"><code>state_effects.py:227</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(
            andL(input_fact(&quot;unfreeze_action&quot;), input_fact(&quot;argument_supplied&quot;)),
            andL(result_fact(&quot;not_frozen&quot;), notL(result_fact(&quot;frozen&quot;))),
            name=&quot;unfreezing_effects&quot;,
        )</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">plugging_in_effects</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Establish plugged_in; remove not_plugged_in.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../utils/state_effects.py#L232"><code>state_effects.py:232</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(
            andL(input_fact(&quot;plugin_action&quot;), input_fact(&quot;argument_supplied&quot;)),
            andL(result_fact(&quot;plugged_in&quot;), notL(result_fact(&quot;not_plugged_in&quot;))),
            name=&quot;plugging_in_effects&quot;,
        )</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">unplugging_effects</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Establish not_plugged_in; remove plugged_in.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../utils/state_effects.py#L237"><code>state_effects.py:237</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(
            andL(input_fact(&quot;plugout_action&quot;), input_fact(&quot;argument_supplied&quot;)),
            andL(result_fact(&quot;not_plugged_in&quot;), notL(result_fact(&quot;plugged_in&quot;))),
            name=&quot;unplugging_effects&quot;,
        )</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">pouring_effects</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Establish inside(source,destination), retaining the hold.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../utils/state_effects.py#L242"><code>state_effects.py:242</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(
            andL(input_fact(&quot;pour_action&quot;), input_fact(&quot;argument_supplied&quot;), input_fact(&quot;source_exists&quot;)),
            result_fact(&quot;poured_inside&quot;),
            name=&quot;pouring_effects&quot;,
        )</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">navigation_effects</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Walk/run establish near and character inside/nextto facts in this abstract model.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../utils/state_effects.py#L252"><code>state_effects.py:252</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(
            andL(input_fact(&quot;walk_action&quot;), input_fact(&quot;argument_supplied&quot;)),
            andL(
                result_fact(&quot;near_object&quot;),
                result_fact(&quot;near_character&quot;),
                result_fact(&quot;character_inside&quot;),
                result_fact(&quot;character_nextto&quot;),
            ),
            name=&quot;navigation_effects&quot;,
        )</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">looking_effects</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Turnto/lookat/watch establish facing(character,argument).</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../utils/state_effects.py#L257"><code>state_effects.py:257</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(
            andL(input_fact(&quot;look_action&quot;), input_fact(&quot;argument_supplied&quot;)),
            result_fact(&quot;facing&quot;),
            name=&quot;looking_effects&quot;,
        )</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">sitting_effects</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Establish sitting(character) and ontop(character,argument).</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../utils/state_effects.py#L262"><code>state_effects.py:262</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(
            andL(input_fact(&quot;sit_action&quot;), input_fact(&quot;argument_supplied&quot;)),
            andL(result_fact(&quot;sitting&quot;), result_fact(&quot;character_ontop&quot;)),
            name=&quot;sitting_effects&quot;,
        )</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">lying_effects</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Establish lying(character) and ontop(character,argument).</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../utils/state_effects.py#L267"><code>state_effects.py:267</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(
            andL(input_fact(&quot;lie_action&quot;), input_fact(&quot;argument_supplied&quot;)),
            andL(result_fact(&quot;lying&quot;), result_fact(&quot;character_ontop&quot;)),
            name=&quot;lying_effects&quot;,
        )</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">slice_effects</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Establish slice result on the argument.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../utils/state_effects.py#L208"><code>state_effects.py:208</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(andL(input_fact(&quot;slice_action&quot;), input_fact(&quot;argument_supplied&quot;)), result_fact(&quot;sliced&quot;), name=&quot;slice_effects&quot;)</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">soak_effects</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Establish soak result on the argument.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../utils/state_effects.py#L208"><code>state_effects.py:208</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(andL(input_fact(&quot;soak_action&quot;), input_fact(&quot;argument_supplied&quot;)), result_fact(&quot;soaked&quot;), name=&quot;soak_effects&quot;)</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">cook_effects</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Establish cook result on the argument.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../utils/state_effects.py#L208"><code>state_effects.py:208</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(andL(input_fact(&quot;cook_action&quot;), input_fact(&quot;argument_supplied&quot;)), result_fact(&quot;cooked&quot;), name=&quot;cook_effects&quot;)</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">touch_effects</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Establish touch result on the argument.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../utils/state_effects.py#L208"><code>state_effects.py:208</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>ifL(andL(input_fact(&quot;touch_action&quot;), input_fact(&quot;argument_supplied&quot;)), result_fact(&quot;touch&quot;), name=&quot;touch_effects&quot;)</code></pre></td></tr>
</tbody>
</table>

For example, result_fact("open") binds the tuple `("open", argument)`; result_fact("source_hand_contents") binds all matching source hold tuples. Action groups are ontology classifications: clean/wipe/scrub/wash/rinse, on/off aliases, navigation, and grasp/release/placement names. <a href="../utils/effect_runtime.py#L11"><code>effect_runtime.py:11</code></a> resolves entity identities and matching tuples. <a href="../utils/effect_runtime.py#L198"><code>effect_runtime.py:198</code></a> executes the compiled implications without using rule names to select effects.

## Task relevance

<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
<colgroup><col style="width: 23%;"><col style="width: 7%;"><col style="width: 48%;"><col style="width: 22%;"></colgroup>
<thead><tr><th style="text-align: left; padding: 0.5em; overflow-wrap: anywhere;">Family</th><th style="text-align: left; padding: 0.5em; overflow-wrap: anywhere;">Count</th><th style="text-align: left; padding: 0.5em; overflow-wrap: anywhere;">Activation/example</th><th style="text-align: left; padding: 0.5em; overflow-wrap: anywhere;">Defined at</th></tr></thead>
<tbody>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Cleaning evidence</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Task context preparation. “Wash the cup” permits clean; “Read the book” does not. Only task_name, natural_language_description, original_goal, tl_goal and transition_model are inspected. Reference actions are excluded.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../utils/task_knowledge.py#L30"><code>task_knowledge.py:30</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code># mentions_* alias the concepts in cues; clean_permitted is the output concept.
equivalenceL(clean_permitted(&quot;task&quot;),
    orL(mentions_clean(&quot;task&quot;), mentions_dust(&quot;task&quot;), mentions_lint(&quot;task&quot;),
        mentions_mop(&quot;task&quot;), mentions_polish(&quot;task&quot;), mentions_rinse(&quot;task&quot;),
        mentions_scrub(&quot;task&quot;), mentions_stain(&quot;task&quot;), mentions_wash(&quot;task&quot;), mentions_wipe(&quot;task&quot;)),
    name=&quot;clean_requires_task_evidence&quot;)</code></pre></td></tr>
</tbody>
</table>

Cue observations preserve case-insensitive substring matching. New task text changes concept values rather than adding rules. This equivalence determines clean permission; the generation implication consumes it.

## SimpleTL satisfaction

Parsing supplies expression structure and the entity domain. Atomic observations test state facts, action identity/arguments and entity-type membership. Boolean composition, finite quantification and temporal witness admissibility use the equations below. Each requested `(kind, arity, count)` template declares **one** equivalence. Repeated evaluation reuses templates, with a cache of at most 256 shapes. Concepts below alias the corresponding result/operand/constant concepts returned by build_goal_operator.

<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
<colgroup><col style="width: 19%;"><col style="width: 12%;"><col style="width: 47%;"><col style="width: 22%;"></colgroup>
<thead><tr><th style="text-align: left; padding: 0.5em; overflow-wrap: anywhere;">Family</th><th style="text-align: left; padding: 0.5em; overflow-wrap: anywhere;">Count</th><th style="text-align: left; padding: 0.5em; overflow-wrap: anywhere;">Knowledge/example</th><th style="text-align: left; padding: 0.5em; overflow-wrap: anywhere;">Defined at</th></tr></thead>
<tbody>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">not</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1 per template</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Negate an atomic or compound subgoal.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../utils/goal_constraints.py#L27"><code>goal_constraints.py:27</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>equivalenceL(goal_satisfied(&quot;evaluation&quot;),
    notL(operand_0_holds(&quot;evaluation&quot;)),
    name=&quot;goal_not_satisfaction&quot;)</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">and</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1 per template</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">All subgoals must hold at this snapshot. Legacy final-goal success also uses this conjunction over correlated required-fact observations and a nonempty-goal observation.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../utils/goal_constraints.py#L27"><code>goal_constraints.py:27</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>equivalenceL(goal_satisfied(&quot;evaluation&quot;),
    andL(operand_0_holds(&quot;evaluation&quot;), operand_1_holds(&quot;evaluation&quot;)),
    name=&quot;goal_and_satisfaction&quot;)</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">or</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1 per template</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">At least one subgoal must hold at this snapshot.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../utils/goal_constraints.py#L27"><code>goal_constraints.py:27</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>equivalenceL(goal_satisfied(&quot;evaluation&quot;),
    orL(operand_0_holds(&quot;evaluation&quot;), operand_1_holds(&quot;evaluation&quot;)),
    name=&quot;goal_or_satisfaction&quot;)</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">forall</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1 per template</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">One operand per entity in the finite grounded domain; all must satisfy the body.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../utils/goal_constraints.py#L27"><code>goal_constraints.py:27</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>equivalenceL(goal_satisfied(&quot;evaluation&quot;),
    andL(operand_0_holds(&quot;evaluation&quot;), operand_1_holds(&quot;evaluation&quot;)),
    name=&quot;goal_forall_satisfaction&quot;)</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">exists</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1 per template</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">One operand per entity; at least one must satisfy the body.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../utils/goal_constraints.py#L27"><code>goal_constraints.py:27</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>equivalenceL(goal_satisfied(&quot;evaluation&quot;),
    orL(operand_0_holds(&quot;evaluation&quot;), operand_1_holds(&quot;evaluation&quot;)),
    name=&quot;goal_exists_satisfaction&quot;)</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">forn</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1 per template</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Exactly N entities satisfy the body; this example uses N=1 over two entities.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../utils/goal_constraints.py#L27"><code>goal_constraints.py:27</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>equivalenceL(goal_satisfied(&quot;evaluation&quot;),
    exactL(operand_0_holds(&quot;evaluation&quot;), operand_1_holds(&quot;evaluation&quot;), 1),
    name=&quot;goal_forn_satisfaction&quot;)</code></pre></td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">temporal_witness</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">1 per template</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Operand 0 is term truth; operand 1 observes whether the snapshot precedes the allowed start. For then, the next start is previous witness index + 1, requiring strictly later snapshots.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../utils/goal_constraints.py#L27"><code>goal_constraints.py:27</code></a></td></tr>
<tr><td colspan="4"><pre style="white-space: pre-wrap; overflow-wrap: anywhere;"><code>equivalenceL(goal_satisfied(&quot;evaluation&quot;),
    andL(operand_0_holds(&quot;evaluation&quot;), notL(operand_1_holds(&quot;evaluation&quot;))),
    name=&quot;goal_temporal_witness_satisfaction&quot;)</code></pre></td></tr>
</tbody>
</table>

Empty conjunctions/universal domains use true_observation; empty disjunctions/existentials use notL(true_observation). Exact counting over an empty domain is true exactly for N=0. Typed syntax `not TYPE(x) or body` first grounds the entity domain. Witness search scans the finite timeline and selects the earliest admissible index. Quantification over temporal bodies uses the same Boolean/counting equations. `open(cup) then closed(cup)` requires distinct ordered snapshots; both in one snapshot are insufficient. Numerical temporal-prefix fractions, recall and reward discounts are scoring, not additional logical constraints.

## Activation and new knowledge

<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
<colgroup><col style="width: 25%;"><col style="width: 39%;"><col style="width: 36%;"></colgroup>
<thead><tr><th style="text-align: left; padding: 0.5em; overflow-wrap: anywhere;">Setting/input change</th><th style="text-align: left; padding: 0.5em; overflow-wrap: anywhere;">What changes</th><th style="text-align: left; padding: 0.5em; overflow-wrap: anywhere;">Example effect</th></tr></thead>
<tbody>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">generation_constraints=always</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Standalone EAI applies token/context DFA during training and evaluation. Joint EAI additionally binds the compiled world predicate monitor.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Invalid token successors are masked; joint grounded decoding additionally rejects observed world violations.</td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">generation_constraints=eval</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Standalone token/context enforcement applies only at evaluation; declarations stay the same.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Training can sample an invalid sequence that constrained evaluation rejects.</td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">generation_constraints=off</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Standalone generation enforcement is bypassed; world reward evaluation is independently configured.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Invalid syntax can still receive a low semantic reward.</td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Default world rules enabled</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">54 preconditions plus 8 grounded families.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Known closed destination or wrong held source violates the applicable formula.</td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">--no-world-constraints / empty custom builders</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Default world rules are omitted. Effect and goal evaluation still run.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">World reward discount disappears; opening still changes abstract state.</td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Custom builders supplied</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Caller constraints replace defaults unless the graph builder explicitly enables both.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">A caller may declare a state mutex using ordinary DomiKnowS syntax.</td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">New explicit planning facts</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Same family rules; changed known/truth values and new requirement nodes.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Adding openable=False turns unknown affordance into a known violation.</td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">New goals/milestones</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Same world families; new operation/completion requirement observations.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Adding open(cooler) blocks premature completion in grounded decoding.</td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">New task text/entity list</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Same relevance/context formulas; changed observations.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">A cleaning cue permits clean; adding bathtub permits bathtub_35 after guarded actions.</td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">New SimpleTL expression</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Instantiate/reuse requested operator templates.</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Changing exists to exactly-one uses exactL, so two matches no longer satisfy it.</td></tr>
</tbody>
</table>

EAI does not inherit VLABench’s grammar/generic/scene/learned/task/metadata ladder. See [KNOWLEDGE_ABLATION.md](../../VLABenchAgentInterface/docs/KNOWLEDGE_ABLATION.md) for that separate domain. Additional input knowledge normally changes facts or grounded requirements, rather than adding world-rule classes.

## Where knowledge enters

<table style="width: 100%; table-layout: fixed; border-collapse: collapse;">
<colgroup><col style="width: 28%;"><col style="width: 27%;"><col style="width: 45%;"></colgroup>
<thead><tr><th style="text-align: left; padding: 0.5em; overflow-wrap: anywhere;">Knowledge source</th><th style="text-align: left; padding: 0.5em; overflow-wrap: anywhere;">Binding/declaration</th><th style="text-align: left; padding: 0.5em; overflow-wrap: anywhere;">How new input changes enforcement</th></tr></thead>
<tbody>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Action/object compatibility learned from examples</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../main.py#L541"><code>main.py:541</code></a></td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Supplied allowed-object sets expand the branches inside the existing generation conjunction; additional training pairs change those sets.</td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Task entities from PDDL objects</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../dataset.py#L249"><code>dataset.py:249</code></a></td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Object type membership changes contextual observations; no per-example policy rule is invented.</td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Explicit scene/goal/requirement observations</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../utils/constraint_compiler.py#L77"><code>constraint_compiler.py:77</code></a></td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">PlanningFacts supplies known/unknown signed observations and per-requirement nodes for the same eight family rules.</td></tr>
<tr><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Legacy correlated benchmark goal grounding</td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;"><a href="../reward.py#L597"><code>reward.py:597</code></a></td><td style="vertical-align: top; padding: 0.5em; overflow-wrap: anywhere;">Reference-derived initial/type/goal inputs remain part of the legacy reward baseline, as qualified below.</td></tr>
</tbody>
</table>

## Source responsibilities and limits

`world_graph.py` declares world ontology and rules and retains compatibility exports. `utils/world_domain.py` holds action/state vocabulary, arity, aliases and semantic categories: ontology records, not extra constraints. `utils/world_facts.py` supplies source observations. `utils/constraint_compiler.py` binds identities and emits operation, requirement and completion events. `utils/world_runtime.py` materializes DataNodes and evaluates actual defaults; arbitrary custom constraints retain the general DomiKnowS verifier. `graph.py` declares token/context rules; effect, task and goal declaration modules above expose the other knowledge.

The legacy reward preparation in `reward.py` still obtains initial facts, type synonyms and correlated goal groundings from reference trajectories. Non-temporal benchmark success retains grounded final-fact recall; ordered goals use SimpleTL satisfaction. Dense reward is recall times ordered-prefix progress, optionally discounted by world compliance. Decoding’s explicit PlanningFacts path does not initialize from that reference-derived PreparedEAIGoal. Declarative effects do not make the abstract simulator an independently observed physical world or repair historical reference-grounding assumptions. This refactor did not recompute historical training/ablation scores.

No new blanket state mutex or action-effect penalties were enabled in the world graph. The established action model is declarative in its separate transition graph. Parsing, entity resolution, tuple enumeration, numeric measurements/ranking, timeline traversal, serialization, caches and reward arithmetic remain runtime work; formulas govern the logical combinations for admissibility, effects and satisfaction.

## Validation

`test/test_declarative_knowledge.py` covers independent Boolean/count/container-access truth tables, strict temporal order, correlated effect updates, actual DataNode observations and formula mutation/removal. Tests demonstrate changes in decoding, fast evaluation and DataNode verification without rule-name dispatch. Existing generation/reward/hand-continuation tests remain applicable. A deterministic comparison against the pre-refactor simulator covered **3,350 cases across all 67 registered actions**, varied arguments, hand holdings and legacy goals. Local tests use no training, simulator assets or dataset downloads; existing tests may read cached benchmark data.
