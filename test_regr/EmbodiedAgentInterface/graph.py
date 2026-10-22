"""EAI token concepts and readable DomiKnowS generation constraints.

Plans start with an action. Object-taking actions are followed by an object;
objects and zero-argument actions are followed by another action or the end
label. Task facts additionally restrict entity-changing actions and cleaning.
Compilation and contextual observation binding use domiknows.generation.
"""

from __future__ import annotations

from test_regr.EmbodiedAgentInterface.dataset import ACTION_VOCAB, EOS_TOKEN, entity_type_for_token


def _disjunction(calls, orL):
    return calls[0] if len(calls) == 1 else orL(*calls)


def _apply_default_constraints(bundle, max_steps, required_tokens, forbidden_tokens):
    """Sequence boundaries, length, required labels and forbidden labels."""
    from domiknows.generation import mark_for_dfa
    from domiknows.graph.logicalConstrain import atLeastAL, atMostAL, ifL, notL

    ctx = bundle.context
    eos_token = bundle.vocabulary.eos_token

    # EOS-closure: once EOS appears at some position, every later position
    # is EOS too.  Same shape as Tasks/hf_generation/graph.py.
    mark_for_dfa(
        ifL(
            ctx.is_before_rel("before"),
            ifL(
                ctx.token_value(eos_token, "x", path=("before", ctx.first_token)),
                ctx.token_value(eos_token, "y", path=("before", ctx.second_token)),
            ),
            name="end_labels_are_closed",
        )
    )

    # ``max_non_eos_count = max_steps - 1`` translates to "at most N
    # positions are not EOS".
    max_non_eos = max(0, int(max_steps) - 1)
    mark_for_dfa(
        atMostAL(notL(ctx.token_value(eos_token, "position")), max_non_eos, name="plan_token_limit")
    )

    # Each required token must appear at least the requested number of
    # times anywhere in the generated sequence.
    for token, count in (required_tokens or {}).items():
        if token not in bundle.vocabulary.tokens:
            continue
        mark_for_dfa(
            atLeastAL(
                ctx.token_value(token, "position"), int(count), name="required_token__" + token
            )
        )

    # Each forbidden token is barred entirely.
    for token in forbidden_tokens or ():
        if token not in bundle.vocabulary.tokens:
            continue
        mark_for_dfa(
            atMostAL(ctx.token_value(token, "position"), 0, name="forbidden_token__" + token)
        )


def _transition_constraint(ctx, triggers, allowed, variable):
    """Build one graph LC for an immediate-successor token-set rule."""
    from domiknows.graph.logicalConstrain import ifL, orL

    trigger_calls = [
        ctx.token_value(token, f"{variable}_trigger_{index}", path=(variable, ctx.first_token))
        for index, token in enumerate(triggers)
    ]
    allowed_calls = [
        ctx.token_value(token, f"{variable}_allowed_{index}", path=(variable, ctx.second_token))
        for index, token in enumerate(allowed)
    ]
    return ifL(
        ctx.is_before_rel(variable),
        ifL(_disjunction(trigger_calls, orL), _disjunction(allowed_calls, orL)),
    )


def _add_action_sequence_constraints(
    graph,
    bundle,
    action_sequence_tokens,
    object_tokens,
    action_object_constraint_tokens,
):
    """Declare the complete EAI plan language as graph logical constraints."""
    from domiknows.generation import mark_for_dfa
    from domiknows.graph.logicalConstrain import andL

    known = set(bundle.vocabulary.tokens)
    actions = tuple(token for token in action_sequence_tokens if token in known)
    zero_argument = tuple(token for token in actions if token in {"sleep", "standup"})
    requiring_objects = tuple(token for token in actions if token not in zero_argument)
    objects = tuple(token for token in object_tokens if token in known)
    if not actions:
        return
    ctx = bundle.context
    with graph:
        start = ctx.starts_with(actions)
        start.name = "plan_starts_with_action"
        mark_for_dfa(start)
        transitions = []
        if requiring_objects and objects:
            transitions.append(
                _transition_constraint(ctx, requiring_objects, objects, "requires_object")
            )
        if objects:
            transitions.append(
                _transition_constraint(
                    ctx,
                    objects,
                    (*actions, bundle.vocabulary.eos_token),
                    "after_object",
                )
            )
        if zero_argument:
            transitions.append(
                _transition_constraint(
                    ctx,
                    zero_argument,
                    (*actions, bundle.vocabulary.eos_token),
                    "after_zero_argument",
                )
            )
        for index, (action, compatible_objects) in enumerate(
            sorted(action_object_constraint_tokens.items())
        ):
            valid_objects = tuple(token for token in compatible_objects if token in objects)
            if action in requiring_objects and valid_objects:
                transitions.append(
                    _transition_constraint(ctx, (action,), valid_objects, f"compatible_{index}")
                )
        if transitions:
            sequence = transitions[0] if len(transitions) == 1 else andL(*transitions)
            sequence.name = "action_argument_sequence"
            mark_for_dfa(sequence)


def _add_contextual_generation_constraints(
    graph,
    bundle,
    object_tokens,
    action_sequence_tokens,
    action_object_constraint_tokens,
):
    """Require available task entities and evidence that cleaning is relevant.

    Contextual markers bind observations from the current example. The formulas
    determine enforcement. These Boolean concepts have no ordinary solver
    sensor, so these two LCs are enforced by the bound generation DFA.
    """
    from domiknows.generation import mark_for_contextual_dfa
    from domiknows.graph import Concept
    from domiknows.graph.logicalConstrain import ifL, orL

    ctx = bundle.context
    known = set(bundle.vocabulary.tokens)
    guarded_objects = tuple(token for token in object_tokens if token in known)
    entity_actions = tuple(
        sorted(token for token in action_object_constraint_tokens if token in known)
    )

    with graph:
        if guarded_objects and entity_actions:
            # Entity types come from the task's transition-model object list.
            # Navigation landmarks remain permitted outside that list.
            entity_available = Concept(name="eai_task_entity_available")
            edge_variable = "task_entity_edge"
            action_predicate = _disjunction(
                [
                    ctx.token_value(
                        token,
                        f"task_entity_action_{index}",
                        path=(edge_variable, bundle.first_token),
                    )
                    for index, token in enumerate(entity_actions)
                ],
                orL,
            )
            object_predicate = _disjunction(
                [
                    ctx.token_value(
                        token,
                        f"task_entity_object_{index}",
                        path=(edge_variable, bundle.second_token),
                    )
                    for index, token in enumerate(guarded_objects)
                ],
                orL,
            )

            object_exists_lc = ifL(
                ctx.is_before_rel(edge_variable),
                ifL(
                    action_predicate,
                    ifL(
                        object_predicate,
                        entity_available("task_entity_object"),
                    ),
                ),
                active=False,
                name="generated_object_exists_in_task_world",
            )

            # For example, bathtub_35 is available when the task lists bathtub.
            mark_for_contextual_dfa(
                object_exists_lc,
                context_key="generation_entity_types",
                token_to_value={token: entity_type_for_token(token) for token in guarded_objects},
                vocabulary=bundle.vocabulary,
                name="generated_object_exists_in_task_world",
                # Some external/custom EAI rows omit a transition model.
                allow_missing_context=True,
                trigger_tokens=entity_actions,
            )

        if "clean" in action_sequence_tokens and "clean" in known:
            # Relevance is declared in utils/task_knowledge.py and grounded
            # from task text and goals, excluding the reference action plan.
            task_action_permitted = Concept(name="eai_task_action_permitted")
            clean_relevant_lc = ifL(
                ctx.token_value("clean", "clean_action"),
                task_action_permitted("clean_action"),
                active=False,
                name="generated_semantic_action_is_task_relevant",
            )

            # An empty permission set forbids cleaning.
            mark_for_contextual_dfa(
                clean_relevant_lc,
                context_key="semantic_action_permissions",
                token_to_value={"clean": "clean"},
                vocabulary=bundle.vocabulary,
                name="generated_semantic_action_is_task_relevant",
                allow_missing_context=False,
            )


def create_generation_graph(
    max_steps=8,
    required_tokens=None,
    forbidden_tokens=None,
    vocab=None,
    object_tokens=None,
    action_tokens=None,
    action_sequence_tokens=None,
    openable_object_tokens=None,
    action_object_constraint_tokens=None,
    enforce_action_object=True,
    enforce_action_object_constraints=True,
):
    from domiknows.generation import GenerationEncoder

    vocab = tuple(vocab or ACTION_VOCAB)
    object_tokens = tuple(object_tokens or ())
    action_tokens = tuple(action_tokens or ())
    action_sequence_tokens = tuple(action_sequence_tokens or action_tokens)
    openable_object_tokens = tuple(openable_object_tokens or ())
    action_object_constraint_tokens = dict(action_object_constraint_tokens or {})
    if openable_object_tokens:
        action_object_constraint_tokens.setdefault("open", openable_object_tokens)

    encoder = GenerationEncoder(
        vocab=list(vocab),
        eos_token=EOS_TOKEN,
        graph_name="eai_generation_graph",
    )
    graph, bundle = encoder.build_graph()

    with graph:
        _apply_default_constraints(
            bundle,
            max_steps=max_steps,
            required_tokens=required_tokens or {},
            forbidden_tokens=forbidden_tokens or [],
        )

    if enforce_action_object:
        _add_action_sequence_constraints(
            graph,
            bundle,
            action_sequence_tokens,
            object_tokens,
            action_object_constraint_tokens if enforce_action_object_constraints else {},
        )
    _add_contextual_generation_constraints(
        graph,
        bundle,
        object_tokens,
        action_sequence_tokens,
        action_object_constraint_tokens,
    )
    return graph, bundle


# Backward-compatible alias for older imports in this folder.
def create_graph(max_steps=8):
    return create_generation_graph(max_steps=max_steps)
