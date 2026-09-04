"""Graph builder for the EmbodiedAgentInterface baseline.

Rebuilt against the current  ``domiknows.generation`` API:

* ``GenerationEncoder`` + ``encoder.build_graph()`` (no constraint-list arg).
* Raw DomiKnowS LCs (``ifL``, ``atMostAL``, ``atLeastAL``, ``notL``,
  ``orL``) are written directly inside ``with graph:`` and discovered by
  :func:`domiknows.generation.constraints_to_dfa_from_graph` at decode time.
* The ``ActionFollowedByObjectConstraint`` helper subclass that used to
  carry a hand-rolled ``to_dfa`` is no longer needed — the multi-token
  ``ifL(rel, ifL(orL(actions), orL(objects)))`` shape it built is now
  natively recognised by ``_path_token_predicate_from_flat`` and compiled
  to the same DFA fragment.
* ``mark_for_dfa(lc)`` takes a single argument (the LC); the second
  positional ``constraint`` argument the old API accepted was dropped when
  enforcement was unified.
"""
from __future__ import annotations

try:
    from .dataset import ACTION_VOCAB, EOS_TOKEN, entity_type_for_token
except ImportError:
    from dataset import ACTION_VOCAB, EOS_TOKEN, entity_type_for_token


def _disjunction(calls, orL):
    return calls[0] if len(calls) == 1 else orL(*calls)


def _apply_default_constraints(bundle, max_steps, required_tokens, forbidden_tokens):
    """Translate the legacy ``default_generation_constraints`` knobs to raw LCs."""
    from domiknows.generation import mark_for_dfa
    from domiknows.graph.logicalConstrain import atLeastAL, atMostAL, ifL, notL

    ctx = bundle.context
    eos_token = bundle.vocabulary.eos_token

    # EOS-closure: once EOS appears at some position, every later position
    # is EOS too.  Same shape as Tasks/hf_generation/graph.py.
    mark_for_dfa(ifL(
        ctx.is_before_rel("before"),
        ifL(
            ctx.token_value(eos_token, "x", path=("before", ctx.first_token)),
            ctx.token_value(eos_token, "y", path=("before", ctx.second_token)),
        ),
    ))

    # ``max_non_eos_count = max_steps - 1`` translates to "at most N
    # positions are not EOS".
    max_non_eos = max(0, int(max_steps) - 1)
    mark_for_dfa(atMostAL(notL(ctx.token_value(eos_token, "x")), max_non_eos))

    # Each required token must appear at least the requested number of
    # times anywhere in the generated sequence.
    for token, count in (required_tokens or {}).items():
        if token not in bundle.vocabulary.tokens:
            continue
        mark_for_dfa(atLeastAL(ctx.token_value(token, "x"), int(count)))

    # Each forbidden token is barred entirely.
    for token in forbidden_tokens or ():
        if token not in bundle.vocabulary.tokens:
            continue
        mark_for_dfa(atMostAL(ctx.token_value(token, "x"), 0))


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
        mark_for_dfa(ctx.starts_with(actions))
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
                    _transition_constraint(
                        ctx, (action,), valid_objects, f"compatible_{index}"
                    )
                )
        if transitions:
            mark_for_dfa(transitions[0] if len(transitions) == 1 else andL(*transitions))


def _add_contextual_generation_constraints(
    graph,
    bundle,
    object_tokens,
    action_sequence_tokens,
    action_object_constraint_tokens,
):
    """Declare task-conditioned policies.

    The two formulas declared below are:

    1. ``semantic_action(x) and immediately_next(x, y) and object(y)``
       ``-> eai_task_entity_available(y)``
    2. ``clean(x) -> eai_task_action_permitted(x)``

    ``mark_for_contextual_dfa`` does not define either formula. It only
    records how the Boolean conclusions are grounded from one example's
    context when the already-compiled DFA is specialized.
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
            # Boolean property of an object-token position. For a concrete
            # task it is true exactly when that label's normalized entity type
            # occurs in the task's ``generation_entity_types`` context fact.
            entity_available = Concept(name="eai_task_entity_available")

            # ``task_entity_edge`` binds the two endpoints of the generation
            # ordering relation. The generation DFA compiler interprets this
            # trigger/second-token form as an immediate-successor policy:
            #
            #   entity-changing action --immediately followed by--> object
            #
            # Navigation/perception actions are intentionally absent from
            # ``entity_actions`` because their scene landmarks need not occur
            # in a task's PDDL ``:objects`` list.
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

            # Declarative DomiKnowS formula:
            #
            #   is_before_rel(edge)
            #     -> (entity_action(first(edge))
            #           -> (object(second(edge))
            #                 -> eai_task_entity_available(second(edge))))
            #
            # The contextual marker below supplies the per-task truth of the
            # final Boolean concept; it does not replace this LC structure.
            object_exists_lc = ifL(
                ctx.is_before_rel(edge_variable),
                ifL(
                    action_predicate,
                    ifL(
                        object_predicate,
                        entity_available("task_entity_object"),
                    ),
                ),
                # No ordinary graph sensor populates this contextual Boolean.
                # Keep solver verification inactive and enforce the marked LC
                # through the task-bound DFA, where the context facts exist.
                active=False,
                name="generated_object_exists_in_task_world",
            )

            # Bind each concrete object label to its normalized PDDL type.
            # Example: ``bathtub_35 -> bathtub``. After an entity-changing
            # action, the bound DFA permits that label only if ``bathtub`` is
            # present in ``generation_entity_types`` for the current task.
            mark_for_contextual_dfa(
                object_exists_lc,
                context_key="generation_entity_types",
                token_to_value={
                    token: entity_type_for_token(token)
                    for token in guarded_objects
                },
                vocabulary=bundle.vocabulary,
                name="generated_object_exists_in_task_world",
                # Some external/custom EAI rows omit a transition model.
                allow_missing_context=True,
                trigger_tokens=entity_actions,
            )

        if "clean" in action_sequence_tokens and "clean" in known:
            # Boolean property saying that a semantic action is relevant to
            # this task. ``semantic_action_permissions`` is derived from the
            # instruction, SimpleTL goal, and transition model—not the gold
            # action trajectory.
            task_action_permitted = Concept(name="eai_task_action_permitted")

            # Declarative DomiKnowS formula:
            #
            #   generated_token(x) == clean
            #       -> eai_task_action_permitted(x)
            #
            # This prevents a globally frequent but irrelevant ``clean`` plan
            # from satisfying the syntax DFA for a reading or navigation task.
            clean_relevant_lc = ifL(
                ctx.token_value("clean", "clean_action"),
                task_action_permitted("clean_action"),
                # As above, the Boolean is supplied while binding the DFA to
                # one example rather than by an ordinary DataNode sensor.
                active=False,
                name="generated_semantic_action_is_task_relevant",
            )

            # An empty permission set deliberately forbids ``clean``. Unlike
            # entity availability, missing action context is not treated as
            # unrestricted because every EAI row constructs this field.
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
