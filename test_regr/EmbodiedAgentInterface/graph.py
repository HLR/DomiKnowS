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

from dataset import ACTION_VOCAB, EOS_TOKEN


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
    return graph, bundle


# Backward-compatible alias for older imports in this folder.
def create_graph(max_steps=8):
    return create_generation_graph(max_steps=max_steps)
