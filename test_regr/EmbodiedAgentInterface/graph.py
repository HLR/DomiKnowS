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
    from domiknows.graph.logicalConstrain import atLeastAL, atMostAL, ifL, notL

    ctx = bundle.context
    eos_token = bundle.vocabulary.eos_token

    # EOS-closure: once EOS appears at some position, every later position
    # is EOS too.  Same shape as Tasks/hf_generation/graph.py.
    ifL(
        ctx.is_before_rel("before"),
        ifL(
            ctx.token_value(eos_token, "x", path=("before", ctx.first_token)),
            ctx.token_value(eos_token, "y", path=("before", ctx.second_token)),
        ),
    )

    # ``max_non_eos_count = max_steps - 1`` translates to "at most N
    # positions are not EOS".
    max_non_eos = max(0, int(max_steps) - 1)
    atMostAL(notL(ctx.token_value(eos_token, "x")), max_non_eos)

    # Each required token must appear at least the requested number of
    # times anywhere in the generated sequence.
    for token, count in (required_tokens or {}).items():
        if token not in bundle.vocabulary.tokens:
            continue
        atLeastAL(ctx.token_value(token, "x"), int(count))

    # Each forbidden token is barred entirely.
    for token in forbidden_tokens or ():
        if token not in bundle.vocabulary.tokens:
            continue
        atMostAL(ctx.token_value(token, "x"), 0)


def _add_action_object_logical_constraint(graph, bundle, action_tokens, object_tokens):
    """Encode "every action token is immediately followed by an object token"."""
    from domiknows.generation import mark_for_dfa
    from domiknows.graph.logicalConstrain import ifL, orL

    valid_actions = tuple(token for token in action_tokens if token in bundle.vocabulary.tokens)
    valid_objects = tuple(token for token in object_tokens if token in bundle.vocabulary.tokens)
    if not valid_actions or not valid_objects:
        return

    ctx = bundle.context
    with graph:
        action_calls = [
            ctx.token_value(action, "x", path=("before", ctx.first_token))
            for action in valid_actions
        ]
        object_calls = [
            ctx.token_value(obj, "y", path=("before", ctx.second_token))
            for obj in valid_objects
        ]
        lc = ifL(
            ctx.is_before_rel("before"),
            ifL(_disjunction(action_calls, orL), _disjunction(object_calls, orL)),
        )
        # The single-arg modern signature; ``mark_for_dfa`` returns *lc* for chaining.
        mark_for_dfa(lc)


def create_generation_graph(
    max_steps=8,
    required_tokens=None,
    forbidden_tokens=None,
    vocab=None,
    object_tokens=None,
    action_tokens=None,
    enforce_action_object=True,
):
    from domiknows.generation import GenerationEncoder

    vocab = tuple(vocab or ACTION_VOCAB)
    object_tokens = tuple(object_tokens or ())
    action_tokens = tuple(action_tokens or ())

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
        _add_action_object_logical_constraint(graph, bundle, action_tokens, object_tokens)
    return graph, bundle


# Backward-compatible alias for older imports in this folder.
def create_graph(max_steps=8):
    return create_generation_graph(max_steps=max_steps)
