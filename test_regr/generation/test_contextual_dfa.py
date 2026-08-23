from domiknows.generation import (
    GenerationEncoder,
    after_token_allowed_dfa,
    bind_contextual_dfa,
    constraints_to_dfa_from_graph,
    declare_contextual_token_constraint,
    discover_contextual_token_constraints,
)


def _graph():
    encoder = GenerationEncoder(
        ["<eos>", "act", "cup", "door"],
        eos_token="<eos>",
        graph_name="test_contextual_generation",
    )
    graph, bundle = encoder.build_graph()
    declare_contextual_token_constraint(
        graph,
        bundle,
        tokens=["cup", "door"],
        context_key="available_types",
        token_to_value={"cup": "cup", "door": "door"},
        allow_missing_context=True,
    )
    return graph, bundle


def test_contextual_constraint_is_graph_declared_and_bound_per_example():
    graph, bundle = _graph()
    specs = discover_contextual_token_constraints(graph)
    assert len(specs) == 1
    assert specs[0].context_key == "available_types"

    base = constraints_to_dfa_from_graph(
        graph, bundle, on_unsupported="raise", minimize=False
    )
    cup = bundle.vocabulary.label_for_token("cup")
    door = bundle.vocabulary.label_for_token("door")
    bound = bind_contextual_dfa(
        base, graph, {"available_types": ("door",)}
    )
    assert bound.accepts([door])
    assert not bound.accepts([cup])
    assert door in bound.allowed_tokens(bound.start_state, remaining_steps=1)
    assert cup not in bound.allowed_tokens(bound.start_state, remaining_steps=1)


def test_contextual_reachability_removes_triggers_with_no_legal_completion():
    graph, bundle = _graph()
    act = bundle.vocabulary.label_for_token("act")
    cup = bundle.vocabulary.label_for_token("cup")
    base = after_token_allowed_dfa(bundle.vocabulary, ["act"], ["cup"])
    bound = bind_contextual_dfa(
        base, graph, {"available_types": ("door",)}
    )
    assert act not in bound.allowed_tokens(bound.start_state, remaining_steps=2)
    assert bound.step(bound.start_state, act) is not None
    assert bound.step(bound.step(bound.start_state, act), cup) is None

    available = bind_contextual_dfa(
        base, graph, {"available_types": ("cup",)}
    )
    assert act in available.allowed_tokens(
        available.start_state, remaining_steps=2
    )


def test_missing_context_can_preserve_unconditioned_behavior():
    graph, bundle = _graph()
    base = constraints_to_dfa_from_graph(
        graph, bundle, on_unsupported="raise", minimize=False
    )
    cup = bundle.vocabulary.label_for_token("cup")
    bound = bind_contextual_dfa(base, graph, {})
    assert bound.accepts([cup])
