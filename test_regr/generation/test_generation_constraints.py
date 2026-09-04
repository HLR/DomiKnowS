from domiknows.generation import (
    GenerationEncoder,
    apply_all_constraints,
    constraints_to_dfa_from_graph,
)


class FakeTokenizer:
    def encode(self, token):
        return {"<eos>": [0], "A": [1], "B": [2], "C": [3]}[token]


def labels(bundle, tokens):
    return [bundle.vocabulary.label_for_token(token) for token in tokens]


def _build():
    encoder = GenerationEncoder(
        ["<eos>", "A", "B", "C"],
        eos_token="<eos>",
        tokenizer=FakeTokenizer(),
    )
    return encoder.build_graph()


def test_apply_all_constraints_adds_domiknows_constraints_to_existing_graph():
    graph, bundle = _build()
    before = len(graph.logicalConstrains)

    with graph:
        constraints = apply_all_constraints(
            bundle.context,
            max_non_eos_count=3,
            required_tokens=["A"],
            forbidden_tokens=["B"],
            conditional_max_non_eos=[("C", 2)],
        )

    assert len(constraints) == 5
    assert len(graph.logicalConstrains) >= before + 5

    dfa = constraints_to_dfa_from_graph(graph, bundle)
    assert dfa.accepts(labels(bundle, ["A", "<eos>"]))
    assert not dfa.accepts(labels(bundle, ["B", "<eos>"]))


def test_apply_all_constraints_default_only_eos_closure():
    graph, bundle = _build()

    with graph:
        constraints = apply_all_constraints(bundle.context)

    assert len(constraints) == 1
    dfa = constraints_to_dfa_from_graph(graph, bundle)
    assert dfa.accepts(labels(bundle, ["A", "<eos>"]))
    assert not dfa.accepts(labels(bundle, ["A", "<eos>", "B"]))
