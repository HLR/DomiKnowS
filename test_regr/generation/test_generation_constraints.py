from domiknows.generation import (
    GenerationEncoder,
    apply_all_constraints,
    constraints_to_dfa,
    default_generation_constraints,
)


class FakeTokenizer:
    def encode(self, token):
        return {"<eos>": [0], "A": [1], "B": [2], "C": [3]}[token]


def labels(bundle, tokens):
    return [bundle.vocabulary.label_for_token(token) for token in tokens]


def test_default_generation_constraints_compile_to_dfa():
    encoder = GenerationEncoder(
        ["<eos>", "A", "B", "C"],
        eos_token="<eos>",
        tokenizer=FakeTokenizer(),
    )
    graph, bundle = encoder.build_graph(
        default_generation_constraints(
            max_non_eos_count=3,
            required_tokens={"A": 1},
            forbidden_tokens=["B"],
            conditional_max_non_eos={"C": 2},
        )
    )

    dfa = constraints_to_dfa(bundle.constraints, bundle.vocabulary)

    assert graph is not None
    assert dfa.accepts(labels(bundle, ["A", "C", "<eos>"]))
    assert not dfa.accepts(labels(bundle, ["B", "<eos>"]))
    assert not dfa.accepts(labels(bundle, ["C", "A", "A", "<eos>"]))


def test_apply_all_constraints_adds_domiknows_constraints_to_existing_graph():
    encoder = GenerationEncoder(
        ["<eos>", "A", "B", "C"],
        eos_token="<eos>",
        tokenizer=FakeTokenizer(),
    )
    graph, bundle = encoder.build_graph()
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

    dfa = constraints_to_dfa(constraints, bundle.vocabulary)
    assert dfa.accepts(labels(bundle, ["A", "<eos>"]))
    assert not dfa.accepts(labels(bundle, ["B", "<eos>"]))
