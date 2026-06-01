import pytest

from domiknows.generation import (
    GenerationEncoder,
    apply_eos_closure_constraint,
    apply_max_non_eos_constraint,
    generation_bundle_from_graph,
)
from domiknows.graph import Concept, EnumConcept, Graph, Relation


class FakeTokenizer:
    def encode(self, token):
        return {"<eos>": [0], "A": [1]}[token]


def test_generation_encoder_builds_domiknows_graph():
    encoder = GenerationEncoder(
        ["<eos>", "A"],
        eos_token="<eos>",
        tokenizer=FakeTokenizer(),
    )
    graph, bundle = encoder.build_graph()
    with graph:
        apply_eos_closure_constraint(bundle.context)
        apply_max_non_eos_constraint(bundle.context, 1)

    assert graph is not None
    assert bundle.vocabulary.label_count == 3
    assert bundle.generated_token.name == "generated_token"
    assert len(graph.logicalConstrains) >= 2


def test_generation_bundle_from_graph_wraps_traditional_shape():
    Graph.clear()
    Concept.clear()
    Relation.clear()

    with Graph("main") as graph:
        text = Concept(name="text")
        token = Concept(name="token")
        contains, = text.contains(token)
        is_before_rel = Concept(name="is_before_rel")
        first_token, second_token = is_before_rel.has_a(arg1=token, arg2=token)
        generated_token = token(
            name="generated_token",
            ConceptClass=EnumConcept,
            values=["0", "1", "2"],
        )

    bundle = generation_bundle_from_graph(
        graph,
        vocab=["<eos>", "A"],
        eos_token="<eos>",
        tokenizer=FakeTokenizer(),
    )

    assert bundle.text is text
    assert bundle.token is token
    assert bundle.contains is contains
    assert bundle.generated_token is generated_token
    assert bundle.is_before_rel is is_before_rel
    assert bundle.first_token is first_token
    assert bundle.second_token is second_token
    assert bundle.vocabulary.label_count == 3


def test_generation_bundle_from_graph_allows_readable_enum_names():
    Graph.clear()
    Concept.clear()
    Relation.clear()

    with Graph("main") as graph:
        plan = Concept(name="plan")
        step = Concept(name="step")
        contains, = plan.contains(step)
        precedes = Concept(name="precedes")
        earlier, later = precedes.has_a(earlier=step, later=step)
        planned_action = step(
            name="planned_action",
            ConceptClass=EnumConcept,
            values=["done", "A", "_other"],
        )

    bundle = generation_bundle_from_graph(
        graph,
        vocab=["<eos>", "A"],
        eos_token="<eos>",
        tokenizer=FakeTokenizer(),
        text_name="plan",
        token_name="step",
        generated_token_name="planned_action",
        before_relation_name="precedes",
        first_role_name="earlier",
        second_role_name="later",
    )

    expr = bundle.context.token_value("<eos>", "x")
    assert bundle.text is plan
    assert bundle.token is step
    assert bundle.contains is contains
    assert bundle.generated_token is planned_action
    assert bundle.is_before_rel is precedes
    assert bundle.first_token is earlier
    assert bundle.second_token is later
    assert expr[0][0] is planned_action
    assert expr[0][1] == "done"
    assert expr[0][2] == 0


def test_generation_bundle_from_graph_requires_other_label():
    Graph.clear()
    Concept.clear()
    Relation.clear()

    with Graph("main") as graph:
        text = Concept(name="text")
        token = Concept(name="token")
        text.contains(token)
        is_before_rel = Concept(name="is_before_rel")
        is_before_rel.has_a(arg1=token, arg2=token)
        token(
            name="generated_token",
            ConceptClass=EnumConcept,
            values=["0", "1"],
        )

    with pytest.raises(ValueError, match="_other"):
        generation_bundle_from_graph(
            graph,
            vocab=["<eos>", "A"],
            eos_token="<eos>",
            tokenizer=FakeTokenizer(),
        )
