"""Independent small-sequence checks for request-bound LC values."""
import pytest
import torch

from domiknows.generation import (
    GenerationEncoder,
    GenerationSemanticContext,
    MissingSemanticBinding,
    compile_generation_constraint_plan,
    constrained_label_greedy_decode,
    mark_for_contextual_dfa,
)
from domiknows.graph.logicalConstrain import (
    andL, differentL, eqL, exactL, fixedL, forAllL, greaterL, iotaL,
    miotaL, notL, orL, queryL, sameL, sumL,
)
from domiknows.graph.dataNode import DataNode
from domiknows.graph import EnumConcept
from domiknows.generation.dfa.semantic import _BOOLEAN_TYPES, _FILTER_TYPES, _VALUE_TYPES


class FakeTokenizer:
    def encode(self, token):
        return {"<eos>": [0], "A": [1], "B": [2]}[token]


def test_every_concrete_logical_constraint_has_one_typed_dispatch_category():
    import inspect
    from domiknows.graph import logicalConstrain as lc_module

    concrete = {
        name for name, obj in vars(lc_module).items()
        if inspect.isclass(obj) and obj.__module__ == lc_module.__name__
        and name.endswith("L") and not name.startswith("_")
    }
    categories = (set(_VALUE_TYPES), _FILTER_TYPES, _BOOLEAN_TYPES)
    assert set.union(*categories) == concrete
    assert all(not left & right for index, left in enumerate(categories)
               for right in categories[index + 1:])


def bundle():
    return GenerationEncoder(
        ["<eos>", "A", "B"], eos_token="<eos>", tokenizer=FakeTokenizer(),
    ).build_graph()


def labels(bundle, tokens):
    return tuple(bundle.vocabulary.label_for_token(token) for token in tokens)


def node(instance_id, **attributes):
    return DataNode(instanceID=instance_id, attributes=attributes)


def test_data_node_context_binds_eq_filter_and_query_expected_answer():
    graph, item = bundle()
    with graph:
        answer = queryL(item.generated_token, iotaL(andL(
            item.context.token_value("A", "x"),
            eqL(item.generated_token, "instanceID", {"chosen"}),
        )))
    plan = compile_generation_constraint_plan(graph, item)
    dfa = plan.bind_data_nodes(
        [node("other"), node("chosen")], max_sequence_length=2,
        expected={answer.lcName: "A"},
    )
    assert dfa.accepts(labels(item, ["B", "A"]))
    assert not dfa.accepts(labels(item, ["A", "B"]))
    assert not dfa.accepts(labels(item, ["B"]))


def test_data_node_context_derives_fixed_truth_for_each_predicate():
    graph, item = bundle()
    with graph:
        fixedL(item.context.token_value("A", "x"))
        fixedL(item.context.token_value("B", "y"))
    key = f"<{item.generated_token.name}>/label"
    plan = compile_generation_constraint_plan(graph, item)
    dfa = plan.bind_data_nodes(
        [node(0, **{key: torch.tensor(item.vocabulary.label_for_token("A"))}),
         node(1, **{key: torch.tensor(item.vocabulary.label_for_token("B"))})],
        max_sequence_length=2,
    )
    assert dfa.accepts(labels(item, ["A", "B"]))
    assert not dfa.accepts(labels(item, ["B", "A"]))
    with pytest.raises(MissingSemanticBinding, match="observed"):
        plan.bind_data_nodes([node(0)], max_sequence_length=1)


def test_data_node_context_reads_soft_miota_probabilities_and_scores():
    graph, item = bundle()
    with graph:
        selector = miotaL(item.context.token_value("A", "x"), threshold=0.6)
        answer = queryL(item.generated_token, selector)
    probability_key = f"<{item.generated_token.name}>/local/softmax"
    score_key = f"{selector.name}/score"
    plan = compile_generation_constraint_plan(graph, item)
    dfa = plan.bind_data_nodes([
        node(0, **{probability_key: torch.tensor([0.05, 0.8, 0.1, 0.05])}),
        node(1, **{score_key: 0.2}),
    ], max_sequence_length=2, expected={answer.lcName: ("A", None)})
    assert dfa.accepts(labels(item, ["A", "B"]))
    assert not dfa.accepts(labels(item, ["B", "A"]))
    with pytest.raises(MissingSemanticBinding, match="needs"):
        plan.bind_data_nodes([node(0)], max_sequence_length=1)


def test_data_node_context_binds_subclass_variables_and_labels():
    graph, item = bundle()
    with graph:
        color = EnumConcept(name="color", values=["red", "blue"])
        sameL(color, "x", "y")
    plan = compile_generation_constraint_plan(graph, item)
    dfa = plan.bind_data_nodes([
        node(0, generation_variables="x", **{"<color>/label": torch.tensor(0)}),
        node(1, generation_variables="y", **{"<color>/label": torch.tensor(0)}),
    ], max_sequence_length=2)
    assert dfa.accepts(labels(item, ["A", "B"]))
    with pytest.raises(MissingSemanticBinding, match="binding for 'y'"):
        plan.bind_data_nodes([
            node(0, generation_variables="x", **{"<color>/label": 0}), node(1),
        ], max_sequence_length=2)
    with pytest.raises(MissingSemanticBinding, match="multiple positions"):
        plan.bind_data_nodes([
            node(0, generation_variables="x"), node(1, generation_variables="x"),
        ], max_sequence_length=2)


def test_eq_filter_and_iota_selection_use_position_attributes():
    graph, item = bundle()
    with graph:
        selector = iotaL(andL(
            item.context.token_value("A", "x"),
            eqL(item.generated_token, "instanceID", {"chosen"}),
        ))
    plan = compile_generation_constraint_plan(graph, item)
    facts = GenerationSemanticContext(attributes={
        0: {"instanceID": "other"}, 1: {"instanceID": "chosen"},
    })
    dfa = plan.bind(facts, max_sequence_length=2)
    good = labels(item, ["A", "A"])
    assert dfa.accepts(good)
    assert plan.evaluate(good, facts)[selector.lcName].value == 1
    assert not dfa.accepts(labels(item, ["A", "B"]))
    with pytest.raises(MissingSemanticBinding):
        plan.evaluate(good, GenerationSemanticContext())


def test_fixed_truth_matches_observed_positive_and_negative_labels():
    graph, item = bundle()
    with graph:
        fixedL(item.context.token_value("A", "x"))
    plan = compile_generation_constraint_plan(graph, item)
    dfa = plan.bind(
        GenerationSemanticContext(fixed_truth={0: True, 1: False}),
        max_sequence_length=2,
    )
    assert dfa.accepts(labels(item, ["A", "B"]))
    assert not dfa.accepts(labels(item, ["B", "A"]))
    assert dfa.allowed_tokens(dfa.start_state, remaining_steps=2) == {item.vocabulary.label_for_token("A")}


def test_sum_is_a_number_and_only_expected_target_constrains_output():
    graph, item = bundle()
    with graph:
        total = sumL(item.context.token_value("A", "x"))
    plan = compile_generation_constraint_plan(graph, item)
    sequence = labels(item, ["A", "B", "A"])
    assert plan.evaluate(sequence, GenerationSemanticContext())[total.lcName].value == 2
    assert plan.bind(GenerationSemanticContext(), max_sequence_length=3).accepts(sequence)
    assert plan.bind(
        GenerationSemanticContext(expected={total.lcName: 2}), max_sequence_length=3,
    ).accepts(sequence)
    assert not plan.bind(
        GenerationSemanticContext(expected={total.lcName: 1}), max_sequence_length=3,
    ).accepts(sequence)


def test_iota_uniqueness_and_query_answer():
    graph, item = bundle()
    with graph:
        answer = queryL(item.generated_token, iotaL(item.context.token_value("A", "x")))
    plan = compile_generation_constraint_plan(graph, item)
    facts = GenerationSemanticContext(expected={answer.lcName: "A"})
    dfa = plan.bind(facts, max_sequence_length=2)
    assert dfa.accepts(labels(item, ["A", "B"]))
    assert not dfa.accepts(labels(item, ["A", "A"]))
    assert not dfa.accepts(labels(item, ["B", "B"]))


def test_miota_threshold_and_candidate_aligned_query():
    graph, item = bundle()
    with graph:
        selector = miotaL(item.context.token_value("A", "x"), threshold=0.6)
        answer = queryL(item.generated_token, selector)
    plan = compile_generation_constraint_plan(graph, item)
    facts = GenerationSemanticContext(membership_scores={selector.name: {0: 0.7, 1: 0.4}})
    values = plan.evaluate(labels(item, ["A", "A"]), facts)
    assert values[answer.lcName].value == ("A", None)
    assert values[answer.lcName].kind == "answer"
    with pytest.raises(MissingSemanticBinding):
        plan.evaluate(labels(item, ["A"]), GenerationSemanticContext())


def test_same_and_different_compare_bound_categorical_values():
    graph, item = bundle()
    with graph:
        same = sameL(item.generated_token, "x", "y")
        different = differentL(item.generated_token, "u", "v")
    plan = compile_generation_constraint_plan(graph, item)
    facts = GenerationSemanticContext(bindings={"x": 0, "y": 1, "u": 0, "v": 1})
    equal = plan.evaluate(labels(item, ["A", "A"]), facts)
    unequal = plan.evaluate(labels(item, ["A", "B"]), facts)
    assert equal[same.lcName].value is True
    assert equal[different.lcName].value is False
    assert unequal[same.lcName].value is False
    assert unequal[different.lcName].value is True


@pytest.mark.parametrize("constraint", [sameL, differentL])
def test_grounded_subclass_comparison_prunes_finite_label_assignments(
    constraint,
):
    graph, item = bundle()
    with graph:
        constraint(item.generated_token, "x", "y")
    dfa = compile_generation_constraint_plan(graph, item).bind(
        GenerationSemanticContext(bindings={"x": 0, "y": 1}),
        max_sequence_length=2,
    )
    first = dfa.step(dfa.start_state, item.vocabulary.label_for_token("A"))
    allowed = dfa.allowed_tokens(first, remaining_steps=1)
    first_label = item.vocabulary.label_for_token("A")
    if constraint is sameL:
        assert allowed == {first_label}
    else:
        assert allowed == set(dfa.alphabet) - {first_label}


def test_same_uses_request_attribute_for_non_token_enum():
    graph, item = bundle()
    with graph:
        color = EnumConcept(name="color", values=["red", "blue"])
        same = sameL(color, "x", "y")
    plan = compile_generation_constraint_plan(graph, item)
    facts = GenerationSemanticContext(
        bindings={"x": 0, "y": 1},
        attributes={0: {"color": "red"}, 1: {"color": "blue"}},
    )
    assert not plan.evaluate(labels(item, ["A", "A"]), facts)[same.lcName].value


def test_sum_inside_exact_count_is_boolean():
    graph, item = bundle()
    with graph:
        exactL(sumL(item.context.token_value("A", "x")), limit=2)
    plan = compile_generation_constraint_plan(graph, item)
    dfa = plan.bind(GenerationSemanticContext(), max_sequence_length=3)
    assert dfa.accepts(labels(item, ["A", "B", "A"]))
    assert not dfa.accepts(labels(item, ["A", "B", "B"]))


def test_sum_of_two_predicates_keeps_additive_count_semantics():
    graph, item = bundle()
    with graph:
        total = sumL(item.context.token_value("A", "x"), item.context.token_value("B", "y"))
    plan = compile_generation_constraint_plan(graph, item)
    value = plan.evaluate(labels(item, ["A", "B", "B"]), GenerationSemanticContext())[total.lcName]
    assert value.kind == "number" and value.value == 3


def test_comparison_consumes_sum_and_token_count():
    graph, item = bundle()
    with graph:
        greaterL(
            sumL(item.context.token_value("A", "x")),
            item.context.token_value("B", "y"),
        )
    plan = compile_generation_constraint_plan(graph, item)
    dfa = plan.bind(GenerationSemanticContext(), max_sequence_length=3)
    assert dfa.accepts(labels(item, ["A", "A", "B"]))
    assert not dfa.accepts(labels(item, ["A", "B", "B"]))


def test_iota_in_boolean_parent_uses_uniqueness_truth():
    graph, item = bundle()
    with graph:
        andL(iotaL(item.context.token_value("A", "x")), exactL(item.context.token_value("B", "y"), limit=1))
    plan = compile_generation_constraint_plan(graph, item)
    dfa = plan.bind(GenerationSemanticContext(), max_sequence_length=3)
    assert dfa.accepts(labels(item, ["A", "B"]))
    assert not dfa.accepts(labels(item, ["A", "A", "B"]))


def test_context_snapshots_data_node_attributes_and_rejects_unknown_targets():
    graph, item = bundle()
    with graph:
        selector = iotaL(andL(
            item.context.token_value("A", "x"),
            eqL(item.generated_token, "instanceID", {"chosen"}),
        ))
    plan = compile_generation_constraint_plan(graph, item)
    nodes = [DataNode(instanceID="other"), DataNode(instanceID="chosen")]
    facts = GenerationSemanticContext.from_data_nodes(nodes)
    assert plan.evaluate(labels(item, ["A", "A"]), facts)[selector.lcName].value == 1
    with pytest.raises(MissingSemanticBinding, match="unknown deferred LCs"):
        plan.bind(GenerationSemanticContext(expected={"typo": 1}), max_sequence_length=2)


def test_path_predicates_require_an_explicit_request_resolver():
    graph, item = bundle()
    with graph:
        selector = iotaL(
            item.is_before_rel("relation"),
            item.context.token_value("A", "x", path=("relation", item.first_token)),
        )
    plan = compile_generation_constraint_plan(graph, item)
    sequence = labels(item, ["A", "A"])
    memberships = {item.is_before_rel.name: frozenset({0, 1})}
    with pytest.raises(MissingSemanticBinding, match="path_resolver"):
        plan.evaluate(sequence, GenerationSemanticContext(memberships=memberships))
    facts = GenerationSemanticContext(
        memberships=memberships, path_resolver=lambda path, labels: {1},
    )
    assert plan.evaluate(sequence, facts)[selector.lcName].value == 1


def test_bound_semantics_masks_compact_decoder_logits():
    graph, item = bundle()
    with graph:
        fixedL(item.context.token_value("A", "x"))
    dfa = compile_generation_constraint_plan(graph, item).bind(
        GenerationSemanticContext(fixed_truth={0: True}), max_sequence_length=1,
    )

    class PreferB:
        label_to_token_id = (0, 1, 2, None)

        def next_label_logits(self, input_ids):
            return torch.tensor([0.0, 1.0, 10.0, 0.0])

        def token_id_for_label(self, label):
            value = self.label_to_token_id[int(label)]
            if value is None:
                raise ValueError("other label is not directly emittable")
            return value

    result = constrained_label_greedy_decode(
        PreferB(), torch.tensor([[0]]), item.vocabulary, dfa, max_new_tokens=1,
    )
    assert result.labels == [item.vocabulary.label_for_token("A")]
    assert result.accepted


def test_value_predicates_preserve_or_and_not_token_sets():
    graph, item = bundle()
    with graph:
        union_count = sumL(orL(
            item.context.token_value("A", "x"),
            item.context.token_value("B", "x"),
        ))
        complement_count = sumL(notL(item.context.token_value("A", "y")))
    plan = compile_generation_constraint_plan(graph, item)
    values = plan.evaluate(labels(item, ["A", "B", "<eos>"]), GenerationSemanticContext())
    assert values[union_count.lcName].value == 2
    assert values[complement_count.lcName].value == 2


def test_for_all_inside_deferred_boolean_parent_is_checked():
    graph, item = bundle()
    with graph:
        andL(
            iotaL(item.context.token_value("A", "x")),
            forAllL(item.context.token_value("B", "y"), item.context.token_value("A", "y")),
        )
    dfa = compile_generation_constraint_plan(graph, item).bind(
        GenerationSemanticContext(), max_sequence_length=2,
    )
    assert dfa.accepts(labels(item, ["A", "<eos>"]))
    assert not dfa.accepts(labels(item, ["A", "B"]))


def test_reachability_budget_reports_exhausted_search():
    graph, item = bundle()
    with graph:
        total = sumL(item.context.token_value("A", "x"))
    dfa = compile_generation_constraint_plan(graph, item).bind(
        GenerationSemanticContext(expected={total.lcName: 2}),
        max_sequence_length=2,
        search_budget=1,
    )
    with pytest.raises(RuntimeError, match="search_budget"):
        dfa.allowed_tokens(dfa.start_state, remaining_steps=2)


def test_request_values_bind_existing_contextual_dfa_markers():
    graph, item = bundle()
    with graph:
        lc = fixedL(item.context.token_value("A", "x"))
    mark_for_contextual_dfa(
        lc, context_key="available", token_to_value={"A": "yes"},
        vocabulary=item.vocabulary,
    )
    plan = compile_generation_constraint_plan(graph, item)
    blocked = plan.bind(
        GenerationSemanticContext(request_values={"available": []}),
        max_sequence_length=1,
    )
    allowed = plan.bind(
        GenerationSemanticContext(request_values={"available": ["yes"]}),
        max_sequence_length=1,
    )
    assert not blocked.accepts(labels(item, ["A"]))
    assert allowed.accepts(labels(item, ["A"]))
