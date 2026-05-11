import pytest

from domiknows.generation import (
    AfterTokenAllowedConstraint,
    AnyOfGenerationConstraint,
    ConditionalMaxNonEosConstraint,
    EosClosureConstraint,
    ForbiddenTokenConstraint,
    GenerationEncoder,
    MaxNonEosConstraint,
    OrderedTokensConstraint,
    RequiredTokenConstraint,
    TokenSetCountConstraint,
    analyze_generation_constraints,
    constraints_to_dfa_from_graph,
    discover_generation_constraints,
    required_token,
)
from domiknows.graph.logicalConstrain import (
    andL,
    atLeastAL,
    atMostAL,
    exactAL,
    existsAL,
    ifL,
    iffL,
    nandL,
    norL,
    notL,
    orL,
    eqL,
    sumL,
    xorL,
)


class FakeTokenizer:
    def encode(self, token):
        return {"<eos>": [0], "A": [1], "B": [2]}[token]


def build_bundle(constraints=()):
    encoder = GenerationEncoder(
        ["<eos>", "A", "B"],
        eos_token="<eos>",
        tokenizer=FakeTokenizer(),
    )
    return encoder.build_graph(constraints)


def labels(bundle, tokens):
    return [bundle.vocabulary.label_for_token(token) for token in tokens]


def test_discovers_raw_eos_closure_constraint():
    graph, bundle = build_bundle()
    ctx = bundle.context
    with graph:
        ifL(
            ctx.is_before_rel("before"),
            ifL(
                ctx.token_value("<eos>", "x", path=("before", ctx.first_token)),
                ctx.token_value("<eos>", "y", path=("before", ctx.second_token)),
            ),
        )

    constraints = discover_generation_constraints(graph, bundle)

    assert any(isinstance(constraint, EosClosureConstraint) for constraint in constraints)
    dfa = constraints_to_dfa_from_graph(graph, bundle)
    assert not dfa.accepts(labels(bundle, ["A", "<eos>", "B"]))


def test_discovers_raw_max_non_eos_constraint():
    graph, bundle = build_bundle()
    with graph:
        atMostAL(bundle.context.non_eos("x"), 2)

    constraints = discover_generation_constraints(graph, bundle)

    assert any(isinstance(constraint, MaxNonEosConstraint) and constraint.max_count == 2 for constraint in constraints)
    dfa = constraints_to_dfa_from_graph(graph, bundle)
    assert dfa.accepts(labels(bundle, ["A", "B", "<eos>"]))
    assert not dfa.accepts(labels(bundle, ["A", "B", "A"]))


def test_discovers_raw_required_token_constraints():
    graph, bundle = build_bundle()
    with graph:
        atLeastAL(bundle.context.token_value("A", "x"), 2)
        existsAL(bundle.context.token_value("B", "x"))

    constraints = discover_generation_constraints(graph, bundle)

    assert any(isinstance(constraint, RequiredTokenConstraint) and constraint.token == "A" and constraint.min_count == 2 for constraint in constraints)
    assert any(isinstance(constraint, RequiredTokenConstraint) and constraint.token == "B" and constraint.min_count == 1 for constraint in constraints)
    dfa = constraints_to_dfa_from_graph(graph, bundle)
    assert dfa.accepts(labels(bundle, ["A", "A", "B"]))
    assert not dfa.accepts(labels(bundle, ["A", "B", "<eos>"]))


def test_discovers_raw_forbidden_token_constraint():
    graph, bundle = build_bundle()
    with graph:
        atMostAL(bundle.context.token_value("B", "x"), 0)

    constraints = discover_generation_constraints(graph, bundle)

    assert any(isinstance(constraint, ForbiddenTokenConstraint) and constraint.token == "B" for constraint in constraints)
    dfa = constraints_to_dfa_from_graph(graph, bundle)
    assert dfa.accepts(labels(bundle, ["A", "<eos>"]))
    assert not dfa.accepts(labels(bundle, ["B", "<eos>"]))


def test_discovers_raw_conditional_max_non_eos_constraint():
    graph, bundle = build_bundle()
    with graph:
        ifL(
            existsAL(bundle.context.token_value("A", "x")),
            atMostAL(bundle.context.non_eos("y"), 2),
        )

    constraints = discover_generation_constraints(graph, bundle)

    assert any(
        isinstance(constraint, ConditionalMaxNonEosConstraint)
        and constraint.token == "A"
        and constraint.max_count == 2
        for constraint in constraints
    )
    dfa = constraints_to_dfa_from_graph(graph, bundle)
    assert dfa.accepts(labels(bundle, ["A", "B", "<eos>"]))
    assert not dfa.accepts(labels(bundle, ["A", "B", "B"]))


def test_discovery_deduplicates_explicit_and_raw_constraints():
    graph, bundle = build_bundle([required_token("A")])
    with graph:
        atLeastAL(bundle.context.token_value("A", "x"), 1)

    constraints = discover_generation_constraints(graph, bundle)
    required_a = [
        constraint
        for constraint in constraints
        if isinstance(constraint, RequiredTokenConstraint) and constraint.token == "A"
    ]

    assert len(required_a) == 1


def test_discovers_supported_and_lc_children_as_intersection_constraints():
    graph, bundle = build_bundle()
    with graph:
        andL(
            atLeastAL(bundle.context.token_value("A", "x"), 1),
            atMostAL(bundle.context.non_eos("y"), 2),
        )

    constraints = discover_generation_constraints(graph, bundle)

    assert any(isinstance(constraint, RequiredTokenConstraint) and constraint.token == "A" for constraint in constraints)
    assert any(isinstance(constraint, MaxNonEosConstraint) and constraint.max_count == 2 for constraint in constraints)
    dfa = constraints_to_dfa_from_graph(graph, bundle)
    assert dfa.accepts(labels(bundle, ["A", "<eos>"]))
    assert not dfa.accepts(labels(bundle, ["B", "<eos>"]))
    assert not dfa.accepts(labels(bundle, ["A", "B", "A"]))


def test_discovers_nested_and_lc_children():
    graph, bundle = build_bundle()
    with graph:
        andL(
            andL(
                atLeastAL(bundle.context.token_value("A", "x"), 1),
                existsAL(bundle.context.token_value("B", "y")),
            ),
            atMostAL(bundle.context.non_eos("z"), 2),
        )

    constraints = discover_generation_constraints(graph, bundle)

    assert any(isinstance(constraint, RequiredTokenConstraint) and constraint.token == "A" for constraint in constraints)
    assert any(isinstance(constraint, RequiredTokenConstraint) and constraint.token == "B" for constraint in constraints)
    assert any(isinstance(constraint, MaxNonEosConstraint) and constraint.max_count == 2 for constraint in constraints)
    dfa = constraints_to_dfa_from_graph(graph, bundle)
    assert dfa.accepts(labels(bundle, ["A", "B", "<eos>"]))
    assert not dfa.accepts(labels(bundle, ["A", "<eos>"]))
    assert not dfa.accepts(labels(bundle, ["A", "B", "A"]))


def test_discovers_or_lc_as_union_constraint():
    graph, bundle = build_bundle()
    with graph:
        orL(
            atLeastAL(bundle.context.token_value("A", "x"), 1),
            atLeastAL(bundle.context.token_value("B", "y"), 1),
        )

    constraints = discover_generation_constraints(graph, bundle)

    assert len(constraints) == 1
    assert isinstance(constraints[0], AnyOfGenerationConstraint)
    dfa = constraints_to_dfa_from_graph(graph, bundle)
    assert dfa.accepts(labels(bundle, ["A", "<eos>"]))
    assert dfa.accepts(labels(bundle, ["B", "<eos>"]))
    assert not dfa.accepts(labels(bundle, ["<eos>"]))


def test_discovers_nested_or_and_formula_exactly():
    graph, bundle = build_bundle()
    with graph:
        orL(
            andL(
                atLeastAL(bundle.context.token_value("A", "x"), 1),
                atMostAL(bundle.context.non_eos("x"), 1),
            ),
            andL(
                atLeastAL(bundle.context.token_value("B", "y"), 1),
                atMostAL(bundle.context.token_value("A", "y"), 0),
            ),
        )

    constraints = discover_generation_constraints(graph, bundle)

    assert len(constraints) == 1
    assert isinstance(constraints[0], AnyOfGenerationConstraint)
    dfa = constraints_to_dfa_from_graph(graph, bundle)
    assert dfa.accepts(labels(bundle, ["A", "<eos>"]))
    assert dfa.accepts(labels(bundle, ["B", "B"]))
    assert not dfa.accepts(labels(bundle, ["A", "B"]))


def test_discovers_negated_regular_constraints():
    graph, bundle = build_bundle()
    with graph:
        notL(existsAL(bundle.context.token_value("A", "x")))
        notL(atMostAL(bundle.context.token_value("B", "y"), 1))

    dfa = constraints_to_dfa_from_graph(graph, bundle)

    assert dfa.accepts(labels(bundle, ["B", "B"]))
    assert not dfa.accepts(labels(bundle, ["A", "B"]))
    assert not dfa.accepts(labels(bundle, ["B"]))


def test_discovers_exact_and_token_set_count_constraints():
    graph, bundle = build_bundle()
    with graph:
        exactAL(bundle.context.token_value("A", "x"), 2)
        atLeastAL(
            orL(
                bundle.context.token_value("A", "y"),
                bundle.context.token_value("B", "y"),
            ),
            3,
        )

    constraints = discover_generation_constraints(graph, bundle)

    assert any(
        isinstance(constraint, TokenSetCountConstraint)
        and constraint.tokens == ("A",)
        and constraint.min_count == 2
        and constraint.max_count == 2
        for constraint in constraints
    )
    assert any(
        isinstance(constraint, TokenSetCountConstraint)
        and set(constraint.tokens) == {"A", "B"}
        and constraint.min_count == 3
        for constraint in constraints
    )
    dfa = constraints_to_dfa_from_graph(graph, bundle)
    assert dfa.accepts(labels(bundle, ["A", "B", "A"]))
    assert not dfa.accepts(labels(bundle, ["A", "B", "B"]))
    assert not dfa.accepts(labels(bundle, ["A", "A"]))


def test_discovers_regular_if_nand_nor_xor_and_iff_constraints():
    graph, bundle = build_bundle()
    with graph:
        ifL(
            existsAL(bundle.context.token_value("A", "x")),
            existsAL(bundle.context.token_value("B", "y")),
        )
        nandL(
            existsAL(bundle.context.token_value("A", "n")),
            atMostAL(bundle.context.token_value("B", "n"), 0),
        )
        norL(
            atLeastAL(bundle.context.token_value("A", "r"), 3),
            atLeastAL(bundle.context.token_value("B", "r"), 3),
        )
        xorL(
            atLeastAL(bundle.context.token_value("A", "o"), 2),
            atLeastAL(bundle.context.token_value("B", "o"), 2),
        )
        iffL(
            atLeastAL(bundle.context.token_value("A", "i"), 2),
            atLeastAL(bundle.context.token_value("B", "i"), 2),
        )

    constraints = discover_generation_constraints(graph, bundle)

    assert len(constraints) == 5
    implies, nand, nor, xor, iff = [constraint.to_dfa(bundle.vocabulary) for constraint in constraints]

    assert implies.accepts(labels(bundle, ["B"]))
    assert implies.accepts(labels(bundle, ["A", "B"]))
    assert not implies.accepts(labels(bundle, ["A"]))

    assert nand.accepts(labels(bundle, ["B"]))
    assert nand.accepts(labels(bundle, ["A", "B"]))
    assert not nand.accepts(labels(bundle, ["A"]))

    assert nor.accepts(labels(bundle, ["A", "B"]))
    assert not nor.accepts(labels(bundle, ["A", "A", "A"]))
    assert not nor.accepts(labels(bundle, ["B", "B", "B"]))

    assert xor.accepts(labels(bundle, ["A", "A"]))
    assert xor.accepts(labels(bundle, ["B", "B"]))
    assert not xor.accepts(labels(bundle, ["A", "A", "B", "B"]))
    assert not xor.accepts(labels(bundle, ["A"]))

    assert iff.accepts(labels(bundle, ["A"]))
    assert iff.accepts(labels(bundle, ["A", "A", "B", "B"]))
    assert not iff.accepts(labels(bundle, ["A", "A"]))


def test_discovers_generalized_before_path_implication():
    graph, bundle = build_bundle()
    ctx = bundle.context
    with graph:
        ifL(
            ctx.is_before_rel("before"),
            ifL(
                ctx.token_value("A", "x", path=("before", ctx.first_token)),
                ctx.token_value("B", "y", path=("before", ctx.second_token)),
            ),
        )

    constraints = discover_generation_constraints(graph, bundle)

    assert any(isinstance(constraint, AfterTokenAllowedConstraint) for constraint in constraints)
    dfa = constraints_to_dfa_from_graph(graph, bundle)
    assert dfa.accepts(labels(bundle, ["A", "B", "B"]))
    assert dfa.accepts(labels(bundle, ["B", "A"]))
    assert not dfa.accepts(labels(bundle, ["A", "B", "A"]))


def test_discovers_ordered_pair_existence_from_before_path():
    graph, bundle = build_bundle()
    ctx = bundle.context
    with graph:
        existsAL(
            andL(
                ctx.is_before_rel("before"),
                ctx.token_value("A", "x", path=("before", ctx.first_token)),
                ctx.token_value("B", "y", path=("before", ctx.second_token)),
            )
        )

    constraints = discover_generation_constraints(graph, bundle)

    assert any(isinstance(constraint, OrderedTokensConstraint) for constraint in constraints)
    dfa = constraints_to_dfa_from_graph(graph, bundle)
    assert dfa.accepts(labels(bundle, ["A", "B"]))
    assert not dfa.accepts(labels(bundle, ["B", "A"]))


def test_analysis_reports_unsupported_regular_fragment_reasons():
    graph, bundle = build_bundle()
    with graph:
        sumL(bundle.context.token_value("A", "x"))
        andL(
            bundle.context.token_value("B", "y"),
            eqL(bundle.generated_token, "instanceID", {"some-id"}),
        )

    analyses = analyze_generation_constraints(graph, bundle, on_unsupported="ignore")

    reasons = [analysis.reason for analysis in analyses if analysis.relevant and not analysis.supported]
    assert any("numeric selection/query semantics" in reason for reason in reasons)
    assert any("eqL path filters" in reason for reason in reasons)


def test_unsupported_generation_relevant_constraints_can_warn_ignore_or_error():
    graph, bundle = build_bundle()
    with graph:
        andL(bundle.context.token_value("A", "x"), bundle.context.token_value("B", "x"))

    with pytest.warns(RuntimeWarning, match="not supported by generation DFA discovery"):
        assert discover_generation_constraints(graph, bundle, on_unsupported="warn") == ()

    assert discover_generation_constraints(graph, bundle, on_unsupported="ignore") == ()
    with pytest.raises(ValueError, match="not supported by generation DFA discovery"):
        discover_generation_constraints(graph, bundle, on_unsupported="error")


def test_unsupported_or_lc_branch_warns_or_errors():
    graph, bundle = build_bundle()
    with graph:
        orL(
            atLeastAL(bundle.context.token_value("A", "x"), 1),
            andL(bundle.context.token_value("A", "x"), bundle.context.token_value("B", "x")),
        )

    with pytest.warns(RuntimeWarning, match="not supported by generation DFA discovery"):
        assert discover_generation_constraints(graph, bundle, on_unsupported="warn") == ()

    with pytest.raises(ValueError, match="not supported by generation DFA discovery"):
        discover_generation_constraints(graph, bundle, on_unsupported="error")
