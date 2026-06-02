import pytest

from domiknows.generation import (
    GenerationEncoder,
    analyze_generation_constraints,
    constraints_to_dfa_from_graph,
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


def build_bundle():
    encoder = GenerationEncoder(
        ["<eos>", "A", "B"],
        eos_token="<eos>",
        tokenizer=FakeTokenizer(),
    )
    return encoder.build_graph()


def labels(bundle, tokens):
    return [bundle.vocabulary.label_for_token(token) for token in tokens]


def supported_lc_types(graph, bundle):
    return [
        analysis.lc_type
        for analysis in analyze_generation_constraints(graph, bundle, on_unsupported="ignore")
        if analysis.supported
    ]


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

    assert "ifL" in supported_lc_types(graph, bundle)
    dfa = constraints_to_dfa_from_graph(graph, bundle)
    assert not dfa.accepts(labels(bundle, ["A", "<eos>", "B"]))


def test_discovers_raw_max_non_eos_constraint():
    graph, bundle = build_bundle()
    with graph:
        atMostAL(bundle.context.non_eos("x"), 2)

    assert "atMostAL" in supported_lc_types(graph, bundle)
    dfa = constraints_to_dfa_from_graph(graph, bundle)
    assert dfa.accepts(labels(bundle, ["A", "B", "<eos>"]))
    assert not dfa.accepts(labels(bundle, ["A", "B", "A"]))


def test_discovers_raw_required_token_constraints():
    graph, bundle = build_bundle()
    with graph:
        atLeastAL(bundle.context.token_value("A", "x"), 2)
        existsAL(bundle.context.token_value("B", "x"))

    types = supported_lc_types(graph, bundle)
    assert "atLeastAL" in types
    assert "existsAL" in types
    dfa = constraints_to_dfa_from_graph(graph, bundle)
    assert dfa.accepts(labels(bundle, ["A", "A", "B"]))
    assert not dfa.accepts(labels(bundle, ["A", "B", "<eos>"]))


def test_discovers_raw_forbidden_token_constraint():
    graph, bundle = build_bundle()
    with graph:
        atMostAL(bundle.context.token_value("B", "x"), 0)

    assert "atMostAL" in supported_lc_types(graph, bundle)
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

    assert "ifL" in supported_lc_types(graph, bundle)
    dfa = constraints_to_dfa_from_graph(graph, bundle)
    assert dfa.accepts(labels(bundle, ["A", "B", "<eos>"]))
    assert not dfa.accepts(labels(bundle, ["A", "B", "B"]))


def test_discovers_supported_and_lc_children_as_intersection_constraints():
    graph, bundle = build_bundle()
    with graph:
        andL(
            atLeastAL(bundle.context.token_value("A", "x"), 1),
            atMostAL(bundle.context.non_eos("y"), 2),
        )

    assert "andL" in supported_lc_types(graph, bundle)
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

    assert "andL" in supported_lc_types(graph, bundle)
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

    assert "orL" in supported_lc_types(graph, bundle)
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

    assert "orL" in supported_lc_types(graph, bundle)
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

    types = supported_lc_types(graph, bundle)
    assert "exactAL" in types
    assert "atLeastAL" in types
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

    analyses = [
        analysis
        for analysis in analyze_generation_constraints(graph, bundle, on_unsupported="ignore")
        if analysis.supported
    ]
    assert len(analyses) == 5
    by_type = {analysis.lc_type: analysis for analysis in analyses}
    implies = by_type["ifL"].dfas[0]
    nand = by_type["nandL"].dfas[0]
    nor = by_type["norL"].dfas[0]
    xor = by_type["xorL"].dfas[0]
    iff = by_type["iffL"].dfas[0]

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

    assert "ifL" in supported_lc_types(graph, bundle)
    dfa = constraints_to_dfa_from_graph(graph, bundle)
    assert dfa.accepts(labels(bundle, ["A", "B", "B"]))
    assert dfa.accepts(labels(bundle, ["B", "A"]))
    assert not dfa.accepts(labels(bundle, ["A", "B", "A"]))


def test_discovers_multi_token_before_path_implication():
    """``ifL(before, ifL(orL(token x, token y), orL(token a, token b)))``
    compiles to ``after_token_allowed_dfa`` over both trigger and allowed sets.

    Exercises the multi-token generalisation: the trigger antecedent and the
    allowed consequent each carry an ``orL`` of path-bound ``token_value``\\ s
    on the same role.  Semantics: for every (p1, p2) under ``is_before_rel``,
    if p1 emits one of the trigger tokens then p2 must emit one of the allowed
    tokens.
    """
    graph, bundle = build_bundle()
    ctx = bundle.context
    with graph:
        ifL(
            ctx.is_before_rel("before"),
            ifL(
                orL(
                    ctx.token_value("A", "x1", path=("before", ctx.first_token)),
                    ctx.token_value("B", "x2", path=("before", ctx.first_token)),
                ),
                orL(
                    ctx.token_value("<eos>", "y1", path=("before", ctx.second_token)),
                    ctx.token_value("A", "y2", path=("before", ctx.second_token)),
                ),
            ),
        )

    assert "ifL" in supported_lc_types(graph, bundle)
    dfa = constraints_to_dfa_from_graph(graph, bundle)
    # No trigger emitted yet: any single token is fine.
    assert dfa.accepts(labels(bundle, ["<eos>"]))
    # Triggered by A; allowed set is {<eos>, A}.
    assert dfa.accepts(labels(bundle, ["A", "A"]))
    assert dfa.accepts(labels(bundle, ["A", "<eos>"]))
    assert not dfa.accepts(labels(bundle, ["A", "B"]))
    # Triggered by B (second trigger token); same allowed set.
    assert dfa.accepts(labels(bundle, ["B", "A"]))
    assert not dfa.accepts(labels(bundle, ["B", "B"]))


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

    assert "existsAL" in supported_lc_types(graph, bundle)
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
        constraints_to_dfa_from_graph(graph, bundle, on_unsupported="warn")

    constraints_to_dfa_from_graph(graph, bundle, on_unsupported="ignore")
    with pytest.raises(ValueError, match="not supported by generation DFA discovery"):
        constraints_to_dfa_from_graph(graph, bundle, on_unsupported="error")


def test_unsupported_or_lc_branch_warns_or_errors():
    graph, bundle = build_bundle()
    with graph:
        orL(
            atLeastAL(bundle.context.token_value("A", "x"), 1),
            andL(bundle.context.token_value("A", "x"), bundle.context.token_value("B", "x")),
        )

    with pytest.warns(RuntimeWarning, match="not supported by generation DFA discovery"):
        constraints_to_dfa_from_graph(graph, bundle, on_unsupported="warn")

    with pytest.raises(ValueError, match="not supported by generation DFA discovery"):
        constraints_to_dfa_from_graph(graph, bundle, on_unsupported="error")


# --------------------------------------------------------------------------- #
# Normalization (step 3 of the LogicalConstraintsToDFAPipeline) tests         #
# --------------------------------------------------------------------------- #


def _build_atmost_b_one_dfa():
    """Reference DFA: ``atMostAL(token_value('B'), 1)`` on its own."""
    graph, bundle = build_bundle()
    with graph:
        atMostAL(bundle.context.token_value("B", "x"), 1)
    return constraints_to_dfa_from_graph(graph, bundle), bundle


def test_double_negation_collapses_to_atom():
    """``notL(notL(atMostAL(B, 1)))`` minimises to the same DFA as the atom alone."""
    reference, _ref_bundle = _build_atmost_b_one_dfa()

    graph, bundle = build_bundle()
    with graph:
        notL(notL(atMostAL(bundle.context.token_value("B", "x"), 1)))
    actual = constraints_to_dfa_from_graph(graph, bundle)

    assert len(actual.states) == len(reference.states)
    # Language equivalence as a sanity check.
    for symbols in [["A", "<eos>"], ["B", "<eos>"], ["B", "B"], ["A", "B", "A"]]:
        assert actual.accepts(labels(bundle, symbols)) == reference.accepts(labels(bundle, symbols))


def test_not_exists_collapses_to_forbidden():
    """``notL(existsAL(t))`` compiles to the same DFA as ``atMostAL(t, 0)``."""
    graph_ref, bundle_ref = build_bundle()
    with graph_ref:
        atMostAL(bundle_ref.context.token_value("A", "x"), 0)
    reference = constraints_to_dfa_from_graph(graph_ref, bundle_ref)

    graph, bundle = build_bundle()
    with graph:
        notL(existsAL(bundle.context.token_value("A", "x")))
    actual = constraints_to_dfa_from_graph(graph, bundle)

    assert len(actual.states) == len(reference.states)
    for symbols in [["B", "<eos>"], ["A"], ["A", "B"], ["<eos>"]]:
        assert actual.accepts(labels(bundle, symbols)) == reference.accepts(labels(bundle, symbols))


def test_duplicate_and_atoms_are_deduped():
    """``andL(A, A)`` produces the same DFA as ``A`` after dedup."""
    reference, _ = _build_atmost_b_one_dfa()

    graph, bundle = build_bundle()
    with graph:
        andL(
            atMostAL(bundle.context.token_value("B", "x"), 1),
            atMostAL(bundle.context.token_value("B", "y"), 1),
        )
    actual = constraints_to_dfa_from_graph(graph, bundle)

    assert len(actual.states) == len(reference.states)
    for symbols in [["A", "<eos>"], ["B", "<eos>"], ["B", "B"], ["A", "B", "A"]]:
        assert actual.accepts(labels(bundle, symbols)) == reference.accepts(labels(bundle, symbols))


def test_contradiction_is_empty_language():
    """``andL(A, notL(A))`` rejects every sequence (constant-folded to bottom)."""
    graph, bundle = build_bundle()
    with graph:
        andL(
            atMostAL(bundle.context.token_value("B", "x"), 1),
            notL(atMostAL(bundle.context.token_value("B", "y"), 1)),
        )
    dfa = constraints_to_dfa_from_graph(graph, bundle)

    # Every reasonable sequence must be rejected.
    for symbols in [[], ["A"], ["B"], ["A", "<eos>"], ["B", "B"], ["A", "B", "A"], ["A", "A", "B", "B", "B"]]:
        assert not dfa.accepts(labels(bundle, symbols)), f"accepted {symbols!r}"


def test_heterogeneous_and_salvages_regular_part():
    """``andL(regular, irregular)`` returns a DFA equivalent to the regular part.

    Without normalization the irregular sibling would cause the whole ``andL``
    to be marked unsupported and dropped from the discovered DFA set.  The
    normalizer salvages the regular subtree and surfaces a warning for the
    irregular sibling.

    The irregular sibling here is an ``andL`` whose direct children are raw
    concept tuples rather than nested LCs — a shape the structural matcher
    cannot compile to a DFA, the same pattern exercised by
    :func:`test_unsupported_or_lc_branch_warns_or_errors`.
    """
    reference, _ = _build_atmost_b_one_dfa()

    graph, bundle = build_bundle()
    with graph:
        andL(
            atMostAL(bundle.context.token_value("B", "x"), 1),
            andL(
                bundle.context.token_value("A", "y"),
                bundle.context.token_value("B", "y"),
            ),
        )

    with pytest.warns(RuntimeWarning, match="not supported by generation DFA discovery"):
        actual = constraints_to_dfa_from_graph(graph, bundle, on_unsupported="warn")

    assert len(actual.states) == len(reference.states)
    for symbols in [["A", "<eos>"], ["B", "<eos>"], ["B", "B"]]:
        assert actual.accepts(labels(bundle, symbols)) == reference.accepts(labels(bundle, symbols))


def test_de_morgan_negation_push_preserves_language():
    """``notL(andL(A, B))`` accepts the same set as ``orL(notL(A), notL(B))``."""
    graph_a, bundle_a = build_bundle()
    with graph_a:
        notL(
            andL(
                existsAL(bundle_a.context.token_value("A", "x")),
                existsAL(bundle_a.context.token_value("B", "y")),
            )
        )
    via_demorgan = constraints_to_dfa_from_graph(graph_a, bundle_a)

    graph_b, bundle_b = build_bundle()
    with graph_b:
        orL(
            notL(existsAL(bundle_b.context.token_value("A", "x"))),
            notL(existsAL(bundle_b.context.token_value("B", "y"))),
        )
    via_explicit_or = constraints_to_dfa_from_graph(graph_b, bundle_b)

    for symbols in [[], ["A"], ["B"], ["A", "B"], ["A", "B", "<eos>"], ["A", "A", "B"], ["<eos>"]]:
        assert via_demorgan.accepts(labels(bundle_a, symbols)) == via_explicit_or.accepts(labels(bundle_b, symbols)), symbols
