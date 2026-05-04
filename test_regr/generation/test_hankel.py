import pytest

from domiknows.generation.automata import (
    DFA,
    WeightedFiniteAutomaton,
    allowed_product_symbols,
    constrained_hankel_matrix,
    hankel_matrix,
    projection_summary,
    start_product_state,
    step_product_state,
)


def build_toy_wfa():
    return WeightedFiniteAutomaton(
        initial=[1.0, 0.0],
        transitions={
            "a": [[0.5, 0.5], [0.0, 0.2]],
            "b": [[0.1, 0.0], [0.3, 0.4]],
        },
        final=[1.0, 0.0],
        symbols=["a", "b"],
    )


def build_no_b_after_a_dfa():
    return DFA(
        states=frozenset({"start", "seen_a"}),
        alphabet=frozenset({"a", "b"}),
        transitions={
            ("start", "a"): "seen_a",
            ("start", "b"): "start",
            ("seen_a", "a"): "seen_a",
        },
        start_state="start",
        accepting_states=frozenset({"start", "seen_a"}),
    )


def test_wfa_probability_matches_manual_matrix_multiplication():
    wfa = build_toy_wfa()

    assert wfa.prefix_state(()) == pytest.approx((1.0, 0.0))
    assert wfa.prefix_state(("a",)) == pytest.approx((0.5, 0.5))
    assert wfa.prefix_state(("a", "b")) == pytest.approx((0.2, 0.2))
    assert wfa.sequence_probability(("a", "b")) == pytest.approx(0.2)


def test_hankel_matrix_entries_are_prefix_suffix_probabilities():
    wfa = build_toy_wfa()
    prefixes = [(), ("a",), ("b",)]
    suffixes = [(), ("a",), ("b",)]

    matrix = hankel_matrix(wfa, prefixes, suffixes)

    assert matrix[0][0] == pytest.approx(wfa.sequence_probability(()))
    assert matrix[1][2] == pytest.approx(wfa.sequence_probability(("a", "b")))
    assert matrix[2][1] == pytest.approx(wfa.sequence_probability(("b", "a")))


def test_constrained_hankel_projects_rejected_strings_to_zero():
    wfa = build_toy_wfa()
    dfa = build_no_b_after_a_dfa()
    prefixes = [(), ("a",), ("b",)]
    suffixes = [(), ("a",), ("b",)]

    original = hankel_matrix(wfa, prefixes, suffixes)
    constrained = constrained_hankel_matrix(wfa, dfa, prefixes, suffixes)

    assert constrained[1][2] == 0.0
    assert original[1][2] == pytest.approx(wfa.sequence_probability(("a", "b")))
    assert constrained[2][1] == pytest.approx(wfa.sequence_probability(("b", "a")))


def test_projection_summary_reports_retained_mass_and_nonzero_counts():
    original = [[1.0, 2.0], [3.0, 4.0]]
    constrained = [[1.0, 0.0], [3.0, 0.0]]

    summary = projection_summary(original, constrained)

    assert summary["original_mass"] == pytest.approx(10.0)
    assert summary["constrained_mass"] == pytest.approx(4.0)
    assert summary["retained_mass"] == pytest.approx(4.0)
    assert summary["retained_fraction"] == pytest.approx(0.4)
    assert summary["original_nonzero"] == 4.0
    assert summary["constrained_nonzero"] == 2.0


def test_product_state_advances_wfa_and_dfa_together():
    wfa = build_toy_wfa()
    dfa = build_no_b_after_a_dfa()

    start = start_product_state(wfa, dfa)
    after_a = step_product_state(wfa, dfa, start, "a")

    assert start.wfa_state == pytest.approx(wfa.initial)
    assert start.dfa_state == "start"
    assert after_a is not None
    assert after_a.wfa_state == pytest.approx(wfa.prefix_state(("a",)))
    assert after_a.dfa_state == "seen_a"
    assert after_a.score == pytest.approx(wfa.sequence_probability(("a",)))


def test_product_state_returns_none_for_blocked_dfa_transition():
    wfa = build_toy_wfa()
    dfa = build_no_b_after_a_dfa()

    after_a = step_product_state(wfa, dfa, start_product_state(wfa, dfa), "a")

    assert after_a is not None
    assert step_product_state(wfa, dfa, after_a, "b") is None


def test_allowed_product_symbols_matches_active_dfa_state():
    wfa = build_toy_wfa()
    dfa = build_no_b_after_a_dfa()

    start = start_product_state(wfa, dfa)
    after_a = step_product_state(wfa, dfa, start, "a")

    assert allowed_product_symbols(wfa, dfa, start) == {"a", "b"}
    assert after_a is not None
    assert allowed_product_symbols(wfa, dfa, after_a) == {"a"}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"initial": []}, "initial must not be empty"),
        ({"final": [1.0]}, "initial and final vectors"),
        ({"symbols": []}, "symbols must not be empty"),
        ({"symbols": ["a", "a"]}, "symbols must be unique"),
        ({"transitions": {"a": [[1.0, 0.0], [0.0, 1.0]]}}, "missing symbol"),
        (
            {
                "transitions": {
                    "a": [[1.0, 0.0], [0.0, 1.0]],
                    "b": [[1.0, 0.0], [0.0, 1.0]],
                    "c": [[1.0, 0.0], [0.0, 1.0]],
                }
            },
            "unknown symbol",
        ),
        ({"transitions": {"a": [[1.0]], "b": [[1.0, 0.0], [0.0, 1.0]]}}, "must have 2 rows"),
        ({"transitions": {"a": [[1.0], [0.0]], "b": [[1.0, 0.0], [0.0, 1.0]]}}, "must be square"),
    ],
)
def test_wfa_rejects_invalid_shapes(kwargs, message):
    args = {
        "initial": [1.0, 0.0],
        "transitions": {
            "a": [[1.0, 0.0], [0.0, 1.0]],
            "b": [[1.0, 0.0], [0.0, 1.0]],
        },
        "final": [1.0, 0.0],
        "symbols": ["a", "b"],
    }
    args.update(kwargs)

    with pytest.raises(ValueError, match=message):
        WeightedFiniteAutomaton(**args)


def test_wfa_rejects_unknown_sequence_symbol():
    wfa = build_toy_wfa()

    with pytest.raises(ValueError, match="unknown symbol 'c'"):
        wfa.sequence_probability(("c",))
