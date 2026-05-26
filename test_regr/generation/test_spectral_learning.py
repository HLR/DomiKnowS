import pytest
import torch

from domiknows.generation.learners import (
    DFA,
    SpectralBasis,
    WeightedFiniteAutomaton,
    build_spectral_basis,
    constrained_hankel_matrix,
    hankel_matrix,
    spectral_learn_from_oracle,
    spectral_learn_from_counts,
    spectral_learn_from_samples,
    start_product_state,
    step_product_state,
)


def build_known_wfa():
    return WeightedFiniteAutomaton(
        initial=[1.0, 0.0],
        transitions={
            "a": [[0.5, 0.5], [0.0, 0.2]],
            "b": [[0.1, 0.0], [0.3, 0.4]],
        },
        final=[1.0, 0.0],
        symbols=["a", "b"],
    )


def build_only_a_dfa():
    return DFA(
        states=frozenset({"ok"}),
        alphabet=frozenset({"a", "b"}),
        transitions={("ok", "a"): "ok"},
        start_state="ok",
        accepting_states=frozenset({"ok"}),
    )


def test_oracle_learning_recovers_known_wfa_hankel_values():
    source = build_known_wfa()
    basis = build_spectral_basis(["a", "b"], max_prefix_len=2, max_suffix_len=2)

    result = spectral_learn_from_oracle(
        source.sequence_probability,
        symbols=["a", "b"],
        rank=2,
        basis=basis,
    )

    expected = hankel_matrix(source, basis.prefixes, basis.suffixes)
    learned = hankel_matrix(result.model, basis.prefixes, basis.suffixes)

    assert result.rank == 2
    assert len(result.singular_values) == min(len(basis.prefixes), len(basis.suffixes))
    assert result.diagnostics["relative_reconstruction_error"] < 1e-8
    assert torch.allclose(learned, expected.to(dtype=learned.dtype), atol=1e-8)


def test_sample_learning_scores_frequent_strings_above_rare_strings():
    basis = build_spectral_basis(["a", "b"], max_prefix_len=1, max_suffix_len=1)
    result = spectral_learn_from_samples(
        [("a",), ("a",), ("a",), ("a",), ("a",), ("b",), ("b",), ("a", "b")],
        symbols=["a", "b"],
        rank=2,
        basis=basis,
    )

    assert result.model.sequence_probability(("a",)) > result.model.sequence_probability(("b",))
    assert result.model.sequence_probability(("a",)) > result.model.sequence_probability(("a", "a"))
    assert result.diagnostics["retained_singular_fraction"] <= 1.0
    assert result.diagnostics["max_score"] >= result.diagnostics["min_score"]


def test_count_learning_matches_repeated_sample_learning():
    basis = build_spectral_basis(["a", "b"], max_prefix_len=1, max_suffix_len=1)
    from_counts = spectral_learn_from_counts(
        {("a",): 5, ("b",): 2, ("a", "b"): 1},
        symbols=["a", "b"],
        rank=2,
        basis=basis,
    )
    from_samples = spectral_learn_from_samples(
        [("a",), ("a",), ("a",), ("a",), ("a",), ("b",), ("b",), ("a", "b")],
        symbols=["a", "b"],
        rank=2,
        basis=basis,
    )

    assert torch.allclose(
        hankel_matrix(from_counts.model, basis.prefixes, basis.suffixes),
        hankel_matrix(from_samples.model, basis.prefixes, basis.suffixes),
        atol=1e-6,
    )


def test_learned_wfa_works_with_hankel_projection_and_product_state():
    source = build_known_wfa()
    basis = build_spectral_basis(["a", "b"], max_prefix_len=1, max_suffix_len=1)
    result = spectral_learn_from_oracle(source.sequence_probability, ["a", "b"], rank=2, basis=basis)
    dfa = build_only_a_dfa()

    projected = constrained_hankel_matrix(result.model, dfa, basis.prefixes, basis.suffixes)
    state = start_product_state(result.model, dfa)
    after_a = step_product_state(result.model, dfa, state, "a")

    assert float(projected[basis.prefixes.index(("a",)), basis.suffixes.index(("b",))]) == 0.0
    assert after_a is not None
    assert after_a.dfa_state == "ok"
    assert step_product_state(result.model, dfa, after_a, "b") is None


def test_rank_truncation_exposes_singular_values_and_diagnostics():
    source = build_known_wfa()
    basis = build_spectral_basis(["a", "b"], max_prefix_len=2, max_suffix_len=2)

    full = spectral_learn_from_oracle(source.sequence_probability, ["a", "b"], rank=2, basis=basis)
    truncated = spectral_learn_from_oracle(source.sequence_probability, ["a", "b"], rank=1, basis=basis)

    assert len(truncated.singular_values) == len(full.singular_values)
    assert truncated.diagnostics["retained_singular_fraction"] < full.diagnostics["retained_singular_fraction"]
    assert truncated.diagnostics["reconstruction_error"] > full.diagnostics["reconstruction_error"]


def test_signed_scores_are_preserved_and_reported():
    basis = build_spectral_basis(["a"], max_prefix_len=1, max_suffix_len=1)

    def signed_oracle(sequence):
        return (-0.5) ** len(sequence)

    result = spectral_learn_from_oracle(signed_oracle, ["a"], rank=1, basis=basis)

    assert result.model.sequence_probability(("a",)) < 0.0
    assert result.diagnostics["negative_score_count"] > 0.0
    assert result.diagnostics["min_score"] < 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"symbols": []}, "symbols must not be empty"),
        ({"symbols": ["a", "a"]}, "symbols must be unique"),
        ({"rank": 0}, "rank must be at least 1"),
        ({"rank": 3}, "rank cannot exceed"),
        (
            {"basis": SpectralBasis(prefixes=[("a",)], suffixes=[()], symbols=["a"])},
            "prefixes must include the empty sequence",
        ),
        (
            {"basis": SpectralBasis(prefixes=[()], suffixes=[("a",)], symbols=["a"])},
            "suffixes must include the empty sequence",
        ),
    ],
)
def test_oracle_learning_rejects_invalid_inputs(kwargs, message):
    args = {
        "probability_fn": lambda sequence: 1.0 if not sequence else 0.0,
        "symbols": ["a"],
        "rank": 1,
        "basis": build_spectral_basis(["a"], max_prefix_len=1, max_suffix_len=1),
    }
    args.update(kwargs)

    with pytest.raises(ValueError, match=message):
        spectral_learn_from_oracle(**args)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sequences": []}, "sequences must not be empty"),
        ({"sequences": [("a", "c")]}, "unknown symbol 'c'"),
        ({"smoothing": -1.0}, "smoothing must be non-negative"),
    ],
)
def test_sample_learning_rejects_invalid_inputs(kwargs, message):
    args = {
        "sequences": [("a",), ("b",)],
        "symbols": ["a", "b"],
        "rank": 1,
        "basis": build_spectral_basis(["a", "b"], max_prefix_len=1, max_suffix_len=1),
    }
    args.update(kwargs)

    with pytest.raises(ValueError, match=message):
        spectral_learn_from_samples(**args)
