from domiknows.generation import (
    AfterTokenAllowedConstraint,
    ComplementGenerationConstraint,
    all_of_constraints,
    any_of_constraints,
    TokenVocabulary,
    TokenSetCountConstraint,
    constraints_to_dfa,
    forbidden_token,
    max_non_eos,
    no_token_after_eos,
    ordered_tokens,
    required_token,
)
from domiknows.generation.automata import complement_dfa, union_dfa


def test_dfa_accepts_and_rejects_basic_sequences():
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>")
    dfa = constraints_to_dfa(
        [
            no_token_after_eos(),
            max_non_eos(2),
            forbidden_token("B"),
        ],
        vocab,
    )

    eos = vocab.label_for_token("<eos>")
    a = vocab.label_for_token("A")
    b = vocab.label_for_token("B")

    assert dfa.accepts([a, a, eos])
    assert not dfa.accepts([a, eos, a])
    assert not dfa.accepts([a, a, a])
    assert not dfa.accepts([b])


def test_remaining_steps_force_required_token():
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>")
    dfa = constraints_to_dfa([required_token("B")], vocab)
    allowed = dfa.allowed_tokens(dfa.start_state, remaining_steps=1)

    assert allowed == {vocab.label_for_token("B")}


def test_ordered_tokens_constraint_tracks_progress():
    vocab = TokenVocabulary(["<eos>", "A", "B", "C"], eos_token="<eos>")
    dfa = ordered_tokens(["A", "C"]).to_dfa(vocab)

    a = vocab.label_for_token("A")
    b = vocab.label_for_token("B")
    c = vocab.label_for_token("C")

    assert dfa.accepts([b, a, b, c])
    assert not dfa.accepts([c, a])


def test_composite_all_of_constraint_intersects_children():
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>")
    dfa = all_of_constraints([required_token("A"), forbidden_token("B")]).to_dfa(vocab)

    eos = vocab.label_for_token("<eos>")
    a = vocab.label_for_token("A")
    b = vocab.label_for_token("B")

    assert dfa.accepts([a, eos])
    assert not dfa.accepts([eos])
    assert not dfa.accepts([a, b])


def test_composite_any_of_constraint_unions_children():
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>")
    dfa = any_of_constraints([required_token("A"), required_token("B")]).to_dfa(vocab)

    eos = vocab.label_for_token("<eos>")
    a = vocab.label_for_token("A")
    b = vocab.label_for_token("B")

    assert dfa.accepts([a, eos])
    assert dfa.accepts([b, eos])
    assert not dfa.accepts([eos])


def test_union_dfa_preserves_acceptance_and_allowed_tokens():
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>")
    dfa = union_dfa([required_token("A").to_dfa(vocab), required_token("B").to_dfa(vocab)])

    eos = vocab.label_for_token("<eos>")
    a = vocab.label_for_token("A")
    b = vocab.label_for_token("B")

    assert dfa.accepts([a])
    assert dfa.accepts([b])
    assert not dfa.accepts([eos])
    assert dfa.allowed_tokens(dfa.start_state, remaining_steps=1) == {a, b}


def test_complement_dfa_flips_acceptance_without_dead_state_pruning():
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>")
    base = required_token("A").to_dfa(vocab)
    dfa = complement_dfa(base)

    eos = vocab.label_for_token("<eos>")
    a = vocab.label_for_token("A")
    b = vocab.label_for_token("B")

    assert dfa.accepts([eos])
    assert dfa.accepts([b, eos])
    assert not dfa.accepts([a])
    allowed = dfa.allowed_tokens(dfa.start_state, remaining_steps=1)
    assert a not in allowed
    assert {eos, b} <= allowed


def test_complement_generation_constraint_wraps_child_dfa():
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>")
    dfa = ComplementGenerationConstraint(required_token("A")).to_dfa(vocab)

    assert dfa.accepts([vocab.label_for_token("B")])
    assert not dfa.accepts([vocab.label_for_token("A")])


def test_token_set_count_constraint_counts_token_sets():
    vocab = TokenVocabulary(["<eos>", "A", "B", "C"], eos_token="<eos>")
    dfa = TokenSetCountConstraint(("A", "B"), min_count=2).to_dfa(vocab)

    eos = vocab.label_for_token("<eos>")
    a = vocab.label_for_token("A")
    b = vocab.label_for_token("B")
    c = vocab.label_for_token("C")

    assert dfa.accepts([a, b, eos])
    assert dfa.accepts([a, a])
    assert not dfa.accepts([a, c, eos])


def test_after_token_allowed_constraint_blocks_later_tokens():
    vocab = TokenVocabulary(["<eos>", "A", "B", "C"], eos_token="<eos>")
    dfa = AfterTokenAllowedConstraint(("A",), ("B",)).to_dfa(vocab)

    a = vocab.label_for_token("A")
    b = vocab.label_for_token("B")
    c = vocab.label_for_token("C")

    assert dfa.accepts([a, b, b])
    assert dfa.accepts([b, a])
    assert not dfa.accepts([a, b, c])
