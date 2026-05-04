from domiknows.generation import (
    all_of_constraints,
    any_of_constraints,
    TokenVocabulary,
    constraints_to_dfa,
    forbidden_token,
    max_non_eos,
    no_token_after_eos,
    ordered_tokens,
    required_token,
)
from domiknows.generation.automata import union_dfa


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
