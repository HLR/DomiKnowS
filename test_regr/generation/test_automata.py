from domiknows.generation.dfa._constraints import (
    after_token_allowed_dfa,
    eos_closure_dfa,
    forbidden_token_dfa,
    max_non_eos_dfa,
    ordered_tokens_dfa,
    required_token_dfa,
    token_set_count_dfa,
)
from domiknows.generation.dfa import complement_dfa, product_dfa, union_dfa
from domiknows.generation.dfa.vocabulary import TokenVocabulary


def test_dfa_accepts_and_rejects_basic_sequences():
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>")
    dfa = product_dfa(
        [
            eos_closure_dfa(vocab),
            max_non_eos_dfa(vocab, 2),
            forbidden_token_dfa(vocab, "B"),
        ]
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
    dfa = required_token_dfa(vocab, "B")
    allowed = dfa.allowed_tokens(dfa.start_state, remaining_steps=1)

    assert allowed == {vocab.label_for_token("B")}


def test_ordered_tokens_constraint_tracks_progress():
    vocab = TokenVocabulary(["<eos>", "A", "B", "C"], eos_token="<eos>")
    dfa = ordered_tokens_dfa(vocab, ["A", "C"])

    a = vocab.label_for_token("A")
    b = vocab.label_for_token("B")
    c = vocab.label_for_token("C")

    assert dfa.accepts([b, a, b, c])
    assert not dfa.accepts([c, a])


def test_composite_all_of_constraint_intersects_children():
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>")
    dfa = product_dfa([required_token_dfa(vocab, "A"), forbidden_token_dfa(vocab, "B")])

    eos = vocab.label_for_token("<eos>")
    a = vocab.label_for_token("A")
    b = vocab.label_for_token("B")

    assert dfa.accepts([a, eos])
    assert not dfa.accepts([eos])
    assert not dfa.accepts([a, b])


def test_composite_any_of_constraint_unions_children():
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>")
    dfa = union_dfa([required_token_dfa(vocab, "A"), required_token_dfa(vocab, "B")])

    eos = vocab.label_for_token("<eos>")
    a = vocab.label_for_token("A")
    b = vocab.label_for_token("B")

    assert dfa.accepts([a, eos])
    assert dfa.accepts([b, eos])
    assert not dfa.accepts([eos])


def test_union_dfa_preserves_acceptance_and_allowed_tokens():
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>")
    dfa = union_dfa([required_token_dfa(vocab, "A"), required_token_dfa(vocab, "B")])

    eos = vocab.label_for_token("<eos>")
    a = vocab.label_for_token("A")
    b = vocab.label_for_token("B")

    assert dfa.accepts([a])
    assert dfa.accepts([b])
    assert not dfa.accepts([eos])
    assert dfa.allowed_tokens(dfa.start_state, remaining_steps=1) == {a, b}


def test_complement_dfa_flips_acceptance_without_dead_state_pruning():
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>")
    base = required_token_dfa(vocab, "A")
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
    dfa = complement_dfa(required_token_dfa(vocab, "A"))

    assert dfa.accepts([vocab.label_for_token("B")])
    assert not dfa.accepts([vocab.label_for_token("A")])


def test_token_set_count_constraint_counts_token_sets():
    vocab = TokenVocabulary(["<eos>", "A", "B", "C"], eos_token="<eos>")
    dfa = token_set_count_dfa(vocab, ("A", "B"), min_count=2)

    eos = vocab.label_for_token("<eos>")
    a = vocab.label_for_token("A")
    b = vocab.label_for_token("B")
    c = vocab.label_for_token("C")

    assert dfa.accepts([a, b, eos])
    assert dfa.accepts([a, a])
    assert not dfa.accepts([a, c, eos])


def test_after_token_allowed_constraint_blocks_later_tokens():
    vocab = TokenVocabulary(["<eos>", "A", "B", "C"], eos_token="<eos>")
    dfa = after_token_allowed_dfa(vocab, ("A",), ("B",))

    a = vocab.label_for_token("A")
    b = vocab.label_for_token("B")
    c = vocab.label_for_token("C")

    assert dfa.accepts([a, b, b])
    assert dfa.accepts([b, a])
    assert not dfa.accepts([a, b, c])
