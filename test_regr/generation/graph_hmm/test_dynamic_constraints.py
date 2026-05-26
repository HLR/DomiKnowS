"""
Tests for dynamic constraints in DomiKnowSAwareHMM.

This module tests context-aware constraints that adapt based on the current state and history:
- Hard masks: Block transitions completely based on the prefix (observed sequence)
- Soft energy: Penalize transitions without blocking them
- Dynamic state space: Tests factorized state representations

Dynamic constraints enable complex reasoning where allowed behaviors depend on what has been
observed so far in the sequence.
"""
import torch
import pytest

from domiknows.generation.learners import (
    DomiKnowSAwareHMM,
    DynamicConstraintContext,
    FactorizedStateSpace,
    apply_transition_energy,
)


def _identity_emission_model(**kwargs):
    """
    Create a simple 2-state HMM where each state can only emit its corresponding symbol.
    
    This is used as a test fixture for dynamic constraint testing. Returns a HMM with:
    - 2 states: "A" and "B"
    - 2 symbols: "x" and "y"
    - Identity emission mask: state A emits x, state B emits y
    """
    return DomiKnowSAwareHMM(
        graph=None,
        n_hidden_states=2,
        state_names=["A", "B"],
        symbols=["x", "y"],
        emission_mask=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
        random_seed=3,
        **kwargs,
    )


def test_dynamic_hard_mask_blocks_paths_during_score_and_viterbi():
    """
    Test that dynamic transition masks completely block forbidden paths.
    
    When prefix is ("x",), only transitions from A->A and B->A/B are allowed.
    This forces the sequence ["x", "y"] to be impossible since state B cannot
    emit "x" in the first position after already being in B.
    """
    def dynamic_transition(context: DynamicConstraintContext):
        if context.prefix == ("x",):
            return torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
        return None

    model = _identity_emission_model(dynamic_transition=dynamic_transition)
    model.fit([["x", "x"], ["x", "x"]], max_iter=2)

    assert model.score(["x", "y"]) == float("-inf")
    assert model.viterbi(["x", "y"]).score == float("-inf")


def test_sampling_never_uses_dynamically_blocked_transition():
    """Test that sampling respects dynamic transition masks."""
    def dynamic_transition(context: DynamicConstraintContext):
        if context.prefix == ("x",):
            return torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
        return None

    model = _identity_emission_model(dynamic_transition=dynamic_transition)
    model.fit([["x", "x"], ["x", "x"]], max_iter=2)
    generator = torch.Generator()
    generator.manual_seed(9)

    assert model.sample(4, generator=generator) == ["x", "x", "x", "x"]


def test_dynamic_all_zero_row_stays_blocked_during_score_viterbi_and_sampling():
    """A dynamically all-zero outgoing row must not be revived by static projection."""

    def dynamic_transition(context: DynamicConstraintContext):
        if context.prefix == ("x",):
            return torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
        return None

    init = {
        "initial": torch.tensor([1.0, 0.0], dtype=torch.float64),
        "transition": torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float64),
        "emission": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
    }
    model = _identity_emission_model(dynamic_transition=dynamic_transition).fit(
        [["x", "x"]],
        max_iter=0,
        init=init,
    )

    assert model.score(["x", "x"]) == float("-inf")
    assert model.viterbi(["x", "x"]).score == float("-inf")
    with pytest.raises(RuntimeError, match="no dynamically allowed outgoing transition"):
        model.sample(2, generator=torch.Generator().manual_seed(0))


def test_soft_transition_energy_penalizes_without_forcing_zero():
    """
    Test that soft energy constraints reduce probability without eliminating it.
    
    Unlike hard masks that set probability to 0, energy penalties reduce the probability
    of certain transitions based on their energy cost and the energy weight.
    """
    def transition_energy(context: DynamicConstraintContext):
        energy = torch.zeros((2, 2), dtype=torch.float64)
        energy[0, 1] = 10.0
        return energy

    init = {
        "initial": torch.tensor([1.0, 0.0], dtype=torch.float64),
        "transition": torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float64),
        "emission": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
    }
    plain = _identity_emission_model().fit([["x", "y"]], max_iter=0, init=init)
    penalized = _identity_emission_model(transition_energy=transition_energy, energy_weight=1.0).fit(
        [["x", "y"]],
        max_iter=0,
        init=init,
    )

    assert penalized.score(["x", "y"]) < plain.score(["x", "y"])
    assert penalized.score(["x", "y"]) != float("-inf")


def test_apply_transition_energy_returns_weighted_matrix():
    """Test that apply_transition_energy correctly weights transitions by energy."""
    transition = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    energy = torch.tensor([[0.0, 2.0]], dtype=torch.float64)
    weighted = apply_transition_energy(transition, energy, weight=1.0)

    assert weighted[0, 0].item() == 0.5
    assert 0.0 < weighted[0, 1].item() < 0.5


def test_factorized_state_space_maps_states_and_builds_masks():
    """
    Test that FactorizedStateSpace correctly manages structured state representations.
    
    Factorized states decompose into multiple independent dimensions (factors), enabling
    constraint specification based on individual dimensions rather than flat state indices.
    """
    space = FactorizedStateSpace.from_factors(
        {
            "entity": ["Person", "Cup"],
            "relation": ["holding", "on_table"],
        }
    )

    assert len(space) == 4
    cup_holding = space.state_id(entity="Cup", relation="holding")
    assert space.state_dict(cup_holding) == {"entity": "Cup", "relation": "holding"}

    no_holding_to_table = space.transition_mask(
        lambda src, dst: not (src["relation"] == "holding" and dst["relation"] == "on_table")
    )
    src = space.state_id(entity="Person", relation="holding")
    dst = space.state_id(entity="Cup", relation="on_table")
    assert no_holding_to_table[src, dst].item() == 0.0
    assert no_holding_to_table[dst, src].item() == 1.0


def test_dynamic_identity_constraint_matches_static_projection_semantics():
    """Ensure static and dynamic paths use equivalent transition projection behavior."""

    init = {
        "initial": torch.tensor([1.0, 0.0], dtype=torch.float64),
        "transition": torch.tensor([[0.0, 0.0], [0.5, 0.5]], dtype=torch.float64),
        "emission": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
    }

    static_model = _identity_emission_model().fit([["x", "x"]], max_iter=0, init=init)
    dynamic_model = _identity_emission_model(
        dynamic_transition=lambda context: torch.ones((2, 2), dtype=torch.float64)
    ).fit([["x", "x"]], max_iter=0, init=init)

    assert static_model.score(["x", "x"]) == dynamic_model.score(["x", "x"])
