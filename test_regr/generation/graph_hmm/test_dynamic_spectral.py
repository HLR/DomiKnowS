"""
Tests for dynamic constraints in GraphSpectralAutomaton.

This module tests context-aware constraints for spectral learning-based automata:
- Operator transforms: Dynamically modify symbol operators based on context
- Operator energy: Add soft penalty terms to symbol operators
- Dynamic context: Verify context information available to constraint callbacks
- Hankel reconstruction: Test that dynamic changes are reflected in matrix reconstructions

Dynamic constraints in spectral learning enable prefix/suffix-aware behaviors where the
meaning of a symbol can depend on what has been observed so far.
"""
import pytest
import torch

from domiknows.generation.graph_hmm import DynamicConstraintContext, GraphSpectralAutomaton


def _fit_dynamic_spectral(**kwargs):
    """
    Create and fit a basic GraphSpectralAutomaton for testing.
    
    This fixture sets up a simple automaton with symbols ["x", "y"] and fits it on
    a small training set with defined prefixes and suffixes for spectral learning.
    """
    automaton = GraphSpectralAutomaton(symbols=["x", "y"], **kwargs)
    automaton.fit(
        [["x"], ["y"], ["x", "y"], ["x", "y"], ["y", "x"]],
        prefixes=[(), ("x",), ("y",)],
        suffixes=[(), ("x",), ("y",)],
        rank=2,
    )
    return automaton


def test_dynamic_operator_transform_changes_score_without_changing_static_operator():
    """
    Test that operator_transform dynamically modifies operators during scoring.
    
    The static operator stored in the automaton should not change, but the effective
    operator used during scoring should be transformed based on the context.
    """
    def operator_transform(context: DynamicConstraintContext, symbol, base):
        if symbol == "y":
            return base * 0.25
        return base

    static = _fit_dynamic_spectral()
    dynamic = _fit_dynamic_spectral(operator_transform=operator_transform)
    static_y = dynamic.operator("y").clone()

    assert dynamic.score(["x", "y"]) != static.score(["x", "y"])
    assert torch.allclose(dynamic.operator("y"), static_y)


def test_dynamic_energy_penalizes_entries_without_changing_sign_semantics():
    """
    Test that operator_energy adds soft penalties while preserving sign semantics.
    
    Energy is added to the operator matrix, which reduces magnitude but preserves
    the sign (positive/negative) of entries.
    """
    def operator_energy(context: DynamicConstraintContext, symbol):
        if symbol != "x":
            return None
        energy = torch.zeros((2, 2), dtype=torch.float64)
        energy[0, 0] = 2.0
        return energy

    automaton = _fit_dynamic_spectral(operator_energy=operator_energy, energy_weight=1.0)
    context = DynamicConstraintContext(step=0, prefix=(), belief=automaton.initial, sequence=("x",))
    base = automaton.operator("x")
    effective = automaton.operator_for_context("x", context)

    assert torch.sign(effective[0, 0]) == torch.sign(base[0, 0])
    assert abs(effective[0, 0].item()) < abs(base[0, 0].item())


def test_dynamic_context_contains_step_prefix_belief_sequence_and_metadata():
    """
    Test that DynamicConstraintContext provides complete state information.
    
    The context should include: current step, observed prefix, current belief state,
    full sequence being scored, and optional metadata passed to the automaton.
    """
    seen = []

    def operator_transform(context: DynamicConstraintContext, symbol, base):
        seen.append((context.step, context.prefix, context.belief.clone(), context.sequence, context.metadata["tag"]))
        return base

    automaton = _fit_dynamic_spectral(operator_transform=operator_transform, dynamic_metadata={"tag": "spectral-test"})
    automaton.score(["x", "y"])

    assert seen[0][0] == 0
    assert seen[0][1] == ()
    assert seen[0][3] == ("x", "y")
    assert seen[0][4] == "spectral-test"
    assert seen[1][0] == 1
    assert seen[1][1] == ("x",)
    assert seen[1][2].shape == (2,)


def test_prefix_state_uses_different_operator_for_same_symbol_at_different_prefixes():
    """
    Test that prefix-dependent transforms change the computed prefix state.
    
    When a symbol is processed at different prefixes with different transformations,
    the resulting state should be different.
    """
    def operator_transform(context: DynamicConstraintContext, symbol, base):
        if symbol == "x" and context.prefix:
            return base * 0.1
        return base

    static = _fit_dynamic_spectral()
    dynamic = _fit_dynamic_spectral(operator_transform=operator_transform)

    assert not torch.allclose(dynamic.prefix_state(["x", "x"]), static.prefix_state(["x", "x"]))
    assert torch.allclose(dynamic.prefix_state(["x"]), static.prefix_state(["x"]))


def test_invalid_dynamic_operator_and_energy_shapes_raise_clear_errors():
    """Test that mismatched shapes in dynamic operators/energy are caught with clear errors."""
    def bad_transform(context: DynamicConstraintContext, symbol, base):
        return torch.ones((3, 3), dtype=torch.float64)

    transform_model = _fit_dynamic_spectral(operator_transform=bad_transform)
    with pytest.raises(ValueError, match="operator_transform"):
        transform_model.score(["x"])

    def bad_energy(context: DynamicConstraintContext, symbol):
        return torch.ones((3, 3), dtype=torch.float64)

    energy_model = _fit_dynamic_spectral(operator_energy=bad_energy)
    with pytest.raises(ValueError, match="transition energy"):
        energy_model.score(["x"])


def test_reconstruct_hankel_supports_dynamic_traversal():
    """
    Test that Hankel reconstruction correctly applies dynamic operators.
    
    When reconstructing the Hankel matrix (which records input-output responses),
    dynamic transformations should result in different reconstructions compared to static.
    """
    def operator_transform(context: DynamicConstraintContext, symbol, base):
        if context.prefix:
            return base * 0.5
        return base

    automaton = _fit_dynamic_spectral(operator_transform=operator_transform)
    static_hankel = automaton.reconstruct_hankel(dynamic=False)
    dynamic_hankel = automaton.reconstruct_hankel(dynamic=True)

    assert static_hankel.shape == dynamic_hankel.shape
    assert not torch.allclose(static_hankel, dynamic_hankel)
