import warnings

import pytest
import torch

from domiknows.generation import (
    AnyOfGenerationConstraint,
    HMMFactorGraphEncoder,
    LabelRef,
    GenerationEncoder,
    GraphLatentCompilerResult,
    LatentWindowSpec,
    LatentTransitionPotential,
    RequiredTokenConstraint,
    constraints_to_dfa,
    discover_generation_enforcement,
    discover_latent_window_specs,
    discover_transition_potentials,
    graph_latent_compiler_result,
    mark_for_both,
    mark_for_dfa,
    mark_for_latent,
    required_token,
    window_formula_loss,
)
from domiknows.graph.logicalConstrain import andL, atLeastAL, existsAL, ifL, notL, orL


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


def test_mark_for_dfa_routes_explicit_constraint():
    graph, bundle = build_bundle()
    with graph:
        lc = andL(bundle.context.token_value("A", "x"), bundle.context.token_value("B", "x"))
        mark_for_dfa(lc, required_token("A"))

    enforcement = discover_generation_enforcement(graph, bundle)

    assert any(
        isinstance(constraint, RequiredTokenConstraint) and constraint.token == "A"
        for constraint in enforcement.dfa_constraints
    )
    assert enforcement.latent_specs == ()


def test_mark_for_latent_routes_only_to_latent_specs():
    graph, bundle = build_bundle()
    a = bundle.vocabulary.label_for_token("A")
    b = bundle.vocabulary.label_for_token("B")
    spec = LatentWindowSpec(if_label=a, formula=b, window=2, weight=0.5)
    with graph:
        lc = ifL(
            existsAL(bundle.context.token_value("A", "x")),
            existsAL(bundle.context.token_value("B", "y")),
        )
        mark_for_latent(lc, spec)

    enforcement = discover_generation_enforcement(graph, bundle)

    assert enforcement.dfa_constraints == ()
    assert enforcement.latent_specs == (spec,)


def test_mark_for_both_routes_to_dfa_and_latent():
    graph, bundle = build_bundle()
    a = bundle.vocabulary.label_for_token("A")
    b = bundle.vocabulary.label_for_token("B")
    spec = LatentWindowSpec(if_label=a, formula=b, window=2)
    with graph:
        lc = atLeastAL(bundle.context.token_value("A", "x"), 1)
        mark_for_both(lc, constraint=required_token("A"), spec=spec)

    enforcement = discover_generation_enforcement(graph, bundle)

    assert any(
        isinstance(constraint, RequiredTokenConstraint) and constraint.token == "A"
        for constraint in enforcement.dfa_constraints
    )
    assert enforcement.latent_specs == (spec,)


def test_auto_discovered_dfa_constraints_still_work_with_enforcement():
    graph, bundle = build_bundle()
    with graph:
        atLeastAL(bundle.context.token_value("A", "x"), 1)

    enforcement = discover_generation_enforcement(graph, bundle)
    dfa = constraints_to_dfa(enforcement.dfa_constraints, bundle.vocabulary)
    a = bundle.vocabulary.label_for_token("A")
    eos = bundle.vocabulary.label_for_token("<eos>")

    assert dfa.accepts([a, eos])
    assert not dfa.accepts([eos])


def test_auto_discovered_boolean_dfa_constraints_work_with_enforcement():
    graph, bundle = build_bundle()
    with graph:
        orL(
            atLeastAL(bundle.context.token_value("A", "x"), 1),
            atLeastAL(bundle.context.token_value("B", "y"), 1),
        )

    enforcement = discover_generation_enforcement(graph, bundle)

    assert len(enforcement.dfa_constraints) == 1
    assert isinstance(enforcement.dfa_constraints[0], AnyOfGenerationConstraint)
    dfa = constraints_to_dfa(enforcement.dfa_constraints, bundle.vocabulary)
    a = bundle.vocabulary.label_for_token("A")
    b = bundle.vocabulary.label_for_token("B")
    eos = bundle.vocabulary.label_for_token("<eos>")

    assert dfa.accepts([a, eos])
    assert dfa.accepts([b, eos])
    assert not dfa.accepts([eos])


def test_latent_loss_matches_window_formula_loss():
    graph, bundle = build_bundle()
    a = bundle.vocabulary.label_for_token("A")
    b = bundle.vocabulary.label_for_token("B")
    spec = LatentWindowSpec(if_label=a, formula=b, window=2, weight=0.25)
    with graph:
        lc = ifL(
            existsAL(bundle.context.token_value("A", "x")),
            existsAL(bundle.context.token_value("B", "y")),
        )
        mark_for_latent(lc, spec)

    probs = torch.zeros((4, 3))
    probs[:, bundle.vocabulary.label_for_token("<eos>")] = 1.0
    probs[0, a] = 0.8
    probs[1, b] = 0.5
    enforcement = discover_generation_enforcement(graph, bundle)

    assert enforcement.latent_loss(probs).item() == pytest.approx(
        0.25 * window_formula_loss(probs, a, b, 2).item()
    )


def test_empty_latent_specs_return_zero_like_loss():
    graph, bundle = build_bundle()
    probs = torch.ones((2, 3), dtype=torch.float64)

    loss = discover_generation_enforcement(graph, bundle).latent_loss(probs)

    assert loss.item() == pytest.approx(0.0)
    assert loss.dtype == torch.float64
    assert loss.device == probs.device


def test_unsupported_unmarked_generation_constraint_still_warns_or_errors():
    graph, bundle = build_bundle()
    with graph:
        andL(bundle.context.token_value("A", "x"), bundle.context.token_value("B", "x"))

    with pytest.warns(RuntimeWarning, match="not supported by generation DFA discovery"):
        discover_generation_enforcement(graph, bundle, on_unsupported="warn")

    with pytest.raises(ValueError, match="not supported by generation DFA discovery"):
        discover_generation_enforcement(graph, bundle, on_unsupported="error")


def test_auto_latent_discovery_compiles_adjacent_factor_rule():
    encoder = HMMFactorGraphEncoder(
        ["<eos>", "A", "B"],
        eos_token="<eos>",
        tokenizer=FakeTokenizer(),
        state_names=("S0", "S1"),
        include_dp_factors=False,
    )
    graph, bundle = encoder.build_graph()
    with graph:
        ifL(
            bundle.context.is_next_rel("next"),
            ifL(
                bundle.context.latent_state_value("S0", "x", path=("next", bundle.context.current_token)),
                bundle.context.latent_state_value("S1", "y", path=("next", bundle.context.next_token)),
            ),
        )

    specs = discover_latent_window_specs(graph, bundle, mode="auto", on_unsupported="ignore")

    assert len(specs) == 1
    assert specs[0].if_label == LabelRef("latent_state", 0)
    assert specs[0].formula == LabelRef("latent_state", 1)
    assert specs[0].window == 1


def test_transition_potential_discovery_compiles_forbidden_adjacent_state():
    encoder = HMMFactorGraphEncoder(
        ["<eos>", "A", "B"],
        eos_token="<eos>",
        tokenizer=FakeTokenizer(),
        state_names=("S0", "S1"),
        include_dp_factors=False,
    )
    graph, bundle = encoder.build_graph()
    with graph:
        ifL(
            bundle.context.is_next_rel("next"),
            ifL(
                bundle.context.latent_state_value("S0", "x", path=("next", bundle.context.current_token)),
                notL(bundle.context.latent_state_value("S1", "y", path=("next", bundle.context.next_token))),
            ),
        )

    potentials = discover_transition_potentials(graph, bundle)

    assert len(potentials) == 1
    values = potentials[0].tensor_for(torch.ones((2, 2)))
    assert values[0, 1].item() == pytest.approx(0.0)
    assert values[1, 1].item() == pytest.approx(1.0)


def test_custom_latent_compiler_adds_window_spec_to_enforcement():
    graph, bundle = build_bundle()
    a = bundle.vocabulary.label_for_token("A")
    b = bundle.vocabulary.label_for_token("B")
    custom_spec = LatentWindowSpec(if_label=a, formula=b, window=2, name="custom_a_then_b")
    with graph:
        andL(bundle.context.token_value("A", "x"), bundle.context.token_value("B", "x"))

    def compiler(lc, bundle):
        if lc.__class__.__name__ != "andL":
            return None
        return graph_latent_compiler_result(latent_specs=custom_spec, compiler_name="project")

    enforcement = discover_generation_enforcement(
        graph,
        bundle,
        on_unsupported="ignore",
        extra_latent_compilers=[compiler],
    )

    assert enforcement.latent_specs == (custom_spec,)


def test_custom_latent_compiler_adds_transition_potential_to_enforcement():
    graph, bundle = build_bundle()
    custom_potential = LatentTransitionPotential(torch.eye(2), name="custom_transition_bias")
    with graph:
        andL(bundle.context.token_value("A", "x"), bundle.context.token_value("B", "x"))

    def compiler(lc, bundle):
        if lc.__class__.__name__ != "andL":
            return None
        return graph_latent_compiler_result(
            transition_potentials=custom_potential,
            compiler_name="project",
        )

    enforcement = discover_generation_enforcement(
        graph,
        bundle,
        on_unsupported="ignore",
        extra_latent_compilers=[compiler],
    )

    assert enforcement.transition_potentials == (custom_potential,)


def test_builtin_latent_compiler_wins_over_custom_compiler():
    encoder = HMMFactorGraphEncoder(
        ["<eos>", "A", "B"],
        eos_token="<eos>",
        tokenizer=FakeTokenizer(),
        state_names=("S0", "S1"),
        include_dp_factors=False,
    )
    graph, bundle = encoder.build_graph()
    with graph:
        ifL(
            bundle.context.is_next_rel("next"),
            ifL(
                bundle.context.latent_state_value("S0", "x", path=("next", bundle.context.current_token)),
                bundle.context.latent_state_value("S1", "y", path=("next", bundle.context.next_token)),
            ),
        )

    custom_spec = LatentWindowSpec(if_label=0, formula=0, window=3, name="should_not_appear")
    calls = []

    def compiler(lc, bundle):
        calls.append(lc)
        return graph_latent_compiler_result(latent_specs=custom_spec, compiler_name="project")

    specs = discover_latent_window_specs(
        graph,
        bundle,
        mode="auto",
        on_unsupported="ignore",
        extra_compilers=[compiler],
    )

    assert calls == []
    assert len(specs) == 1
    assert specs[0].name != "should_not_appear"
    assert specs[0].if_label == LabelRef("latent_state", 0)


def test_custom_latent_compiler_deduplicates_specs():
    graph, bundle = build_bundle()
    a = bundle.vocabulary.label_for_token("A")
    b = bundle.vocabulary.label_for_token("B")
    custom_spec = LatentWindowSpec(if_label=a, formula=b, window=2, name="duplicate")
    with graph:
        andL(bundle.context.token_value("A", "x"), bundle.context.token_value("B", "x"))

    def compiler(lc, bundle):
        return graph_latent_compiler_result(
            latent_specs=(custom_spec, custom_spec),
            compiler_name="project",
        )

    specs = discover_latent_window_specs(
        graph,
        bundle,
        on_unsupported="ignore",
        extra_compilers=[compiler],
    )

    assert specs == (custom_spec,)


def test_custom_latent_compiler_relevant_unsupported_warns_or_errors():
    graph, bundle = build_bundle()
    with graph:
        andL(bundle.context.token_value("A", "x"), bundle.context.token_value("B", "x"))

    def compiler(lc, bundle):
        return GraphLatentCompilerResult(
            relevant=True,
            supported=False,
            reason="project path requires runtime context",
            compiler_name="project",
        )

    with pytest.warns(RuntimeWarning, match="project path requires runtime context"):
        discover_latent_window_specs(graph, bundle, extra_compilers=[compiler], on_unsupported="warn")

    with pytest.raises(ValueError, match="project path requires runtime context"):
        discover_latent_window_specs(graph, bundle, extra_compilers=[compiler], on_unsupported="error")


def test_custom_latent_compiler_irrelevant_result_is_ignored():
    graph, bundle = build_bundle()
    with graph:
        andL(bundle.context.token_value("A", "x"), bundle.context.token_value("B", "x"))

    def compiler(lc, bundle):
        return graph_latent_compiler_result(
            relevant=False,
            supported=False,
            reason="ignored",
            compiler_name="project",
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        specs = discover_latent_window_specs(graph, bundle, extra_compilers=[compiler], on_unsupported="warn")

    assert specs == ()
    assert caught == []
