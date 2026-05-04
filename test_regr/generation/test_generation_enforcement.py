import pytest
import torch

from domiknows.generation import (
    AnyOfGenerationConstraint,
    GenerationEncoder,
    LatentWindowSpec,
    RequiredTokenConstraint,
    constraints_to_dfa,
    discover_generation_enforcement,
    mark_for_both,
    mark_for_dfa,
    mark_for_latent,
    required_token,
    window_formula_loss,
)
from domiknows.graph.logicalConstrain import andL, atLeastAL, existsAL, ifL, orL


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
