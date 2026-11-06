import warnings

import pytest
import torch

from domiknows.generation import (
    GenerationEncoder,
    HMMFactorGraphEncoder,
    LabelRef,
    LatentWindowSpec,
    SpectralWFAFactorGraphEncoder,
    adjacent_implication_recipe,
    bounded_lookahead_recipe,
    combined_transition_potential_from_graph,
    common_latent_compiler_recipes,
    discover_generation_enforcement,
    discover_latent_window_specs,
    discover_transition_potentials,
    forbidden_transition_potential_recipe,
    mark_for_latent,
    register_latent_compilers,
    registered_latent_compilers,
    window_formula_loss,
)
from domiknows.generation.learners.wfa.factors import apply_wfa_factor_consistency_constraints
from domiknows.graph.logicalConstrain import andL, atLeastAL, ifL, nandL, notL, orL

S0 = LabelRef("latent_state", 0)
S1 = LabelRef("latent_state", 1)


class FakeTokenizer:
    def encode(self, token):
        return {"<eos>": [0], "A": [1], "B": [2]}[token]


def token_bundle():
    return GenerationEncoder(["<eos>", "A", "B"], eos_token="<eos>", tokenizer=FakeTokenizer()).build_graph()


def hmm_bundle():
    return HMMFactorGraphEncoder(
        ["<eos>", "A", "B"],
        eos_token="<eos>",
        tokenizer=FakeTokenizer(),
        state_names=("S0", "S1"),
        include_dp_factors=False,
    ).build_graph()


def current(ctx, state, variable="x"):
    return ctx.latent_state_value(state, variable, path=("next", ctx.current_token))


def following(ctx, state, variable="y"):
    return ctx.latent_state_value(state, variable, path=("next", ctx.next_token))


def add_token_lc(graph, bundle):
    with graph:
        andL(bundle.context.token_value("A", "x"), bundle.context.token_value("B", "x"))


def add_adjacent_rule(graph, bundle):
    ctx = bundle.context
    with graph:
        return ifL(ctx.is_next_rel("next"), ifL(current(ctx, "S0"), following(ctx, "S1")))


def test_registered_recipe_applies_without_being_passed_to_discovery():
    graph, bundle = token_bundle()
    add_token_lc(graph, bundle)
    register_latent_compilers(graph, adjacent_implication_recipe(lc_class_name="andL", if_token="A", then_token="B"))

    enforcement = discover_generation_enforcement(graph, bundle, on_unsupported="ignore")
    specs = discover_latent_window_specs(graph, bundle, on_unsupported="ignore")

    a = bundle.vocabulary.label_for_token("A")
    b = bundle.vocabulary.label_for_token("B")
    assert [(s.if_label, s.formula, s.window) for s in enforcement.latent_specs] == [(a, b, 1)]
    assert specs == enforcement.latent_specs


def test_registered_recipe_reaches_every_transition_potential_entry_point():
    graph, bundle = hmm_bundle()
    with graph:
        andL(bundle.context.latent_state_value("S0", "x"), bundle.context.latent_state_value("S1", "y"))
    register_latent_compilers(
        graph,
        forbidden_transition_potential_recipe(lc_class_name="andL", from_state="S0", to_state="S1"),
    )

    enforcement = discover_generation_enforcement(graph, bundle, on_unsupported="ignore")

    assert len(enforcement.transition_potentials) == 1
    assert len(discover_transition_potentials(graph, bundle, on_unsupported="ignore")) == 1
    assert combined_transition_potential_from_graph(graph, bundle, on_unsupported="ignore") is not None


def test_registered_compiler_runs_once_per_rule_during_enforcement():
    graph, bundle = token_bundle()
    add_token_lc(graph, bundle)
    calls = []

    def compiler(lc, bundle):
        calls.append(lc)
        return None

    register_latent_compilers(graph, compiler)
    discover_generation_enforcement(graph, bundle, on_unsupported="ignore")

    assert len(calls) == 1


def test_registration_is_scoped_to_its_graph():
    graph, bundle = token_bundle()
    add_token_lc(graph, bundle)
    other_graph, other_bundle = token_bundle()
    add_token_lc(other_graph, other_bundle)
    register_latent_compilers(graph, adjacent_implication_recipe(lc_class_name="andL", if_token="A", then_token="B"))

    assert registered_latent_compilers(other_graph) == ()
    assert discover_latent_window_specs(other_graph, other_bundle, on_unsupported="ignore") == ()


def test_register_accepts_recipe_tuples_and_ignores_repeats():
    graph, _ = token_bundle()
    recipes = common_latent_compiler_recipes(
        adjacent_lc_class_name="andL",
        adjacent_if_token="A",
        adjacent_then_token="B",
        bounded_lc_class_name="orL",
        bounded_if_token="A",
        bounded_then_token="B",
    )

    assert register_latent_compilers(graph, recipes) is graph
    register_latent_compilers(graph, recipes[0])

    assert registered_latent_compilers(graph) == recipes


def test_register_rejects_non_callable():
    graph, _ = token_bundle()

    with pytest.raises(TypeError, match="callable"):
        register_latent_compilers(graph, "not a compiler")


def test_per_call_compiler_overrides_registered_compiler():
    graph, bundle = token_bundle()
    add_token_lc(graph, bundle)
    register_latent_compilers(
        graph,
        bounded_lookahead_recipe(lc_class_name="andL", if_token="A", then_token="B", window=5, name="registered"),
    )

    specs = discover_latent_window_specs(
        graph,
        bundle,
        on_unsupported="ignore",
        extra_compilers=[
            bounded_lookahead_recipe(lc_class_name="andL", if_token="A", then_token="B", window=2, name="per_call"),
        ],
    )

    assert [s.name for s in specs] == ["per_call"]


def test_registered_compilers_run_in_marked_mode():
    graph, bundle = token_bundle()
    add_token_lc(graph, bundle)
    register_latent_compilers(graph, adjacent_implication_recipe(lc_class_name="andL", if_token="A", then_token="B"))

    assert len(discover_latent_window_specs(graph, bundle, mode="marked", on_unsupported="ignore")) == 1


def test_adjacent_rule_is_discovered_by_default():
    graph, bundle = hmm_bundle()
    add_adjacent_rule(graph, bundle)

    enforcement = discover_generation_enforcement(graph, bundle, on_unsupported="ignore")

    assert [(s.if_label, s.formula, s.window) for s in enforcement.latent_specs] == [(S0, S1, 1)]


def test_marked_mode_keeps_marked_only_behaviour():
    graph, bundle = hmm_bundle()
    add_adjacent_rule(graph, bundle)

    enforcement = discover_generation_enforcement(graph, bundle, on_unsupported="ignore", latent_mode="marked")

    assert enforcement.latent_specs == ()


def test_marked_specs_replace_automatic_spec_by_default():
    graph, bundle = hmm_bundle()
    lc = add_adjacent_rule(graph, bundle)
    marked = LatentWindowSpec(if_label=S0, formula=S1, window=3, weight=0.25, concept="latent_state")
    mark_for_latent(lc, marked)

    assert discover_latent_window_specs(graph, bundle, on_unsupported="ignore") == (marked,)
    assert len(discover_latent_window_specs(graph, bundle, mode="marked_and_auto", on_unsupported="ignore")) == 2


def test_default_mode_stays_quiet_about_rules_it_does_not_compile():
    graph, bundle = token_bundle()
    with graph:
        atLeastAL(bundle.context.token_value("A", "x"), 1)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        enforcement = discover_generation_enforcement(graph, bundle, on_unsupported="error")

    assert enforcement.latent_specs == ()
    with pytest.raises(ValueError, match="not supported by latent auto-discovery"):
        discover_latent_window_specs(graph, bundle, mode="auto", on_unsupported="error")


def test_default_mode_works_without_a_bundle():
    graph, bundle = hmm_bundle()
    add_adjacent_rule(graph, bundle)

    assert discover_latent_window_specs(graph) == ()
    with pytest.raises(ValueError, match="bundle is required"):
        discover_latent_window_specs(graph, mode="auto")


@pytest.mark.parametrize("mode", ["default", "auto", "marked_and_auto"])
def test_negated_consequent_is_never_compiled_to_positive_window_spec(mode):
    graph, bundle = hmm_bundle()
    ctx = bundle.context
    with graph:
        ifL(ctx.is_next_rel("next"), ifL(current(ctx, "S0"), notL(following(ctx, "S1"))))

    assert discover_latent_window_specs(graph, bundle, mode=mode, on_unsupported="ignore") == ()
    potentials = discover_transition_potentials(graph, bundle)
    assert len(potentials) == 1
    assert potentials[0].tensor_for(torch.ones((2, 2)))[0, 1].item() == pytest.approx(0.0)


@pytest.mark.parametrize(
    "consequent",
    [
        lambda ctx: orL(following(ctx, "S0"), notL(following(ctx, "S1", "z"))),
        lambda ctx: andL(following(ctx, "S0"), notL(following(ctx, "S1", "z"))),
        lambda ctx: nandL(following(ctx, "S0"), following(ctx, "S1", "z")),
    ],
    ids=["or_with_not", "and_with_not", "nand"],
)
def test_consequent_with_unrepresentable_part_is_rejected_whole(consequent):
    graph, bundle = hmm_bundle()
    ctx = bundle.context
    with graph:
        ifL(ctx.is_next_rel("next"), ifL(current(ctx, "S0"), consequent(ctx)))

    assert discover_latent_window_specs(graph, bundle, on_unsupported="ignore") == ()


def test_negated_antecedent_is_not_compiled():
    graph, bundle = hmm_bundle()
    ctx = bundle.context
    with graph:
        ifL(ctx.is_next_rel("next"), ifL(notL(current(ctx, "S0")), following(ctx, "S1")))
        ifL(ctx.is_next_rel("other"), ifL(notL(current_on(ctx, "S0", "other")), notL(following_on(ctx, "S1", "other"))))

    assert discover_latent_window_specs(graph, bundle, on_unsupported="ignore") == ()
    assert discover_transition_potentials(graph, bundle) == ()


def current_on(ctx, state, relation):
    return ctx.latent_state_value(state, "u", path=(relation, ctx.current_token))


def following_on(ctx, state, relation):
    return ctx.latent_state_value(state, "v", path=(relation, ctx.next_token))


def test_and_or_consequents_still_compile():
    graph, bundle = hmm_bundle()
    ctx = bundle.context
    with graph:
        ifL(ctx.is_next_rel("next"), ifL(current(ctx, "S0"), orL(following(ctx, "S0"), following(ctx, "S1", "z"))))

    specs = discover_latent_window_specs(graph, bundle, on_unsupported="ignore")

    assert [s.formula for s in specs] == [("or", S0, S1)]


def test_wfa_transition_pair_consistency_rules_do_not_become_state_implications():
    graph, bundle = SpectralWFAFactorGraphEncoder(
        vocab=["<eos>", "A"],
        eos_token="<eos>",
        tokenizer=FakeTokenizer(),
        state_names=["S0", "S1"],
        include_transition_pairs=True,
    ).build_graph()
    with graph:
        apply_wfa_factor_consistency_constraints(bundle)

    assert discover_latent_window_specs(graph, bundle, on_unsupported="ignore") == ()


def test_automatic_state_rule_is_skipped_for_bare_token_tensor():
    graph, bundle = hmm_bundle()
    add_adjacent_rule(graph, bundle)
    enforcement = discover_generation_enforcement(graph, bundle, on_unsupported="ignore")
    token_probs = torch.tensor([[0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1]], dtype=torch.float64)

    with pytest.warns(RuntimeWarning, match="latent_state"):
        loss = enforcement.latent_loss(token_probs)
    with pytest.warns(RuntimeWarning, match="latent_state"):
        breakdown = enforcement.latent_breakdown(token_probs)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        enforcement.latent_loss(token_probs)

    assert loss.item() == 0.0
    assert loss.dtype == torch.float64
    assert breakdown.items == ()


def test_automatic_state_rule_is_skipped_when_concept_missing_from_mapping():
    graph, bundle = hmm_bundle()
    add_adjacent_rule(graph, bundle)
    enforcement = discover_generation_enforcement(graph, bundle, on_unsupported="ignore")

    with pytest.warns(RuntimeWarning, match="latent_state"):
        loss = enforcement.latent_loss({"generated_token": torch.full((3, 3), 1 / 3)})

    assert loss.item() == 0.0


def test_automatic_state_rule_is_enforced_when_concept_is_supplied():
    graph, bundle = hmm_bundle()
    add_adjacent_rule(graph, bundle)
    enforcement = discover_generation_enforcement(graph, bundle, on_unsupported="ignore")
    state_probs = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.5, 0.5]])
    probs = {"generated_token": torch.full((3, 3), 1 / 3), "latent_state": state_probs}

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loss = enforcement.latent_loss(probs)

    expected = window_formula_loss(probs, S0, S1, 1, concept="latent_state", empty_window_policy="ignore")
    assert loss.item() == pytest.approx(expected.item())
    assert loss.item() > 0.0


def test_marked_state_spec_still_evaluates_bare_tensor_as_given():
    graph, bundle = hmm_bundle()
    lc = add_adjacent_rule(graph, bundle)
    spec = LatentWindowSpec(if_label=0, formula=1, window=1, concept="latent_state")
    mark_for_latent(lc, spec)
    enforcement = discover_generation_enforcement(graph, bundle, on_unsupported="ignore")
    state_probs = torch.tensor([[0.9, 0.1], [0.8, 0.2]])

    assert enforcement.latent_loss(state_probs).item() == pytest.approx(
        window_formula_loss(state_probs, 0, 1, 1, concept="latent_state").item()
    )


def test_integer_eos_label_refers_to_tokens_for_state_rules():
    graph, bundle = hmm_bundle()
    add_adjacent_rule(graph, bundle)
    enforcement = discover_generation_enforcement(graph, bundle, on_unsupported="ignore")
    eos = bundle.vocabulary.eos_label
    assert eos == 0
    # No token is EOS, yet state 0 (the same index) dominates from the start,
    # so reading the id as a state would clip everything after position 0.
    tokens = torch.zeros((3, 3))
    tokens[:, 1] = 1.0
    states = torch.tensor([[0.9, 0.1], [0.6, 0.4], [0.7, 0.3]])
    probs = {"generated_token": tokens, "latent_state": states}

    # Mean of S0(t) * (1 - S1(t+1)) over both adjacent pairs.
    expected = (0.9 * (1 - 0.4) + 0.6 * (1 - 0.3)) / 2
    assert enforcement.latent_loss(probs, eos_label=eos).item() == pytest.approx(expected)
    assert enforcement.latent_breakdown(probs, eos_label=eos).total.item() == pytest.approx(expected)
