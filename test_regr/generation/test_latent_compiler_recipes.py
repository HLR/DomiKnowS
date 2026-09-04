import pytest
import torch

from domiknows.generation import (
    HMMFactorGraphEncoder,
    LabelRef,
    GenerationEncoder,
    UnsupportedRecipeMatch,
    WindowRecipeMatch,
    adjacent_implication_recipe,
    bounded_lookahead_recipe,
    common_latent_compiler_recipes,
    cooccurrence_recipe,
    discover_generation_enforcement,
    discover_latent_window_specs,
    forbidden_transition_potential_recipe,
)
from domiknows.graph.logicalConstrain import andL, ifL


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


def build_hmm_bundle():
    encoder = HMMFactorGraphEncoder(
        ["<eos>", "A", "B"],
        eos_token="<eos>",
        tokenizer=FakeTokenizer(),
        state_names=("S0", "S1"),
        include_dp_factors=False,
    )
    return encoder.build_graph()


def test_adjacent_implication_recipe_emits_window_one_spec():
    graph, bundle = build_bundle()
    with graph:
        andL(bundle.context.token_value("A", "x"), bundle.context.token_value("B", "x"))

    recipe = adjacent_implication_recipe(lc_class_name="andL", if_token="A", then_token="B")
    enforcement = discover_generation_enforcement(
        graph,
        bundle,
        on_unsupported="ignore",
        extra_latent_compilers=[recipe],
    )

    a = bundle.vocabulary.label_for_token("A")
    b = bundle.vocabulary.label_for_token("B")
    assert len(enforcement.latent_specs) == 1
    assert enforcement.latent_specs[0].if_label == a
    assert enforcement.latent_specs[0].formula == b
    assert enforcement.latent_specs[0].window == 1


def test_bounded_lookahead_recipe_uses_configured_window():
    graph, bundle = build_bundle()
    with graph:
        andL(bundle.context.token_value("A", "x"), bundle.context.token_value("B", "x"))

    recipe = bounded_lookahead_recipe(
        lc_class_name="andL",
        if_token="A",
        then_token="B",
        window=3,
        name="a_then_b_soon",
    )
    specs = discover_latent_window_specs(
        graph,
        bundle,
        on_unsupported="ignore",
        extra_compilers=[recipe],
    )

    assert len(specs) == 1
    assert specs[0].window == 3
    assert specs[0].name == "a_then_b_soon"


def test_cooccurrence_recipe_builds_and_or_formula():
    graph, bundle = build_bundle()
    with graph:
        andL(bundle.context.token_value("A", "x"), bundle.context.token_value("B", "x"))

    recipe = cooccurrence_recipe(
        lc_class_name="andL",
        if_token="A",
        candidate_tokens=("A", "B"),
        mode="and",
        window=4,
    )
    specs = discover_latent_window_specs(
        graph,
        bundle,
        on_unsupported="ignore",
        extra_compilers=[recipe],
    )

    a = bundle.vocabulary.label_for_token("A")
    b = bundle.vocabulary.label_for_token("B")
    assert len(specs) == 1
    assert specs[0].if_label == a
    assert specs[0].formula == ("and", a, b)
    assert specs[0].window == 4


def test_forbidden_transition_recipe_emits_transition_potential():
    graph, bundle = build_hmm_bundle()
    with graph:
        andL(
            bundle.context.latent_state_value("S0", "x"),
            bundle.context.latent_state_value("S1", "y"),
        )

    recipe = forbidden_transition_potential_recipe(
        lc_class_name="andL",
        from_state="S0",
        to_state="S1",
    )
    enforcement = discover_generation_enforcement(
        graph,
        bundle,
        on_unsupported="ignore",
        extra_latent_compilers=[recipe],
    )

    assert len(enforcement.transition_potentials) == 1
    values = enforcement.transition_potentials[0].tensor_for(torch.ones((2, 2)))
    assert values[0, 1].item() == pytest.approx(0.0)
    assert values[1, 1].item() == pytest.approx(1.0)


def test_recipe_outputs_deduplicate_through_discovery():
    graph, bundle = build_bundle()
    with graph:
        andL(bundle.context.token_value("A", "x"), bundle.context.token_value("B", "x"))
        andL(bundle.context.token_value("A", "u"), bundle.context.token_value("B", "u"))

    recipe = adjacent_implication_recipe(lc_class_name="andL", if_token="A", then_token="B")
    specs = discover_latent_window_specs(
        graph,
        bundle,
        on_unsupported="ignore",
        extra_compilers=[recipe],
    )

    assert len(specs) == 1


def test_builtin_latent_discovery_wins_over_recipe():
    graph, bundle = build_hmm_bundle()
    with graph:
        ifL(
            bundle.context.is_next_rel("next"),
            ifL(
                bundle.context.latent_state_value("S0", "x", path=("next", bundle.context.current_token)),
                bundle.context.latent_state_value("S1", "y", path=("next", bundle.context.next_token)),
            ),
        )

    def matcher(lc, bundle):
        return WindowRecipeMatch(LabelRef("latent_state", 0), LabelRef("latent_state", 0), window=5, name="custom")

    recipe = adjacent_implication_recipe(matcher, lc_class_name="ifL", concept="latent_state")
    specs = discover_latent_window_specs(
        graph,
        bundle,
        mode="auto",
        on_unsupported="ignore",
        extra_compilers=[recipe],
    )

    assert len(specs) == 1
    assert specs[0].if_label == LabelRef("latent_state", 0)
    assert specs[0].formula == LabelRef("latent_state", 1)
    assert specs[0].window == 1


def test_recipe_unsupported_relevant_result_warns_or_errors():
    graph, bundle = build_bundle()
    with graph:
        andL(bundle.context.token_value("A", "x"), bundle.context.token_value("B", "x"))

    def matcher(lc, bundle):
        return UnsupportedRecipeMatch("project recipe cannot resolve this LC")

    recipe = adjacent_implication_recipe(matcher, lc_class_name="andL")

    with pytest.warns(RuntimeWarning, match="project recipe cannot resolve this LC"):
        discover_latent_window_specs(graph, bundle, extra_compilers=[recipe], on_unsupported="warn")

    with pytest.raises(ValueError, match="project recipe cannot resolve this LC"):
        discover_latent_window_specs(graph, bundle, extra_compilers=[recipe], on_unsupported="error")


def test_common_latent_compiler_recipes_work_with_enforcement():
    graph, bundle = build_bundle()
    with graph:
        andL(bundle.context.token_value("A", "x"), bundle.context.token_value("B", "x"))

    recipes = common_latent_compiler_recipes(
        adjacent_lc_class_name="andL",
        adjacent_if_token="A",
        adjacent_then_token="B",
    )
    enforcement = discover_generation_enforcement(
        graph,
        bundle,
        on_unsupported="ignore",
        extra_latent_compilers=recipes,
    )

    assert len(recipes) == 1
    assert len(enforcement.latent_specs) == 1
