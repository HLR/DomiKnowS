"""The ``official`` knowledge level: the official protocol's rules as world-graph LCs, for training and evaluation."""
import json
from argparse import Namespace
from dataclasses import replace

import pytest
from PIL import Image

from ..knowledge import (
    LEVELS, SIDE_LEVELS, apply_knowledge_argument, default_level, default_task_identity, group_active,
    official_observations, set_default_level, set_default_task_identity, split_level,
)
from ..world_graph import build_vlabench_world_graph, materialize_plan, set_knowledge_level, verify_plan_constraints
from .test_constraint_equivalence import encode, op
from .test_causal_knowledge_level import CAUSAL, active

ENTITIES = ("numbered object 0", "numbered object 1", "numbered object 2", "numbered object 3")
OFFICIAL = CAUSAL + ("official__observation_supported_references", "official__instruction_candidates_bound_targets")
# The instruction names objects 1 and 2 for pick, and 0 or 3 as the place destination.
CANDIDATES = (("pick", "target_entity_name", (1, 2)), ("place", "target_container_name", (0, 3)))


def test_official_is_causal_plus_its_own_group_outside_the_chain():
    assert "official" not in LEVELS and SIDE_LEVELS["official"] == ("causal", "official")
    assert [g for g in ("causal", "generic", "scene", "task", "official") if group_active("official", g)] == [
        "causal", "official"]
    assert not any(group_active(level, "official") for level in LEVELS)
    assert split_level("official") == ("official", None)
    assert official_observations("official") and not official_observations("task")
    world = build_vlabench_world_graph("official_level_groups", knowledge_level="official")
    assert active(world) == set(OFFICIAL)


@pytest.mark.parametrize("level", (*LEVELS, "official"))
def test_toggling_to_and_from_official_matches_a_fresh_graph(level):
    toggled = build_vlabench_world_graph("official_toggle_" + level, knowledge_level="official")
    set_knowledge_level(toggled, level)
    assert active(toggled) == active(build_vlabench_world_graph("official_fresh_" + level, knowledge_level=level))
    set_knowledge_level(toggled, "official")
    assert active(toggled) == set(OFFICIAL)


def policy(world, context):
    from ..graph import PlanVocabulary, bind_vlabench_contextual_dfa, compile_planner_dfa, create_planner_generation_graph
    vocab = PlanVocabulary.from_world(world, max_entities=len(ENTITIES))
    graph, generation = create_planner_generation_graph(world, vocab, max_operations=2)
    grammar = compile_planner_dfa(graph, generation, world, vocab, max_operations=2)
    return vocab, grammar, bind_vlabench_contextual_dfa(grammar, vocab, {"entities": ENTITIES, **context})


def verified(plan, world, context):
    result = verify_plan_constraints(materialize_plan(plan, ENTITIES, world, task_context=context), world)
    return result is None or result.score == 1.0


CASES = [
    ({"object_candidates": CANDIDATES}, [op("pick", 1), op("place", 3)], True),
    ({"object_candidates": CANDIDATES}, [op("pick", 0), op("place", 3)], False),    # not an instruction candidate
    ({"object_candidates": CANDIDATES}, [op("pick", 2), op("place", 1)], False),    # destination not a candidate
    ({"object_candidates": CANDIDATES}, [op("pick", 1), op("pour", 1)], False),     # causal: self-placement
    ({"object_candidates": CANDIDATES}, [op("pick", 2), op("pour", 3)], True),      # pour roles not restricted
    ({"visible_entities": (1, 3)}, [op("pick", 1), op("place", 3)], True),
    ({"visible_entities": (1, 3)}, [op("pick", 2), op("place", 3)], False),         # not seen in the image
    ({}, [op("pick", 0), op("place", 1)], True),                                    # no observation: no restriction
]


@pytest.mark.parametrize("context,plan,expected", CASES)
def test_graph_dfa_and_verifier_agree_on_the_official_rules(context, plan, expected):
    world = build_vlabench_world_graph("official_level_cases", knowledge_level="official")
    vocab, grammar, monitor = policy(world, context)
    sequence = encode(plan, vocab, world, ENTITIES)
    assert grammar.accepts(sequence), "the case must exercise the knowledge rules, not syntax"
    assert verified(plan, world, context) == expected
    assert monitor.accepts(sequence) == expected


def test_the_official_rules_are_off_at_other_levels():
    context = {"object_candidates": CANDIDATES, "visible_entities": (1, 3)}
    plan = [op("pick", 0), op("place", 2)]
    for level, expected in (("causal", True), ("official", False)):
        world = build_vlabench_world_graph(f"official_off_{level}", knowledge_level=level)
        vocab, _, monitor = policy(world, context)
        assert monitor.accepts(encode(plan, vocab, world, ENTITIES)) == expected
        assert verified(plan, world, context) == expected


class FakeModel:
    from ..research.perception import PerceptionConfig
    config = PerceptionConfig(model_id="frozen")

    def infer(self, raw, numbered, instruction):
        objects = [{"id": i, "confidence": .95, "bbox": [.1 * i, .1, .1 * i + .05, .2], "category": kind,
                    "description": "thing", "states": {"visible": True}}
                   for i, kind in ((0, "tomato"), (1, "apple"), (2, "bowl"), (3, "plate"))]
        return json.loads(json.dumps({"objects": objects, "requests": [], "hypotheses": [], "complete": True}))


def test_perception_enrichment_stores_the_official_candidates(tmp_path):
    from ..dataset import PlanningExample
    from ..research.perception import enrich_examples
    paths = []
    for name in ("input", "input_mask"):
        paths.append(tmp_path / f"{name}.png")
        Image.new("RGB", (8, 8)).save(paths[-1])
    example = PlanningExample(episode_id="e", instruction="Put the apple in the bowl", operation_sequence=(),
                              image_paths=(paths[0],), segmented_image_paths=(paths[1],), env_config_path=None,
                              entities=("a", "b", "c", "d"), perception_image_paths=tuple(paths))
    enriched, = enrich_examples([example], FakeModel())
    assert enriched.visible_entities == (0, 1, 2, 3)
    candidates = dict(((skill, role), ids) for skill, role, ids in enriched.object_candidates)
    assert candidates[("pick", "target_entity_name")] == (1,)
    assert candidates[("place", "target_container_name")] == (2,)
    assert enriched.as_reward_item()["object_candidates"] == enriched.object_candidates


def research_args(**values):
    from ..research.config import add_research_arguments
    import argparse
    parser = argparse.ArgumentParser()
    add_research_arguments(parser)
    parser.add_argument("--knowledge")
    parser.add_argument("--task-identity")
    parser.add_argument("--planner-model")
    return parser.parse_args([f"--{k.replace('_', '-')}={v}" for k, v in values.items()])


def test_official_level_turns_image_perception_on_for_training_and_evaluation():
    from ..research.config import apply_research_arguments, default_perception_options
    previous = (default_level(), default_task_identity())
    try:
        args = research_args(knowledge="official", planner_model="frozen")
        apply_research_arguments(args)
        assert args.research_perception == "image-vlm" and default_perception_options()["model_id"] == "frozen"
        assert apply_knowledge_argument(args) == "official"
        with pytest.raises(ValueError, match="image perception"):
            apply_research_arguments(research_args(knowledge="official", research_perception="off"))
        apply_research_arguments(research_args(knowledge="causal"))
        assert default_perception_options() is None
        with pytest.raises(ValueError, match="image perception"):
            apply_knowledge_argument(Namespace(knowledge="official", task_identity=None))
    finally:
        set_default_level(previous[0])
        set_default_task_identity(previous[1])
        apply_research_arguments(research_args())


def test_official_protocol_accepts_the_official_level_as_causal_with_perception():
    from ..main import official_constraints
    assert official_constraints(Namespace(knowledge="official", constraints="none")) == "causal"
    with pytest.raises(ValueError, match="conflicts"):
        official_constraints(Namespace(knowledge="official", constraints="grammar"))
    with pytest.raises(ValueError, match="grammar, causal or official"):
        official_constraints(Namespace(knowledge="task", constraints="none"))


def test_both_joint_evaluation_entry_points_accept_the_official_level():
    from test_regr.JointEmbodiedAgentInterface.eval_planning import build_eval_parser
    from test_regr.JointEmbodiedAgentInterface.main import build_parser
    standalone = build_eval_parser().parse_args(["--checkpoint", "x.pt", "--knowledge", "official"])
    command = build_parser().parse_args(["evaluate-planning", "--checkpoint", "x.pt", "--knowledge", "official"])
    assert standalone.knowledge == command.knowledge == "official"
    assert standalone.research_perception == "auto"


def test_training_runtime_reads_the_candidates_through_the_joint_planner_context():
    from test_regr.JointEmbodiedAgentInterface.utils.planner_context import planner_context
    from ..dataset import PlanningExample
    from ..plan_codec import plan_to_tokens
    from ..training import build_constraint_runtime
    previous = default_level()
    try:
        set_default_level("official")
        runtime = build_constraint_runtime(max_entities=4, max_operations=2, name_prefix="official_level_runtime")
    finally:
        set_default_level(previous)
    example = PlanningExample("e", "put the apple in the bowl", (), (), (), None, entities=ENTITIES,
                              visible_entities=(0, 1, 2, 3), object_candidates=CANDIDATES)
    context = planner_context("vlabench", example)
    assert context["object_candidates"] == CANDIDATES
    policy = runtime.dfa_for("vlabench", {**context, "entities": ENTITIES})

    def accepted(plan):
        return policy.accepts([runtime.vocabulary.label_for_token(t)
                               for t in plan_to_tokens(plan, ENTITIES, world=runtime.world_bundle)])
    assert accepted([op("pick", 1), op("place", 3)])
    assert not accepted([op("pick", 0), op("place", 3)])
