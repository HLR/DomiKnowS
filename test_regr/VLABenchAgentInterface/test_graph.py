import torch

from test_regr.VLABenchAgentInterface.graph import (
    PlanVocabulary,
    compile_planner_dfa,
    create_planner_generation_graph,
    dfa_accepts_plan,
)
from test_regr.VLABenchAgentInterface.world_graph import (
    EOS_TOKEN,
    SKILL_ARGUMENTS,
    build_vlabench_world_graph,
    materialize_plan,
    verify_plan_constraints,
)


PLAN = [
    {"name": "pick", "params": {"target_entity_name": 0}},
    {"name": "place", "params": {"target_container_name": 1}},
]


def _descendants(root):
    pending = list(root.getChildDataNodes() or [])
    result = []
    while pending:
        node = pending.pop()
        result.append(node)
        pending.extend(node.getChildDataNodes() or [])
    return result


def test_generation_graph_compiles_and_enforces_eos_closure():
    world = build_vlabench_world_graph("test_vlabench_generation_world")
    vocabulary = PlanVocabulary.from_world(world, max_entities=4)
    graph, bundle = create_planner_generation_graph(
        world,
        vocabulary,
        max_operations=3,
        graph_name="test_vlabench_generation",
    )
    dfa = compile_planner_dfa(graph, bundle, world, vocabulary, max_operations=3)
    assert dfa_accepts_plan(dfa, bundle, PLAN, ("apple", "bowl"), world=world)
    labels = [
        bundle.vocabulary.label_for_token("skill:pick"),
        bundle.vocabulary.label_for_token(EOS_TOKEN),
        bundle.vocabulary.label_for_token("obj:0"),
    ]
    assert not dfa.accepts(labels)

    # A complete EOS-closed token sequence can still be rejected by the
    # graph-derived domain automaton when its skill pattern is illegal.
    illegal = [
        bundle.vocabulary.label_for_token("skill:place"),
        bundle.vocabulary.label_for_token("arg:target_container_name"),
        bundle.vocabulary.label_for_token("obj:0"),
        bundle.vocabulary.label_for_token(EOS_TOKEN),
    ]
    assert not dfa.accepts(illegal)


def test_world_graph_defines_grounded_plan_and_named_constraints():
    bundle = build_vlabench_world_graph("test_vlabench_world")
    assert bundle.plan.name == "vlabench_plan"
    assert bundle.operation.name == "vlabench_operation"
    assert set(bundle.skills) >= {"pick", "place", "insert", "pour"}
    assert bundle.skill_arguments == SKILL_ARGUMENTS
    assert len(bundle.domain_checksum) == 64
    assert bundle.has_constraints
    root = materialize_plan(PLAN, ("apple", "bowl"), bundle)
    evaluation = verify_plan_constraints(root, bundle)
    assert evaluation.score == 1.0
    assert evaluation.results["operation_schema_is_valid"]["satisfied"] == 100.0
    assert evaluation.results["transition_matches_subtask_pattern"]["satisfied"] == 100.0

    operations = [node for node in _descendants(root) if node.ontologyNode is bundle.operation]
    assert len(operations) == 2
    by_index = {node.instanceID: node for node in operations}
    assert torch.argmax(by_index[0].attributes["<skill__pick>"]).item() == 1
    assert torch.argmax(by_index[0].attributes["<skill__place>"]).item() == 0
    assert by_index[0].relationLinks[bundle.operation_roles["target_entity"].name][0].instanceValue == "apple"


def test_world_constraints_reject_bad_pointer_and_unknown_skill():
    bundle = build_vlabench_world_graph("test_vlabench_invalid_world")
    invalid_pointer = [
        {"name": "pick", "params": {"target_entity_name": 99}},
        {"name": "place", "params": {"target_container_name": 1}},
    ]
    evaluation = verify_plan_constraints(materialize_plan(invalid_pointer, ("apple", "bowl"), bundle), bundle)
    assert evaluation.score == 0.0
    assert evaluation.results["operation_pointer_is_valid"]["satisfied"] < 100.0

    unknown = [{"name": "teleport", "params": {} }]
    unknown_evaluation = verify_plan_constraints(materialize_plan(unknown, (), bundle), bundle)
    assert unknown_evaluation.score == 0.0
    assert unknown_evaluation.results["operation_exactly_one_skill"]["satisfied"] == 0.0


def test_multiple_world_graph_builds_keep_relation_roles_isolated():
    first = build_vlabench_world_graph("test_vlabench_world_first")
    second = build_vlabench_world_graph("test_vlabench_world_second")
    assert first.operation_roles["target_entity"].name != second.operation_roles["target_entity"].name
    assert verify_plan_constraints(materialize_plan(PLAN, ("apple", "bowl"), first), first).score == 1.0
    assert verify_plan_constraints(materialize_plan(PLAN, ("apple", "bowl"), second), second).score == 1.0


def test_schema_module_is_removed_and_vocabulary_is_world_derived():
    from pathlib import Path

    assert not (Path(__file__).parent / "schema.py").exists()
    world = build_vlabench_world_graph("test_vlabench_schema_source")
    vocabulary = PlanVocabulary.from_world(world, max_entities=7)
    assert vocabulary.skill_argument_map == world.skill_arguments
    assert vocabulary.domain_checksum == world.domain_checksum
