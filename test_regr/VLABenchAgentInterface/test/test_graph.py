import torch

from test_regr.VLABenchAgentInterface.graph import (
    COMPLEX_TASK_SCHEMAS,
    PlanVocabulary,
    bind_vlabench_contextual_dfa,
    compile_planner_dfa,
    create_planner_generation_graph,
    dfa_accepts_plan,
)
from test_regr.VLABenchAgentInterface.reward import score_vlabench_plan
from test_regr.VLABenchAgentInterface.world_graph import (
    EOS_TOKEN,
    PRIMITIVE_TASK_PATTERNS,
    SKILL_ARGUMENTS,
    TASK_CONTRACTS,
    build_vlabench_world_graph,
    materialize_plan,
    task_contract_for,
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
    assert evaluation.score < 1.0
    # Pointer errors belong to schema/scene constraints, not pattern progress.
    assert evaluation.results["transition_matches_subtask_pattern"]["satisfied"] == 100.0
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

    assert not (Path(__file__).parent.parent / "schema.py").exists()
    world = build_vlabench_world_graph("test_vlabench_schema_source")
    vocabulary = PlanVocabulary.from_world(world, max_entities=7)
    assert vocabulary.skill_argument_map == world.skill_arguments
    assert vocabulary.domain_checksum == world.domain_checksum


def test_every_adapter_primitive_has_an_explicit_task_contract():
    assert set(TASK_CONTRACTS) == set(PRIMITIVE_TASK_PATTERNS)
    for task, pattern in PRIMITIVE_TASK_PATTERNS.items():
        contract = task_contract_for(task)
        assert contract is not None
        assert contract.family == "primitive"
        assert contract.skill_sequence == pattern
        assert contract.success_predicate == "task.should_terminate_episode(physics)"
        assert "get_task_progress" in contract.progress_signals
        assert "target_distance" in contract.progress_signals


def test_task_contract_preserves_upstream_painting_and_book_semantics():
    assert task_contract_for("select_painting").target_attributes == ("target_button", "target_entity")
    assert task_contract_for("select_painting").skill_sequence == ("press",)
    assert task_contract_for("select_book").skill_sequence == ("pick", "pull")
    assert task_contract_for("unknown_task") is None


def test_world_graph_verifies_causal_constraints():
    bundle = build_vlabench_world_graph("test_vlabench_causal_world")
    entities = ("apple", "bowl", "kitchen_counter")

    # 1. Valid plan satisfies all causal constraints
    valid_plan = [
        {"name": "pick", "params": {"target_entity_name": 0}},
        {"name": "place", "params": {"target_container_name": 1}},
    ]
    eval_valid = verify_plan_constraints(materialize_plan(valid_plan, entities, bundle), bundle)
    assert eval_valid.score == 1.0
    assert eval_valid.results["anti_self_placement"]["satisfied"] == 100.0
    assert eval_valid.results["pick_excludes_static_fixtures"]["satisfied"] == 100.0

    # 2. Self-placement violation (placing into self)
    self_placement_plan = [
        {"name": "place", "params": {"target_entity_name": 0, "target_container_name": 0}},
    ]
    eval_self = verify_plan_constraints(materialize_plan(self_placement_plan, entities, bundle), bundle)
    assert eval_self.score < 1.0
    assert eval_self.results["anti_self_placement"]["satisfied"] == 0.0

    # 3. Static fixture violation (picking counter)
    static_pick_plan = [
        {"name": "pick", "params": {"target_entity_name": 2}},
        {"name": "place", "params": {"target_container_name": 1}},
    ]
    eval_static = verify_plan_constraints(materialize_plan(static_pick_plan, entities, bundle), bundle)
    assert eval_static.score < 1.0
    assert eval_static.results["pick_excludes_static_fixtures"]["satisfied"] < 100.0

    # 4. Redundant repeated pick violation
    repeated_pick_plan = [
        {"name": "pick", "params": {"target_entity_name": 0}},
        {"name": "pick", "params": {"target_entity_name": 0}},
    ]
    eval_repeated = verify_plan_constraints(materialize_plan(repeated_pick_plan, entities, bundle), bundle)
    assert eval_repeated.score == 0.0
    assert eval_repeated.results["no_redundant_repeated_picks"]["satisfied"] == 0.0

    # 5. Inconsistent container destination violation
    inconsistent_container_plan = [
        {"name": "place", "params": {"target_entity_name": 0, "target_container_name": 1}},
        {"name": "place", "params": {"target_entity_name": 0, "target_container_name": 2}},
    ]
    eval_inconsistent = verify_plan_constraints(materialize_plan(inconsistent_container_plan, entities, bundle, task_context={"task_target_container": "bowl"}), bundle)
    assert eval_inconsistent.score < 1.0
    assert eval_inconsistent.results["container_destination_consistency"]["satisfied"] == 0.0


def test_world_graph_verifies_composite_and_access_preconditions():
    bundle = build_vlabench_world_graph("test_vlabench_composite_world")
    entities = ("soda_can", "cooler", "kitchen_counter")

    # 1. Unopened container placement violation: placing into cooler without open_door
    unopened_placement_plan = [
        {"name": "pick", "params": {"target_entity_name": 0}},
        {"name": "place", "params": {"target_container_name": 1}},
    ]
    eval_unopened = verify_plan_constraints(
        materialize_plan(unopened_placement_plan, entities, bundle), bundle
    )
    assert eval_unopened.score < 1.0
    assert eval_unopened.results["open_before_enclosed_placement"]["satisfied"] == 0.0

    # 2. Valid access sequence: pick cooler -> open_door cooler -> pick soda -> place in cooler
    valid_access_plan = [
        {"name": "pick", "params": {"target_entity_name": 1}},
        {"name": "open_door", "params": {"target_entity_name": 1}},
        {"name": "pick", "params": {"target_entity_name": 0}},
        {"name": "place", "params": {"target_container_name": 1}},
    ]
    eval_valid = verify_plan_constraints(
        materialize_plan(valid_access_plan, entities, bundle), bundle
    )
    assert eval_valid.score == 1.0
    assert eval_valid.results["open_before_enclosed_placement"]["satisfied"] == 100.0

    # 3. Premature composite EOS violation: composite schema requires 4 steps and ("open_door", "pick", "place")
    truncated_plan = [
        {"name": "pick", "params": {"target_entity_name": 1}},
        {"name": "open_door", "params": {"target_entity_name": 1}},
    ]
    eval_truncated = verify_plan_constraints(
        materialize_plan(
            truncated_plan,
            entities,
            bundle,
            task_name="cool_drink",
            min_operations=4,
            required_skills=("open_door", "pick", "place"),
        ),
        bundle,
    )
    assert eval_truncated.score < 1.0
    assert eval_truncated.results["composite_subgoal_completion"]["satisfied"] < 100.0

    # 4. Completed composite plan satisfies composite_subgoal_completion
    eval_complete = verify_plan_constraints(
        materialize_plan(
            valid_access_plan,
            entities,
            bundle,
            task_name="cool_drink",
            min_operations=4,
            required_skills=("open_door", "pick", "place"),
        ),
        bundle,
    )
    assert eval_complete.score == 1.0
    assert eval_complete.results["composite_subgoal_completion"]["satisfied"] == 100.0


def test_composite_planning_dfa_enforces_sequence_and_preconditions():
    world = build_vlabench_world_graph("test_composite_dfa_world")
    vocabulary = PlanVocabulary.from_world(world, max_entities=4)
    graph, gen_bundle = create_planner_generation_graph(
        world, vocabulary, max_operations=5, graph_name="test_composite_dfa_gen"
    )
    base_dfa = compile_planner_dfa(graph, gen_bundle, world, vocabulary, max_operations=5)

    entities = ("soda", "cooler", "table")
    context = {"task": "cool_drink", "entities": entities, "instruction": "Cool the drink in the cooler"}
    dfa = bind_vlabench_contextual_dfa(base_dfa, vocabulary, context)
    from ..constraint_compiler import WorldConstraintDFA
    assert isinstance(dfa, WorldConstraintDFA)

    # 1. Truncated plan without place (only pick cooler, open_door cooler, pick soda) must be rejected
    truncated_tokens = [
        vocabulary.label_for_token("skill:pick"),
        vocabulary.label_for_token("arg:target_entity_name"),
        vocabulary.label_for_token("obj:1"),
        vocabulary.label_for_token("skill:open_door"),
        vocabulary.label_for_token("arg:target_entity_name"),
        vocabulary.label_for_token("obj:1"),
        vocabulary.label_for_token("skill:pick"),
        vocabulary.label_for_token("arg:target_entity_name"),
        vocabulary.label_for_token("obj:0"),
        vocabulary.label_for_token(EOS_TOKEN),
    ]
    assert not dfa.accepts(truncated_tokens)

    # 2. Plan placing into unopened cooler (skipping open_door) must be rejected
    unopened_tokens = [
        vocabulary.label_for_token("skill:pick"),
        vocabulary.label_for_token("arg:target_entity_name"),
        vocabulary.label_for_token("obj:0"),
        vocabulary.label_for_token("skill:place"),
        vocabulary.label_for_token("arg:target_container_name"),
        vocabulary.label_for_token("obj:1"),
        vocabulary.label_for_token(EOS_TOKEN),
    ]
    assert not dfa.accepts(unopened_tokens)

    # 3. Complete valid plan (pick cooler -> open_door -> pick soda -> place -> EOS) must be accepted
    valid_tokens = [
        vocabulary.label_for_token("skill:pick"),
        vocabulary.label_for_token("arg:target_entity_name"),
        vocabulary.label_for_token("obj:1"),
        vocabulary.label_for_token("skill:open_door"),
        vocabulary.label_for_token("arg:target_entity_name"),
        vocabulary.label_for_token("obj:1"),
        vocabulary.label_for_token("skill:pick"),
        vocabulary.label_for_token("arg:target_entity_name"),
        vocabulary.label_for_token("obj:0"),
        vocabulary.label_for_token("skill:place"),
        vocabulary.label_for_token("arg:target_container_name"),
        vocabulary.label_for_token("obj:1"),
        vocabulary.label_for_token(EOS_TOKEN),
    ]
    assert dfa.accepts(valid_tokens)


def test_causal_masking_never_relaxes_hard_constraints():
    world = build_vlabench_world_graph("test_causal_compiled_world")
    vocabulary = PlanVocabulary.from_world(world, max_entities=4)
    graph, generation = create_planner_generation_graph(world, vocabulary, max_operations=2)
    grammar = compile_planner_dfa(graph, generation, world, vocabulary, max_operations=2)
    policy = bind_vlabench_contextual_dfa(grammar, vocabulary, {"entities": ("apple", "cooler", "table")})
    def advance(*tokens):
        state = policy.start_state
        for token in tokens:
            state = policy.step(state, vocabulary.label_for_token(token))
            assert state is not None
        return state
    pick = advance("skill:pick", "arg:target_entity_name")
    assert vocabulary.label_for_token("obj:2") not in policy.allowed_tokens(pick)
    place = advance("skill:pick", "arg:target_entity_name", "obj:0", "skill:place", "arg:target_container_name")
    assert vocabulary.label_for_token("obj:0") not in policy.allowed_tokens(place)
    assert vocabulary.label_for_token("obj:1") not in policy.allowed_tokens(place)


def test_materialize_plan_and_score_vlabench_plan_auto_resolve_task_name():
    bundle = build_vlabench_world_graph("test_auto_resolve_world")
    entities = ("apple", "bowl", "kitchen_counter")

    # A 2-step plan for cook_dishes (which requires min 4 steps) must fail post-generation verification
    short_plan = [
        {"name": "pick", "params": {"target_entity_name": 0}},
        {"name": "place", "params": {"target_container_name": 1}},
    ]

    # materialize_plan auto-resolves schema when task_name="cook_dishes" is passed
    root = materialize_plan(short_plan, entities, bundle, task_name="cook_dishes")
    evaluation = verify_plan_constraints(root, bundle)
    assert evaluation.score < 1.0
    assert evaluation.results["task__cook_dishes__obligations"]["satisfied"] < 100.0

    # score_vlabench_plan also propagates task_name and marks short composite plans invalid
    ref_plan = [
        {"name": "pick", "params": {"target_entity_name": 0}},
        {"name": "place", "params": {"target_container_name": 1}},
        {"name": "pick", "params": {"target_entity_name": 0}},
        {"name": "place", "params": {"target_container_name": 1}},
    ]
    breakdown = score_vlabench_plan(
        short_plan,
        ref_plan,
        entity_table=entities,
        world_bundle=bundle,
        task_name="cook_dishes",
    )
    assert not breakdown.valid
    assert any("task__cook_dishes__obligations" in err or "constraints scored" in err for err in breakdown.errors)


def test_score_planning_metrics_grounds_entity_indices_against_names():
    from test_regr.VLABenchAgentInterface.planning_metrics import score_planning_metrics

    # Generated plan contains entity indices; reference contains names
    pred_plan = [
        {"name": "pick", "params": {"target_entity_name": 0}},
        {"name": "place", "params": {"target_container_name": 1}},
    ]
    ref_plan = [
        {"name": "pick", "params": {"target_entity_name": "apple"}},
        {"name": "place", "params": {"target_container_name": "bowl"}},
    ]
    entities = ("apple", "bowl")

    # With entity_table provided, pick 0 -> place 1 scores 100/100
    scores = score_planning_metrics(pred_plan, ref_plan, entity_table=entities)
    assert scores["SR"] == 100.0
    assert scores["PR"] == 100.0
    assert scores["SPR"] == 100.0
    assert scores["PM"] == 100.0
    assert scores["score"] == 100.0


def test_decoding_excludes_nonexistent_entity_pointers_and_floor():
    from ..training import build_constraint_runtime
    runtime = build_constraint_runtime(max_entities=4, max_operations=2, name_prefix="compiled_pointer_mask")
    vocab = runtime.vocabulary
    def advance(policy, *tokens):
        state = policy.start_state
        for token in tokens:
            state = policy.step(state, vocab.label_for_token(token))
            assert state is not None
        return state
    policy = runtime.dfa_for("vlabench", {"entities": ("apple", "cooler")})
    place = advance(policy, "skill:pick", "arg:target_entity_name", "obj:0", "skill:place", "arg:target_container_name")
    assert policy.allowed_tokens(place) == set()
    floor = runtime.dfa_for("vlabench", {"entities": ("apple", "cooler", "kitchen_floor")})
    pick = advance(floor, "skill:pick", "arg:target_entity_name")
    assert vocab.label_for_token("obj:2") not in floor.allowed_tokens(pick)
    assert vocab.label_for_token("obj:3") not in floor.allowed_tokens(pick)


def test_composite_completion_rejects_premature_eos_and_is_not_accepting():
    from ..training import build_constraint_runtime
    runtime = build_constraint_runtime(max_entities=2, max_operations=2, name_prefix="compiled_completion")
    policy = runtime.dfa_for("vlabench", {"entities": ("apple", "bowl"), "task": "cook_dishes"})
    state = policy.start_state
    for token in ("skill:pick", "arg:target_entity_name", "obj:0", "skill:place", "arg:target_container_name", "obj:1"):
        state = policy.step(state, runtime.vocabulary.label_for_token(token))
        assert state is not None
    assert runtime.vocabulary.eos_label not in policy.allowed_tokens(state)
    assert not policy.is_accepting(state)


def test_verifier_catches_self_placement_across_operations():
    bundle = build_vlabench_world_graph("test_self_placement_across_ops_world")
    entities = ("apple", "bowl")

    # pick apple -> place into apple
    self_placement_plan = [
        {"name": "pick", "params": {"target_entity_name": "apple"}},
        {"name": "place", "params": {"target_container_name": "apple"}},
    ]
    root = materialize_plan(self_placement_plan, entities, bundle)
    evaluation = verify_plan_constraints(root, bundle)
    assert evaluation.score < 1.0
    assert evaluation.results["anti_self_placement"]["satisfied"] < 100.0


def test_destination_consistency_catches_divergent_containers():
    bundle = build_vlabench_world_graph("test_dest_consistency_world")
    entities = ("apple", "pear", "bowl", "plate")

    # pick apple -> place bowl -> pick pear -> place plate
    divergent_plan = [
        {"name": "pick", "params": {"target_entity_name": "apple"}},
        {"name": "place", "params": {"target_container_name": "bowl"}},
        {"name": "pick", "params": {"target_entity_name": "pear"}},
        {"name": "place", "params": {"target_container_name": "plate"}},
    ]
    root_bad = materialize_plan(divergent_plan, entities, bundle, task_context={"task_target_container": "bowl"})
    eval_bad = verify_plan_constraints(root_bad, bundle)
    assert eval_bad.score < 1.0
    assert eval_bad.results["container_destination_consistency"]["satisfied"] < 100.0

    # Consistent destination: place into bowl both times
    consistent_plan = [
        {"name": "pick", "params": {"target_entity_name": "apple"}},
        {"name": "place", "params": {"target_container_name": "bowl"}},
        {"name": "pick", "params": {"target_entity_name": "pear"}},
        {"name": "place", "params": {"target_container_name": "bowl"}},
    ]
    root_good = materialize_plan(consistent_plan, entities, bundle, task_context={"task_target_container": "bowl"})
    eval_good = verify_plan_constraints(root_good, bundle)
    assert eval_good.results["container_destination_consistency"]["satisfied"] == 100.0


