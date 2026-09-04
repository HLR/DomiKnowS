from collections import OrderedDict

import pytest
import torch

from domiknows.graph import (
    Concept,
    DataNode,
    EnumConcept,
    Graph,
    Relation,
    andL,
    execute,
    existsL,
    iotaL,
    queryL,
    sumL,
)
from domiknows.solver import ilpOntSolverFactory


@pytest.fixture(autouse=True)
def _reset_graph_state():
    ilpOntSolverFactory.clear()
    Graph.clear()
    Concept.clear()
    Relation.clear()
    DataNode.collectedConceptsAndRelations = None
    yield
    ilpOntSolverFactory.clear()
    Graph.clear()
    Concept.clear()
    Relation.clear()
    DataNode.collectedConceptsAndRelations = None


def _binary_scene(name, executable_factory):
    with Graph(name) as graph:
        scene = Concept(name=f"{name}_scene")
        item = Concept(name=f"{name}_item")
        scene.contains(item)
        flag = item(name=f"{name}_flag")
        executable_factory(flag)

    root = DataNode(instanceID=0, ontologyNode=scene)
    root.current_device = "cpu"
    return graph, root, item, flag


def _add_binary_items(root, item, flag, probabilities):
    children = []
    for index, probability in enumerate(probabilities):
        child = DataNode(instanceID=index, ontologyNode=item)
        child.current_device = "cpu"
        logits = torch.log(
            torch.tensor([1.0 - probability, probability], dtype=torch.float32)
        )
        child.attributes[f"<{flag.name}>"] = logits
        child.attributes[f"<{flag.name}>/local/softmax"] = torch.tensor(
            [1.0 - probability, probability], dtype=torch.float32
        )
        root.addChildDataNode(child)
        children.append(child)
    return children


def _add_constraint_child(root, *names):
    constraint = DataNode(
        instanceID=0,
        ontologyNode=root.graph.get_constraint_concept(),
    )
    constraint.current_device = "cpu"
    for name in names:
        # The value activates the ELC but must not force its inferred answer.
        constraint.attributes[f"{name}/label"] = torch.tensor(0)
    root.addChildDataNode(constraint)
    return constraint


def test_tnorm_boolean_returns_and_persists_answer_probability():
    _, root, item, flag = _binary_scene(
        "soft_exists", lambda concept: execute(existsL(concept("x")))
    )
    children = _add_binary_items(root, item, flag, (0.8, 0.3))
    constraint = _add_constraint_child(root, "ELC0")

    results = root.inferExecutableResults(flag, mode="tnorm", tnorm="P")

    result = results["ELC0"]
    assert result["type"] == "boolean"
    assert result["answer"] is True
    assert result["probability"] == pytest.approx(0.86, abs=1e-6)
    assert torch.allclose(
        result["distribution"], torch.tensor([0.14, 0.86]), atol=1e-6
    )
    assert constraint.attributes["ELC0/answer"] is True
    assert constraint.attributes["ELC0/probability"] == pytest.approx(0.86)
    assert all(
        not any("/ILP" in key for key in child.attributes)
        for child in children
    )


@pytest.mark.parametrize("mode", ["tnorm", "circuit"])
def test_count_search_ignores_activation_label_and_returns_distribution(mode):
    _, root, item, flag = _binary_scene(
        f"{mode}_count", lambda concept: execute(sumL(concept("x")))
    )
    _add_binary_items(root, item, flag, (0.8, 0.3))
    constraint = _add_constraint_child(root, "ELC0")
    assert constraint.attributes["ELC0/label"].item() == 0

    results = root.inferExecutableResults(flag, mode=mode, tnorm="P")

    result = results["ELC0"]
    assert result["type"] == "count"
    assert result["answer"] == 1
    assert isinstance(result["answer"], int)
    assert not isinstance(result["answer"], bool)
    assert result["probability"] == pytest.approx(0.62, abs=1e-6)
    assert torch.allclose(
        result["distribution"],
        torch.tensor([0.14, 0.62, 0.24]),
        atol=1e-6,
    )
    assert result["distribution"].sum().item() == pytest.approx(1.0)
    assert constraint.attributes["ELC0/answer"] == 1
    assert constraint.attributes["ELC0/probability"] == pytest.approx(0.62)
    assert result["exact"] is (mode == "circuit")


@pytest.mark.parametrize("mode", ["tnorm", "circuit"])
def test_query_returns_class_name_probability_and_distribution(mode):
    with Graph(f"{mode}_query") as graph:
        scene = Concept(name=f"{mode}_query_scene")
        item = Concept(name=f"{mode}_query_item")
        scene.contains(item)
        target = item(name=f"{mode}_target")
        color = item(
            name=f"{mode}_color",
            ConceptClass=EnumConcept,
            values=["red", "blue"],
        )
        execute(queryL(color, iotaL(target("x"))))

    root = DataNode(instanceID=0, ontologyNode=scene)
    root.current_device = "cpu"
    child = DataNode(instanceID=0, ontologyNode=item)
    child.current_device = "cpu"
    child.attributes[f"<{target.name}>"] = torch.log(torch.tensor([0.1, 0.9]))
    child.attributes[f"<{target.name}>/local/softmax"] = torch.tensor(
        [0.1, 0.9]
    )
    child.attributes[f"<{color.name}>"] = torch.log(torch.tensor([0.7, 0.3]))
    child.attributes[f"<{color.name}>/local/softmax"] = torch.tensor(
        [0.7, 0.3]
    )
    root.addChildDataNode(child)
    constraint = _add_constraint_child(root, "ELC0")

    result = root.inferExecutableResults(
        target, color, mode=mode, tnorm="P"
    )["ELC0"]

    assert result["type"] == "query"
    assert result["answer"] == "red"
    assert result["classNames"] == ["red", "blue"]
    assert result["probability"] == pytest.approx(
        result["distribution"][0].item()
    )
    assert result["distribution"].shape == (2,)
    assert constraint.attributes["ELC0/answer"] == "red"
    assert constraint.attributes["ELC0/probability"] == pytest.approx(
        result["probability"]
    )


def test_populate_false_is_non_mutating_and_empty_active_set_clears_stale_data():
    _, root, item, flag = _binary_scene(
        "lifecycle", lambda concept: execute(existsL(concept("x")))
    )
    _add_binary_items(root, item, flag, (0.8,))
    constraint = _add_constraint_child(root, "ELC0")

    result = root.inferExecutableResults(
        flag, populate=False
    )["ELC0"]
    assert result["answer"] is True
    assert "ELC0/answer" not in constraint.attributes
    assert "ELC0/probability" not in constraint.attributes

    root.inferExecutableResults(flag)
    assert "ELC0/answer" in constraint.attributes
    assert "ELC0/probability" in constraint.attributes

    constraint.attributes.pop("ELC0/label")
    assert root.inferExecutableResults(flag) == {}
    assert "ELC0/answer" not in constraint.attributes
    assert "ELC0/probability" not in constraint.attributes


def test_explicit_constraint_selection_works_without_activation_label():
    _, root, item, flag = _binary_scene(
        "explicit", lambda concept: execute(existsL(concept("x")))
    )
    _add_binary_items(root, item, flag, (0.2,))
    constraint = _add_constraint_child(root)

    result = root.inferExecutableResults(
        flag, constraints="ELC0"
    )["ELC0"]

    assert result["answer"] is False
    assert result["probability"] == pytest.approx(0.8, abs=1e-6)
    assert constraint.attributes["ELC0/answer"] is False
    assert constraint.attributes["ELC0/probability"] == pytest.approx(0.8)


def test_circuit_size_limit_falls_back_to_product_tnorm():
    _, root, item, flag = _binary_scene(
        "fallback", lambda concept: execute(existsL(concept("x")))
    )
    _add_binary_items(root, item, flag, (0.8, 0.3))
    _add_constraint_child(root, "ELC0")

    with pytest.warns(RuntimeWarning, match="Falling back to Product t-norm"):
        result = root.inferExecutableResults(
            flag,
            mode="circuit",
            circuitBackend="bdd",
            circuitMaxNodes=1,
            circuitSizeLimitAction="raise",
        )["ELC0"]

    assert result["answer"] is True
    assert result["probability"] == pytest.approx(0.86, abs=1e-6)
    assert result["mode"] == "tnorm"
    assert result["exact"] is False
    assert result["fallback"] == "circuit-size-limit"


@pytest.mark.gurobi
def test_ilp_mode_delegates_and_normalizes_persisted_boolean_answer():
    _, root, item, flag = _binary_scene(
        "generic_ilp", lambda concept: execute(existsL(concept("x")))
    )
    children = _add_binary_items(root, item, flag, (0.8, 0.3))
    constraint = _add_constraint_child(root, "ELC0")
    constraint.attributes["ELC0/probability"] = 0.123

    result = root.inferExecutableResults(flag, mode="ilp")["ELC0"]

    assert result == {
        "type": "boolean",
        "answer": True,
        "probability": None,
        "distribution": None,
        "mode": "ilp",
        "exact": None,
    }
    assert constraint.attributes["ELC0/answer"] is True
    assert "ELC0/probability" not in constraint.attributes
    assert all(
        f"<{flag.name}>/ILP" in child.attributes for child in children
    )


@pytest.mark.gurobi
def test_ilp_mode_returns_native_count_through_generic_interface():
    _, root, item, flag = _binary_scene(
        "generic_ilp_count", lambda concept: execute(sumL(concept("x")))
    )
    _add_binary_items(root, item, flag, (0.8, 0.3))
    _add_constraint_child(root, "ELC0")

    result = root.inferExecutableResults(flag, mode="ilp")["ELC0"]

    assert result["type"] == "count"
    assert result["answer"] == 1
    assert isinstance(result["answer"], int)
    assert not isinstance(result["answer"], bool)
    assert result["probability"] is None
    assert result["distribution"] is None


@pytest.mark.gurobi
def test_ilp_mode_returns_query_class_name_through_generic_interface():
    with Graph("generic_ilp_query") as graph:
        scene = Concept(name="generic_ilp_query_scene")
        item = Concept(name="generic_ilp_query_item")
        scene.contains(item)
        target = item(name="generic_ilp_query_target")
        color = item(
            name="generic_ilp_query_color",
            ConceptClass=EnumConcept,
            values=["red", "blue"],
        )
        execute(queryL(color, iotaL(target("x"))))

    root = DataNode(instanceID=0, ontologyNode=scene)
    root.current_device = "cpu"
    child = DataNode(instanceID=0, ontologyNode=item)
    child.current_device = "cpu"
    child.attributes[f"<{target.name}>"] = torch.tensor([0.1, 2.0])
    child.attributes[f"<{color.name}>"] = torch.tensor([2.0, 0.1])
    root.addChildDataNode(child)
    _add_constraint_child(root, "ELC0")

    result = root.inferExecutableResults(
        target, color, mode="ilp"
    )["ELC0"]

    assert result["type"] == "query"
    assert result["answer"] == "red"
    assert result["classNames"] == ["red", "blue"]
    assert result["probability"] is None
    assert result["distribution"] is None


def test_ilp_mode_rejects_non_mutating_and_mismatched_selection_options():
    _, root, item, flag = _binary_scene(
        "generic_ilp_options",
        lambda concept: execute(existsL(concept("x"))),
    )
    _add_binary_items(root, item, flag, (0.8,))
    _add_constraint_child(root, "ELC0")

    with pytest.raises(ValueError, match="requires populate=True"):
        root.inferExecutableResults(flag, mode="ilp", populate=False)

    with pytest.raises(ValueError, match="must exactly match"):
        root.inferExecutableResults(
            flag, mode="ilp", constraints=[]
        )


def test_adhoc_single_dsl_tnorm_is_return_only_and_excludes_registered_query():
    graph, root, item, flag = _binary_scene(
        "adhoc_tnorm", lambda concept: execute(sumL(concept("x")))
    )
    _add_binary_items(root, item, flag, (0.8, 0.3))
    constraint = _add_constraint_child(root, "ELC0")
    constraint.attributes["ELC0/answer"] = 17
    constraint.attributes["ELC0/probability"] = 0.25

    graph_state = (
        list(graph.logicalConstrains.items()),
        list(graph.executableLCs.items()),
        dict(graph.executableLCsLabels),
        dict(constraint.attributes),
        graph.executableLCs["ELC0"].active,
    )

    results = root.inferExecutableResults(
        flag,
        queries='existsL(asked_flag("x"))',
        queryNamespace={"asked_flag": flag},
        populate=True,
    )

    assert list(results) == ["ADHOC0"]
    assert results["ADHOC0"]["answer"] is True
    assert results["ADHOC0"]["probability"] == pytest.approx(0.86)
    assert list(graph.logicalConstrains.items()) == graph_state[0]
    assert list(graph.executableLCs.items()) == graph_state[1]
    assert graph.executableLCsLabels == graph_state[2]
    assert constraint.attributes == graph_state[3]
    assert graph.executableLCs["ELC0"].active is graph_state[4]

    captured_context_result = root.inferExecutableResults(
        flag, queries='existsL(flag("x"))'
    )["ADHOC0"]
    assert captured_context_result["answer"] is True
    assert constraint.attributes == graph_state[3]


def test_adhoc_named_object_and_string_queries_in_circuit_mode():
    with Graph("adhoc_circuit") as graph:
        scene = Concept(name="adhoc_circuit_scene")
        item = Concept(name="adhoc_circuit_item")
        scene.contains(item)
        target = item(name="adhoc_circuit_target")
        color = item(
            name="adhoc_circuit_color",
            ConceptClass=EnumConcept,
            values=["red", "blue"],
        )

    with graph:
        count_expression = sumL(target("x"))

    root = DataNode(instanceID=0, ontologyNode=scene)
    root.current_device = "cpu"
    child = DataNode(instanceID=0, ontologyNode=item)
    child.current_device = "cpu"
    child.attributes[f"<{target.name}>"] = torch.log(torch.tensor([0.1, 0.9]))
    child.attributes[f"<{target.name}>/local/softmax"] = torch.tensor(
        [0.1, 0.9]
    )
    child.attributes[f"<{color.name}>"] = torch.log(torch.tensor([0.7, 0.3]))
    child.attributes[f"<{color.name}>/local/softmax"] = torch.tensor(
        [0.7, 0.3]
    )
    root.addChildDataNode(child)
    logical_state = list(graph.logicalConstrains.items())
    concepts = root.collectConceptsAndRelations((target, color))
    solver, _ = root.getILPSolver(concepts)
    circuit_processor = solver.myCircuitBooleanMethods
    circuit_cache = dict(solver.circuitLossCalculator._compile_cache)

    results = root.inferExecutableResults(
        target,
        color,
        mode="circuit",
        queries=OrderedDict(
            [
                ("count", count_expression),
                (
                    "color",
                    'queryL(answer_color, iotaL(selector("x")))',
                ),
            ]
        ),
        queryNamespace={"answer_color": color, "selector": target},
    )

    assert list(results) == ["count", "color"]
    assert results["count"]["answer"] == 1
    assert results["count"]["distribution"].shape == (2,)
    assert results["color"]["answer"] == "red"
    assert results["color"]["classNames"] == ["red", "blue"]
    assert results["color"]["distribution"].shape == (2,)
    assert list(graph.logicalConstrains.items()) == logical_state
    assert graph.executableLCs == {}
    assert root._getExecutableConstraintDataNode() is None
    assert solver.myCircuitBooleanMethods is circuit_processor
    assert solver.circuitLossCalculator._compile_cache == circuit_cache


def test_adhoc_batch_samples_are_evaluated_independently():
    with Graph("adhoc_batch") as graph:
        batch = Concept(name="adhoc_batch_root", batch=True)
        scene = Concept(name="adhoc_batch_scene")
        item = Concept(name="adhoc_batch_item")
        batch.contains(scene)
        scene.contains(item)
        flag = item(name="adhoc_batch_flag")

    batch_root = DataNode(instanceID=0, ontologyNode=batch)
    for scene_index, probability in enumerate((0.8, 0.2)):
        scene_root = DataNode(instanceID=scene_index, ontologyNode=scene)
        item_dn = DataNode(instanceID=scene_index, ontologyNode=item)
        item_dn.attributes[f"<{flag.name}>"] = torch.log(
            torch.tensor([1.0 - probability, probability])
        )
        item_dn.attributes[f"<{flag.name}>/local/softmax"] = torch.tensor(
            [1.0 - probability, probability]
        )
        scene_root.addChildDataNode(item_dn)
        batch_root.addChildDataNode(scene_root)

    results = batch_root.inferExecutableResults(
        flag,
        queries='existsL(query_flag("x"))',
        queryNamespace={"query_flag": flag},
        populate=True,
    )

    assert isinstance(results, list)
    assert [sample["ADHOC0"]["answer"] for sample in results] == [
        True,
        False,
    ], results
    assert all(
        scene_root._getExecutableConstraintDataNode() is None
        for scene_root in batch_root.getChildDataNodes()
    )


def test_adhoc_validation_and_parse_failure_restore_graph_state():
    graph, root, item, flag = _binary_scene(
        "adhoc_validation", lambda concept: execute(existsL(concept("x")))
    )
    _add_binary_items(root, item, flag, (0.8,))
    constraint = _add_constraint_child(root, "ELC0")
    constraint.attributes["ELC0/answer"] = True
    executable_state = list(graph.executableLCs.items())
    attribute_state = dict(constraint.attributes)

    with pytest.raises(ValueError, match="mutually exclusive"):
        root.inferExecutableResults(
            flag, constraints="ELC0", queries='existsL(flag("x"))'
        )
    with pytest.raises(NameError, match="queryNamespace"):
        root.inferExecutableResults(
            flag, queries='existsL(missing_symbol("x"))'
        )
    with pytest.raises(ValueError, match="non-empty strings"):
        root.inferExecutableResults(flag, queries={"": 'existsL(flag("x"))'})
    with pytest.raises(TypeError, match="must be a mapping"):
        root.inferExecutableResults(
            flag, queries='existsL(flag("x"))', queryNamespace=[]
        )
    with pytest.raises(TypeError, match="DSL string"):
        root.inferExecutableResults(flag, queries={"bad": object()})

    assert list(graph.executableLCs.items()) == executable_state
    assert constraint.attributes == attribute_state


def test_adhoc_duplicate_object_and_foreign_graph_are_rejected_and_restored():
    graph, root, item, flag = _binary_scene(
        "adhoc_objects", lambda concept: execute(existsL(concept("x")))
    )
    _add_binary_items(root, item, flag, (0.8,))
    with graph:
        expression = sumL(flag("x"))
    logical_state = list(graph.logicalConstrains.items())
    expression_name = expression.lcName

    with pytest.raises(ValueError, match="multiple ad hoc query names"):
        root.inferExecutableResults(
            flag,
            queries=OrderedDict([("first", expression), ("second", expression)]),
        )

    assert list(graph.logicalConstrains.items()) == logical_state
    assert expression.lcName == expression_name

    class DuplicateNamesMapping(dict):
        def items(self):
            return [
                ("same", 'existsL(flag("x"))'),
                ("same", 'sumL(flag("x"))'),
            ]

    with pytest.raises(ValueError, match="Duplicate ad hoc query name"):
        root.inferExecutableResults(flag, queries=DuplicateNamesMapping())
    assert list(graph.logicalConstrains.items()) == logical_state

    with Graph("foreign_adhoc") as foreign_graph:
        foreign_item = Concept(name="foreign_adhoc_item")
        foreign_flag = foreign_item(name="foreign_adhoc_flag")
        foreign_expression = existsL(foreign_flag("x"))

    with pytest.raises(ValueError, match="another graph"):
        root.inferExecutableResults(flag, queries=foreign_expression)
    assert list(graph.logicalConstrains.items()) == logical_state
    assert foreign_expression in foreign_graph.logicalConstrains.values()


def test_adhoc_inference_failure_restores_registered_state(monkeypatch):
    graph, root, item, flag = _binary_scene(
        "adhoc_failure", lambda concept: execute(existsL(concept("x")))
    )
    children = _add_binary_items(root, item, flag, (0.8,))
    constraint = _add_constraint_child(root, "ELC0")
    constraint.attributes["ELC0/answer"] = False
    constraint.attributes["ELC0/probability"] = 0.75
    children[0].attributes[f"<{flag.name}>/ILP"] = torch.tensor(0.125)
    graph_state = list(graph.executableLCs.items())
    attribute_state = dict(constraint.attributes)

    from domiknows.solver.executableInference import ExecutableInference

    def fail_inference(*args, **kwargs):
        raise RuntimeError("forced ad hoc inference failure")

    monkeypatch.setattr(ExecutableInference, "infer", fail_inference)
    with pytest.raises(RuntimeError, match="forced ad hoc inference failure"):
        root.inferExecutableResults(
            flag,
            queries='existsL(query_flag("x"))',
            queryNamespace={"query_flag": flag},
        )

    assert list(graph.executableLCs.items()) == graph_state
    assert constraint.attributes == attribute_state
    assert children[0].attributes[f"<{flag.name}>/ILP"].item() == 0.125


@pytest.mark.gurobi
def test_adhoc_ilp_joint_queries_return_native_answers_without_constraint_node():
    with Graph("adhoc_ilp_joint") as graph:
        scene = Concept(name="adhoc_ilp_joint_scene")
        item = Concept(name="adhoc_ilp_joint_item")
        scene.contains(item)
        target = item(name="adhoc_ilp_joint_target")
        color = item(
            name="adhoc_ilp_joint_color",
            ConceptClass=EnumConcept,
            values=["red", "blue"],
        )

    root = DataNode(instanceID=0, ontologyNode=scene)
    child = DataNode(instanceID=0, ontologyNode=item)
    child.attributes[f"<{target.name}>"] = torch.tensor([0.1, 2.0])
    child.attributes[f"<{color.name}>"] = torch.tensor([2.0, 0.1])
    child.attributes[f"<{target.name}>/ILP"] = torch.tensor(0.25)
    root.addChildDataNode(child)
    assert root._getExecutableConstraintDataNode() is None

    results = root.inferExecutableResults(
        target,
        color,
        mode="ilp",
        populate=False,
        queries=OrderedDict(
            [
                ("present", 'existsL(selector("x"))'),
                ("count", 'sumL(selector("x"))'),
                (
                    "color",
                    'queryL(answer_color, iotaL(selector("x")))',
                ),
            ]
        ),
        queryNamespace={"selector": target, "answer_color": color},
    )

    assert list(results) == ["present", "count", "color"]
    assert results["present"]["answer"] is True
    assert results["count"]["answer"] == 1
    assert isinstance(results["count"]["answer"], int)
    assert results["color"]["answer"] == "red"
    assert results["color"]["classNames"] == ["red", "blue"]
    assert all(
        result["probability"] is None and result["distribution"] is None
        for result in results.values()
    )
    assert root._getExecutableConstraintDataNode() is None
    assert graph.logicalConstrains == {}
    assert graph.executableLCs == {}
    assert child.attributes[f"<{target.name}>/ILP"].item() == 0.25
    assert f"<{color.name}>/ILP" not in child.attributes


@pytest.mark.gurobi
def test_adhoc_ilp_failure_removes_temporary_constraint_node_and_registry():
    with Graph("adhoc_ilp_failure") as graph:
        scene = Concept(name="adhoc_ilp_failure_scene")
        item = Concept(name="adhoc_ilp_failure_item")
        scene.contains(item)
        flag = item(name="adhoc_ilp_failure_flag")

    root = DataNode(instanceID=0, ontologyNode=scene)
    child = DataNode(instanceID=0, ontologyNode=item)
    child.attributes[f"<{flag.name}>"] = torch.tensor([0.1, 2.0])
    root.addChildDataNode(child)
    children_state = list(root.getChildDataNodes())

    with pytest.raises(Exception, match="andL|unsupported|Unsupported"):
        root.inferExecutableResults(
            flag,
            mode="ilp",
            queries='andL(query_flag("x"), query_flag("x"))',
            queryNamespace={"query_flag": flag},
        )

    assert list(root.getChildDataNodes()) == children_state
    assert root._getExecutableConstraintDataNode() is None
    assert graph.logicalConstrains == {}
    assert graph.executableLCs == {}
    assert not any("/ILP" in key for key in child.attributes)


@pytest.mark.gurobi
def test_adhoc_ilp_restores_registered_labels_answers_and_local_probabilities():
    graph, root, item, flag = _binary_scene(
        "adhoc_ilp_restore", lambda concept: execute(sumL(concept("x")))
    )
    children = _add_binary_items(root, item, flag, (0.8, 0.3))
    constraint = _add_constraint_child(root, "ELC0")
    constraint.attributes["ELC0/answer"] = 99
    constraint.attributes["ELC0/probability"] = 0.125
    children[0].attributes[f"<{flag.name}>/ILP"] = torch.tensor(0.625)
    executable_state = list(graph.executableLCs.items())
    active_state = graph.executableLCs["ELC0"].active
    constraint_state = dict(constraint.attributes)
    child_states = [dict(child.attributes) for child in children]

    result = root.inferExecutableResults(
        flag,
        mode="ilp",
        queries='existsL(query_flag("x"))',
        queryNamespace={"query_flag": flag},
    )["ADHOC0"]

    assert result["answer"] is True
    assert list(graph.executableLCs.items()) == executable_state
    assert graph.executableLCs["ELC0"].active is active_state
    assert constraint.attributes == constraint_state
    for child, state in zip(children, child_states):
        assert child.attributes.keys() == state.keys()
        for key, value in state.items():
            if torch.is_tensor(value):
                assert child.attributes[key] is value
            else:
                assert child.attributes[key] == value


@pytest.mark.gurobi
def test_adhoc_ilp_batch_answers_each_sample_independently_and_cleans_up():
    with Graph("adhoc_ilp_batch") as graph:
        batch = Concept(name="adhoc_ilp_batch_root", batch=True)
        scene = Concept(name="adhoc_ilp_batch_scene")
        item = Concept(name="adhoc_ilp_batch_item")
        batch.contains(scene)
        scene.contains(item)
        flag = item(name="adhoc_ilp_batch_flag")

    batch_root = DataNode(instanceID=0, ontologyNode=batch)
    scenes = []
    for scene_index, logits in enumerate(((0.1, 2.0), (2.0, 0.1))):
        scene_root = DataNode(instanceID=scene_index, ontologyNode=scene)
        item_dn = DataNode(instanceID=scene_index, ontologyNode=item)
        item_dn.attributes[f"<{flag.name}>"] = torch.tensor(logits)
        scene_root.addChildDataNode(item_dn)
        batch_root.addChildDataNode(scene_root)
        scenes.append(scene_root)

    results = batch_root.inferExecutableResults(
        flag,
        mode="ilp",
        queries='existsL(query_flag("x"))',
        queryNamespace={"query_flag": flag},
    )

    assert [sample["ADHOC0"]["answer"] for sample in results] == [True, False]
    assert all(scene._getExecutableConstraintDataNode() is None for scene in scenes)
    assert graph.logicalConstrains == {}
    assert graph.executableLCs == {}
    assert all(
        not any("/ILP" in key for key in scene.getChildDataNodes()[0].attributes)
        for scene in scenes
    )
