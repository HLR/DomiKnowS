import importlib.util
import itertools
import math
import sys

import pytest
import torch

from domiknows.solver.bdd import CircuitSizeLimitExceeded
from domiknows.solver.circuitBooleanMethods import (
    CircuitLeaf,
    circuitBooleanMethods,
)
from domiknows.solver.logicalConstraintConstructor import LogicalConstraintConstructor


def _binary(name, probability, instance=0):
    probability = torch.as_tensor(probability)
    key = (name, instance, 0)
    return CircuitLeaf(
        key,
        probability,
        ("binary", key),
        1,
        (1.0 - probability, probability),
    )


def _categorical(name, probabilities, instance=0):
    probabilities = tuple(probabilities)
    return [
        CircuitLeaf(
            (name, instance, index),
            probability,
            ("categorical", name, instance),
            index,
            probabilities,
            categorical=True,
        )
        for index, probability in enumerate(probabilities)
    ]


def _brute_binary(probabilities, predicate):
    total = 0.0
    for assignment in itertools.product((False, True), repeat=len(probabilities)):
        weight = math.prod(
            probability if value else 1.0 - probability
            for probability, value in zip(probabilities, assignment)
        )
        if predicate(*assignment):
            total += weight
    return total


@pytest.mark.parametrize(
    "builder,predicate",
    [
        (lambda b, x: b.andVar(None, *x), lambda a, c, d: a and c and d),
        (lambda b, x: b.orVar(None, *x), lambda a, c, d: a or c or d),
        (lambda b, x: b.nandVar(None, *x), lambda a, c, d: not (a and c and d)),
        (lambda b, x: b.norVar(None, *x), lambda a, c, d: not (a or c or d)),
        (lambda b, x: b.xorVar(None, *x), lambda a, c, d: sum((a, c, d)) == 1),
        (lambda b, x: b.ifVar(None, x[0], x[1]), lambda a, c, d: (not a) or c),
        (
            lambda b, x: b.equivalenceVar(None, *x),
            lambda a, c, d: (a and c and d) or not (a or c or d),
        ),
    ],
)
def test_boolean_operator_wmc_matches_enumeration(builder, predicate):
    probabilities = [0.2, 0.55, 0.8]
    backend = circuitBooleanMethods(backend="bdd")
    backend.begin_evaluation()
    variables = [_binary(f"x{i}", p) for i, p in enumerate(probabilities)]
    actual = backend.wmc(builder(backend, variables))
    expected = _brute_binary(probabilities, predicate)
    assert actual.item() == pytest.approx(expected, abs=1e-7)


def test_repeated_grounded_leaf_keeps_identity():
    backend = circuitBooleanMethods(backend="bdd")
    backend.begin_evaluation()
    probability = torch.tensor(0.37)
    leaf = _binary("shared", probability)
    root = backend.andVar(None, leaf, leaf, backend.orVar(None, leaf, leaf))
    assert backend.wmc(root).item() == pytest.approx(0.37)


def test_fixed_var_uses_forced_label_instead_of_model_probability():
    backend = circuitBooleanMethods(backend="bdd")
    backend.begin_evaluation()
    base = _binary("fixed", torch.tensor(0.2))
    forced_true = CircuitLeaf(
        base.key,
        base.probability,
        base.variable_key,
        base.value_index,
        base.probabilities,
        fixed_value=1,
    )
    forced_false = CircuitLeaf(
        base.key,
        base.probability,
        base.variable_key,
        base.value_index,
        base.probabilities,
        fixed_value=0,
    )
    assert backend.wmc(backend.fixedVar(None, forced_true)).item() == 1.0
    assert backend.wmc(backend.fixedVar(None, forced_false)).item() == 0.0


@pytest.mark.parametrize(
    "operation,limit,predicate",
    [
        (">=", 2, lambda count: count >= 2),
        ("<=", 1, lambda count: count <= 1),
        ("==", 2, lambda count: count == 2),
    ],
)
def test_cardinality_wmc_matches_enumeration(operation, limit, predicate):
    probabilities = [0.15, 0.4, 0.65, 0.9]
    backend = circuitBooleanMethods(backend="bdd")
    backend.begin_evaluation()
    variables = [_binary(f"x{i}", p) for i, p in enumerate(probabilities)]
    root = backend.countVar(None, *variables, limitOp=operation, limit=limit)
    expected = _brute_binary(probabilities, lambda *a: predicate(sum(a)))
    assert backend.wmc(root).item() == pytest.approx(expected, abs=1e-7)


def test_compare_counts_and_sum_wmc_match_enumeration():
    probabilities = [0.2, 0.7, 0.4, 0.8]
    backend = circuitBooleanMethods(backend="bdd")
    backend.begin_evaluation()
    variables = [_binary(f"x{i}", p) for i, p in enumerate(probabilities)]
    comparison = backend.compareCountsVar(
        None, variables[:2], variables[2:], compareOp=">=", diff=1
    )
    expected_comparison = _brute_binary(
        probabilities, lambda *a: sum(a[:2]) - sum(a[2:]) >= 1
    )
    assert backend.wmc(comparison).item() == pytest.approx(expected_comparison)

    summation = backend.summationVar(None, *variables, label=2)
    expected_sum = _brute_binary(probabilities, lambda *a: sum(a) == 2)
    assert backend.wmc(summation).item() == pytest.approx(expected_sum)


def test_multiclass_same_and_different_use_categorical_wmc():
    first_probs = tuple(torch.tensor(v) for v in (0.2, 0.3, 0.5))
    second_probs = tuple(torch.tensor(v) for v in (0.6, 0.1, 0.3))
    first = _categorical("color", first_probs, instance=0)
    second = _categorical("color", second_probs, instance=1)
    backend = circuitBooleanMethods(backend="bdd")
    backend.begin_evaluation()
    same = backend.sameVar(None, None, [0, 1, 2], first, second)
    expected = sum(a.item() * b.item() for a, b in zip(first_probs, second_probs))
    assert backend.wmc(same).item() == pytest.approx(expected)
    different = backend.notVar(None, same)
    assert backend.wmc(different).item() == pytest.approx(1.0 - expected)


def test_iota_query_has_exact_unique_selection_semantics():
    condition_probabilities = [0.2, 0.5]
    class_probabilities = [(0.7, 0.3), (0.1, 0.9)]
    backend = circuitBooleanMethods(backend="bdd")
    backend.begin_evaluation()
    conditions = [
        _binary("condition", p, instance=i)
        for i, p in enumerate(condition_probabilities)
    ]
    selections = backend.iotaVar(None, *conditions)
    subclass_data = [
        _categorical("material", tuple(torch.tensor(v) for v in probs), instance=i)
        for i, probs in enumerate(class_probabilities)
    ]
    query = backend.queryVar(
        None,
        None,
        ["rubber", "metal"],
        selections,
        subclass_data=subclass_data,
    )

    unique_probability = (
        condition_probabilities[0] * (1 - condition_probabilities[1])
        + (1 - condition_probabilities[0]) * condition_probabilities[1]
    )
    expected_metal = (
        condition_probabilities[0]
        * (1 - condition_probabilities[1])
        * class_probabilities[0][1]
        + (1 - condition_probabilities[0])
        * condition_probabilities[1]
        * class_probabilities[1][1]
    )
    assert backend.wmc(backend.iotaVar(None, *conditions, onlyConstrains=True)).item() == pytest.approx(
        unique_probability
    )
    assert backend.wmc(query[1]).item() == pytest.approx(expected_metal)


def test_semantic_loss_is_differentiable_and_hand_computed():
    logits = torch.tensor([0.2, -0.4], requires_grad=True)
    probabilities = logits.sigmoid()
    backend = circuitBooleanMethods(backend="bdd")
    backend.begin_evaluation()
    left, right = (_binary("x", probabilities[i], i) for i in range(2))
    probability = backend.wmc(backend.ifVar(None, left, right))
    loss = -torch.log(probability)
    expected = -math.log(1.0 - probabilities[0].item() * (1.0 - probabilities[1].item()))
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum().item() > 0


def test_bdd_structure_is_reused_and_budget_is_guarded():
    backend = circuitBooleanMethods(backend="bdd")
    probability = torch.tensor(0.4)
    leaf = _binary("x", probability)
    backend.begin_evaluation()
    first = backend.andVar(None, leaf, backend.notVar(None, backend.notVar(None, leaf)))
    count = backend.node_count
    backend.begin_evaluation()
    second = backend.andVar(None, leaf, backend.notVar(None, backend.notVar(None, leaf)))
    assert first is second
    assert backend.node_count == count

    limited = circuitBooleanMethods(backend="bdd", max_nodes=2)
    limited.begin_evaluation()
    with pytest.raises(CircuitSizeLimitExceeded):
        limited.andVar(None, _binary("limited", torch.tensor(0.5)))


def test_auto_backend_and_forced_bdd_fallback():
    forced = circuitBooleanMethods(backend="bdd")
    assert forced.backend_name == "bdd"
    automatic = circuitBooleanMethods(backend="auto")
    expected = "pysdd" if importlib.util.find_spec("pysdd") else "bdd"
    assert automatic.backend_name == expected


def test_auto_backend_falls_back_when_pysdd_is_forced_off(monkeypatch):
    monkeypatch.setitem(sys.modules, "pysdd", None)
    automatic = circuitBooleanMethods(backend="auto")
    assert automatic.backend_name == "bdd"


@pytest.mark.skipif(importlib.util.find_spec("pysdd") is None, reason="optional pysdd not installed")
def test_pysdd_and_bdd_have_matching_multiclass_values_and_logit_gradients():
    def evaluate(backend_name):
        logits = torch.tensor([0.4, -0.2, 0.8], requires_grad=True)
        probabilities = logits.softmax(dim=0)
        backend = circuitBooleanMethods(backend=backend_name)
        backend.begin_evaluation()
        leaves = _categorical("enum", tuple(probabilities))
        probability = backend.wmc(backend.orVar(None, leaves[0], leaves[2]))
        loss = -torch.log(probability)
        loss.backward()
        return probability.detach(), logits.grad.detach()

    bdd_probability, bdd_gradient = evaluate("bdd")
    sdd_probability, sdd_gradient = evaluate("pysdd")
    assert torch.allclose(bdd_probability, sdd_probability, atol=1e-7)
    assert torch.allclose(bdd_gradient, sdd_gradient, atol=1e-7)


@pytest.mark.skipif(importlib.util.find_spec("pysdd") is None, reason="optional pysdd not installed")
def test_pysdd_does_not_reuse_stale_categorical_weights_between_batches():
    backend = circuitBooleanMethods(backend="pysdd")
    backend.begin_evaluation()
    first = _categorical("first", tuple(torch.tensor(v) for v in (0.2, 0.8)))
    assert backend.wmc(first[1]).item() == pytest.approx(0.8)

    backend.begin_evaluation()
    second = _categorical("second", tuple(torch.tensor(v) for v in (0.65, 0.35)))
    assert backend.wmc(second[0]).item() == pytest.approx(0.65)


def test_product_tnorm_conjunction_is_consistent_with_exact_wmc():
    from domiknows.solver.lcLossBooleanMethods import lcLossBooleanMethods

    probabilities = [torch.tensor(0.25), torch.tensor(0.6), torch.tensor(0.8)]
    tnorm = lcLossBooleanMethods()
    tnorm.current_device = torch.device("cpu")
    tnorm.setTNorm("P")
    product_success = tnorm.andVar(None, *probabilities, onlyConstrains=False).squeeze()

    backend = circuitBooleanMethods(backend="bdd")
    backend.begin_evaluation()
    leaves = [_binary(f"p{index}", value) for index, value in enumerate(probabilities)]
    exact_probability = backend.wmc(backend.andVar(None, *leaves))
    assert exact_probability.item() == pytest.approx(product_success.item())
    assert -torch.log(exact_probability).item() == pytest.approx(
        -math.log(math.prod(value.item() for value in probabilities))
    )


def test_constructor_circuit_mode_returns_stable_leaf_key():
    class Logger:
        def __getattr__(self, _):
            return lambda *args, **kwargs: None

    class Concept:
        name = "predicate"
        _out = {}

    class Node:
        ontologyNode = type("OntologyNode", (), {"name": "entity"})()

        def __init__(self):
            self.attributes = {"<predicate>/local/softmax": torch.tensor([0.25, 0.75])}

        def getAttribute(self, key, *rest):
            return self.attributes.get(key)

        def getAttributes(self):
            return self.attributes

        def getInstanceID(self):
            return 7

    constructor = LogicalConstraintConstructor(Logger())
    node = Node()
    first = constructor.getMLResult(
        node,
        "<predicate>/local/softmax",
        ("predicate", 1, 0),
        0,
        circuit=True,
        concept=Concept(),
    )
    second = constructor.getMLResult(
        node,
        "<predicate>/local/softmax",
        ("predicate", 1, 0),
        0,
        circuit=True,
        concept=Concept(),
    )
    assert first.key == second.key == ("predicate", 7, 0)
    assert first.probability.item() == pytest.approx(0.75)


def test_public_semantic_loss_program_export():
    from domiknows.program import SemanticLossModel, SemanticLossProgram

    assert SemanticLossModel.__name__ == "SemanticLossModel"
    assert SemanticLossProgram.DEFAULTCMODEL is SemanticLossModel


def test_iota_query_dsl_traversal_matches_direct_exact_formula():
    from domiknows.graph import Concept as GraphConcept
    from domiknows.graph import Graph, Relation
    from domiknows.graph.concept import EnumConcept
    from domiknows.graph.logicalConstrain import iotaL, queryL
    from domiknows.program import LearningBasedProgram
    from domiknows.program.model.pytorch import PoiModel
    from domiknows.sensor.pytorch.learners import TorchLearner
    from domiknows.sensor.pytorch.relation_sensors import EdgeSensor
    from domiknows.sensor.pytorch.sensors import ReaderSensor

    Graph.clear(); GraphConcept.clear(); Relation.clear()
    with Graph("circuit_iota_query") as graph:
        scene = GraphConcept(name="scene")
        entity = GraphConcept(name="entity")
        (contains,) = scene.contains(entity)
        selected = entity(name="selected")
        material = entity(
            name="material",
            ConceptClass=EnumConcept,
            values=["metal", "rubber"],
        )
        material_query = queryL(material, iotaL(selected("x")), name="material_query")

    class SelectionLearner(TorchLearner):
        def forward(self, values):
            logits = torch.tensor(
                [[2.0, -1.0], [-0.5, 1.5], [1.0, 0.0]],
                device=values.device,
            )
            return logits[: len(values)]

    class MaterialLearner(TorchLearner):
        def forward(self, values):
            logits = torch.tensor(
                [[1.5, 0.0], [0.0, 1.0], [0.2, 0.3]],
                device=values.device,
            )
            return logits[: len(values)]

    graph.detach()
    scene["index"] = ReaderSensor(keyword="scene")
    entity["index"] = ReaderSensor(keyword="entities")
    entity[contains] = EdgeSensor(
        entity["index"],
        scene["index"],
        relation=contains,
        forward=lambda entities, _: torch.ones_like(entities).unsqueeze(-1),
    )
    entity[selected] = SelectionLearner("index")
    entity[material] = MaterialLearner("index")
    program = LearningBasedProgram(graph, PoiModel, poi=[scene, entity])
    datanode = next(program.populate(dataset=[{"scene": [0], "entities": [0, 1, 2]}]))

    result = datanode.calculateLcLoss(circuit=True)[material_query.lcName]
    entity_nodes = datanode.findDatanodes(
        select=datanode.findRootConceptOrRelation(entity.name)
    )
    selected_probabilities = [
        node.getAttribute("<selected>/local/softmax").squeeze()[1]
        for node in entity_nodes
    ]
    material_probabilities = [
        node.getAttribute("<material>/local/softmax").squeeze()
        for node in entity_nodes
    ]
    expected = []
    for class_index in range(2):
        class_probability = selected_probabilities[0].new_zeros(())
        for entity_index, selected_probability in enumerate(selected_probabilities):
            unique_selection = selected_probability
            for other_index, other_probability in enumerate(selected_probabilities):
                if other_index != entity_index:
                    unique_selection = unique_selection * (1.0 - other_probability)
            class_probability = (
                class_probability
                + unique_selection * material_probabilities[entity_index][class_index]
            )
        expected.append(class_probability)

    assert result["queryProbabilities"] is not None
    assert torch.allclose(result["queryProbabilities"], torch.stack(expected), atol=1e-6)
