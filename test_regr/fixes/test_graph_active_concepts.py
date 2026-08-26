import pytest
import torch

from domiknows.graph import Concept, Graph, Relation, andL
from domiknows.graph.dataNode import DataNode, DataNodeBuilder
from domiknows.program.model.pytorch import SolverModel
from domiknows.sensor.pytorch import ModuleLearner
from domiknows.sensor.pytorch.sensors import FunctionalReaderSensor
from domiknows.solver.ilpOntSolverFactory import ilpOntSolverFactory


def _clear_state():
    Graph.clear()
    Concept.clear()
    Relation.clear()
    DataNode.clear()
    DataNodeBuilder.clear()
    ilpOntSolverFactory.clear()


@pytest.fixture(autouse=True)
def clear_domiknows_state():
    _clear_state()
    yield
    _clear_state()


def _build_concept_graph(name="active_concepts"):
    with Graph(name) as graph:
        obj = Concept(name="obj")
        red = obj(name="red")
        dog = obj(name="dog")
        cat = obj(name="cat")
        tree = obj(name="tree")
    return graph, obj, red, dog, cat, tree


def test_active_concepts_default_switch_dependency_closure_and_reset():
    graph, obj, red, dog, cat, tree = _build_concept_graph()

    assert graph.get_active_concepts() == tuple(graph.concepts.values())
    assert all(graph.is_concept_active(concept) for concept in graph.concepts.values())

    active = graph.set_active_concepts(["red", dog])
    assert tuple(concept.name for concept in active) == ("constraint", "obj", "red", "dog")
    assert graph.is_concept_active(graph.constraint)
    assert graph.is_concept_active(obj)
    assert graph.is_concept_active(red)
    assert graph.is_concept_active(dog)
    assert not graph.is_concept_active(cat)
    assert not graph.is_concept_active(tree)

    active = graph.set_active_concepts([cat, "tree"])
    assert tuple(concept.name for concept in active) == ("constraint", "obj", "cat", "tree")
    assert not graph.is_concept_active(red)
    assert not graph.is_concept_active(dog)

    graph.set_active_concepts(None)
    assert graph.get_active_concepts() == tuple(graph.concepts.values())


def test_active_concepts_reject_invalid_and_foreign_values():
    graph, *_ = _build_concept_graph()
    with Graph("foreign_graph"):
        foreign = Concept(name="foreign")

    with pytest.raises(ValueError, match="Unknown concept"):
        graph.set_active_concepts(["missing"])
    with pytest.raises(ValueError, match="does not belong"):
        graph.set_active_concepts([foreign])
    with pytest.raises(TypeError, match="iterable"):
        graph.set_active_concepts("red")
    with pytest.raises(TypeError, match="Concept instances"):
        graph.set_active_concepts([object()])


def test_root_activation_applies_to_subgraphs_and_keeps_their_constraints():
    with Graph("activation_root") as graph:
        obj = Concept(name="obj")
        unrelated = obj(name="unrelated")
        with Graph("activation_subgraph") as subgraph:
            child = obj(name="child")

    graph.set_active_concepts([child])

    assert graph.is_concept_active(obj)
    assert subgraph.is_concept_active(child)
    assert graph.is_concept_active(graph.constraint)
    assert subgraph.is_concept_active(subgraph.constraint)
    assert not graph.is_concept_active(unrelated)


def test_root_activation_preserves_same_named_sibling_concepts_by_identity():
    with Graph("duplicate_activation_root") as graph:
        with Graph("left_generation") as left:
            left_text = Concept(name="text")
        with Graph("right_generation") as right:
            right_text = Concept(name="text")

    graph.set_active_concepts([left_text])

    assert left.is_concept_active(left_text)
    assert not right.is_concept_active(right_text)
    assert graph.is_concept_active(left_text.fullname)
    assert not graph.is_concept_active(right_text.fullname)
    with pytest.raises(ValueError, match="Unknown concept"):
        graph.is_concept_active("text")


def test_activation_concept_index_is_cached_and_invalidated_on_concept_addition():
    graph, obj, *_ = _build_concept_graph()

    first = graph._activation_concepts()
    assert graph._activation_concepts() is first

    with graph:
        late = obj(name="late")

    second = graph._activation_concepts()
    assert second is not first
    assert second["late"] is late
    assert graph._activation_concepts() is second


def test_subgraph_mutation_invalidates_parent_activation_index():
    with Graph("activation_cache_root") as graph:
        with Graph("child") as child_graph:
            original = Concept(name="original")

    first = graph._activation_concepts()
    with child_graph:
        late = Concept(name="late_child")

    second = graph._activation_concepts()
    assert second is not first
    assert original in second.values()
    assert late in second.values()


def test_constraint_effective_activation_preserves_explicit_flag():
    graph, _, red, dog, _, _ = _build_concept_graph()
    with graph:
        enabled = andL(red("x"), dog(path="x"))
        explicitly_disabled = andL(red("x"), dog(path="x"), active=False)

    assert enabled.declared_active
    assert enabled.active
    assert not explicitly_disabled.declared_active
    assert not explicitly_disabled.active

    graph.set_active_concepts([red])
    assert enabled.declared_active
    assert not enabled.active

    graph.set_active_concepts([red, dog])
    assert enabled.active
    assert not explicitly_disabled.active


class _CountingClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, features):
        self.calls += 1
        score = features.float().view(-1)
        return torch.stack((-score, score), dim=-1)


def test_inactive_concept_sensors_are_skipped_and_can_be_reenabled():
    with Graph("active_sensor_execution") as graph:
        obj = Concept(name="obj")
        red = obj(name="red")
        dog = obj(name="dog")

    obj["features"] = FunctionalReaderSensor(
        keyword="features",
        forward=lambda data: torch.as_tensor(data, dtype=torch.float32),
    )
    red_model = _CountingClassifier()
    dog_model = _CountingClassifier()
    obj[red] = ModuleLearner("features", module=red_model)
    obj[dog] = ModuleLearner("features", module=dog_model)

    model = SolverModel(graph, poi=[obj, graph.constraint], inferTypes=[], device="cpu")

    graph.set_active_concepts([red])
    model({"features": [1.0, 0.0]})
    assert red_model.calls == 1
    assert dog_model.calls == 0

    graph.set_active_concepts([dog])
    model({"features": [1.0, 0.0]})
    assert red_model.calls == 1
    assert dog_model.calls == 1
