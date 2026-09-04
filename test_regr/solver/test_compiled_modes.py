"""Regression coverage for shared compiled plans outside fuzzy t-norm loss."""

import pytest
import torch

from domiknows.graph import (
    Concept,
    DataNode,
    Graph,
    Relation,
    execute,
    existsL,
)
from domiknows.solver import ilpOntSolverFactory
from domiknows.solver.logicalConstraintVerifier import LogicalConstraintVerifier
from domiknows.solver.sampleLossCalculator import SampleLossCalculator


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


def _scene(name, *, executable=False):
    with Graph(name) as graph:
        scene = Concept(name=f"{name}_scene")
        item = Concept(name=f"{name}_item")
        scene.contains(item)
        flag = item(name=f"{name}_flag")
        formula = existsL(flag("x"), active=True)
        if executable:
            execute(formula)

    root = DataNode(instanceID=0, ontologyNode=scene)
    root.current_device = "cpu"
    for index, probability in enumerate((0.8, 0.3)):
        child = DataNode(instanceID=index, ontologyNode=item)
        child.current_device = "cpu"
        child.attributes[f"<{flag.name}>"] = torch.log(torch.tensor(
            [1.0 - probability, probability], dtype=torch.float32))
        child.attributes[f"<{flag.name}>/local/softmax"] = torch.tensor(
            [1.0 - probability, probability], dtype=torch.float32)
        root.addChildDataNode(child)

    solver, concepts = root.getILPSolver(
        conceptsRelations=root.collectConceptsAndRelations((flag,)))
    solver.current_device = root.current_device
    return graph, root, flag, formula, solver, concepts


def _forbid_interpreter(monkeypatch, solver):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("logical-constraint interpreter was called")

    monkeypatch.setattr(
        solver.constraintConstructor,
        "constructLogicalConstrains",
        unexpected,
    )


def test_compiled_sampling_does_not_bypass_to_interpreter(monkeypatch):
    _, root, _, formula, solver, concepts = _scene("compiled_sample")
    _forbid_interpreter(monkeypatch, solver)

    result = SampleLossCalculator(solver).calculateSampleLoss(
        root,
        sampleSize=16,
        sampleGlobalLoss=False,
        conceptsRelations=concepts,
        compiled=True,
    )

    assert formula.lcName in result
    assert result[formula.lcName]["lossTensor"]
    assert solver._compiled_formula_plan_cache.info()["misses"] > 0


def test_compiled_sampling_matches_interpreter_samples():
    _, root, _, formula, solver, concepts = _scene("compiled_sample_parity")
    calculator = SampleLossCalculator(solver)
    torch.manual_seed(17)
    reference = calculator.calculateSampleLoss(
        root, 32, False, concepts, compiled=False)
    compiled = calculator.calculateSampleLoss(
        root, 32, False, concepts, compiled=True)

    assert torch.equal(
        reference["globalSuccesses"], compiled["globalSuccesses"])
    reference_loss = reference[formula.lcName]["lossTensor"][0]
    compiled_loss = compiled[formula.lcName]["lossTensor"][0]
    assert torch.allclose(reference_loss, compiled_loss, equal_nan=True)


def test_compiled_semantic_sampling_uses_complete_assignment_table():
    _, root, _, formula, solver, concepts = _scene("compiled_semantic_sample")
    result = SampleLossCalculator(solver).calculateSampleLoss(
        root, -1, False, concepts, compiled=True)

    # Two independent binary item predictions produce all 2**2 assignments.
    assert result["globalSuccesses"].shape == (4,)
    assert result[formula.lcName]["lcSuccesses"][0].shape == (4,)


def test_compiled_circuit_does_not_bypass_to_interpreter(monkeypatch):
    _, root, _, formula, solver, _ = _scene("compiled_circuit")
    _forbid_interpreter(monkeypatch, solver)

    result = solver.calculateCircuitLoss(
        root, backend="bdd", compiled=True)

    assert result[formula.lcName]["backend"] == "bdd"
    assert result[formula.lcName]["probability"].item() == pytest.approx(0.86)
    features = result[formula.lcName]["groundingFeatures"]
    assert features.shape[0] == result[formula.lcName]["lossTensor"].numel()


def test_compiled_verification_does_not_bypass_to_interpreter(monkeypatch):
    _, root, _, formula, solver, _ = _scene("compiled_verify")
    root.inferLocal(keys=("argmax",))
    _forbid_interpreter(monkeypatch, solver)
    verifier = LogicalConstraintVerifier(solver)

    result = verifier.verifySingleConstraint(
        formula,
        solver.booleanMethodsCalculator,
        root,
        key="/local/argmax",
        compiled=True,
    )

    assert result["satisfied"] == 100.0


@pytest.mark.parametrize("mode", ["tnorm", "circuit"])
def test_compiled_executable_inference_does_not_bypass(
        monkeypatch, mode):
    _, root, flag, _, solver, _ = _scene(
        f"compiled_exec_{mode}", executable=True)
    constraint = DataNode(
        instanceID=0,
        ontologyNode=root.graph.get_constraint_concept(),
    )
    constraint.attributes["ELC0/label"] = torch.tensor(0)
    root.addChildDataNode(constraint)
    _forbid_interpreter(monkeypatch, solver)

    result = root.inferExecutableResults(
        flag, mode=mode, circuitBackend="bdd", compiled=True)

    assert result["ELC0"]["answer"] is True


@pytest.mark.gurobi
def test_compiled_ilp_does_not_bypass_to_interpreter(monkeypatch):
    _, root, flag, _, solver, _ = _scene("compiled_ilp")
    _forbid_interpreter(monkeypatch, solver)

    root.inferILPResults(flag, compiled=True)

    assert solver._compiled_formula_plan_cache.info()["misses"] > 0


def test_formula_plan_is_shared_between_tnorm_and_circuit_execution():
    _, root, flag, _, solver, _ = _scene(
        "compiled_shared_plan", executable=True)
    constraint = DataNode(
        instanceID=0,
        ontologyNode=root.graph.get_constraint_concept(),
    )
    constraint.attributes["ELC0/label"] = torch.tensor(0)
    root.addChildDataNode(constraint)

    root.inferExecutableResults(flag, mode="tnorm", compiled=True)
    after_tnorm = solver._compiled_formula_plan_cache.info().copy()
    root.inferExecutableResults(
        flag, mode="circuit", circuitBackend="bdd", compiled=True)
    after_circuit = solver._compiled_formula_plan_cache.info()

    assert after_tnorm["misses"] > 0
    assert after_circuit["hits"] > after_tnorm["hits"]
    assert after_circuit["size"] == after_tnorm["size"]
