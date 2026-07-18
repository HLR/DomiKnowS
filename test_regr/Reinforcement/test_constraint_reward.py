"""Tests for reward-from-constraints: the reward is derived from the graph's
declared logical constraints instead of (or in addition to) a reward function.
"""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from domiknows.graph import Graph, Concept, Relation, EnumConcept
from domiknows.graph.logicalConstrain import exactL
from domiknows.sensor.pytorch.sensors import ReaderSensor
from domiknows.sensor.pytorch.relation_sensors import EdgeSensor
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.program import ReinforcementProgram
from domiknows.reinforcement import constraint_satisfaction_reward


class _Net(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.l = torch.nn.Linear(n, 2)

    def forward(self, rel, x):
        return self.l(x)


N, M, EXPECTED_ZEROS = 6, 8, 3


def _build(estimator="importance_weighted", num_samples=32):
    Graph.clear(); Concept.clear(); Relation.clear()
    with Graph("constraint_rl") as graph:
        a = Concept(name="a")
        b = Concept(name="b")
        (a_contain_b,) = a.contains(b)
        b_answer = b(name="answer_b", ConceptClass=EnumConcept, values=["zero", "one"])
        # Declared constraint: exactly EXPECTED_ZEROS of the b's must be "zero".
        exactL(b_answer.__getattr__("zero"), EXPECTED_ZEROS)

    a["index"] = ReaderSensor(keyword="a")
    b["index"] = ReaderSensor(keyword="b")
    b[a_contain_b] = EdgeSensor(
        b["index"], a["index"], relation=a_contain_b,
        forward=lambda bb, _: torch.ones(len(bb)).unsqueeze(-1))
    b[b_answer] = ModuleLearner(a_contain_b, "index", module=_Net(N))

    program = ReinforcementProgram(
        graph,
        targets=[b_answer],
        reward_from_constraints=True,   # <-- reward comes from the graph constraint
        num_samples=num_samples,
        estimator=estimator,
        poi=[a, b, b_answer],
        device="cpu",
    )
    dataset = [{
        "a": [0],
        "b": [((np.random.rand(N) - np.random.rand(N))).tolist() for _ in range(M)],
    }]
    return program, dataset, b_answer


def test_constraint_reward_distinguishes_satisfying_samples():
    np.random.seed(0); torch.manual_seed(0)
    program, dataset, b_answer = _build()
    program.to("cpu")
    _, _, datanode, _ = program.model(dataset[0])

    good = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1])  # exactly 3 zeros -> satisfied
    bad = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1])   # zero zeros -> violated
    r_good = constraint_satisfaction_reward(datanode, {b_answer: good}, [b_answer])
    r_bad = constraint_satisfaction_reward(datanode, {b_answer: bad}, [b_answer])
    assert r_good == 1.0
    assert r_bad == 0.0


def test_constraint_reward_works_without_reward_function():
    np.random.seed(0); torch.manual_seed(0)
    program, dataset, _ = _build(estimator="importance_weighted", num_samples=48)
    baseline = program.evaluate_reward(dataset, num_samples=300)
    program.train(
        dataset, train_epoch_num=200,
        Optim=lambda p: torch.optim.Adam(p, lr=5e-3), device="cpu")
    trained = program.evaluate_reward(dataset, num_samples=300)
    assert trained > baseline, f"constraint reward did not improve: {baseline} -> {trained}"


def test_exact_cardinality_circuit_matches_poisson_binomial_and_backprops():
    """The exactL integration path agrees with its hand-written DP oracle."""
    np.random.seed(0); torch.manual_seed(0)
    program, dataset, b_answer = _build()
    _mloss, _metric, datanode, _builder = program.model(dataset[0])

    constraint = next(iter(program.graph.logicalConstrains.values()))
    result = datanode.calculateLcLoss(circuit=True)[constraint.lcName]

    root_concept = datanode.findRootConceptOrRelation(b_answer.name)
    b_nodes = datanode.findDatanodes(select=root_concept)
    zero_probabilities = [
        node.getAttribute(f"<{b_answer.name}>/local/softmax").squeeze()[0]
        for node in b_nodes
    ]
    distribution = zero_probabilities[0].new_zeros(len(zero_probabilities) + 1)
    distribution[0] = 1.0
    for probability in zero_probabilities:
        previous = distribution.clone()
        distribution[0] = previous[0] * (1.0 - probability)
        for count in range(1, len(zero_probabilities) + 1):
            distribution[count] = (
                previous[count] * (1.0 - probability)
                + previous[count - 1] * probability
            )

    assert result["probability"].detach().item() == pytest.approx(
        distribution[EXPECTED_ZEROS].detach().item()
    )
    cached = datanode.calculateLcLoss(circuit=True)[constraint.lcName]
    assert cached["cacheHit"] is True
    assert cached["probability"].detach().item() == pytest.approx(
        result["probability"].detach().item()
    )
    enumeration = datanode.calculateLcLoss(
        sample=True,
        sampleSize=-1,
        sampleGlobalLoss=False,
    )[constraint.lcName]
    assignment_weights = enumeration["lossTensor"][0]
    satisfying = enumeration["lcSuccesses"][0].bool()
    assert result["probability"].detach().item() == pytest.approx(
        assignment_weights[satisfying].sum().detach().item(),
        abs=1e-6,
    )
    with pytest.warns(RuntimeWarning, match="Falling back to Product t-norm"):
        fallback = datanode.calculateLcLoss(
            circuit=True,
            circuitBackend="bdd",
            circuitMaxNodes=2,
        )[constraint.lcName]
    assert fallback["backend"] == "tnorm"
    assert fallback["fallback"] == "circuit-size-limit"
    assert fallback["exact"] is False
    assert torch.isfinite(fallback["loss"]).all()
    result["loss"].backward()
    gradients = [
        parameter.grad
        for parameter in program.model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_semantic_loss_program_improves_exact_constraint_probability():
    from domiknows.program import SemanticLossProgram
    from domiknows.program.model.pytorch import PoiModel

    np.random.seed(0); torch.manual_seed(0)
    base_program, dataset, b_answer = _build()
    graph = base_program.graph
    program = SemanticLossProgram(
        graph,
        PoiModel,
        poi=[graph.concepts["a"], graph.concepts["b"], b_answer],
        circuit_backend="bdd",
        device="cpu",
    )
    program.to("cpu")
    constraint = next(iter(graph.logicalConstrains.values()))

    def satisfaction_probability():
        _mloss, _metric, datanode, _builder = program.model(dataset[0])
        return datanode.calculateLcLoss(circuit=True)[constraint.lcName][
            "probability"
        ].detach().item()

    before = satisfaction_probability()
    program.train(
        dataset,
        train_epoch_num=12,
        Optim=lambda parameters: torch.optim.Adam(parameters, lr=2e-2),
        c_lr=1e-2,
        c_warmup_iters=0,
        device="cpu",
    )
    after = satisfaction_probability()
    assert after > before + 0.02, f"exact satisfaction did not improve: {before} -> {after}"
