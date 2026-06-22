"""Tests for reward-from-constraints: the reward is derived from the graph's
declared logical constraints instead of (or in addition to) a reward function.
"""
from pathlib import Path
import sys

import numpy as np
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
