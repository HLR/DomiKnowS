import pytest
import torch

from domiknows.graph import Concept, Graph, existsL, miotaL, queryL
from domiknows.solver.answerModule import AnswerSolver
from domiknows.solver.booleanMethodsCalculator import booleanMethodsCalculator
from domiknows.solver.compiled.formula import CompiledLossCalculator
from domiknows.solver.gurobiILPBooleanMethods import gurobiILPBooleanProcessor
from domiknows.solver.lcLossBooleanMethods import lcLossBooleanMethods
from domiknows.solver.lcLossSampleBooleanMethods import lcLossSampleBooleanMethods

from .example import GOLD_ANSWERS
from .example_multiAnswers import (
    MULTI_HOT_LABEL,
    build_multi_answer_example,
    predict_answer_set,
    predict_answer_vector,
)


def test_miota_example_returns_one_multi_answer_vector():
    example = build_multi_answer_example()

    assert predict_answer_vector(example) == MULTI_HOT_LABEL.to(torch.int64).tolist()
    assert predict_answer_set(example) == GOLD_ANSWERS
    assert len(example.dataset) == 1
    assert "miotaL" in example.logic_string

    metrics = example.program.evaluate_condition(
        example.dataset, device="cpu", return_dict=True
    )
    assert metrics["accuracy"] == 100.0
    assert metrics["miota_exact_accuracy"] == 100.0
    assert metrics["miota_position_accuracy"] == 100.0


def test_miota_custom_threshold_can_return_empty_set():
    example = build_multi_answer_example(threshold=1.0)
    assert predict_answer_set(example) == set()


@pytest.mark.parametrize(
    "label,error",
    [
        ([1, 0], "has 2 values"),
        ([1, 0.5, 0], "binary multi-hot vector"),
    ],
)
def test_miota_rejects_invalid_vector_labels(label, error):
    example = build_multi_answer_example(label=label)
    with pytest.raises(ValueError, match=error):
        example.program.evaluate_condition(
            example.dataset, device="cpu", return_dict=True
        )


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_miota_rejects_out_of_range_threshold(threshold):
    with pytest.raises(ValueError, match=r"threshold must be in \[0, 1\]"):
        miotaL(object(), threshold=threshold)


def test_miota_threshold_is_inclusive_and_allows_multiple_answers():
    processor = booleanMethodsCalculator()
    assert processor.miotaVar(None, 0.5, 0.49, 0.9, threshold=0.5) == [1, 0, 1]


def test_miota_hard_mode_uses_straight_through_gradients():
    processor = lcLossBooleanMethods()
    processor.current_device = "cpu"
    source = torch.tensor([0.2, 0.8], requires_grad=True)
    selected = processor.miotaVar(None, source, threshold=0.5, hard=True)
    assert selected.tolist() == [0.0, 1.0]
    selected.sum().backward()
    assert source.grad is not None
    assert torch.all(source.grad != 0)


def test_query_rejects_multi_answer_selector():
    Graph.clear()
    Concept.clear()
    with Graph("miota_query_rejection"):
        obj = Concept(name="obj")
        color = Concept(name="color")
        red = Concept(name="red")
        red.is_a(color)
        selector = miotaL(obj("x"))
        with pytest.raises(ValueError, match="queryL does not support miotaL"):
            queryL(color, selector)


def test_miota_nests_in_boolean_and_counting_constraints():
    Graph.clear()
    Concept.clear()
    with Graph("miota_nesting"):
        obj = Concept(name="obj")
        selector = miotaL(obj("x"))
        parent = existsL(selector)

    processor = lcLossBooleanMethods()
    processor.current_device = "cpu"
    processor.setTNorm("P")
    selected = selector(
        None, processor, {"x": [[torch.tensor([0.2, 0.8])]]}, headConstrain=False
    )
    parent_loss = parent(
        None, processor, {"selected": selected}, headConstrain=True
    )
    assert selected[0][0].tolist() == pytest.approx([0.2, 0.8])
    assert parent_loss[0][0].item() == pytest.approx(0.16)


def test_miota_sample_and_ilp_backends_keep_all_memberships():
    sampler = lcLossSampleBooleanMethods()
    sampler.current_device = "cpu"
    samples = sampler.miotaVar(
        None,
        torch.tensor([0.0, 1.0]),
        torch.tensor([1.0, 1.0]),
    )
    assert samples.tolist() == [[0.0, 1.0], [1.0, 1.0]]

    ilp = gurobiILPBooleanProcessor()
    assert ilp.miotaVar(None, 1, None, 0, 1) == [1, 0, 0, 1]


def test_miota_compiled_and_circuit_paths_match_interpreter():
    example = build_multi_answer_example()
    datanode = next(example.program.populate(example.dataset, device="cpu"))
    context = datanode._prepareLcLossContext("P")
    reference = context["lossCalculator"].calculate_single_lc_loss(
        example.constraint, datanode, "/local/softmax", tnorm="P"
    )
    compiled = CompiledLossCalculator(context["solver"]).calculate_single_lc_loss(
        example.constraint, datanode, "/local/softmax", tnorm="P"
    )
    assert torch.allclose(
        reference["selectionDistribution"], compiled["selectionDistribution"]
    )

    circuit = datanode.calculateLcLoss(
        circuit=True, includeExecutable=True
    )[example.constraint.lcName]
    assert torch.allclose(
        reference["selectionDistribution"],
        circuit["selectionDistribution"],
        atol=1e-6,
    )
    assert torch.isfinite(circuit["loss"])


@pytest.mark.parametrize("hard", [False, True])
def test_miota_vector_bce_runs_in_training(hard):
    example = build_multi_answer_example(hard=hard)
    steps = list(
        example.program.train_epoch(
            example.dataset, print_loss=False, training_mode="standard"
        )
    )
    assert len(steps) == 1
    assert torch.isfinite(steps[0][0])


def test_miota_answer_solver_persists_vector_without_powerset_search():
    example = build_multi_answer_example()
    datanode = next(example.program.populate(example.dataset, device="cpu"))
    datanode.inferLocal()
    solver = AnswerSolver(example.program.graph)

    result = solver.solve_active_constraints(
        datanode, [example.constraint.lcName]
    )

    expected = MULTI_HOT_LABEL.to(torch.int64).tolist()
    assert result["hypotheses"][example.constraint.lcName] == expected
    assert datanode.getExecutableConstraintLabels()[
        f"{example.constraint.lcName}/answer"
    ] == expected
