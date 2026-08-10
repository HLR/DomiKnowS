import pytest
import torch

from domiknows.graph import Concept, Graph, andL, existsL, miotaL, queryL
from domiknows.graph.concept import EnumConcept
from domiknows.solver.answerModule import AnswerSolver
from domiknows.solver.booleanMethodsCalculator import booleanMethodsCalculator
from domiknows.solver.compiled.formula import CompiledLossCalculator
from domiknows.solver.gurobiILPBooleanMethods import gurobiILPBooleanProcessor
from domiknows.solver.lcLossBooleanMethods import lcLossBooleanMethods
from domiknows.solver.lcLossSampleBooleanMethods import lcLossSampleBooleanMethods
from domiknows.solver.lossCalculator import LossCalculator

from .example import GOLD_ANSWERS
from .example_multiAnswers import (
    MULTI_HOT_LABEL,
    build_multi_answer_example,
    predict_answer_set,
    predict_answer_vector,
)
from .example_multiAnswerQuery import (
    MULTI_QUERY_LABEL,
    build_multi_query_example,
    predict_class_vector,
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


def test_multi_query_joint_nll_gradients_reach_selector_and_classes():
    processor = lcLossBooleanMethods()
    processor.current_device = "cpu"
    processor.setTNorm("P")
    memberships = torch.tensor([0.8, 0.2], requires_grad=True)
    class_scores = torch.tensor(
        [[0.25, 0.75], [0.6, 0.4]], requires_grad=True
    )
    distribution = processor.queryVar(
        None,
        None,
        [(None, "a", 0), (None, "b", 1)],
        [memberships],
        subclass_data=[list(class_scores[0]), list(class_scores[1])],
        multi_answer=True,
    )
    loss = -torch.log(distribution[0, 1]) - torch.log(
        1.0 - distribution[1].sum()
    )
    loss.backward()

    assert memberships.grad is not None
    assert torch.all(memberships.grad != 0)
    assert class_scores.grad is not None
    assert class_scores.grad[0].abs().sum() > 0


def test_multi_query_allows_an_empty_candidate_axis():
    processor = lcLossBooleanMethods()
    processor.current_device = "cpu"
    result = processor.queryVar(
        None,
        None,
        [(None, "a", 0), (None, "b", 1)],
        [],
        subclass_data=[],
        multi_answer=True,
    )
    assert result.shape == (0, 2)


def test_multi_query_decoding_uses_inclusive_membership_threshold():
    distribution = torch.tensor([[0.2, 0.3], [0.3, 0.19]])
    assert LossCalculator._decode_multi_query(
        distribution, 0.5
    ).tolist() == [1, -1]


def test_query_accepts_direct_multi_answer_selector_and_preserves_rows():
    example = build_multi_query_example()
    datanode = next(example.program.populate(example.dataset, device="cpu"))
    result = datanode.calculateSingleLcLoss(example.constraint.lcName, tnorm="P")

    assert result["queryDistribution"].shape == (3, 2)
    assert result["queryAnswer"].tolist() == MULTI_QUERY_LABEL.tolist()
    assert predict_class_vector(example) == MULTI_QUERY_LABEL.tolist()
    assert torch.allclose(
        result["queryDistribution"].sum(dim=-1),
        torch.tensor([0.9647, 0.9647, 0.0177]),
        atol=1e-3,
    )


def test_multi_query_custom_threshold_can_change_to_empty_answer():
    example = build_multi_query_example(threshold=0.99)
    assert predict_class_vector(example) == [-1, -1, -1]


@pytest.mark.parametrize(
    "label,error",
    [
        ([0, 1], "has 2 values"),
        ([0, 0.5, -1], "integer class IDs"),
        ([0, 2, -1], "must be -1 or class IDs"),
        ([0, -2, -1], "must be -1 or class IDs"),
    ],
)
def test_multi_query_rejects_invalid_aligned_labels(label, error):
    example = build_multi_query_example(label=label)
    with pytest.raises(ValueError, match=error):
        example.program.evaluate_condition(
            example.dataset, device="cpu", return_dict=True
        )


def test_multi_query_rejects_indirect_mixed_and_logical_nesting():
    Graph.clear()
    Concept.clear()
    with Graph("miota_query_composition_errors"):
        obj = Concept(name="obj")
        color = obj(
            name="color", ConceptClass=EnumConcept, values=["red", "blue"]
        )
        selector = miotaL(obj("x"))
        with pytest.raises(ValueError, match="exactly one direct miotaL"):
            queryL(color, selector, obj("y"))
        with pytest.raises(ValueError, match="one direct miotaL selector"):
            queryL(color, andL(selector))
        value = queryL(color, selector)
        with pytest.raises(ValueError, match="value-returning expression"):
            existsL(value)


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


def test_multi_query_verification_sampling_and_ilp_preserve_candidate_rows():
    subclasses = [(None, "dog_kind", 0), (None, "cat_kind", 1)]
    discrete_rows = [[1, 0], [0, 1], [0, 1]]

    verifier = booleanMethodsCalculator()
    assert verifier.queryVar(
        None, None, subclasses, [1, 0, 1],
        subclass_data=discrete_rows, multi_answer=True,
    ) == [[1, 0], [0, 0], [0, 1]]

    sampler = lcLossSampleBooleanMethods()
    sampler.current_device = "cpu"
    sampler.sampleSize = 2
    sampled = sampler.queryVar(
        None,
        None,
        subclasses,
        [torch.tensor([[1.0, 0.0], [0.0, 1.0]])],
        subclass_data=[
            [torch.tensor([1.0, 1.0]), torch.tensor([0.0, 0.0])],
            [torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0])],
        ],
        multi_answer=True,
    )
    assert sampled.shape == (2, 2, 2)
    assert sampled.tolist() == [
        [[1.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 1.0]],
    ]

    ilp = gurobiILPBooleanProcessor()
    assert ilp.queryVar(
        None, None, subclasses, [1, 0, 1],
        subclass_data=discrete_rows, multi_answer=True,
    ) == [[1, 0], [0, 0], [0, 1]]


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


def test_multi_query_compiled_and_circuit_paths_match_interpreter():
    example = build_multi_query_example()
    datanode = next(example.program.populate(example.dataset, device="cpu"))
    context = datanode._prepareLcLossContext("P")
    reference = context["lossCalculator"].calculate_single_lc_loss(
        example.constraint, datanode, "/local/softmax", tnorm="P"
    )
    compiled = CompiledLossCalculator(context["solver"]).calculate_single_lc_loss(
        example.constraint, datanode, "/local/softmax", tnorm="P"
    )
    assert torch.allclose(
        reference["queryDistribution"], compiled["queryDistribution"]
    )
    assert torch.allclose(reference["lossTensor"], compiled["lossTensor"])
    assert torch.allclose(reference["loss"], compiled["loss"])
    assert compiled["queryAnswer"].tolist() == [0, 1, -1]

    circuit = datanode.calculateLcLoss(
        circuit=True, includeExecutable=True
    )[example.constraint.lcName]
    assert torch.allclose(
        reference["queryDistribution"], circuit["queryDistribution"], atol=1e-6
    )
    assert circuit["queryAnswer"].tolist() == [0, 1, -1]
    assert torch.allclose(reference["loss"], circuit["loss"], atol=1e-6)
    assert torch.isfinite(circuit["loss"])


def test_multi_query_evaluation_reports_exact_selection_and_class_metrics():
    example = build_multi_query_example()
    metrics = example.program.evaluate_condition(
        example.dataset, device="cpu", return_dict=True
    )
    assert metrics["accuracy"] == 100.0
    assert metrics["query_accuracy"] == 100.0
    assert metrics["multi_query_position_accuracy"] == 100.0
    assert metrics["multi_query_selection_exact_accuracy"] == 100.0
    assert metrics["multi_query_selection_position_accuracy"] == 100.0
    assert metrics["multi_query_class_accuracy"] == 100.0


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


@pytest.mark.parametrize("hard", [False, True])
def test_multi_query_joint_nll_runs_in_training(hard):
    example = build_multi_query_example(hard=hard)
    steps = list(
        example.program.train_epoch(
            example.dataset, print_loss=False, training_mode="standard"
        )
    )
    assert len(steps) == 1
    assert torch.isfinite(steps[0][0])


def test_multi_query_all_unselected_label_has_finite_joint_loss():
    example = build_multi_query_example(label=[-1, -1, -1])
    steps = list(
        example.program.train_epoch(
            example.dataset, print_loss=False, training_mode="standard"
        )
    )
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


def test_multi_query_answer_solver_persists_aligned_class_ids():
    example = build_multi_query_example()
    datanode = next(example.program.populate(example.dataset, device="cpu"))
    datanode.inferLocal()
    solver = AnswerSolver(example.program.graph)
    original_calculate = solver.solver._calculateILPSelection
    solve_calls = 0

    def counted_calculate(*args, **kwargs):
        nonlocal solve_calls
        solve_calls += 1
        return original_calculate(*args, **kwargs)

    solver.solver._calculateILPSelection = counted_calculate

    result = solver.solve_active_constraints(
        datanode, [example.constraint.lcName]
    )

    expected = MULTI_QUERY_LABEL.tolist()
    assert result["hypotheses"][example.constraint.lcName] == expected
    assert datanode.getExecutableConstraintLabels()[
        f"{example.constraint.lcName}/answer"
    ] == expected
    assert solve_calls == 1
