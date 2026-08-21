import pytest
import torch

from domiknows.solver.compiled.formula import CompiledLossCalculator
from domiknows.solver.answerModule import AnswerSolver

from .example_relationAnswers import (
    EXPECTED_MULTI_HOT,
    LEFT_OBJECT_IDS,
    UNIQUE_RED_LEFT_ID,
    build_relation_answer_example,
    ilp_relation_answers,
    relation_answers,
)


def _datanode(example):
    return next(example.program.populate(example.dataset, device="cpu"))


NESTED_RELATION_LOGIC = (
    'miotaL(andL(object("x"), '
    'left("r1", path=("x", pair_src.reversed)), '
    'ball("y", path=("r1", pair_dst)), '
    'left("r2", path=("y", pair_src.reversed)), '
    'ball("z", path=("r2", pair_dst))), '
    'threshold=0.5, hard=False)'
)


def test_relational_iota_and_miota_are_object_aligned():
    example = build_relation_answer_example()
    unique_id, answers, vector, unique_dist, multi_dist = relation_answers(example)

    assert unique_dist.shape == (4,)
    assert multi_dist.shape == (4,)
    assert unique_id == UNIQUE_RED_LEFT_ID
    assert answers == LEFT_OBJECT_IDS
    assert vector.cpu().tolist() == EXPECTED_MULTI_HOT.tolist()


@pytest.mark.parametrize("hard", [False, True])
def test_relation_miota_keeps_training_gradients(hard):
    example = build_relation_answer_example(hard=hard)
    datanode = _datanode(example)
    distribution = datanode.calculateSingleLcLoss(
        example.multiple.lcName, tnorm="P"
    )["selectionDistribution"]

    assert distribution.requires_grad
    distribution.sum().backward()


def test_duplicate_qualifying_relation_uses_one_output_position():
    example = build_relation_answer_example(
        second_ball=True,
        left_pairs={(1, 3), (1, 4)},
    )
    datanode = _datanode(example)
    distribution = datanode.calculateSingleLcLoss(
        example.multiple.lcName, tnorm="P"
    )["selectionDistribution"]

    assert distribution.shape == (4,)
    assert (distribution >= example.multiple.threshold).to(torch.int64).tolist() == [
        1, 0, 0, 0
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"left_pairs": set()},
        {"threshold": 1.0},
    ],
)
def test_relation_miota_supports_empty_answers(kwargs):
    example = build_relation_answer_example(**kwargs)
    datanode = _datanode(example)
    distribution = datanode.calculateSingleLcLoss(
        example.multiple.lcName, tnorm="P"
    )["selectionDistribution"]
    assert not bool((distribution >= example.multiple.threshold).any())


def test_interpreter_compiled_and_exact_circuit_distributions_match():
    example = build_relation_answer_example()
    datanode = _datanode(example)
    context = datanode._prepareLcLossContext("P")
    reference = context["lossCalculator"].calculate_single_lc_loss(
        example.multiple, datanode, "/local/softmax", tnorm="P"
    )["selectionDistribution"]
    compiled = CompiledLossCalculator(context["solver"]).calculate_single_lc_loss(
        example.multiple, datanode, "/local/softmax", tnorm="P"
    )["selectionDistribution"]
    circuit = datanode.calculateLcLoss(tnorm="P", circuit=True)[
        example.multiple.lcName
    ]["selectionDistribution"]

    assert reference.shape == (4,)
    assert torch.allclose(reference, compiled, atol=1e-6)
    assert torch.allclose(reference, circuit, atol=1e-6)


@pytest.mark.parametrize(
    "builder_kwargs,expected",
    [
        ({}, [0, 0, 0, 0]),
        (
            {"second_ball": True, "left_pairs": {(1, 3), (2, 3), (3, 4)}},
            [1, 1, 0, 0],
        ),
    ],
)
def test_nested_relation_miota_stays_on_the_primary_object_axis(
    builder_kwargs, expected
):
    example = build_relation_answer_example(
        executable=True,
        logic=NESTED_RELATION_LOGIC,
        logic_label=expected,
        **builder_kwargs,
    )
    datanode = _datanode(example)
    distribution = datanode.calculateSingleLcLoss(
        example.multiple.lcName, tnorm="G"
    )["selectionDistribution"]

    assert distribution.shape == (4,)
    assert (distribution >= 0.5).to(torch.int64).tolist() == expected

    # The compiled and exact-circuit paths must preserve the same four-object
    # axis even though exact WMC and fuzzy Product t-norm need not be numeric twins.
    context = datanode._prepareLcLossContext("P")
    reference = context["lossCalculator"].calculate_single_lc_loss(
        example.multiple, datanode, "/local/softmax", tnorm="P"
    )["selectionDistribution"]
    compiled = CompiledLossCalculator(context["solver"]).calculate_single_lc_loss(
        example.multiple, datanode, "/local/softmax", tnorm="P"
    )["selectionDistribution"]
    circuit = datanode.calculateLcLoss(tnorm="P", circuit=True)[
        example.multiple.lcName
    ]["selectionDistribution"]
    assert reference.shape == compiled.shape == circuit.shape == (4,)
    assert torch.allclose(reference, compiled, atol=1e-6)

    sample_processor = context["solver"].myLcLossSampleBooleanMethods
    sample_processor.sampleSize = 8
    sample_processor.current_device = datanode.current_device
    sample_processor.current_dtype = datanode.current_dtype
    sampled, *_ = context["solver"].constraintConstructor.constructLogicalConstrains(
        example.multiple, sample_processor, None, datanode, 8,
        key="/local/softmax", headLC=False, loss=True, sample=True,
    )
    assert sampled[0][0].shape == (8, 4)

    datanode.inferLocal()
    datanode.inferILPResults(fun=None, minimizeObjective=False)
    ilp_answer = datanode.getExecutableConstraintLabels()[
        f"{example.multiple.lcName}/answer"
    ]
    assert len(ilp_answer) == 4
    if expected == [0, 0, 0, 0]:
        assert ilp_answer == expected


def test_relation_miota_evaluation_and_vector_training():
    evaluation_example = build_relation_answer_example(executable=True)
    metrics = evaluation_example.program.evaluate_condition(
        evaluation_example.dataset, device="cpu", return_dict=True
    )
    assert metrics["miota_exact_accuracy"] == 100.0
    assert metrics["miota_position_accuracy"] == 100.0

    training_example = build_relation_answer_example(executable=True)
    steps = list(training_example.program.train_epoch(
        training_example.dataset, print_loss=False, training_mode="standard"
    ))
    assert steps and torch.isfinite(steps[0][0])


def test_verification_and_sampling_accept_relation_aligned_selector():
    example = build_relation_answer_example()
    datanode = _datanode(example)

    verified = datanode.verifySingleConstraint(
        example.multiple.lcName, key="/local/argmax"
    )
    context = datanode._prepareLcLossContext("P")
    processor = context["solver"].myLcLossSampleBooleanMethods
    processor.sampleSize = 8
    processor.current_device = datanode.current_device
    processor.current_dtype = datanode.current_dtype
    sampled, *_ = context["solver"].constraintConstructor.constructLogicalConstrains(
        example.multiple, processor, None, datanode, 8,
        key="/local/softmax", headLC=False, loss=True, sample=True,
    )

    assert verified is not None
    assert sampled[0][0].shape == (8, 4)


@pytest.mark.gurobi
def test_relation_aligned_selectors_construct_in_ilp():
    example = build_relation_answer_example()
    datanode = _datanode(example)
    datanode.inferILPResults(fun=None, minimizeObjective=False)

    assert datanode is not None


@pytest.mark.gurobi
def test_example_reports_relation_aligned_ilp_answers():
    unique_id, answers, vector = ilp_relation_answers()

    assert unique_id == UNIQUE_RED_LEFT_ID
    assert answers == LEFT_OBJECT_IDS
    assert vector.tolist() == EXPECTED_MULTI_HOT.tolist()


@pytest.mark.gurobi
def test_relation_miota_answer_is_persisted_on_the_object_axis():
    example = build_relation_answer_example(executable=True)
    datanode = _datanode(example)
    datanode.inferLocal()

    result = AnswerSolver(example.program.graph).solve_active_constraints(
        datanode, [example.multiple.lcName]
    )
    expected = EXPECTED_MULTI_HOT.tolist()

    assert result["hypotheses"][example.multiple.lcName] == expected
    assert datanode.getExecutableConstraintLabels()[
        f"{example.multiple.lcName}/answer"
    ] == expected
