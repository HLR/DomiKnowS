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
    # Materialize the fixture's single batch so tests exercise the populated graph.
    return next(example.program.populate(example.dataset, device="cpu"))


# Two relation hops must still produce one answer for each starting object.
NESTED_RELATION_LOGIC = (
    'miotaL(andL(object("x"), '
    'left("r1", path=("x", pair_src.reversed)), '
    'ball("y", path=("r1", pair_dst)), '
    'left("r2", path=("y", pair_src.reversed)), '
    'ball("z", path=("r2", pair_dst))), '
    'threshold=0.5, hard=False)'
)


def _multi_property_hop_logic(hops):
    # Require a red ball at every destination to stress relation/property alignment.
    conditions = ['object("x")']
    source = "x"
    for hop in range(1, hops + 1):
        relation = f"r{hop}"
        destination = f"y{hop}"
        conditions.extend([
            f'left("{relation}", path=("{source}", pair_src.reversed))',
            f'ball("{destination}", path=("{relation}", pair_dst))',
            f'red("red_at_{hop}", path=("{destination}",))',
        ])
        source = destination
    return "miotaL(andL(" + ", ".join(conditions) + "), threshold=0.5)"


def _two_hop_color_logic(color):
    # The color predicate applies only to the intermediate object in the chain.
    return (
        "miotaL(andL("
        'object("x"), '
        'left("r1", path=("x", pair_src.reversed)), '
        'ball("y", path=("r1", pair_dst)), '
        f'{color}("color_y", path=("y",)), '
        'left("r2", path=("y", pair_src.reversed)), '
        'ball("z", path=("r2", pair_dst))'
        "), threshold=0.5)"
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
    # This must remain differentiable for both soft and hard selector modes.
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
    # Two qualifying targets for object 1 must not create an extra answer slot.
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

    # All evaluator backends must retain the same relation-aligned distribution.
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


@pytest.mark.parametrize(
    "hops",
    [3, 4, 5],
)
def test_multi_property_relation_chains_remain_entity_aligned(hops):
    object_ids = (1, 2, 3)
    # Every object is simultaneously red and a ball. A cycle provides a valid
    # path of every tested depth without creating a needlessly huge fixture.
    features = torch.ones((len(object_ids), 4), dtype=torch.float32)
    expected = [1, 1, 1]
    example = build_relation_answer_example(
        executable=True,
        logic=_multi_property_hop_logic(hops),
        logic_label=expected,
        object_ids=object_ids,
        features=features,
        left_pairs={(1, 2), (2, 3), (3, 1)},
    )
    datanode = _datanode(example)
    distribution = datanode.calculateSingleLcLoss(
        example.multiple.lcName, tnorm="G"
    )["selectionDistribution"]

    assert distribution.shape == (len(object_ids),)
    assert (distribution >= 0.5).to(torch.int64).tolist() == expected
    assert distribution.requires_grad
    distribution.sum().backward()

    # Removing the closing edge leaves fewer than three hops from every
    # candidate, so all 3/4/5-hop constraints must return an empty set while
    # retaining the same three-object output axis.
    empty_example = build_relation_answer_example(
        executable=True,
        logic=_multi_property_hop_logic(hops),
        logic_label=[0, 0, 0],
        object_ids=object_ids,
        features=features,
        left_pairs={(1, 2), (2, 3)},
    )
    empty_distribution = _datanode(empty_example).calculateSingleLcLoss(
        empty_example.multiple.lcName, tnorm="G"
    )["selectionDistribution"]
    assert empty_distribution.shape == (len(object_ids),)
    assert (empty_distribution >= 0.5).to(torch.int64).tolist() == [0, 0, 0]


def test_existing_hop_fails_on_yellow_ball_when_blue_is_required():
    object_ids = (1, 2, 3)
    # All relation targets exist. Object 2 is a yellow ball rather than a blue
    # ball; object 3 is a blue ball at the end of the second hop.
    features = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 0.0],
    ])
    left_pairs = {(1, 2), (2, 3)}

    blue_example = build_relation_answer_example(
        executable=True,
        logic=_two_hop_color_logic("blue"),
        logic_label=[0, 0, 0],
        object_ids=object_ids,
        features=features,
        left_pairs=left_pairs,
    )
    blue_distribution = _datanode(blue_example).calculateSingleLcLoss(
        blue_example.multiple.lcName, tnorm="G"
    )["selectionDistribution"]
    assert blue_distribution.shape == (3,)
    assert (blue_distribution >= 0.5).to(torch.int64).tolist() == [0, 0, 0]

    # Changing only the required color proves that the first chain was rejected
    # by the false blue predicate, not by a missing relation candidate.
    yellow_example = build_relation_answer_example(
        executable=True,
        logic=_two_hop_color_logic("yellow"),
        logic_label=[1, 0, 0],
        object_ids=object_ids,
        features=features,
        left_pairs=left_pairs,
    )
    yellow_distribution = _datanode(yellow_example).calculateSingleLcLoss(
        yellow_example.multiple.lcName, tnorm="G"
    )["selectionDistribution"]
    assert yellow_distribution.shape == (3,)
    assert (yellow_distribution >= 0.5).to(torch.int64).tolist() == [1, 0, 0]


def test_relation_miota_evaluation_and_vector_training():
    evaluation_example = build_relation_answer_example(executable=True)
    metrics = evaluation_example.program.evaluate_condition(
        evaluation_example.dataset, device="cpu", return_dict=True
    )
    assert metrics["miota_exact_accuracy"] == 100.0
    assert metrics["miota_position_accuracy"] == 100.0

    # Training consumes the vector-valued answer without producing a non-finite loss.
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

    # Verification and sampled Boolean construction use the same four-object axis.
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

    # The solver result and persisted executable label must agree exactly.
    assert result["hypotheses"][example.multiple.lcName] == expected
    assert datanode.getExecutableConstraintLabels()[
        f"{example.multiple.lcName}/answer"
    ] == expected
