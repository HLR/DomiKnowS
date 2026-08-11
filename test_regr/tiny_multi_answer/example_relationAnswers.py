"""Relation-aware ``iotaL`` and ``miotaL`` over an object-aligned axis.

Four objects are considered in order.  Objects 1 and 2 are left of object 3,
the reference ball, while only object 1 is red.  The two selectors therefore
return object 1 and ``{1, 2}``, respectively.
"""

from dataclasses import dataclass

import torch

from domiknows.graph import Concept, Graph, Relation, andL, iotaL, miotaL
from domiknows.graph.dataNode import DataNode, DataNodeBuilder
from domiknows.program.lossprogram import InferenceProgram
from domiknows.program.model.pytorch import SolverModel
from domiknows.sensor.pytorch import ModuleLearner
from domiknows.sensor.pytorch.query_sensor import DataNodeReaderSensor
from domiknows.sensor.pytorch.relation_sensors import (
    CompositionCandidateSensor,
    EdgeSensor,
)
from domiknows.sensor.pytorch.sensors import FunctionalReaderSensor
from domiknows.solver.ilpOntSolverFactory import ilpOntSolverFactory

from .example import TinyConceptClassifier


OBJECT_IDS = (1, 2, 3, 4)
REFERENCE_BALL_ID = 3
UNIQUE_RED_LEFT_ID = 1
LEFT_OBJECT_IDS = {1, 2}
EXPECTED_MULTI_HOT = torch.tensor([1, 1, 0, 0], dtype=torch.int64)
UNIQUE_CONSTRAINT_TEXT = (
    'iotaL(andL(red("x"), '
    'left("r", path=("x", pair_src.reversed)), '
    'ball("y", path=("r", pair_dst))))'
)
MULTI_CONSTRAINT_TEXT = (
    'miotaL(andL(object("x"), '
    'left("r", path=("x", pair_src.reversed)), '
    'ball("y", path=("r", pair_dst))), threshold=0.5, hard=False)'
)
FEATURES = torch.tensor([
    # red, ball, blue, yellow
    [1.0, 0.0, 0.0, 0.0],  # object 1: red, left of the ball
    [0.0, 0.0, 0.0, 0.0],  # object 2: non-red, left of the ball
    [0.0, 1.0, 1.0, 0.0],  # object 3: the reference blue ball
    [0.0, 0.0, 0.0, 1.0],  # object 4: yellow distractor
])


@dataclass
class RelationAnswerExample:
    dataset: object
    program: InferenceProgram
    unique: iotaL
    multiple: miotaL
    object_ids: tuple = OBJECT_IDS


def reset_domiknows_state():
    # These registries are global, so clear them before rebuilding fixture concepts.
    Graph.clear()
    Concept.clear()
    Relation.clear()
    DataNode.clear()
    DataNodeBuilder.clear()
    ilpOntSolverFactory.clear()


def build_relation_answer_example(
    device="cpu", threshold=0.5, hard=False, *, left_pairs=None,
    second_ball=False, executable=False, logic=None, logic_label=None,
    object_ids=OBJECT_IDS, features=None,
):
    # Rebuild global DomiKnowS state for each independent regression example.
    reset_domiknows_state()
    with Graph("tiny_relation_answers") as graph:
        # Object concepts carry unary properties; pair nodes carry relation edges.
        scene = Concept(name="scene")
        object_node = Concept(name="object")
        (contains,) = scene.contains(object_node)
        red = object_node(name="red")
        ball = object_node(name="ball")
        blue = object_node(name="blue")
        yellow = object_node(name="yellow")

        pair = Concept(name="pair")
        (pair_src, pair_dst) = pair.has_a(
            pair_src=object_node, pair_dst=object_node
        )
        left = pair(name="left")

        # Both selectors return one value per object x after traversing x -> pair -> ball.
        unique = iotaL(andL(
            red("x"),
            left("r", path=("x", pair_src.reversed)),
            ball("y", path=("r", pair_dst)),
        ))
        multiple = miotaL(andL(
            object_node("x"),
            left("r", path=("x", pair_src.reversed)),
            ball("y", path=("r", pair_dst)),
        ), threshold=threshold, hard=hard)

    scene["index"] = FunctionalReaderSensor(
        keyword="scene",
        forward=lambda data: torch.as_tensor(data, dtype=torch.long, device=device),
    )
    object_node["index"] = FunctionalReaderSensor(
        keyword="object_ids",
        forward=lambda data: torch.as_tensor(data, dtype=torch.long, device=device),
    )
    object_node["features"] = FunctionalReaderSensor(
        keyword="features",
        forward=lambda data: torch.as_tensor(data, dtype=torch.float32, device=device),
    )
    # Each object belongs to the only scene in this synthetic data row.
    object_node[contains] = EdgeSensor(
        object_node["index"], scene["index"], relation=contains,
        forward=lambda objects, _: torch.ones(
            (len(objects), 1), dtype=torch.float32, device=objects.device
        ),
    )
    object_node[red] = ModuleLearner(
        "features", module=TinyConceptClassifier(0, feature_count=4).to(device)
    )
    object_node[ball] = ModuleLearner(
        "features", module=TinyConceptClassifier(1, feature_count=4).to(device)
    )
    object_node[blue] = ModuleLearner(
        "features", module=TinyConceptClassifier(2, feature_count=4).to(device)
    )
    object_node[yellow] = ModuleLearner(
        "features", module=TinyConceptClassifier(3, feature_count=4).to(device)
    )
    # Materialize all object-pair candidates; read_left supplies their truth values.
    pair[pair_src.reversed, pair_dst.reversed] = CompositionCandidateSensor(
        object_node["index"],
        relations=(pair_src.reversed, pair_dst.reversed),
        forward=lambda *_, **__: True,
    )

    def read_left(*_, data, datanode):
        # Recover the candidate endpoints and encode false/true relation logits.
        src = datanode.relationLinks[pair_src.name][0]
        dst = datanode.relationLinks[pair_dst.name][0]
        edge = (src.getAttribute("index").item(), dst.getAttribute("index").item())
        truth = edge in data
        return torch.tensor([not truth, truth], dtype=torch.float32, device=device)

    pair[left] = DataNodeReaderSensor(
        pair_src.reversed, pair_dst.reversed,
        keyword="left_pairs", forward=read_left,
    )

    # Keep IDs, feature rows, and relation edges in the same deterministic order.
    object_ids = tuple(object_ids)
    features = FEATURES.clone() if features is None else torch.as_tensor(
        features, dtype=torch.float32
    ).clone()
    if len(features) != len(object_ids):
        raise ValueError("features must contain one row per object ID")
    if features.ndim != 2 or features.shape[1] != 4:
        raise ValueError("features must have red, ball, blue, and yellow columns")
    if second_ball:
        # Used by the duplicate-path regression case.
        features[3, 1] = 1.0
    if left_pairs is None:
        left_pairs = {(1, REFERENCE_BALL_ID), (2, REFERENCE_BALL_ID)}
    rows = [{
        "scene": torch.tensor([0], device=device),
        "object_ids": torch.tensor(object_ids, device=device),
        "features": features.to(device),
        "left_pairs": set(left_pairs),
    }]
    poi = [scene, object_node, pair, red, ball, blue, yellow, left]
    if executable:
        # Compile the same relation-aware selector to exercise labels and training.
        if logic is None:
            logic = (
                "miotaL(andL("
                'object("x"), '
                'left("r", path=("x", pair_src.reversed)), '
                'ball("y", path=("r", pair_dst))'
                f"), threshold={float(threshold)!r}, hard={bool(hard)!r})"
            )
        if logic_label is None:
            logic_label = EXPECTED_MULTI_HOT
        rows[0].update({
            "logic_str": logic,
            "logic_label": torch.as_tensor(
                logic_label, device=device, dtype=torch.float32
            ),
        })
        dataset = graph.compile_executable(
            rows,
            logic_keyword="logic_str",
            logic_label_keyword="logic_label",
            extra_namespace_values={
                "object": object_node,
                "left": left,
                "ball": ball,
                "blue": blue,
                "yellow": yellow,
                "pair_src": pair_src,
                "pair_dst": pair_dst,
            },
        )
        multiple = next(iter(graph.executableLCs.values())).innerLC
        # Include the compiled constraint in the program's points of interest.
        poi.append(graph.constraint)
    else:
        dataset = rows
    program = InferenceProgram(
        graph, SolverModel,
        poi=poi,
        device=device, inferTypes=["local/argmax"], beta=1.0,
    )
    return RelationAnswerExample(dataset, program, unique, multiple, object_ids)


def relation_answers(example, device="cpu"):
    datanode = next(example.program.populate(example.dataset, device=device))
    context = datanode._prepareLcLossContext("G")
    processor = context["solver"].myLcLossBooleanMethods
    processor.setTNorm("G")
    # iotaL needs its explicit selection distribution for argmax decoding.
    example.unique.returnsSelection = True
    unique_output, _ = context["solver"].constraintConstructor.constructLogicalConstrains(
        example.unique, processor, None, datanode, 0,
        key="/local/softmax", headLC=False, loss=True, sample=False,
    )
    unique_distribution = unique_output[0][0].detach().reshape(-1)
    # miotaL exposes its answer vector directly through the loss calculation.
    multi_result = datanode.calculateSingleLcLoss(example.multiple.lcName, tnorm="G")
    multi_distribution = multi_result["selectionDistribution"].detach().reshape(-1)
    # Distributions share OBJECT_IDS order; miotaL keeps every qualifying position.
    unique_id = example.object_ids[int(unique_distribution.argmax().item())]
    multi_hot = (multi_distribution >= example.multiple.threshold).to(torch.int64)
    answer_set = {
        object_id for object_id, selected in zip(example.object_ids, multi_hot.tolist())
        if selected
    }
    return unique_id, answer_set, multi_hot, unique_distribution, multi_distribution


def ilp_relation_answers(device="cpu"):
    """Infer both relation answers through the executable-constraint ILP path."""
    example = build_relation_answer_example(device=device, executable=True)
    datanode = next(example.program.populate(example.dataset, device=device))
    datanode.inferLocal()
    # The active executable miotaL dispatches inferILPResults through
    # AnswerSolver; the ordinary iotaL remains a hard graph constraint.
    datanode.inferILPResults(fun=None, minimizeObjective=False)

    graph = example.program.graph
    object_node = graph["object"]
    pair = graph["pair"]
    red = graph["red"]
    ball = graph["ball"]
    left = graph["left"]
    pair_src, pair_dst = pair.has_a()

    def selected(node, concept):
        # ILP attributes are binary tensors, one value for this concrete datanode.
        value = node.getAttribute(concept, "ILP")
        return value is not None and bool(value.detach().reshape(-1)[0] >= 0.5)

    objects = datanode.findDatanodes(select=object_node)
    red_ids = {
        node.getAttribute("index").item() for node in objects if selected(node, red)
    }
    ball_ids = {
        node.getAttribute("index").item() for node in objects if selected(node, ball)
    }
    unique_candidates = set()
    # Reconstruct iotaL's sole answer from selected predicates and relation edges.
    for pair_node in datanode.findDatanodes(select=pair):
        if not selected(pair_node, left):
            continue
        src_id = pair_node.relationLinks[pair_src.name][0].getAttribute("index").item()
        dst_id = pair_node.relationLinks[pair_dst.name][0].getAttribute("index").item()
        if src_id in red_ids and dst_id in ball_ids:
            unique_candidates.add(src_id)
    if len(unique_candidates) != 1:
        raise RuntimeError(
            f"iotaL expected one ILP-qualified object, found {sorted(unique_candidates)}"
        )

    labels = datanode.getExecutableConstraintLabels()
    # The executable miotaL answer remains aligned with the original object order.
    multi_hot = torch.as_tensor(
        labels[f"{example.multiple.lcName}/answer"], dtype=torch.int64
    )
    answer_set = {
        object_id for object_id, chosen in zip(example.object_ids, multi_hot.tolist())
        if chosen
    }
    return next(iter(unique_candidates)), answer_set, multi_hot


def run_example(device="cpu"):
    example = build_relation_answer_example(device=device)
    unique_id, answers, predicted, _, _ = relation_answers(example, device=device)
    exact_accuracy = float(torch.equal(predicted.cpu(), EXPECTED_MULTI_HOT) * 100)
    position_accuracy = float(
        (predicted.cpu() == EXPECTED_MULTI_HOT).float().mean().item() * 100
    )
    return unique_id, answers, exact_accuracy, position_accuracy


if __name__ == "__main__":
    unique_id, answers, exact_accuracy, position_accuracy = run_example()
    ilp_unique_id, ilp_answers, _ = ilp_relation_answers()
    # Print the full source-like constraints so each reported answer is
    # understandable without opening the graph declaration above.
    print(f"unique_constraint={UNIQUE_CONSTRAINT_TEXT}")
    print(f"multi_constraint={MULTI_CONSTRAINT_TEXT}")
    print(f"unique_id={unique_id}")
    print(f"multi_answer_set={sorted(answers)}")
    print(f"exact_set_accuracy={exact_accuracy:.1f}")
    print(f"per_position_accuracy={position_accuracy:.1f}")
    print(f"ilp_unique_id={ilp_unique_id}")
    print(f"ilp_multi_answer_set={sorted(ilp_answers)}")
