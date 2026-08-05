"""The same tiny set-valued question expressed by one ``miotaL`` query."""

from dataclasses import dataclass

import torch

from domiknows.graph import Concept, Graph, Relation, miotaL
from domiknows.graph.dataNode import DataNode, DataNodeBuilder
from domiknows.program.lossprogram import InferenceProgram
from domiknows.program.model.pytorch import SolverModel
from domiknows.sensor.pytorch import ModuleLearner
from domiknows.sensor.pytorch.sensors import FunctionalReaderSensor
from domiknows.solver.ilpOntSolverFactory import ilpOntSolverFactory

from .example import (
    FEATURES,
    GOLD_ANSWERS,
    OBJECT_IDS,
    TinyConceptClassifier,
)


MULTI_HOT_LABEL = torch.tensor([1, 1, 0], dtype=torch.float32)


@dataclass
class MiotaExample:
    dataset: object
    program: InferenceProgram
    constraint: miotaL
    logic_string: str


def reset_domiknows_state():
    Graph.clear()
    Concept.clear()
    Relation.clear()
    DataNode.clear()
    DataNodeBuilder.clear()
    ilpOntSolverFactory.clear()


def build_multi_answer_example(
    device="cpu", threshold=0.5, hard=False, label=MULTI_HOT_LABEL
):
    reset_domiknows_state()
    with Graph("tiny_multi_answer_miota") as graph:
        obj = Concept(name="obj")
        red = obj(name="red")
        dog = obj(name="dog")
        cat = obj(name="cat")

    obj["features"] = FunctionalReaderSensor(
        keyword="features",
        forward=lambda data: torch.as_tensor(data, dtype=torch.float32, device=device),
    )
    obj[red] = ModuleLearner(
        "features", module=TinyConceptClassifier(0).to(device)
    )
    obj[dog] = ModuleLearner(
        "features", module=TinyConceptClassifier(1).to(device)
    )
    obj[cat] = ModuleLearner(
        "features", module=TinyConceptClassifier(2).to(device)
    )

    logic = (
        "miotaL(andL("
        'red("o"), orL(dog(path="o"), cat(path="o"))'
        f"), threshold={float(threshold)!r}, hard={bool(hard)!r})"
    )
    rows = [{
        "features": FEATURES.to(device),
        "logic_str": logic,
        "logic_label": torch.as_tensor(label, dtype=torch.float32, device=device),
    }]
    dataset = graph.compile_executable(
        rows,
        logic_keyword="logic_str",
        logic_label_keyword="logic_label",
        extra_namespace_values={"red": red, "dog": dog, "cat": cat},
    )
    constraint = next(iter(graph.executableLCs.values())).innerLC
    program = InferenceProgram(
        graph,
        SolverModel,
        poi=[obj, red, dog, cat, graph.constraint],
        device=device,
        inferTypes=["local/argmax"],
        beta=1.0,
    )
    return MiotaExample(dataset, program, constraint, logic)


def predict_answer_vector(example, device="cpu"):
    datanode = next(example.program.populate(example.dataset, device=device))
    result = datanode.calculateSingleLcLoss(example.constraint.lcName, tnorm="P")
    probabilities = result["selectionDistribution"].detach().reshape(-1)
    return (probabilities >= example.constraint.threshold).to(torch.int64).tolist()


def predict_answer_set(example, device="cpu"):
    vector = predict_answer_vector(example, device=device)
    return {object_id for object_id, selected in zip(OBJECT_IDS, vector) if selected}


def run_example(device="cpu"):
    example = build_multi_answer_example(device=device)
    answers = predict_answer_set(example, device=device)
    metrics = example.program.evaluate_condition(
        example.dataset, device=device, return_dict=True
    )
    return answers, metrics


if __name__ == "__main__":
    answers, evaluation = run_example()
    print(
        f"answers={sorted(answers)} "
        f"exact_accuracy={evaluation['miota_exact_accuracy']:.1f} "
        f"position_accuracy={evaluation['miota_position_accuracy']:.1f}"
    )
