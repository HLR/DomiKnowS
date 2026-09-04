"""Tiny candidate-aligned multi-answer ``queryL(miotaL(...))`` example."""

from dataclasses import dataclass

import torch

from domiknows.graph import Concept, Graph, miotaL, queryL
from domiknows.graph.concept import EnumConcept
from domiknows.program.lossprogram import InferenceProgram
from domiknows.program.model.pytorch import SolverModel
from domiknows.sensor.pytorch import ModuleLearner
from domiknows.sensor.pytorch.relation_sensors import EdgeSensor
from domiknows.sensor.pytorch.sensors import FunctionalReaderSensor

from .example import FEATURES, TinyConceptClassifier
from .example_multiAnswers import reset_domiknows_state


MULTI_QUERY_LABEL = torch.tensor([0, 1, -1], dtype=torch.long)


@dataclass
class MultiQueryExample:
    dataset: object
    program: InferenceProgram
    constraint: queryL
    logic_string: str


class TinyKindClassifier(torch.nn.Module):
    """Classify dog-like rows as 0 and cat-like rows as 1."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.zero_()
            self.linear.weight[0, 1] = 4.0
            self.linear.weight[0, 2] = -4.0
            self.linear.weight[1, 1] = -4.0
            self.linear.weight[1, 2] = 4.0

    def forward(self, features):
        return self.linear(features.float())


def build_multi_query_example(
    device="cpu", threshold=0.5, hard=False, label=MULTI_QUERY_LABEL
):
    # Clear global graph state so repeated regression runs build the same graph.
    reset_domiknows_state()
    with Graph("tiny_multi_answer_query") as graph:
        scene = Concept(name="scene")
        obj = Concept(name="obj")
        (contains,) = scene.contains(obj)
        red = obj(name="red")
        dog = obj(name="dog")
        cat = obj(name="cat")
        kind = obj(
            name="kind",
            ConceptClass=EnumConcept,
            values=["dog_kind", "cat_kind"],
        )

    scene["scene"] = FunctionalReaderSensor(
        keyword="scene",
        forward=lambda data: torch.as_tensor(data, dtype=torch.long, device=device),
    )
    obj["features"] = FunctionalReaderSensor(
        keyword="features",
        forward=lambda data: torch.as_tensor(data, dtype=torch.float32, device=device),
    )
    # Attach every object row to the single scene so each row remains a query candidate.
    obj[contains] = EdgeSensor(
        obj["features"],
        scene["scene"],
        relation=contains,
        forward=lambda features, _: torch.ones(
            (len(features), 1), dtype=torch.float32, device=features.device
        ),
    )
    # Fixed tiny classifiers make the selected candidates and their class IDs deterministic.
    obj[red] = ModuleLearner("features", module=TinyConceptClassifier(0).to(device))
    obj[dog] = ModuleLearner("features", module=TinyConceptClassifier(1).to(device))
    obj[cat] = ModuleLearner("features", module=TinyConceptClassifier(2).to(device))
    obj[kind] = ModuleLearner("features", module=TinyKindClassifier().to(device))

    # miotaL selects qualifying objects independently; queryL returns kind IDs on that same axis.
    logic = (
        "queryL(kind, miotaL(andL("
        'red("o"), orL(dog(path="o"), cat(path="o"))'
        f"), threshold={float(threshold)!r}, hard={bool(hard)!r}))"
    )
    # Labels are candidate-aligned: -1 marks an object excluded by the selector.
    rows = [{
        "scene": torch.tensor([0], device=device),
        "features": FEATURES.to(device),
        "logic_str": logic,
        "logic_label": torch.as_tensor(label, device=device),
    }]
    # The parsed expression refers to the concepts by their symbolic names.
    dataset = graph.compile_executable(
        rows,
        logic_keyword="logic_str",
        logic_label_keyword="logic_label",
        extra_namespace_values={
            "red": red,
            "dog": dog,
            "cat": cat,
            "kind": kind,
        },
    )
    constraint = next(iter(graph.executableLCs.values())).innerLC
    program = InferenceProgram(
        graph,
        SolverModel,
        poi=[scene, obj, red, dog, cat, graph.constraint],
        device=device,
        inferTypes=["local/argmax"],
        beta=1.0,
    )
    return MultiQueryExample(dataset, program, constraint, logic)


def predict_class_vector(example, device="cpu"):
    datanode = next(example.program.populate(example.dataset, device=device))
    result = datanode.calculateSingleLcLoss(example.constraint.lcName, tnorm="P")
    # queryAnswer preserves the candidate order and uses -1 for unselected rows.
    return result["queryAnswer"].detach().cpu().tolist()


def run_example(device="cpu"):
    example = build_multi_query_example(device=device)
    answers = predict_class_vector(example, device=device)
    metrics = example.program.evaluate_condition(
        example.dataset, device=device, return_dict=True
    )
    return answers, metrics


if __name__ == "__main__":
    answers, evaluation = run_example()
    print(
        f"class_ids={answers} "
        f"exact_accuracy={evaluation['query_accuracy']:.1f} "
        f"position_accuracy={evaluation['multi_query_position_accuracy']:.1f}"
    )
