"""A three-object, multi-answer executable-query example.

``iotaL`` selects one entity, so set-valued answers are represented as one
executable membership query per candidate.  The returned set contains every
candidate whose ``existsL`` query evaluates to true.
"""

from dataclasses import dataclass

import torch

from domiknows.graph import Concept, Graph, Relation
from domiknows.graph.dataNode import DataNode, DataNodeBuilder
from domiknows.program.lossprogram import InferenceProgram
from domiknows.program.model.pytorch import SolverModel
from domiknows.sensor.pytorch import ModuleLearner
from domiknows.sensor.pytorch.sensors import FunctionalReaderSensor
from domiknows.solver.ilpOntSolverFactory import ilpOntSolverFactory


OBJECT_IDS = ("o1", "o2", "o3")
GOLD_ANSWERS = {"o1", "o2"}
FEATURES = torch.tensor(
    [
        [1.0, 1.0, 0.0],  # o1: red dog
        [1.0, 0.0, 1.0],  # o2: red cat
        [0.0, 1.0, 0.0],  # o3: dog, but not red
    ]
)


class TinyConceptClassifier(torch.nn.Module):
    """A learnable linear yes/no classifier for one feature column."""

    def __init__(self, feature_index, feature_count=3):
        super().__init__()
        self.linear = torch.nn.Linear(feature_count, 2)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.copy_(torch.tensor([2.0, -2.0]))
            self.linear.weight[0, feature_index] = -4.0
            self.linear.weight[1, feature_index] = 4.0

    def forward(self, features):
        return self.linear(features.float())


class CandidateClassifier(torch.nn.Module):
    """Identify one candidate in the object list."""

    def __init__(self, candidate_index):
        super().__init__()
        self.candidate_index = candidate_index

    def forward(self, indices):
        score = (indices.view(-1) == self.candidate_index).float() * 8.0 - 4.0
        return torch.stack((-score, score), dim=-1)


@dataclass
class MultiAnswerExample:
    dataset: object
    program: InferenceProgram
    concept_models: dict
    concept_learners: dict
    logic_strings: tuple


def reset_domiknows_state():
    """Clear process-global registries before declaring an independent graph."""

    Graph.clear()
    Concept.clear()
    Relation.clear()
    DataNode.clear()
    DataNodeBuilder.clear()
    ilpOntSolverFactory.clear()


def build_multi_answer_example(device="cpu"):
    reset_domiknows_state()
    with Graph("tiny_multi_answer") as graph:
        obj = Concept(name="obj")
        red = obj(name="red")
        dog = obj(name="dog")
        cat = obj(name="cat")
        candidates = [obj(name=f"candidate_{index}") for index in range(len(OBJECT_IDS))]

    obj["features"] = FunctionalReaderSensor(
        keyword="features",
        forward=lambda data: torch.as_tensor(data, dtype=torch.float32, device=device),
    )
    obj["index"] = FunctionalReaderSensor(
        keyword="indices",
        forward=lambda data: torch.as_tensor(data, dtype=torch.long, device=device),
    )

    concept_models = {
        "red": TinyConceptClassifier(0).to(device),
        "dog": TinyConceptClassifier(1).to(device),
        "cat": TinyConceptClassifier(2).to(device),
    }
    concept_learners = {}
    for name, concept in (("red", red), ("dog", dog), ("cat", cat)):
        learner = ModuleLearner("features", module=concept_models[name])
        obj[concept] = learner
        concept_learners[name] = learner
    for index, candidate in enumerate(candidates):
        obj[candidate] = ModuleLearner("index", module=CandidateClassifier(index).to(device))

    namespace = {"red": red, "dog": dog, "cat": cat}
    namespace.update({candidate.name: candidate for candidate in candidates})
    rows = []
    logic_strings = []
    for index, object_id in enumerate(OBJECT_IDS):
        logic = (
            "existsL(andL("
            'red("o"), orL(dog(path="o"), cat(path="o")), '
            f'candidate_{index}(path="o")))'
        )
        logic_strings.append(logic)
        rows.append(
            {
                "features": FEATURES.to(device),
                "indices": torch.arange(len(OBJECT_IDS), device=device),
                "logic_str": logic,
                "logic_label": torch.tensor(
                    [int(object_id in GOLD_ANSWERS)], dtype=torch.long, device=device
                ),
            }
        )

    dataset = graph.compile_executable(
        rows,
        logic_keyword="logic_str",
        logic_label_keyword="logic_label",
        extra_namespace_values=namespace,
    )
    program = InferenceProgram(
        graph,
        SolverModel,
        poi=[obj, red, dog, cat, *candidates, graph.constraint],
        device=device,
        inferTypes=["local/argmax"],
        beta=1.0,
    )
    return MultiAnswerExample(
        dataset,
        program,
        concept_models,
        concept_learners,
        tuple(logic_strings),
    )


def predict_answer_set(example):
    """Decode the tiny predicate logits into the corresponding answer set."""

    with torch.no_grad():
        predictions = {
            name: model(FEATURES.to(next(model.parameters()).device)).argmax(dim=-1).bool()
            for name, model in example.concept_models.items()
        }
    selected = predictions["red"] & (predictions["dog"] | predictions["cat"])
    return {object_id for object_id, keep in zip(OBJECT_IDS, selected.tolist()) if keep}


def run_example(device="cpu"):
    example = build_multi_answer_example(device=device)
    accuracy = example.program.evaluate_condition(example.dataset, device=device)
    return predict_answer_set(example), accuracy


if __name__ == "__main__":
    answers, executable_accuracy = run_example()
    print(f"answers={sorted(answers)} executable_accuracy={executable_accuracy:.1f}")
