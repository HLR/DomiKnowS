"""Reuse one complete graph while changing active concepts per example step."""

from dataclasses import dataclass

import torch

from domiknows.graph import Concept, Graph
from domiknows.program.lossprogram import InferenceProgram
from domiknows.program.model.pytorch import SolverModel
from domiknows.sensor.pytorch import ModuleLearner
from domiknows.sensor.pytorch.sensors import FunctionalReaderSensor

from test_regr.tiny_dynamic_graph.example import (
    CONCEPT_VOCABULARY,
    EXAMPLES,
    CandidateClassifier,
    ConceptView,
    SharedConceptClassifier,
    reset_domiknows_state,
)


@dataclass
class ReusableDynamicGraphContext:
    graph: Graph
    concepts: dict
    candidates: tuple
    datasets: dict
    program: InferenceProgram
    shared_model: SharedConceptClassifier
    optimizer: torch.optim.Optimizer


def _rows_for_spec(spec, device):
    features = torch.tensor(spec.features, dtype=torch.float32, device=device)
    rows = []
    for candidate_index in range(len(spec.features)):
        predicates = [f'{spec.query_concepts[0]}("o")']
        predicates.extend(f'{name}(path="o")' for name in spec.query_concepts[1:])
        predicates.append(f'candidate_{candidate_index}(path="o")')
        rows.append(
            {
                "features": features,
                "indices": torch.arange(len(spec.features), device=device),
                "logic_str": f'existsL(andL({", ".join(predicates)}))',
                "logic_label": torch.tensor(
                    [int(candidate_index == spec.answer_index)],
                    dtype=torch.long,
                    device=device,
                ),
            }
        )
    return rows


def build_reusable_dynamic_graph(specs=EXAMPLES, device="cpu", learning_rate=1e-3):
    """Build the union schema, executable datasets, and program exactly once."""

    specs = tuple(specs)
    reset_domiknows_state()
    shared_model = SharedConceptClassifier().to(device)

    with Graph("tiny_dynamic_reusable") as graph:
        obj = Concept(name="obj")
        # concept = ("red", "dog", "cat", "tree")
        concepts = {name: obj(name=name) for name in CONCEPT_VOCABULARY}
        candidate_count = max(len(spec.features) for spec in specs)
        candidates = tuple(obj(name=f"candidate_{index}") for index in range(candidate_count))

    obj["features"] = FunctionalReaderSensor(
        keyword="features",
        forward=lambda data: torch.as_tensor(data, dtype=torch.float32, device=device),
    )
    obj["index"] = FunctionalReaderSensor(
        keyword="indices",
        forward=lambda data: torch.as_tensor(data, dtype=torch.long, device=device),
    )

    for name, concept in concepts.items():
        obj[concept] = ModuleLearner(
            "features",
            module=ConceptView(shared_model, CONCEPT_VOCABULARY.index(name)).to(device),
        )
    for index, candidate in enumerate(candidates):
        obj[candidate] = ModuleLearner(
            "index",
            module=CandidateClassifier(index).to(device),
        )

    namespace = dict(concepts)
    namespace.update({candidate.name: candidate for candidate in candidates})
    rows = []
    dataset_ranges = {}
    for spec in specs:
        start = len(rows)
        rows.extend(_rows_for_spec(spec, device))
        dataset_ranges[spec.example_id] = range(start, len(rows))

    compiled_dataset = graph.compile_executable(
        rows,
        logic_keyword="logic_str",
        logic_label_keyword="logic_label",
        extra_namespace_values=namespace,
    )
    datasets = {
        example_id: tuple(compiled_dataset[index] for index in indices)
        for example_id, indices in dataset_ranges.items()
    }
    program = InferenceProgram(
        graph,
        SolverModel,
        poi=[obj, *concepts.values(), *candidates, graph.constraint],
        device=device,
        inferTypes=["local/argmax"],
        beta=1.0,
    )
    optimizer = torch.optim.Adam(shared_model.parameters(), lr=learning_rate)
    program.opt = optimizer
    return ReusableDynamicGraphContext(
        graph,
        concepts,
        candidates,
        datasets,
        program,
        shared_model,
        optimizer,
    )


def _activate_example(context, spec):
    requested = [context.concepts[name] for name in spec.active_concepts]
    requested.extend(context.candidates[:len(spec.features)])
    context.graph.set_active_concepts(requested)
    return tuple(
        name for name in CONCEPT_VOCABULARY
        if context.graph.is_concept_active(context.concepts[name])
    )


def train_each_dynamic_graph(
    context=None,
    specs=EXAMPLES,
    device="cpu",
    learning_rate=1e-3,
):
    """Train every example by switching one reusable graph's active concepts."""

    if context is None:
        context = build_reusable_dynamic_graph(
            specs=specs,
            device=device,
            learning_rate=learning_rate,
        )
    before = context.shared_model.linear.weight.detach().clone()
    graph_summaries = []
    for spec in specs:
        active_vocabulary = _activate_example(context, spec)
        context.program.opt = context.optimizer
        context.program.train(
            context.datasets[spec.example_id],
            warmup_epochs=0,
            constraint_epochs=1,
            device=device,
            c_lr=learning_rate,
        )
        graph_summaries.append((context.graph.name, active_vocabulary))
    parameters_changed = not torch.equal(
        before.cpu(),
        context.shared_model.linear.weight.detach().cpu(),
    )
    return context, tuple(graph_summaries), parameters_changed


def infer_each_dynamic_graph(context, specs=EXAMPLES, device="cpu"):
    """Evaluate every example using the same graph and program as training."""

    accuracies = {}
    graph_summaries = []
    for spec in specs:
        active_vocabulary = _activate_example(context, spec)
        accuracies[spec.example_id] = context.program.evaluate_condition(
            context.datasets[spec.example_id],
            device=device,
        )
        graph_summaries.append((context.graph.name, active_vocabulary))
    return accuracies, tuple(graph_summaries)


if __name__ == "__main__":
    reusable, trained_graphs, changed = train_each_dynamic_graph()
    results, inferred_graphs = infer_each_dynamic_graph(reusable)
    print(f"parameters_changed={changed}")
    print(f"trained_graphs={trained_graphs}")
    print(f"inferred_graphs={inferred_graphs}")
    print(f"accuracies={results}")
