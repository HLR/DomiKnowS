"""CLEVR-style DomiKnowS InferenceProgram wrapper for TemporalRelation.

This module keeps the MATRES/TemporalQA adapter on the same execution path as
``test_regr.Clever``: build a graph, compile executable ``logic_str`` queries,
attach sensors/learners, and evaluate through ``InferenceProgram``.
"""

from __future__ import annotations

import torch

from domiknows.program.lossprogram import InferenceProgram
from domiknows.program.model.pytorch import SolverModel
from domiknows.sensor.pytorch import EdgeSensor, ModuleLearner
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from domiknows.sensor.pytorch.sensors import FunctionalReaderSensor, ReaderSensor

from .execution import create_executable_instance
from .graph import TEMPORAL_LABELS, create_temporal_graph, unpack_pair


class BinaryOracleLearner(torch.nn.Module):
    """Convert 0/1 oracle labels into two-class probabilities.

    This mirrors the CLEVR dummy learner style: the learner still lives behind a
    DomiKnowS ``ModuleLearner``, but for the example it emits perfect predicate
    probabilities so the InferenceProgram path can be validated end to end.
    """

    def forward(self, labels):
        labels = labels.float()
        return torch.stack([1.0 - labels, labels], dim=-1)


def temporal_program_declaration(instances, device="cpu", tnorm="G", infer_types=None):
    """Create a CLEVR-style ``InferenceProgram`` for TemporalRelation examples."""

    infer_types = infer_types or ["local/argmax"]
    instances = list(instances)
    ctx = create_temporal_graph(instances)
    graph = ctx.graph

    _attach_temporal_sensors(ctx, device=device)
    dataset = compile_temporal_program_dataset(instances, ctx, device=device)

    poi = [
        ctx.document,
        ctx.sentence,
        ctx.token,
        ctx.event,
        ctx.query_event1,
        ctx.query_event2,
        ctx.event_pair,
        ctx.temporal_relation,
        *ctx.label_concepts.values(),
        graph.constraint,
    ]
    program = InferenceProgram(
        graph,
        SolverModel,
        poi=poi,
        device=device,
        tnorm=tnorm,
        inferTypes=infer_types,
    )
    return dataset, ctx, program


def compile_temporal_program_dataset(instances, ctx, device="cpu"):
    executable = [_to_program_data(instance, device=device) for instance in instances]
    return ctx.graph.compile_executable(
        executable,
        logic_keyword="logic_str",
        logic_label_keyword="logic_label",
        extra_namespace_values=ctx.namespace,
    )


def make_packed_left_example():
    """Small TemporalQA example used by docs/tests."""

    return {
        "doc_id": "toy_temporal_1",
        "text": "John packed his bag before he left the house.",
        "tokens": [
            {"id": "e1", "text": "packed"},
            {"id": "e2", "text": "left"},
        ],
        "events": [
            {"id": "e1", "token_id": "e1", "text": "packed"},
            {"id": "e2", "token_id": "e2", "text": "left"},
        ],
        "event_pairs": [
            {"e1": "e1", "e2": "e2", "label": "Before"},
            {"e1": "e2", "e2": "e1", "label": "After"},
        ],
        "query_pair": {"e1": "e1", "e2": "e2", "label": "Before"},
    }


def evaluate_packed_left_example(device="cpu"):
    dataset, _ctx, program = temporal_program_declaration([make_packed_left_example()], device=device)
    return program.evaluate_condition(dataset, device=device)


def _attach_temporal_sensors(ctx, device="cpu"):
    # Label for the executable queryL/iotaL constraint. Without it no constraint
    # datanode exists, so getExecutableConstraintLabels() returns {} and
    # evaluate_condition skips every item. Matches test_regr/Clever/main.py.
    ctx.graph.constraint["label"] = ReaderSensor(keyword="logic_label", label=True)

    ctx.document["index"] = FunctionalReaderSensor(
        keyword="document_indices",
        forward=lambda data: _tensor(data, device=device),
    )
    ctx.sentence["index"] = FunctionalReaderSensor(
        keyword="sentence_indices",
        forward=lambda data: _tensor(data, device=device),
    )
    ctx.token["index"] = FunctionalReaderSensor(
        keyword="event_indices",
        forward=lambda data: _tensor(data, device=device),
    )

    ctx.sentence[ctx.document_contains_sentence] = EdgeSensor(
        ctx.sentence["index"],
        ctx.document["index"],
        relation=ctx.document_contains_sentence,
        forward=lambda sentence, _document: torch.ones_like(sentence).unsqueeze(-1),
    )
    ctx.token[ctx.sentence_contains_token] = EdgeSensor(
        ctx.token["index"],
        ctx.sentence["index"],
        relation=ctx.sentence_contains_token,
        forward=lambda token, _sentence: torch.ones_like(token).unsqueeze(-1),
    )

    ctx.token["event_label"] = FunctionalReaderSensor(
        keyword="is_event",
        forward=lambda data: _tensor(data, device=device),
    )
    ctx.token[ctx.event] = ModuleLearner("event_label", module=BinaryOracleLearner(), device=device)

    ctx.token["query_event1_label"] = FunctionalReaderSensor(
        keyword="is_query_event1",
        forward=lambda data: _tensor(data, device=device),
    )
    ctx.token[ctx.query_event1] = ModuleLearner("query_event1_label", module=BinaryOracleLearner(), device=device)

    ctx.token["query_event2_label"] = FunctionalReaderSensor(
        keyword="is_query_event2",
        forward=lambda data: _tensor(data, device=device),
    )
    ctx.token[ctx.query_event2] = ModuleLearner("query_event2_label", module=BinaryOracleLearner(), device=device)

    ctx.event_pair[ctx.pair_event1.reversed, ctx.pair_event2.reversed] = CompositionCandidateSensor(
        ctx.token["index"],
        relations=(ctx.pair_event1.reversed, ctx.pair_event2.reversed),
        forward=_candidate_event_pair,
    )

    for label, concept in ctx.label_concepts.items():
        ctx.event_pair[concept] = FunctionalReaderSensor(
            keyword=f"is_temporal_{label}",
            forward=lambda data, _device=device: torch.as_tensor(data, dtype=torch.float32, device=_device),
        )


def _to_program_data(instance, device="cpu"):
    converted = create_executable_instance(instance)
    events = list(instance.get("events", []))
    event_ids = [_event_id(event) for event in events]
    event_index = {event_id: index for index, event_id in enumerate(event_ids)}
    query_pair = converted.get("query_pair") or instance.get("query_pair") or instance.get("event_pairs", [None])[0]
    query_e1, query_e2, _label = unpack_pair(query_pair)

    pair_labels = {}
    for pair in instance.get("event_pairs", []):
        e1, e2, label = unpack_pair(pair)
        if e1 in event_index and e2 in event_index and label in TEMPORAL_LABELS:
            pair_labels[(event_index[e1], event_index[e2])] = label

    candidate_pairs = [
        (left, right)
        for left in range(len(events))
        for right in range(len(events))
        if left != right
    ]
    temporal_label_tensors = {}
    for label in TEMPORAL_LABELS:
        rows = []
        for pair in candidate_pairs:
            value = 1 if pair_labels.get(pair) == label else 0
            rows.append([1 - value, value])
        temporal_label_tensors[f"is_temporal_{label}"] = torch.tensor(rows, dtype=torch.float32, device=device)

    converted.update(
        {
            "document_indices": torch.tensor([0], dtype=torch.long, device=device),
            "sentence_indices": torch.tensor([0], dtype=torch.long, device=device),
            "event_indices": torch.arange(len(events), dtype=torch.long, device=device),
            "is_event": torch.ones(len(events), dtype=torch.long, device=device),
            "is_query_event1": torch.tensor(
                [1 if event_id == query_e1 else 0 for event_id in event_ids],
                dtype=torch.long,
                device=device,
            ),
            "is_query_event2": torch.tensor(
                [1 if event_id == query_e2 else 0 for event_id in event_ids],
                dtype=torch.long,
                device=device,
            ),
            **temporal_label_tensors,
        }
    )
    if converted.get("logic_label") is not None:
        converted["logic_label"] = torch.LongTensor([int(converted["logic_label"])]).to(device)
    return converted


def _candidate_event_pair(arg1=None, arg2=None, **kwargs):
    if arg1 is not None and arg2 is not None:
        return int(arg1.getAttribute("index")) != int(arg2.getAttribute("index"))
    nodes = [value for value in kwargs.values() if hasattr(value, "getAttribute")]
    if len(nodes) >= 2:
        return int(nodes[0].getAttribute("index")) != int(nodes[1].getAttribute("index"))
    return True


def _tensor(data, device="cpu"):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    return torch.as_tensor(data, dtype=torch.long, device=device)


def _event_id(event):
    return event.get("id") if isinstance(event, dict) else event


if __name__ == "__main__":
    print(evaluate_packed_left_example())
