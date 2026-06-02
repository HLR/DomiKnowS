"""PMD learning program for the nested-constraints demo.

Mirrors :mod:`Tasks.real_hmm_pmd_learning.learning_program` but builds the
graph from :mod:`Tasks.nested_constraints_demo.graph` (which registers the
three head LCs from :mod:`Tasks.nested_constraints_demo.constraints`) and uses
``on_unsupported="warn"`` so the heterogeneous-andL salvage path in LC #2
surfaces its warning at program-build time rather than at decode time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from domiknows.generation import discover_generation_enforcement
from domiknows.generation.learners import (
    CompactLabelGenerationHead,
    EnergyCompactLabelGenerationHead,
    GraphHMMGenerationHead,
    HMMGenerationHead,
)
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.lossprogram import PrimalDualProgram
from domiknows.program.metric import MacroAverageTracker
from domiknows.program.model.pytorch import SolverModel
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from domiknows.sensor.pytorch.sensors import FunctionalSensor, JointSensor, ReaderSensor

try:
    from .graph import build_bundle
    from .stream_generator import (
        GeneratorTrainingSource,
        PROMPT_VOCAB_SIZE,
        StreamTrainingExample,
        prompt_spec,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from graph import build_bundle
    from stream_generator import (
        GeneratorTrainingSource,
        PROMPT_VOCAB_SIZE,
        StreamTrainingExample,
        prompt_spec,
    )


@dataclass
class NestedConstraintsArtifacts:
    """Objects produced by :func:`build_learning_program`."""

    program: PrimalDualProgram = field(metadata={"description": "PMD program wrapping the compact-label head."})
    graph: object = field(metadata={"description": "DomiKnowS graph with the three nested + path LCs registered."})
    bundle: object = field(metadata={"description": "Generation adapter view of the graph."})
    model: CompactLabelGenerationHead = field(metadata={"description": "Trainable compact-label head attached via ModuleLearner."})
    learner_name: str = field(metadata={"description": "Selected learner: 'discrete-hmm', 'graph-hmm', or 'energy'."})
    training_source: GeneratorTrainingSource = field(metadata={"description": "Mock stream generator that produces valid+invalid batches."})
    stream_examples: tuple[StreamTrainingExample, ...] = field(metadata={"description": "Most recently materialized training batch."})
    dfa: object = field(metadata={"description": "Compiled hard DFA verifier built from the three head LCs."})
    enforcement: object = field(metadata={"description": "Discovered enforcement summary (per-LC analyses)."})
    stream_seed: int = field(metadata={"description": "Seed controlling the deterministic stream."})
    inference_prompt_name: str = field(metadata={"description": "Prompt name for post-training greedy inference."})
    inference_prompt_text: str = field(metadata={"description": "Human-readable prompt description."})
    inference_prompt_token_id: int = field(metadata={"description": "Compact prompt token id passed to the head."})


def label_token_id_map(vocabulary) -> tuple[int | None, ...]:
    """Use compact labels as concrete token ids in this toy offline task."""
    return tuple(None if label == vocabulary.other_label else label for label in range(vocabulary.label_count))


def build_compact_learner(
    learner: str = "discrete-hmm",
    *,
    graph,
    bundle,
    dfa=None,
    pad_size: int,
    random_seed: int | None = 0,
) -> CompactLabelGenerationHead:
    """Instantiate the trainable head selected by *learner*."""
    learner = _normalise_learner_name(learner)
    label_to_token_id = label_token_id_map(bundle.vocabulary)
    if learner == "discrete-hmm":
        return HMMGenerationHead(
            label_count=bundle.vocabulary.label_count,
            state_count=3,
            pad_size=pad_size,
            label_to_token_id=label_to_token_id,
            prompt_conditioning="initial",
            prompt_vocab_size=PROMPT_VOCAB_SIZE,
            trainable=True,
            random_seed=random_seed,
            dynamics_conditioning="gated",
            dynamics_expert_count=2,
        )
    if learner == "graph-hmm":
        return GraphHMMGenerationHead.from_bundle(
            bundle,
            graph=graph,
            dfa=dfa,
            trainable=True,
            pad_size=pad_size,
            label_to_token_id=label_to_token_id,
            prompt_conditioning="initial",
            prompt_vocab_size=PROMPT_VOCAB_SIZE,
            random_seed=random_seed,
        )
    if learner == "energy":
        return EnergyCompactLabelGenerationHead(
            label_count=bundle.vocabulary.label_count,
            pad_size=pad_size,
            label_to_token_id=label_to_token_id,
            vocab_size=PROMPT_VOCAB_SIZE,
            random_seed=random_seed,
        )
    raise ValueError("learner must be 'discrete-hmm', 'graph-hmm', or 'energy'")


build_compact_head = build_compact_learner


def build_learning_program(
    *,
    learner: str = "discrete-hmm",
    head: str | None = None,
    stream_count: int = 4,
    stream_seed: int = 0,
    inference_prompt: str = "with_A",
    pad_size: int | None = None,
    random_seed: int | None = 0,
    beta: float = 0.3,
) -> NestedConstraintsArtifacts:
    """Build the PMD program + trainable head over the nested constraint graph."""
    if head is not None:
        learner = head
    learner = _normalise_learner_name(learner)
    if stream_count <= 0:
        raise ValueError("stream_count must be positive")
    inference_prompt_info = prompt_spec(inference_prompt)

    graph, bundle = build_bundle()
    pad_size = int(pad_size or 6)
    # ``warn`` (not ``error``) -- LC #2 surfaces the irregular sibling via the
    # heterogeneous-andL salvage path and we want the demo to keep running.
    enforcement = discover_generation_enforcement(graph, bundle, on_unsupported="warn")
    dfa = enforcement.dfa

    text = bundle.text
    token = bundle.token
    contains = bundle.contains
    generated_symbol = bundle.generated_token
    precedes = bundle.is_before_rel
    earlier = bundle.first_token
    later = bundle.second_token

    text["instruction_tokens"] = ReaderSensor(keyword="instruction_tokens")
    text["sequence_labels_input"] = ReaderSensor(keyword="sequence_labels_input")

    def add_sequence(sequence_labels_input):
        flat = sequence_labels_input[0] if getattr(sequence_labels_input, "dim", lambda: 0)() == 2 else sequence_labels_input
        labels = [int(label) for label in flat[:pad_size]]
        pad_label = bundle.vocabulary.other_label
        if len(labels) < pad_size:
            labels.extend([pad_label] * (pad_size - len(labels)))
        return torch.ones((pad_size, 1)), torch.tensor(labels, dtype=torch.long), torch.arange(pad_size)

    token[contains, "sequence_labels", "token_index"] = JointSensor(
        text["sequence_labels_input"],
        forward=add_sequence,
    )
    token[generated_symbol] = FunctionalSensor(
        token[contains],
        "sequence_labels",
        forward=lambda _contains, labels: labels.long(),
        label=True,
    )

    model = build_compact_learner(
        learner,
        graph=graph,
        bundle=bundle,
        dfa=dfa,
        pad_size=pad_size,
        random_seed=random_seed,
    )
    learner_name = learner
    if learner_name not in {"discrete-hmm", "graph-hmm", "energy"}:
        raise ValueError("learner must be 'discrete-hmm', 'graph-hmm', or 'energy'")

    token[generated_symbol] = ModuleLearner(
        token[contains],
        text["instruction_tokens"],
        "sequence_labels",
        module=model,
    )

    def precedes_edges(*_args, **candidates):
        earlier_node = candidates.get("earlier") or candidates.get("arg1")
        later_node = candidates.get("later") or candidates.get("arg2")
        if earlier_node is None or later_node is None:
            values = list(candidates.values())
            if len(values) < 2:
                raise ValueError("expected two position candidates")
            earlier_node, later_node = values[0], values[1]
        return earlier_node.getAttribute("token_index") < later_node.getAttribute("token_index")

    precedes[earlier.reversed, later.reversed] = CompositionCandidateSensor(
        relations=(earlier.reversed, later.reversed),
        forward=precedes_edges,
    )

    program = PrimalDualProgram(
        graph,
        SolverModel,
        poi=(text, token, precedes),
        inferTypes=["local/argmax"],
        loss=MacroAverageTracker(NBCrossEntropyLoss()),
        beta=float(beta),
        device="cpu",
        tnorm="P",
        counting_tnorm="P",
    )

    training_source = GeneratorTrainingSource(
        bundle,
        stream_count=stream_count,
        seed=stream_seed,
        max_length=pad_size,
    )

    return NestedConstraintsArtifacts(
        program=program,
        graph=graph,
        bundle=bundle,
        model=model,
        learner_name=learner_name,
        training_source=training_source,
        stream_examples=training_source.next_batch(step=0),
        dfa=dfa,
        enforcement=enforcement,
        stream_seed=stream_seed,
        inference_prompt_name=inference_prompt,
        inference_prompt_text=str(inference_prompt_info["text"]),
        inference_prompt_token_id=int(inference_prompt_info["token_id"]),
    )


def _normalise_learner_name(learner: str) -> str:
    learner = str(learner).replace("_", "-").lower()
    if learner == "hmm":
        return "discrete-hmm"
    return learner
