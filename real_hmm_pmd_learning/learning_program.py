"""Small PMD learning program for one graph rule: B appears at most once."""
from __future__ import annotations

from dataclasses import dataclass, field
import torch

from domiknows.generation import (
    discover_generation_enforcement,
)
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
    from .stream_generator import GeneratorTrainingSource, PROMPT_VOCAB_SIZE, StreamTrainingExample, prompt_spec
    from .graph import build_bundle
except ImportError:  # pragma: no cover - direct script execution fallback
    from stream_generator import GeneratorTrainingSource, PROMPT_VOCAB_SIZE, StreamTrainingExample, prompt_spec
    from graph import build_bundle


@dataclass
class RealHMMPMDArtifacts:
    """Objects produced by ``build_learning_program``.
    """

    program: PrimalDualProgram = field(
        metadata={"description": "The standard DomiKnowS PMD program. It owns the statistical model, PMD constraint model, and training loop."}
    )
    graph: object = field(
        metadata={"description": "The declarative DomiKnowS graph with concepts, relation, enum symbol labels, and the single B-at-most-once rule."}
    )
    bundle: object = field(
        metadata={"description": "The generation adapter view of the graph. It gives generic generation code stable names for string, position, symbol, and order fields."}
    )
    model: CompactLabelGenerationHead = field(
        metadata={"description": "The trainable compact-label learner attached through ModuleLearner to predict generated_symbol probabilities."}
    )
    learner_name: str = field(
        metadata={
            "description": "The selected compact-label learner: discrete-hmm, graph-hmm, or energy.",
            "purpose": "Lets the demo compare a prompt-conditioned DiscreteHMM-backed learner, a graph-shaped HMM learner, and a neural local energy scorer without changing PMD wiring.",
        }
    )
    training_source: GeneratorTrainingSource = field(
        metadata={
            "description": "The task-local generator source. It reads fresh mock generator batches and converts them into DomiKnowS PMD samples.",
            "purpose": "Separates live data supply from the standard PrimalDualProgram.train(...) loop.",
        }
    )
    stream_examples: tuple[StreamTrainingExample, ...] = field(
        metadata={
            "description": "The latest materialized generator batch. Each item contains symbols and the DomiKnowS sample dict.",
            "purpose": "Gives the demo something concrete to print while training data itself comes from training_source.",
        }
    )
    dfa: object = field(
        metadata={"description": "The hard DFA verifier compiled from the graph rule. It rejects strings where B appears more than once."}
    )
    enforcement: object = field(
        metadata={
            "description": "The discovered generation enforcement summary, including the graph constraints used to build the DFA.",
            "purpose": "Shows which declarative graph rules were found and converted into hard generation enforcement.",
        }
    )
    stream_seed: int = field(
        metadata={
            "description": "Seed controlling the deterministic mock generator stream.",
            "purpose": "Makes the live stream reproducible while still being generated as batches during the demo.",
        }
    )
    inference_prompt_name: str = field(
        metadata={
            "description": "Prompt used when the learned compact-label model performs greedy inference after training.",
            "purpose": "Shows that generation is conditioned by a prompt, not only by random sequence frequencies.",
        }
    )
    inference_prompt_text: str = field(
        metadata={"description": "Human-readable text for the inference prompt."}
    )
    inference_prompt_token_id: int = field(
        metadata={"description": "Compact prompt token id passed to instruction_tokens for inference."}
    )


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
    """Create the compact-label learner used by ModuleLearner."""
    learner = _normalise_learner_name(learner)
    label_to_token_id = label_token_id_map(bundle.vocabulary)
    if learner == "discrete-hmm":
        # Use gated dynamics so emission/transition (not just the initial state)
        # depend on the prompt.  Without this the prompt only steers the first
        # hidden state, and with limited training data the initial_projector
        # softmax saturates to one state for every prompt — collapsing AB / CD /
        # short into identical outputs.
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
            dynamics_expert_count=3,
        )
    if learner == "graph-hmm":
        # The graph-HMM learner starts from the bundle-specific DFA support when available.
        model = GraphHMMGenerationHead.from_bundle(
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
        return model
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
    inference_prompt: str = "AB",
    pad_size: int | None = None,
    random_seed: int | None = 0,
    beta: float = 2.0,
) -> RealHMMPMDArtifacts:
    """Build a PMD program with a trainable compact-label learner."""
    if head is not None:
        learner = head
    learner = _normalise_learner_name(learner)
    if stream_count <= 0:
        raise ValueError("stream_count must be positive")
    inference_prompt_info = prompt_spec(inference_prompt)

    # 1. DomiKnows graph -> generation bundle -> graph-discovered DFA constraints.
    graph, bundle = build_bundle()
    pad_size = int(pad_size or 6)
    enforcement = discover_generation_enforcement(graph, bundle, on_unsupported="error")
    dfa = enforcement.dfa

    text = bundle.text
    token = bundle.token
    contains = bundle.contains
    generated_symbol = bundle.generated_token
    precedes = bundle.is_before_rel
    earlier = bundle.first_token
    later = bundle.second_token

    # 2. Sensors create one sequence of token DataNodes from generated sequence labels.
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

    # 3. Create the trainable compact-label learner.
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

    # 4. ModuleLearner lets PMD read compact-learner probabilities on DataNodes.
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

    return RealHMMPMDArtifacts(
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
    """Normalize user-facing learner aliases."""
    learner = str(learner).replace("_", "-").lower()
    if learner == "hmm":
        return "discrete-hmm"
    return learner
