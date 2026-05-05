"""Collie-style DomiKnowS learning path for the hf_generation task."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from domiknows.generation import constraints_to_dfa, discover_generation_enforcement
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.lossprogram import PrimalDualProgram
from domiknows.program.metric import MacroAverageTracker
from domiknows.program.model.pytorch import SolverModel
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from domiknows.sensor.pytorch.sensors import FunctionalSensor, JointSensor, ReaderSensor

try:
    from .graph import VOCAB, build_generation_graph
    from .learning_model import FrozenBackboneGenerationHead
    from .mock_hf import MockFrozenBackbone, MockTokenizer
    from .run_demo import generation_vocab_for_tokenizer, load_backend
except ImportError:
    from graph import VOCAB, build_generation_graph
    from learning_model import FrozenBackboneGenerationHead
    from mock_hf import MockFrozenBackbone, MockTokenizer
    from run_demo import generation_vocab_for_tokenizer, load_backend


@dataclass
class LearningArtifacts:
    """Objects produced by ``build_learning_program`` for inspection/tests."""

    program: PrimalDualProgram
    graph: object
    bundle: object
    tokenizer: object
    model: FrozenBackboneGenerationHead
    sample_data: dict
    dfa: object


@dataclass
class LearningOptimizers:
    """Optimizers used by the tiny learning demo."""

    head: torch.optim.Optimizer
    constraints: torch.optim.Optimizer


def make_sample_data(tokenizer, target_tokens=None) -> dict:
    """Create one tiny prompt/target sample for the learning demo."""
    target_tokens = target_tokens or [" The", " cat", " mat", tokenizer.eos_token]
    target_ids = [tokenizer.encode(token)[0] for token in target_tokens]
    return {
        "instruction_tokens": tokenizer("Once", return_tensors="pt").input_ids,
        "target_token_ids": torch.tensor([target_ids], dtype=torch.long),
    }


def _build_backbone(real_hf: bool, model_name: str, quiet_transformers: bool):
    if not real_hf:
        tokenizer = MockTokenizer()
        return tokenizer, MockFrozenBackbone(vocab_size=len(tokenizer.token_to_id))

    tokenizer, model = load_backend(
        real_hf=True,
        model_name=model_name,
        quiet_transformers=quiet_transformers,
    )
    return tokenizer, model


def label_token_id_map(vocabulary) -> tuple[int | None, ...]:
    """Map compact generation labels back to raw tokenizer ids when possible."""
    token_ids: list[int | None] = []
    for label in range(vocabulary.label_count):
        try:
            token_ids.append(vocabulary.token_id_for_label(label))
        except ValueError:
            token_ids.append(None)
    return tuple(token_ids)


def build_learning_program(
    *,
    real_hf: bool = False,
    model_name: str = "roneneldan/TinyStories-1M",
    pad_size: int = 4,
    constrained_decoding: bool = False,
    quiet_transformers: bool = True,
    random_seed: int | None = 0,
) -> LearningArtifacts:
    """Build a small Collie-style PMD program for generation learning."""
    tokenizer, backbone = _build_backbone(real_hf, model_name, quiet_transformers)
    vocab, eos_token = generation_vocab_for_tokenizer(tokenizer, real_hf=real_hf)
    graph, bundle = build_generation_graph(tokenizer, vocab, eos_token=eos_token)
    enforcement = discover_generation_enforcement(graph, bundle, on_unsupported="error")
    dfa = constraints_to_dfa(enforcement.dfa_constraints, bundle.vocabulary)

    text = bundle.text
    token = bundle.token
    contains = bundle.contains
    generated_token = bundle.generated_token
    is_before_rel = bundle.is_before_rel
    first_token = bundle.first_token
    second_token = bundle.second_token

    text["instruction_tokens"] = ReaderSensor(keyword="instruction_tokens")
    text["target_token_ids"] = ReaderSensor(keyword="target_token_ids")

    def add_sequence(target_token_ids):
        flat = target_token_ids[0] if getattr(target_token_ids, "dim", lambda: 0)() == 2 else target_token_ids
        labels = [bundle.vocabulary.label_for_token_id(int(token_id)) for token_id in flat[:pad_size]]
        eos_label = bundle.vocabulary.eos_label
        if len(labels) < pad_size:
            labels.extend([eos_label] * (pad_size - len(labels)))
        return torch.ones((pad_size, 1)), torch.tensor(labels, dtype=torch.long), torch.arange(pad_size)

    token[contains, "target_labels", "token_index"] = JointSensor(
        text["target_token_ids"],
        forward=add_sequence,
    )
    token[generated_token] = FunctionalSensor(
        token[contains],
        "target_labels",
        forward=lambda _contains, labels: labels.long(),
        label=True,
    )

    if random_seed is None:
            model = FrozenBackboneGenerationHead(
                backbone=backbone,
                label_count=bundle.vocabulary.label_count,
                pad_size=pad_size,
                label_to_token_id=label_token_id_map(bundle.vocabulary),
            )
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(random_seed)
            model = FrozenBackboneGenerationHead(
                backbone=backbone,
                label_count=bundle.vocabulary.label_count,
                pad_size=pad_size,
                label_to_token_id=label_token_id_map(bundle.vocabulary),
            )
    token[generated_token] = ModuleLearner(
        token[contains],
        text["instruction_tokens"],
        "target_labels",
        module=model,
    )

    def is_before_edges(*_args, arg1, arg2):
        return arg1.getAttribute("token_index") < arg2.getAttribute("token_index")

    is_before_rel[first_token.reversed, second_token.reversed] = CompositionCandidateSensor(
        relations=(first_token.reversed, second_token.reversed),
        forward=is_before_edges,
    )

    if constrained_decoding:
        # The learning path reports the DFA too; hard masking during the
        # trainable head forward pass is intentionally left to the simple
        # decoder demo so gradients remain easy to inspect.
        model.constrained_dfa = dfa

    program = PrimalDualProgram(
        graph,
        SolverModel,
        poi=(text, token, is_before_rel),
        inferTypes=["local/argmax"],
        loss=MacroAverageTracker(NBCrossEntropyLoss()),
        beta=10,
        device="cpu",
        tnorm="P",
        counting_tnorm="P",
    )

    return LearningArtifacts(
        program=program,
        graph=graph,
        bundle=bundle,
        tokenizer=tokenizer,
        model=model,
        sample_data=make_sample_data(tokenizer),
        dfa=dfa,
    )


def make_optimizers(artifacts: LearningArtifacts, lr: float = 1e-2) -> LearningOptimizers:
    """Create optimizers for the trainable head and PMD constraint model."""
    return LearningOptimizers(
        head=torch.optim.Adam((p for p in artifacts.model.parameters() if p.requires_grad), lr=lr),
        constraints=torch.optim.Adam(artifacts.program.cmodel.parameters(), lr=lr),
    )


def run_one_training_step(
    artifacts: LearningArtifacts,
    lr: float = 1e-2,
    optimizers: LearningOptimizers | None = None,
    *,
    supervised_weight: float = 1.0,
    constraint_weight: float = 1.0,
) -> dict[str, float]:
    """Run one PMD-style optimization step and return loss scalars."""
    if supervised_weight < 0:
        raise ValueError("supervised_weight must be non-negative")
    if constraint_weight < 0:
        raise ValueError("constraint_weight must be non-negative")

    optimizers = optimizers or make_optimizers(artifacts, lr=lr)
    optimizers.head.zero_grad()
    optimizers.constraints.zero_grad()

    model_loss, _, *output = artifacts.program.model(artifacts.sample_data)
    constraint_loss, *_ = artifacts.program.cmodel(output[1])
    total = supervised_weight * model_loss
    if torch.is_tensor(constraint_loss):
        total = total + constraint_weight * constraint_loss

    if torch.is_tensor(total) and total.requires_grad:
        total.backward()
        optimizers.head.step()
        optimizers.constraints.step()

    return {
        "model_loss": float(model_loss.detach().item()) if torch.is_tensor(model_loss) else float(model_loss or 0.0),
        "constraint_loss": (
            float(constraint_loss.detach().item())
            if torch.is_tensor(constraint_loss)
            else float(constraint_loss or 0.0)
        ),
        "total_loss": float(total.detach().item()) if torch.is_tensor(total) else float(total or 0.0),
    }
