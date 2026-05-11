"""Prompt-conditioned HMM/WFA learning helpers for hf_generation."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from domiknows.generation import (
    GenerationLossWeights,
    HMMGenerationHead,
    PromptConditionedHMMGenerationHead,
    PromptConditionedSpectralWFAGenerationHead,
    SpectralWFAGenerationHead,
    allowed_mass_loss,
    compute_generation_training_loss,
    constrained_label_greedy_decode,
    constraints_to_dfa,
    discover_generation_enforcement,
    hmm_sequence_nll,
    token_probs_from_log_probs,
    wfa_sequence_energy_loss,
)
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.lossprogram import PrimalDualProgram
from domiknows.program.metric import MacroAverageTracker
from domiknows.program.model.pytorch import SolverModel
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from domiknows.sensor.pytorch.sensors import FunctionalSensor, JointSensor, ReaderSensor

try:
    from .graph import build_generation_graph
    from .learning_program import _build_backbone, label_token_id_map, make_sample_data
    from .run_demo import generation_vocab_for_tokenizer
except ImportError:
    from graph import build_generation_graph
    from learning_program import _build_backbone, label_token_id_map, make_sample_data
    from run_demo import generation_vocab_for_tokenizer


PromptAutomataHead = PromptConditionedHMMGenerationHead | PromptConditionedSpectralWFAGenerationHead
BaselineAutomataHead = HMMGenerationHead | SpectralWFAGenerationHead


@dataclass
class PromptAutomataLearningArtifacts:
    """Objects produced by ``build_prompt_automata_learning_program``."""

    program: PrimalDualProgram
    graph: object
    bundle: object
    tokenizer: object
    model: PromptAutomataHead
    baseline_model: BaselineAutomataHead
    sample_data: dict
    dfa: object
    kind: str
    encoder_kind: str
    dynamics_conditioning: str
    step_dynamics_conditioning: str
    enforcement: object
    backbone: torch.nn.Module | None = None


@dataclass
class PromptAutomataLearningOptimizers:
    """Optimizers for the prompt-conditioned automata head and PMD model."""

    head: torch.optim.Optimizer
    constraints: torch.optim.Optimizer


def target_labels_for_sample(artifacts: PromptAutomataLearningArtifacts) -> torch.Tensor:
    """Return padded compact target labels for the demo sample."""
    target_ids = artifacts.sample_data["target_token_ids"][0][: artifacts.model.pad_size]
    labels = [artifacts.bundle.vocabulary.label_for_token_id(int(token_id)) for token_id in target_ids]
    if len(labels) < artifacts.model.pad_size:
        labels.extend([artifacts.bundle.vocabulary.eos_label] * (artifacts.model.pad_size - len(labels)))
    return torch.tensor(labels, dtype=torch.long)


def build_prompt_automata_learning_program(
    *,
    kind: str = "hmm",
    encoder_kind: str = "embedding",
    real_hf: bool = False,
    model_name: str = "roneneldan/TinyStories-1M",
    pad_size: int = 4,
    state_count: int = 3,
    dynamics_conditioning: str = "none",
    dynamics_expert_count: int = 2,
    step_dynamics_conditioning: str = "none",
    trainable: bool = True,
    random_seed: int = 0,
    quiet_transformers: bool = True,
    latent_mode: str = "marked",
) -> PromptAutomataLearningArtifacts:
    """Build a tiny PMD program with prompt-conditioned HMM/WFA head."""
    kind = kind.lower()
    if kind not in {"hmm", "wfa"}:
        raise ValueError("kind must be 'hmm' or 'wfa'")
    encoder_kind = encoder_kind.lower().replace("-", "_")
    if encoder_kind not in {"embedding", "frozen_backbone"}:
        raise ValueError("encoder_kind must be 'embedding' or 'frozen_backbone'")
    dynamics_conditioning = dynamics_conditioning.lower().replace("-", "_")
    if dynamics_conditioning not in {"none", "gated"}:
        raise ValueError("dynamics_conditioning must be 'none' or 'gated'")
    step_dynamics_conditioning = step_dynamics_conditioning.lower().replace("-", "_")
    if step_dynamics_conditioning not in {"none", "prefix_gated"}:
        raise ValueError("step_dynamics_conditioning must be 'none' or 'prefix_gated'")

    tokenizer, backbone = _build_backbone(real_hf, model_name, quiet_transformers)
    vocab, eos_token = generation_vocab_for_tokenizer(tokenizer, real_hf=real_hf)
    graph, bundle = build_generation_graph(tokenizer, vocab, eos_token=eos_token)
    enforcement = discover_generation_enforcement(graph, bundle, on_unsupported="error", latent_mode=latent_mode)
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
        if len(labels) < pad_size:
            labels.extend([bundle.vocabulary.eos_label] * (pad_size - len(labels)))
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

    common_kwargs = {
        "label_count": bundle.vocabulary.label_count,
        "state_count": state_count,
        "pad_size": pad_size,
        "label_to_token_id": label_token_id_map(bundle.vocabulary),
        "trainable": trainable,
        "random_seed": random_seed,
    }
    if kind == "hmm":
        baseline_model: BaselineAutomataHead = HMMGenerationHead(**common_kwargs)
        model_cls = PromptConditionedHMMGenerationHead
    else:
        baseline_model = SpectralWFAGenerationHead(**common_kwargs)
        model_cls = PromptConditionedSpectralWFAGenerationHead

    prompt_kwargs = dict(common_kwargs)
    if encoder_kind == "embedding":
        prompt_kwargs.update(
            prompt_encoder_type="embedding",
            prompt_vocab_size=_tokenizer_vocab_size(tokenizer),
            prompt_hidden_size=8,
        )
    else:
        prompt_kwargs.update(prompt_encoder_type="frozen_backbone", backbone=backbone)
    prompt_kwargs.update(
        dynamics_conditioning=dynamics_conditioning,
        dynamics_expert_count=dynamics_expert_count,
        step_dynamics_conditioning=step_dynamics_conditioning,
    )
    model: PromptAutomataHead = model_cls(**prompt_kwargs)

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

    return PromptAutomataLearningArtifacts(
        program=program,
        graph=graph,
        bundle=bundle,
        tokenizer=tokenizer,
        model=model,
        baseline_model=baseline_model,
        sample_data=make_sample_data(tokenizer),
        dfa=dfa,
        kind=kind,
        encoder_kind=encoder_kind,
        dynamics_conditioning=dynamics_conditioning,
        step_dynamics_conditioning=step_dynamics_conditioning,
        enforcement=enforcement,
        backbone=backbone,
    )


def make_optimizers(
    artifacts: PromptAutomataLearningArtifacts,
    lr: float = 1e-2,
) -> PromptAutomataLearningOptimizers:
    """Create optimizers for the trainable prompt-conditioned head and PMD model."""
    trainable_params = [p for p in artifacts.model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("prompt-conditioned automata head has no trainable parameters")
    return PromptAutomataLearningOptimizers(
        head=torch.optim.Adam(trainable_params, lr=lr),
        constraints=torch.optim.Adam(artifacts.program.cmodel.parameters(), lr=lr),
    )


def prompt_automata_auxiliary_loss(artifacts: PromptAutomataLearningArtifacts) -> torch.Tensor:
    """Return the prompt-conditioned HMM/WFA supervised auxiliary loss."""
    labels = target_labels_for_sample(artifacts)
    instruction_tokens = artifacts.sample_data["instruction_tokens"]
    if isinstance(artifacts.model, PromptConditionedHMMGenerationHead):
        return hmm_sequence_nll(artifacts.model, labels, instruction_tokens=instruction_tokens)
    return wfa_sequence_energy_loss(artifacts.model, labels, instruction_tokens=instruction_tokens)


def run_one_prompt_automata_training_step(
    artifacts: PromptAutomataLearningArtifacts,
    lr: float = 1e-2,
    optimizers: PromptAutomataLearningOptimizers | None = None,
    *,
    supervised_weight: float = 1.0,
    constraint_weight: float = 1.0,
    automata_weight: float = 1.0,
    latent_weight: float = 0.0,
    allowed_mass_weight: float = 0.0,
    latent_diagnostics: bool = False,
) -> dict[str, float]:
    """Run one PMD-style step with prompt-conditioned automata auxiliary loss."""
    weights = GenerationLossWeights(
        supervised=supervised_weight,
        pmd=constraint_weight,
        latent=latent_weight,
        allowed_mass=allowed_mass_weight,
        automata=automata_weight,
    )

    optimizers = optimizers or make_optimizers(artifacts, lr=lr)
    optimizers.head.zero_grad()
    optimizers.constraints.zero_grad()

    model_loss, _, *output = artifacts.program.model(artifacts.sample_data)
    constraint_loss, *_ = artifacts.program.cmodel(output[1])
    aux_loss = prompt_automata_auxiliary_loss(artifacts)
    labels = target_labels_for_sample(artifacts)
    log_probs = artifacts.model(None, artifacts.sample_data["instruction_tokens"], labels)
    probs = token_probs_from_log_probs(log_probs)
    latent_breakdown = artifacts.enforcement.latent_breakdown(
        probs,
        eos_label=artifacts.bundle.vocabulary.eos_label,
    )
    mass_loss = allowed_mass_loss(probs, artifacts.dfa) if allowed_mass_weight else probs.new_zeros(())
    breakdown = compute_generation_training_loss(
        supervised_loss=model_loss,
        pmd_loss=constraint_loss,
        latent_loss=latent_breakdown.total,
        allowed_mass_loss_value=mass_loss,
        automata_aux_loss=aux_loss,
        weights=weights,
        latent_items=latent_breakdown.items,
    )
    total = breakdown.total

    if torch.is_tensor(total) and total.requires_grad:
        total.backward()
        optimizers.head.step()
        optimizers.constraints.step()

    all_values = breakdown.as_float_dict()
    values = {
        "model_loss": all_values["model_loss"],
        "constraint_loss": all_values["constraint_loss"],
        "automata_aux_loss": all_values["automata_aux_loss"],
        "total_loss": all_values["total_loss"],
    }
    if latent_weight or latent_diagnostics:
        values["latent_loss"] = all_values["latent_loss"]
    if allowed_mass_weight:
        values["allowed_mass_loss"] = all_values["allowed_mass_loss"]
    if latent_diagnostics:
        values["latent_terms"] = float(len(latent_breakdown.items))
    return values


def constrained_decode(artifacts: PromptAutomataLearningArtifacts):
    """Decode the prompt-conditioned head with graph-discovered DFA enforcement."""
    return constrained_label_greedy_decode(
        artifacts.model,
        artifacts.sample_data["instruction_tokens"],
        artifacts.bundle.vocabulary,
        artifacts.dfa,
        max_new_tokens=artifacts.model.pad_size,
    )


def _tokenizer_vocab_size(tokenizer) -> int:
    if hasattr(tokenizer, "token_to_id"):
        return len(tokenizer.token_to_id)
    value = getattr(tokenizer, "vocab_size", None)
    if value is not None:
        return int(value)
    try:
        return len(tokenizer.get_vocab())
    except AttributeError:
        return 1024


def _as_float(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().item())
    return float(value or 0.0)
