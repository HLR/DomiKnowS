"""DomiKnowS PMD program using an explicit HMM factor graph."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from domiknows.generation import (
    HMMFactorGraphEncoder,
    HMMFactorGraphHead,
    constrained_label_greedy_decode,
    constraints_to_dfa,
    discover_generation_enforcement,
    hmm_factor_sequence_nll,
)
from domiknows.graph.logicalConstrain import ifL, notL
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.lossprogram import PrimalDualProgram
from domiknows.program.metric import MacroAverageTracker
from domiknows.program.model.pytorch import SolverModel
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from domiknows.sensor.pytorch.sensors import FunctionalSensor, JointSensor, ReaderSensor

try:
    from .graph import EOS_TOKEN, VOCAB, apply_graph_constraints
    from .learning_program import label_token_id_map, make_sample_data
    from .mock_hf import MockTokenizer
except ImportError:
    from graph import EOS_TOKEN, VOCAB, apply_graph_constraints
    from learning_program import label_token_id_map, make_sample_data
    from mock_hf import MockTokenizer


@dataclass
class HMMFactorArtifacts:
    """Objects produced by ``build_hmm_factor_program``."""

    program: PrimalDualProgram
    graph: object
    bundle: object
    tokenizer: MockTokenizer
    head: HMMFactorGraphHead
    generated_model: torch.nn.Module
    latent_model: torch.nn.Module
    sample_data: dict
    dfa: object


@dataclass
class HMMFactorOptimizers:
    """Optimizers for the shared HMM parameters and PMD constraints."""

    head: torch.optim.Optimizer
    constraints: torch.optim.Optimizer


def apply_hmm_factor_constraints(bundle) -> None:
    """Add toy latent-state constraints over adjacent token positions."""
    ctx = bundle.context

    # PER(t) => not LOC(t + 1)
    ifL(
        ctx.is_next_rel("next"),
        ifL(
            ctx.latent_state_value("PER", "x", path=("next", ctx.current_token)),
            notL(ctx.latent_state_value("LOC", "y", path=("next", ctx.next_token))),
        ),
    )

    # LOC(t) => generated token is " mat" in this tiny toy vocabulary.
    ifL(
        ctx.latent_state_value("LOC", "x"),
        ctx.token_value(" mat", "x"),
    )


def build_hmm_factor_graph(tokenizer=None, state_names=("PER", "O", "LOC")):
    """Build the hf_generation graph with generated-token and HMM latent nodes."""
    tokenizer = tokenizer or MockTokenizer()
    encoder = HMMFactorGraphEncoder(
        vocab=VOCAB,
        eos_token=EOS_TOKEN,
        tokenizer=tokenizer,
        state_names=state_names,
        graph_name="hf_generation_hmm_factor",
    )
    graph, bundle = encoder.build_graph()
    with graph:
        apply_graph_constraints(bundle)
        apply_hmm_factor_constraints(bundle)
    return graph, bundle


def target_labels_for_sample(artifacts: HMMFactorArtifacts) -> torch.Tensor:
    """Return padded compact generated-token labels for the demo sample."""
    target_ids = artifacts.sample_data["target_token_ids"][0][: artifacts.head.pad_size]
    labels = [artifacts.bundle.vocabulary.label_for_token_id(int(token_id)) for token_id in target_ids]
    if len(labels) < artifacts.head.pad_size:
        labels.extend([artifacts.bundle.vocabulary.eos_label] * (artifacts.head.pad_size - len(labels)))
    return torch.tensor(labels, dtype=torch.long)


def build_hmm_factor_program(
    *,
    pad_size: int = 4,
    state_names=("PER", "O", "LOC"),
    trainable: bool = True,
    random_seed: int = 0,
) -> HMMFactorArtifacts:
    """Build the opt-in HMM factor-graph PMD demo."""
    tokenizer = MockTokenizer()
    graph, bundle = build_hmm_factor_graph(tokenizer, state_names=state_names)
    # Latent-state constraints are for PMD/cmodel, not DFA token masking. The
    # DFA bridge should therefore ignore those non-token factor constraints.
    enforcement = discover_generation_enforcement(graph, bundle, on_unsupported="ignore")
    dfa = constraints_to_dfa(enforcement.dfa_constraints, bundle.vocabulary)

    text = bundle.text
    token = bundle.token
    contains = bundle.contains
    generated_token = bundle.generated_token
    latent_state = bundle.latent_state
    is_before_rel = bundle.is_before_rel
    first_token = bundle.first_token
    second_token = bundle.second_token
    is_next_rel = bundle.is_next_rel
    current_token = bundle.current_token
    next_token = bundle.next_token

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

    head = HMMFactorGraphHead(
        label_count=bundle.vocabulary.label_count,
        state_names=bundle.state_names,
        pad_size=pad_size,
        label_to_token_id=label_token_id_map(bundle.vocabulary),
        trainable=trainable,
        random_seed=random_seed,
    )
    generated_model = head.generated_module()
    latent_model = head.latent_module()
    token[generated_token] = ModuleLearner(
        token[contains],
        text["instruction_tokens"],
        "target_labels",
        module=generated_model,
    )
    token[latent_state] = ModuleLearner(
        token[contains],
        text["instruction_tokens"],
        "target_labels",
        module=latent_model,
    )

    def before_edges(*_args, **candidates):
        arg1, arg2 = _candidate_pair(candidates)
        return arg1.getAttribute("token_index") < arg2.getAttribute("token_index")

    def next_edges(*_args, **candidates):
        arg1, arg2 = _candidate_pair(candidates)
        return arg2.getAttribute("token_index") == arg1.getAttribute("token_index") + 1

    is_before_rel[first_token.reversed, second_token.reversed] = CompositionCandidateSensor(
        relations=(first_token.reversed, second_token.reversed),
        forward=before_edges,
    )
    is_next_rel[current_token.reversed, next_token.reversed] = CompositionCandidateSensor(
        relations=(current_token.reversed, next_token.reversed),
        forward=next_edges,
    )

    program = PrimalDualProgram(
        graph,
        SolverModel,
        poi=(text, token, is_before_rel, is_next_rel),
        inferTypes=["local/argmax"],
        loss=MacroAverageTracker(NBCrossEntropyLoss()),
        beta=10,
        device="cpu",
        tnorm="P",
        counting_tnorm="P",
    )

    return HMMFactorArtifacts(
        program=program,
        graph=graph,
        bundle=bundle,
        tokenizer=tokenizer,
        head=head,
        generated_model=generated_model,
        latent_model=latent_model,
        sample_data=make_sample_data(tokenizer),
        dfa=dfa,
    )


def make_optimizers(artifacts: HMMFactorArtifacts, lr: float = 1e-2) -> HMMFactorOptimizers:
    """Create optimizers for shared HMM parameters and PMD constraints."""
    return HMMFactorOptimizers(
        head=torch.optim.Adam((p for p in artifacts.head.parameters() if p.requires_grad), lr=lr),
        constraints=torch.optim.Adam(artifacts.program.cmodel.parameters(), lr=lr),
    )


def run_one_hmm_factor_step(
    artifacts: HMMFactorArtifacts,
    lr: float = 1e-2,
    optimizers: HMMFactorOptimizers | None = None,
    *,
    supervised_weight: float = 1.0,
    constraint_weight: float = 1.0,
    hmm_weight: float = 1.0,
) -> dict[str, float]:
    """Run one optimization step with supervised, PMD, and HMM NLL losses."""
    optimizers = optimizers or make_optimizers(artifacts, lr=lr)
    optimizers.head.zero_grad()
    optimizers.constraints.zero_grad()

    model_loss, _, *output = artifacts.program.model(artifacts.sample_data)
    constraint_loss, *_ = artifacts.program.cmodel(output[1])
    hmm_loss = hmm_factor_sequence_nll(artifacts.head, target_labels_for_sample(artifacts))
    total = supervised_weight * model_loss + hmm_weight * hmm_loss
    if torch.is_tensor(constraint_loss):
        total = total + constraint_weight * constraint_loss

    if torch.is_tensor(total) and total.requires_grad:
        total.backward()
        optimizers.head.step()
        optimizers.constraints.step()

    return {
        "model_loss": _as_float(model_loss),
        "constraint_loss": _as_float(constraint_loss),
        "hmm_factor_nll": _as_float(hmm_loss),
        "total_loss": _as_float(total),
    }


def constrained_decode(artifacts: HMMFactorArtifacts):
    """Decode the generated-token projection with graph-discovered DFA enforcement."""
    return constrained_label_greedy_decode(
        artifacts.generated_model,
        artifacts.sample_data["instruction_tokens"],
        artifacts.bundle.vocabulary,
        artifacts.dfa,
        max_new_tokens=artifacts.head.pad_size,
    )


def _as_float(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().item())
    return float(value or 0.0)


def _candidate_pair(candidates: dict):
    arg1 = candidates.get("arg1") or candidates.get("arg1-1")
    arg2 = candidates.get("arg2") or candidates.get("arg2-1")
    if arg1 is None or arg2 is None:
        values = list(candidates.values())
        if len(values) < 2:
            raise ValueError("expected two token candidates")
        arg1, arg2 = values[0], values[1]
    return arg1, arg2
