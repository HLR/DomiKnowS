"""Graph-HMM learning helpers for the package shipping demo."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from domiknows.generation.learners import DomiKnowSAwareHMM, GraphHMMGenerationHead
from domiknows.generation.planning import PlanningBundle, encode_plan, planning_hmm_masks_from_graph


@dataclass
class ShippingHMMArtifacts:
    """Objects produced by fitting the graph-aware HMM."""

    hmm: DomiKnowSAwareHMM
    transition_mask: torch.Tensor
    emission_mask: torch.Tensor


def fit_graph_hmm(
    bundle: PlanningBundle,
    *,
    max_iter: int = 20,
    random_seed: int = 0,
) -> ShippingHMMArtifacts:
    """Fit a DomiKnowS-aware HMM from graph-declared reference plans."""

    transition_mask, emission_mask = planning_hmm_masks_from_graph(bundle)
    hmm = DomiKnowSAwareHMM(
        graph=bundle.graph,
        n_hidden_states=len(bundle.phase_names),
        transition_mask=transition_mask,
        emission_mask=emission_mask,
        symbols=bundle.action_names,
        state_names=bundle.phase_names,
        random_seed=random_seed,
    )
    hmm.fit(list(bundle.reference_plans.values()), max_iter=max_iter)
    return ShippingHMMArtifacts(hmm=hmm, transition_mask=transition_mask, emission_mask=emission_mask)


def build_graph_hmm_head(
    bundle: PlanningBundle,
    *,
    pad_size: int | None = None,
    random_seed: int = 0,
) -> GraphHMMGenerationHead:
    """Build a PMD-compatible graph-HMM head from the declarative graph."""

    transition_mask, emission_mask = planning_hmm_masks_from_graph(bundle)
    return GraphHMMGenerationHead(
        graph=bundle.graph,
        n_hidden_states=len(bundle.phase_names),
        label_count=len(bundle.action_names),
        symbols=bundle.action_names,
        state_names=bundle.phase_names,
        transition_mask=transition_mask,
        emission_mask=emission_mask,
        pad_size=pad_size or max(len(plan) for plan in bundle.reference_plans.values()),
        label_to_token_id=tuple(range(len(bundle.action_names))),
        random_seed=random_seed,
    )


def supervised_head_loss(head: GraphHMMGenerationHead, bundle: PlanningBundle, plan=None) -> torch.Tensor:
    """Return a teacher-forced NLL for one graph-declared shipping plan."""

    action_plan = tuple(plan or bundle.selected_reference_plan)
    labels = torch.tensor(encode_plan(bundle, action_plan), dtype=torch.long)
    log_probs = head(None, torch.zeros((1, 1), dtype=torch.long), labels)
    chosen = log_probs.gather(1, labels.view(-1, 1)).squeeze(1)
    return -chosen.mean()


def run_one_head_step(
    head: GraphHMMGenerationHead,
    bundle: PlanningBundle,
    *,
    lr: float = 0.05,
) -> dict[str, float]:
    """Run one differentiable update on the graph-HMM head."""

    optimizer = torch.optim.Adam((p for p in head.parameters() if p.requires_grad), lr=lr)
    optimizer.zero_grad()
    loss = supervised_head_loss(head, bundle)
    loss.backward()
    optimizer.step()
    return {"model_loss": float(loss.detach().cpu())}
