"""Head-only pairwise training and completed-plan reranking."""

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_sequence

from .features import ACTIONS, PROGRESS, StepFeatures, candidate_features


FORMAT = "shared-plan-reranker-v1"
REGRESSION_RULE = "prefer_preserving_known_public_goal_literals"
LATENT_WEIGHT = 0.5
RESIDUAL_LIMIT = 4.


@lru_cache(maxsize=1)
def goal_transition_potential():
    """Compile finite log factors for transitions between public truth values."""
    from domiknows.generation import HMMFactorGraphEncoder, discover_transition_potentials
    from domiknows.graph.logicalConstrain import ifL, notL

    names = ("unknown", "violated", "satisfied")
    graph, bundle = HMMFactorGraphEncoder(
        ["<eos>"], eos_token="<eos>", state_names=names, include_dp_factors=False,
        clear_graph=False, graph_name="shared_reranker_public_goal",
    ).build_graph()
    ctx = bundle.context
    with graph:
        ifL(ctx.is_next_rel("edge"), ifL(
            ctx.latent_state_value("satisfied", "before", path=("edge", ctx.current_token)),
            notL(ctx.latent_state_value("violated", "after", path=("edge", ctx.next_token))),
        ), name=REGRESSION_RULE)
    compiled, = discover_transition_potentials(
        graph, bundle, strength=math.exp(-LATENT_WEIGHT), on_unsupported="error"
    )
    return tuple(compiled.tensor_for(torch.ones((len(names), len(names)), dtype=torch.float64))
                 .log().flatten().tolist())


@dataclass(frozen=True)
class PlanFeatures:
    """Same frozen baseline score and observable trace in training and inference."""

    log_probability: float
    steps: tuple[StepFeatures, ...]

    def __post_init__(self):
        if not math.isfinite(self.log_probability) or not self.steps or self.steps[0].action != "start":
            raise ValueError("expected finite planner score and trace with an initial state")


@dataclass(frozen=True)
class ReplayPool:
    """Independent replay outcomes are targets alongside, outside the feature tensors."""

    episode_id: str
    candidates: tuple[PlanFeatures, ...]
    outcomes: tuple[bool | None, ...]


def _row(step):
    return (*[step.action == name for name in ACTIONS], *step.roles, *step.before, *step.after,
            *[step.progress == name for name in PROGRESS])


class SharedPlanReranker(nn.Module):
    """Add a bounded learned correction and a compiled soft preference to each plan."""

    def __init__(self, hidden_size=32):
        super().__init__()
        self.hidden_size = hidden_size
        self.trace_encoder = nn.GRU(len(ACTIONS) + 8 + 6 + len(PROGRESS), hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.register_buffer("potential", torch.tensor(goal_transition_potential()), persistent=False)

    def components(self, candidates):
        if not candidates:
            raise ValueError("expected at least one legal candidate")
        rows = [self.potential.new_tensor([_row(step) for step in candidate.steps])
                for candidate in candidates]
        _, hidden = self.trace_encoder(pack_sequence(rows, enforce_sorted=False))
        learned = RESIDUAL_LIMIT * self.output(hidden[0]).squeeze(-1).tanh()
        counts = self.potential.new_tensor([
            [sum(column) for column in zip(*(step.transitions for step in candidate.steps))]
            for candidate in candidates
        ])
        latent = counts @ self.potential
        baseline = self.potential.new_tensor([candidate.log_probability for candidate in candidates],
                                             dtype=torch.float64)
        return baseline, learned, latent

    def forward(self, candidates):
        baseline, learned, latent = self.components(candidates)
        return baseline + learned + latent

    def save(self, path):
        with Path(path).open("xb") as stream:
            torch.save({"format": FORMAT, "hidden_size": self.hidden_size, "weights": self.state_dict(),
                        "potential": goal_transition_potential()}, stream)

    @classmethod
    def load(cls, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format") != FORMAT or payload.get("potential") != goal_transition_potential():
            raise ValueError("shared reranker checkpoint schema or compiled preference differs")
        head = cls(payload["hidden_size"])
        head.load_state_dict(payload["weights"], strict=True)
        return head.eval()


@dataclass(frozen=True)
class RerankDecision:
    index: int
    planner_score: float
    learned_residual: float
    latent_preference: float

    @property
    def score(self):
        return self.planner_score + self.learned_residual + self.latent_preference


def encode_candidates(candidates, policy, vocabulary, *, goal=None):
    """Build the same feature records for frozen replay training and live ranking."""
    options = {} if goal is None else {"goal": goal}
    return tuple(PlanFeatures(candidate.log_probability, candidate_features(
        candidate.labels, policy, vocabulary, **options
    )) for candidate in candidates)


@torch.inference_mode()
def rerank_candidates(head, candidates, policy, vocabulary, *, goal=None):
    """Return ranked original beam indices. Every candidate must pass the active DFA."""
    features = encode_candidates(candidates, policy, vocabulary, goal=goal)
    parts = (component.tolist() for component in head.components(features))
    decisions = [RerankDecision(index, *scores) for index, scores in enumerate(zip(*parts))]
    if any(not math.isfinite(entry.score) for entry in decisions):
        raise ValueError("shared reranker produced a nonfinite score")
    return tuple(sorted(decisions, key=lambda entry: -entry.score))


def fit_shared_reranker(head, pools, *, source_train_ids, epochs=4, lr=1e-3, seed=42):
    """Fit only the shared head with within-episode replay success/failure pairs."""
    if epochs <= 0 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("expected positive training epochs and learning rate")
    groups, seen = [], set()
    for pool in pools:
        if pool.episode_id not in source_train_ids or pool.episode_id in seen:
            raise ValueError("duplicate or non-training episode in replay pools")
        seen.add(pool.episode_id)
        if (len(pool.candidates) != len(pool.outcomes)
                or any(value is not None and type(value) is not bool for value in pool.outcomes)):
            raise ValueError("expected an independent Boolean or unknown outcome per candidate")
        positive = [i for i, value in enumerate(pool.outcomes) if value is True]
        negative = [i for i, value in enumerate(pool.outcomes) if value is False]
        pairs = [(i, j) for i in positive for j in negative
                 if pool.candidates[i].steps != pool.candidates[j].steps]
        if pairs:
            groups.append((pool.candidates, pairs))
    if not groups:
        raise ValueError("no replay-labeled success/failure pairs with distinguishable observations")
    generator = random.Random(seed)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    head.train()
    history = []
    for _ in range(epochs):
        generator.shuffle(groups)
        total = 0.
        for candidates, pairs in groups:
            scores = head(candidates)
            positive, negative = zip(*pairs)
            loss = F.softplus(scores[list(negative)] - scores[list(positive)]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        history.append(total / len(groups))
    return {"loss": history, "episodes": len(groups), "pairs": sum(len(pairs) for _, pairs in groups)}
