from __future__ import annotations

import importlib
from types import SimpleNamespace

import torch

from domiknows.generation import (
    ConstraintBundle,
    GenerationCandidate,
    HybridController,
    HybridScoreWeights,
    LatentLossBreakdown,
    ManualConstraintSelector,
    discover_generation_enforcement,
    eos_closure_dfa,
    forbidden_token_dfa,
    product_dfa,
    required_token_dfa,
)


def import_hf_module(name: str):
    return importlib.import_module(f"Tasks.hf_generation.{name}")


class StepwiseScorer(torch.nn.Module):
    def __init__(self, logits_by_prefix, label_to_token_id):
        super().__init__()
        self.logits_by_prefix = {tuple(key): torch.tensor(value, dtype=torch.float32) for key, value in logits_by_prefix.items()}
        self.label_to_token_id = tuple(label_to_token_id)
        self.label_count = len(label_to_token_id)

    def token_id_for_label(self, label):
        token_id = self.label_to_token_id[int(label)]
        if token_id is None:
            raise ValueError("not emittable")
        return int(token_id)

    def next_label_logits(self, input_ids):
        ids = tuple(int(item) for item in input_ids.reshape(-1).tolist())
        return self.logits_by_prefix.get(ids, torch.zeros(self.label_count))


class SequenceScorer(torch.nn.Module):
    def __init__(self, label_count: int):
        super().__init__()
        self.label_count = label_count

    def sequence_log_probs(self, labels):
        labels = labels.long().reshape(-1)
        rows = torch.full((labels.numel(), self.label_count), -5.0)
        for index, label in enumerate(labels.tolist()):
            rows[index, int(label)] = -0.1
        return torch.log_softmax(rows, dim=-1)


def build_controller(scorer=None, *, enforcement=None, weights=None):
    graph_module = import_hf_module("graph")
    mock_hf = import_hf_module("mock_hf")
    tokenizer = mock_hf.MockTokenizer()
    graph, bundle = graph_module.build_generation_graph(tokenizer)
    discovered = discover_generation_enforcement(graph, bundle, on_unsupported="error")
    dfa = discovered.dfa
    scorer = scorer or SequenceScorer(bundle.vocabulary.label_count)
    return HybridController(
        vocabulary=bundle.vocabulary,
        dfa=dfa,
        scorer_head=scorer,
        enforcement=enforcement,
        tokenizer=tokenizer,
        weights=weights,
        constraints=discovered.dfa_constraints,
    ), tokenizer


def test_valid_candidates_rank_above_rejected_by_default():
    controller, tokenizer = build_controller()
    prompt_ids = tokenizer("Once", return_tensors="pt").input_ids

    ranked = controller.rerank_candidates(
        prompt_ids,
        [
            GenerationCandidate(text=" The dog<eos>", token_ids=[1, 4, 0], labels=[1, 4, 0]),
            GenerationCandidate(text=" The cat<eos>", token_ids=[1, 2, 0], labels=[1, 2, 0]),
        ],
        keep_rejected=True,
        explain=True,
    )

    assert ranked[0].score.accepted
    assert ranked[0].candidate.labels == [1, 2, 0]
    assert not ranked[1].score.accepted
    assert ranked[1].score.rejection


def test_stepwise_compact_head_scoring_path_is_used():
    graph_module = import_hf_module("graph")
    mock_hf = import_hf_module("mock_hf")
    tokenizer = mock_hf.MockTokenizer()
    _graph, bundle = graph_module.build_generation_graph(tokenizer)
    scorer = StepwiseScorer(
        {
            (5,): [-5, 5, 1, 0, -5, -5],
            (5, 1): [-5, -5, 5, 0, -5, -5],
            (5, 1, 2): [5, -5, -5, 0, -5, -5],
        },
        [0, 1, 2, 3, 4, None],
    )
    controller, _tokenizer = build_controller(scorer)
    prompt_ids = tokenizer("Once", return_tensors="pt").input_ids

    good = controller.score_candidate(prompt_ids, GenerationCandidate(text=" The cat<eos>", token_ids=[1, 2, 0], labels=[1, 2, 0]))
    poor = controller.score_candidate(prompt_ids, GenerationCandidate(text=" cat The<eos>", token_ids=[2, 1, 0], labels=[2, 1, 0]))

    assert good.head_logprob > poor.head_logprob


def test_latent_preference_weight_can_change_ranking():
    def latent_breakdown(probs, **_kwargs):
        # Penalize " mat" probability; candidates without mat should rank higher
        # when latent preference is weighted strongly.
        return LatentLossBreakdown(total=probs[:, 3].mean(), items=())

    enforcement = SimpleNamespace(latent_specs=(object(),), latent_breakdown=latent_breakdown)
    controller, tokenizer = build_controller(
        enforcement=enforcement,
        weights=HybridScoreWeights(head_logprob=0.0, validity=10.0, latent_preference=5.0, risk=0.0),
    )
    prompt_ids = tokenizer("Once", return_tensors="pt").input_ids

    ranked = controller.rerank_candidates(
        prompt_ids,
        [
            GenerationCandidate(text=" The cat mat<eos>", token_ids=[1, 2, 3, 0], labels=[1, 2, 3, 0]),
            GenerationCandidate(text=" The cat<eos>", token_ids=[1, 2, 0], labels=[1, 2, 0]),
        ],
        keep_rejected=True,
    )

    assert ranked[0].candidate.labels == [1, 2, 0]


def test_failure_risk_increases_when_scorer_prefers_dfa_blocked_label():
    scorer = StepwiseScorer({(5,): [-5, 0, 0, 0, 8, -5]}, [0, 1, 2, 3, 4, None])
    controller, tokenizer = build_controller(scorer, weights=HybridScoreWeights(risk=1.0))
    prompt_ids = tokenizer("Once", return_tensors="pt").input_ids

    risk = controller.predict_failure_risk(prompt_ids, [])

    assert risk > 0.9


def test_repair_suggestions_find_missing_and_forbidden_tokens():
    controller, tokenizer = build_controller()
    prompt_ids = tokenizer("Once", return_tensors="pt").input_ids

    repair = controller.suggest_repair(
        GenerationCandidate(text=" The dog<eos>", token_ids=[1, 4, 0], labels=[1, 4, 0]),
        prompt_ids=prompt_ids,
    )

    assert not repair["accepted"]
    assert any("add ' cat'" in item for item in repair["suggestions"])


def test_constraint_selector_returns_expected_named_bundle():
    controller, _tokenizer = build_controller(
        weights=HybridScoreWeights(),
    )
    controller.constraint_selector = ManualConstraintSelector({"strict": "strict"}, default="loose")
    vocab = controller.vocabulary
    loose_dfa = product_dfa([eos_closure_dfa(vocab), required_token_dfa(vocab, " cat")])
    strict_dfa = product_dfa(
        [
            eos_closure_dfa(vocab),
            required_token_dfa(vocab, " cat"),
            forbidden_token_dfa(vocab, " dog"),
        ]
    )
    bundles = [
        ConstraintBundle("loose", loose_dfa),
        ConstraintBundle("strict", strict_dfa),
    ]

    selected = controller.select_constraints("use strict constraints", bundles)

    assert selected.name == "strict"
