from __future__ import annotations

import torch

from domiknows.generation import (
    PromptConditionedHMMGenerationHead,
    PromptConditionedSpectralWFAGenerationHead,
    hmm_sequence_nll,
    wfa_sequence_energy_loss,
)
from Tasks.hf_generation.mock_hf import MockFrozenBackbone


def _head_kwargs():
    return {
        "label_count": 5,
        "state_count": 3,
        "pad_size": 4,
        "label_to_token_id": (0, 1, 2, 3, 4),
        "prompt_vocab_size": 8,
        "prompt_hidden_size": 6,
        "random_seed": 7,
    }


def test_prompt_conditioned_hmm_returns_normalized_log_probs():
    head = PromptConditionedHMMGenerationHead(**_head_kwargs())

    log_probs = head(None, torch.tensor([[5]]), torch.tensor([1, 2, 3, 0]))

    assert log_probs.shape == (4, 5)
    assert torch.isfinite(log_probs).all()
    assert torch.allclose(log_probs.exp().sum(dim=-1), torch.ones(4), atol=1e-5)


def test_prompt_conditioned_wfa_returns_finite_log_probs():
    head = PromptConditionedSpectralWFAGenerationHead(**_head_kwargs())

    log_probs = head(None, torch.tensor([[5]]), torch.tensor([1, 2, 3, 0]))

    assert log_probs.shape == (4, 5)
    assert torch.isfinite(log_probs).all()
    assert torch.allclose(log_probs.exp().sum(dim=-1), torch.ones(4), atol=1e-5)


def test_different_prompts_change_hmm_initial_state_and_logits():
    head = PromptConditionedHMMGenerationHead(**_head_kwargs())

    state_a = head.prompt_initial_probs(torch.tensor([[5]]))
    state_b = head.prompt_initial_probs(torch.tensor([[6]]))
    logits_a = head.next_label_logits(torch.tensor([5, 1]))
    logits_b = head.next_label_logits(torch.tensor([6, 1]))

    assert not torch.allclose(state_a, state_b)
    assert not torch.allclose(logits_a, logits_b)


def test_different_prompts_change_wfa_initial_state_and_logits():
    head = PromptConditionedSpectralWFAGenerationHead(**_head_kwargs())

    state_a = head.prompt_initial_state(torch.tensor([[5]]))
    state_b = head.prompt_initial_state(torch.tensor([[6]]))
    logits_a = head.next_label_logits(torch.tensor([5, 1]))
    logits_b = head.next_label_logits(torch.tensor([6, 1]))

    assert not torch.allclose(state_a, state_b)
    assert not torch.allclose(logits_a, logits_b)


def test_frozen_backbone_prompt_encoder_keeps_backbone_frozen():
    backbone = MockFrozenBackbone(vocab_size=8, hidden_size=6)
    head = PromptConditionedHMMGenerationHead(
        label_count=5,
        state_count=3,
        pad_size=4,
        label_to_token_id=(0, 1, 2, 3, 4),
        prompt_encoder_type="frozen_backbone",
        backbone=backbone,
        trainable=True,
    )

    assert all(not parameter.requires_grad for parameter in backbone.parameters())
    names = head.trainable_parameter_names()
    assert "initial_projector.weight" in names
    assert "transition_logits" in names
    assert not any(name.startswith("prompt_encoder.backbone") for name in names)


def test_embedding_prompt_encoder_is_trainable_in_offline_mode():
    head = PromptConditionedSpectralWFAGenerationHead(**_head_kwargs())

    names = head.trainable_parameter_names()

    assert "prompt_encoder.embedding.weight" in names
    assert "initial_projector.weight" in names
    assert "transitions" in names


def test_prompt_conditioned_losses_accept_instruction_tokens():
    hmm = PromptConditionedHMMGenerationHead(**_head_kwargs())
    wfa = PromptConditionedSpectralWFAGenerationHead(**_head_kwargs())
    labels = torch.tensor([1, 2, 3, 0])
    prompt = torch.tensor([[5]])

    hmm_loss = hmm_sequence_nll(hmm, labels, instruction_tokens=prompt)
    wfa_loss = wfa_sequence_energy_loss(wfa, labels, instruction_tokens=prompt)
    (hmm_loss + wfa_loss).backward()

    assert torch.isfinite(hmm_loss)
    assert torch.isfinite(wfa_loss)
    assert any(parameter.grad is not None for parameter in hmm.parameters() if parameter.requires_grad)
    assert any(parameter.grad is not None for parameter in wfa.parameters() if parameter.requires_grad)


def test_gated_hmm_returns_normalized_dynamics_weights_and_matrices():
    head = PromptConditionedHMMGenerationHead(
        **_head_kwargs(),
        dynamics_conditioning="gated",
        dynamics_expert_count=3,
    )

    weights = head.prompt_dynamics_weights(torch.tensor([[5]]))
    transition = head.prompt_transition_probs(torch.tensor([[5]]))
    emission = head.prompt_emission_probs(torch.tensor([[5]]))
    log_probs = head(None, torch.tensor([[5]]), torch.tensor([1, 2, 3, 0]))

    assert weights.shape == (3,)
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5)
    assert transition.shape == (3, 3)
    assert emission.shape == (3, 5)
    assert torch.allclose(transition.sum(dim=-1), torch.ones(3), atol=1e-5)
    assert torch.allclose(emission.sum(dim=-1), torch.ones(3), atol=1e-5)
    assert torch.isfinite(log_probs).all()


def test_gated_hmm_prompts_change_transition_and_emission_matrices():
    head = PromptConditionedHMMGenerationHead(
        **_head_kwargs(),
        dynamics_conditioning="gated",
        dynamics_expert_count=3,
    )

    transition_a = head.prompt_transition_probs(torch.tensor([[5]]))
    transition_b = head.prompt_transition_probs(torch.tensor([[6]]))
    emission_a = head.prompt_emission_probs(torch.tensor([[5]]))
    emission_b = head.prompt_emission_probs(torch.tensor([[6]]))

    assert not torch.allclose(transition_a, transition_b)
    assert not torch.allclose(emission_a, emission_b)


def test_gated_hmm_single_expert_matches_initial_only_base_dynamics():
    head = PromptConditionedHMMGenerationHead(
        **_head_kwargs(),
        dynamics_conditioning="gated",
        dynamics_expert_count=1,
    )

    assert torch.allclose(head.prompt_dynamics_weights(torch.tensor([[5]])), torch.ones(1))
    assert torch.allclose(head.prompt_transition_probs(torch.tensor([[5]])), head.transition_probs)
    assert torch.allclose(head.prompt_emission_probs(torch.tensor([[5]])), head.emission_probs)


def test_none_dynamics_uses_initial_only_base_dynamics():
    head = PromptConditionedHMMGenerationHead(**_head_kwargs(), dynamics_conditioning="none")

    assert torch.allclose(head.prompt_dynamics_weights(torch.tensor([[5]])), torch.ones(1))
    assert torch.allclose(head.prompt_transition_probs(torch.tensor([[5]])), head.transition_probs)
    assert torch.allclose(head.prompt_emission_probs(torch.tensor([[5]])), head.emission_probs)


def test_gated_wfa_returns_finite_signed_dynamics():
    head = PromptConditionedSpectralWFAGenerationHead(
        **_head_kwargs(),
        dynamics_conditioning="gated",
        dynamics_expert_count=3,
    )

    weights = head.prompt_dynamics_weights(torch.tensor([[5]]))
    transitions = head.prompt_transitions(torch.tensor([[5]]))
    final = head.prompt_final(torch.tensor([[5]]))
    log_probs = head(None, torch.tensor([[5]]), torch.tensor([1, 2, 3, 0]))

    assert weights.shape == (3,)
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5)
    assert transitions.shape == (5, 3, 3)
    assert final.shape == (3,)
    assert torch.isfinite(transitions).all()
    assert torch.isfinite(final).all()
    assert torch.isfinite(log_probs).all()


def test_gated_wfa_prompts_change_transition_and_final_scores():
    head = PromptConditionedSpectralWFAGenerationHead(
        **_head_kwargs(),
        dynamics_conditioning="gated",
        dynamics_expert_count=3,
    )

    transitions_a = head.prompt_transitions(torch.tensor([[5]]))
    transitions_b = head.prompt_transitions(torch.tensor([[6]]))
    final_a = head.prompt_final(torch.tensor([[5]]))
    final_b = head.prompt_final(torch.tensor([[6]]))

    assert not torch.allclose(transitions_a, transitions_b)
    assert not torch.allclose(final_a, final_b)


def test_step_adaptive_hmm_returns_finite_log_probs_and_prefix_changes_dynamics():
    head = PromptConditionedHMMGenerationHead(
        **_head_kwargs(),
        dynamics_conditioning="gated",
        dynamics_expert_count=3,
        step_dynamics_conditioning="prefix_gated",
    )

    log_probs = head(None, torch.tensor([[5]]), torch.tensor([1, 2, 3, 0]))
    weights_empty = head.step_dynamics_weights(torch.tensor([[5]]), [])
    weights_prefix = head.step_dynamics_weights(torch.tensor([[5]]), [1, 2])
    transition_empty = head.step_transition_probs(torch.tensor([[5]]), [])
    transition_prefix = head.step_transition_probs(torch.tensor([[5]]), [1, 2])
    emission_empty = head.step_emission_probs(torch.tensor([[5]]), [])
    emission_prefix = head.step_emission_probs(torch.tensor([[5]]), [1, 2])

    assert log_probs.shape == (4, 5)
    assert torch.isfinite(log_probs).all()
    assert torch.allclose(log_probs.exp().sum(dim=-1), torch.ones(4), atol=1e-5)
    assert weights_empty.shape == (3,)
    assert torch.allclose(weights_empty.sum(), torch.tensor(1.0), atol=1e-5)
    assert not torch.allclose(weights_empty, weights_prefix)
    assert not torch.allclose(transition_empty, transition_prefix)
    assert not torch.allclose(emission_empty, emission_prefix)


def test_step_adaptive_wfa_returns_finite_log_probs_and_prefix_changes_dynamics():
    head = PromptConditionedSpectralWFAGenerationHead(
        **_head_kwargs(),
        dynamics_conditioning="gated",
        dynamics_expert_count=3,
        step_dynamics_conditioning="prefix_gated",
    )

    log_probs = head(None, torch.tensor([[5]]), torch.tensor([1, 2, 3, 0]))
    weights_empty = head.step_dynamics_weights(torch.tensor([[5]]), [])
    weights_prefix = head.step_dynamics_weights(torch.tensor([[5]]), [1, 2])
    transitions_empty = head.step_transitions(torch.tensor([[5]]), [])
    transitions_prefix = head.step_transitions(torch.tensor([[5]]), [1, 2])
    final_empty = head.step_final(torch.tensor([[5]]), [])
    final_prefix = head.step_final(torch.tensor([[5]]), [1, 2])

    assert log_probs.shape == (4, 5)
    assert torch.isfinite(log_probs).all()
    assert torch.allclose(log_probs.exp().sum(dim=-1), torch.ones(4), atol=1e-5)
    assert weights_empty.shape == (3,)
    assert torch.allclose(weights_empty.sum(), torch.tensor(1.0), atol=1e-5)
    assert not torch.allclose(weights_empty, weights_prefix)
    assert not torch.allclose(transitions_empty, transitions_prefix)
    assert not torch.allclose(final_empty, final_prefix)


def test_step_adaptive_weights_change_with_prompt_and_next_logits_change_with_prefix():
    head = PromptConditionedHMMGenerationHead(
        **_head_kwargs(),
        dynamics_conditioning="gated",
        dynamics_expert_count=3,
        step_dynamics_conditioning="prefix_gated",
    )

    weights_a = head.step_dynamics_weights(torch.tensor([[5]]), [1])
    weights_b = head.step_dynamics_weights(torch.tensor([[6]]), [1])
    logits_a = head.next_label_logits(torch.tensor([5, 1]))
    logits_b = head.next_label_logits(torch.tensor([5, 1, 2]))

    assert not torch.allclose(weights_a, weights_b)
    assert not torch.allclose(logits_a, logits_b)


def test_step_adaptive_requires_gated_dynamics():
    try:
        PromptConditionedHMMGenerationHead(
            **_head_kwargs(),
            dynamics_conditioning="none",
            step_dynamics_conditioning="prefix_gated",
        )
    except ValueError as exc:
        assert "requires dynamics_conditioning='gated'" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_step_adaptive_single_expert_is_valid_noop():
    head = PromptConditionedHMMGenerationHead(
        **_head_kwargs(),
        dynamics_conditioning="gated",
        dynamics_expert_count=1,
        step_dynamics_conditioning="prefix_gated",
    )

    assert torch.allclose(head.step_dynamics_weights(torch.tensor([[5]]), [1, 2]), torch.ones(1))
    assert torch.allclose(head.step_transition_probs(torch.tensor([[5]]), [1]), head.transition_probs)
    assert torch.allclose(head.step_emission_probs(torch.tensor([[5]]), [1]), head.emission_probs)
