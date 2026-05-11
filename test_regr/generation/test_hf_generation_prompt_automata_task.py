from __future__ import annotations

import importlib

import torch


def import_task_module(name: str):
    return importlib.import_module(f"Tasks.hf_generation.{name}")


def _build(
    kind: str = "hmm",
    encoder_kind: str = "embedding",
    dynamics_conditioning: str = "none",
    step_dynamics_conditioning: str = "none",
):
    program = import_task_module("prompt_automata_program")
    return program.build_prompt_automata_learning_program(
        kind=kind,
        encoder_kind=encoder_kind,
        dynamics_conditioning=dynamics_conditioning,
        dynamics_expert_count=3,
        step_dynamics_conditioning=step_dynamics_conditioning,
        pad_size=4,
        state_count=3,
    )


def test_prompt_automata_program_populates_pmd_constraint_loss():
    program = import_task_module("prompt_automata_program")
    artifacts = _build("hmm")

    model_loss, _, *output = artifacts.program.model(artifacts.sample_data)
    constraint_loss, *_ = artifacts.program.cmodel(output[1])
    aux_loss = program.prompt_automata_auxiliary_loss(artifacts)

    assert torch.isfinite(model_loss)
    assert torch.isfinite(constraint_loss)
    assert torch.isfinite(aux_loss)


def test_prompt_wfa_program_populates_pmd_constraint_loss():
    program = import_task_module("prompt_automata_program")
    artifacts = _build("wfa")

    model_loss, _, *output = artifacts.program.model(artifacts.sample_data)
    constraint_loss, *_ = artifacts.program.cmodel(output[1])
    aux_loss = program.prompt_automata_auxiliary_loss(artifacts)

    assert torch.isfinite(model_loss)
    assert torch.isfinite(constraint_loss)
    assert torch.isfinite(aux_loss)


def test_prompt_automata_step_updates_conditioning_parameters():
    program = import_task_module("prompt_automata_program")
    artifacts = _build("hmm")
    before = artifacts.model.initial_projector.weight.detach().clone()

    losses = program.run_one_prompt_automata_training_step(artifacts, lr=0.5)
    after = artifacts.model.initial_projector.weight.detach()

    assert set(losses) == {"model_loss", "constraint_loss", "automata_aux_loss", "total_loss"}
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    assert not torch.allclose(before, after)


def test_gated_prompt_automata_step_updates_gate_parameters():
    program = import_task_module("prompt_automata_program")
    artifacts = _build("hmm", dynamics_conditioning="gated")
    before = artifacts.model.dynamics_gate.weight.detach().clone()

    program.run_one_prompt_automata_training_step(artifacts, lr=0.5)
    after = artifacts.model.dynamics_gate.weight.detach()

    assert artifacts.model.prompt_dynamics_weights(artifacts.sample_data["instruction_tokens"]).shape == (3,)
    assert not torch.allclose(before, after)


def test_gated_prompt_wfa_program_populates_constraint_loss():
    program = import_task_module("prompt_automata_program")
    artifacts = _build("wfa", dynamics_conditioning="gated")

    model_loss, _, *output = artifacts.program.model(artifacts.sample_data)
    constraint_loss, *_ = artifacts.program.cmodel(output[1])
    aux_loss = program.prompt_automata_auxiliary_loss(artifacts)

    assert torch.isfinite(model_loss)
    assert torch.isfinite(constraint_loss)
    assert torch.isfinite(aux_loss)


def test_step_adaptive_prompt_hmm_program_populates_constraint_loss_and_updates_step_gate():
    program = import_task_module("prompt_automata_program")
    artifacts = _build("hmm", dynamics_conditioning="gated", step_dynamics_conditioning="prefix_gated")
    before_gate = artifacts.model.step_dynamics_gate.weight.detach().clone()
    before_embedding = artifacts.model.prefix_embedding.weight.detach().clone()

    losses = program.run_one_prompt_automata_training_step(artifacts, lr=0.5)

    assert artifacts.step_dynamics_conditioning == "prefix_gated"
    assert set(losses) == {"model_loss", "constraint_loss", "automata_aux_loss", "total_loss"}
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    assert not torch.allclose(before_gate, artifacts.model.step_dynamics_gate.weight.detach())
    assert not torch.allclose(before_embedding, artifacts.model.prefix_embedding.weight.detach())


def test_step_adaptive_prompt_wfa_program_populates_constraint_loss():
    program = import_task_module("prompt_automata_program")
    artifacts = _build("wfa", dynamics_conditioning="gated", step_dynamics_conditioning="prefix_gated")

    model_loss, _, *output = artifacts.program.model(artifacts.sample_data)
    constraint_loss, *_ = artifacts.program.cmodel(output[1])
    aux_loss = program.prompt_automata_auxiliary_loss(artifacts)
    result = program.constrained_decode(artifacts)

    assert torch.isfinite(model_loss)
    assert torch.isfinite(constraint_loss)
    assert torch.isfinite(aux_loss)
    assert result.accepted


def test_prompt_automata_frozen_backbone_mode_does_not_train_backbone():
    artifacts = _build("hmm", encoder_kind="frozen_backbone")

    assert artifacts.backbone is not None
    assert all(not parameter.requires_grad for parameter in artifacts.backbone.parameters())
    assert not any(
        name.startswith("prompt_encoder.backbone")
        for name in artifacts.model.trainable_parameter_names()
    )


def test_prompt_automata_dfa_decode_is_accepted():
    program = import_task_module("prompt_automata_program")
    artifacts = _build("hmm")

    result = program.constrained_decode(artifacts)

    dog_label = artifacts.bundle.vocabulary.label_for_token(" dog")
    assert result.accepted
    assert dog_label not in result.labels
    assert artifacts.dfa.accepts(result.labels)


def test_prompt_automata_demo_runs_short_mock_loop(capsys):
    demo = import_task_module("prompt_automata_demo")

    assert demo.main(["--kind", "hmm", "--steps", "1", "--pad-size", "4"]) == 0
    captured = capsys.readouterr().out

    assert "Prompt-conditioned automata path" in captured
    assert "Dynamics conditioning: gated" in captured
    assert "Step dynamics conditioning: prefix_gated" in captured
    assert "Step dynamics weights:" in captured
    assert "Dynamics weights:" in captured
    assert "Baseline non-prompt:" in captured
    assert "Step 1:" in captured
    assert "After DFA-constrained:" in captured


def test_prompt_automata_demo_wfa_runs_short_mock_loop(capsys):
    demo = import_task_module("prompt_automata_demo")

    assert demo.main(["--kind", "wfa", "--steps", "1", "--pad-size", "4"]) == 0
    captured = capsys.readouterr().out

    assert "WFA head" in captured
    assert "Step 1:" in captured
    assert "After DFA-constrained:" in captured
