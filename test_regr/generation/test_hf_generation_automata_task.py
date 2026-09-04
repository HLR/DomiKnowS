from __future__ import annotations

import importlib

import torch

from domiknows.generation import constrained_label_greedy_decode


def import_task_module(name: str):
    return importlib.import_module(f"Tasks.hf_generation.{name}")


def _build(kind: str):
    automata_program = import_task_module("automata_program")
    return automata_program.build_automata_learning_program(kind=kind, pad_size=4, state_count=3)


def test_hmm_automata_program_populates_constraint_loss():
    automata_program = import_task_module("automata_program")
    artifacts = _build("hmm")

    model_loss, _, *output = artifacts.program.model(artifacts.sample_data)
    constraint_loss, *_ = artifacts.program.cmodel(output[1])
    aux_loss = automata_program.automata_auxiliary_loss(artifacts)

    assert torch.isfinite(model_loss)
    assert torch.isfinite(constraint_loss)
    assert torch.isfinite(aux_loss)


def test_wfa_automata_program_populates_constraint_loss():
    automata_program = import_task_module("automata_program")
    artifacts = _build("wfa")

    model_loss, _, *output = artifacts.program.model(artifacts.sample_data)
    constraint_loss, *_ = artifacts.program.cmodel(output[1])
    aux_loss = automata_program.automata_auxiliary_loss(artifacts)

    assert torch.isfinite(model_loss)
    assert torch.isfinite(constraint_loss)
    assert torch.isfinite(aux_loss)


def test_automata_step_updates_trainable_hmm_parameters():
    automata_program = import_task_module("automata_program")
    artifacts = _build("hmm")
    before = next(parameter.detach().clone() for parameter in artifacts.model.parameters() if parameter.requires_grad)

    losses = automata_program.run_one_automata_training_step(artifacts, lr=0.5)
    after = next(parameter.detach() for parameter in artifacts.model.parameters() if parameter.requires_grad)

    assert set(losses) == {"model_loss", "constraint_loss", "automata_aux_loss", "total_loss"}
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    assert not torch.allclose(before, after)


def test_automata_step_updates_trainable_wfa_parameters():
    automata_program = import_task_module("automata_program")
    artifacts = _build("wfa")
    before = next(parameter.detach().clone() for parameter in artifacts.model.parameters() if parameter.requires_grad)

    automata_program.run_one_automata_training_step(artifacts, lr=0.5)
    after = next(parameter.detach() for parameter in artifacts.model.parameters() if parameter.requires_grad)

    assert not torch.allclose(before, after)


def test_hmm_head_decodes_with_graph_discovered_dfa():
    artifacts = _build("hmm")

    result = constrained_label_greedy_decode(
        artifacts.model,
        artifacts.sample_data["instruction_tokens"],
        artifacts.bundle.vocabulary,
        artifacts.dfa,
        max_new_tokens=4,
    )

    dog_label = artifacts.bundle.vocabulary.label_for_token(" dog")
    assert result.accepted
    assert dog_label not in result.labels
    assert artifacts.dfa.accepts(result.labels)


def test_wfa_head_decodes_with_graph_discovered_dfa():
    artifacts = _build("wfa")

    result = constrained_label_greedy_decode(
        artifacts.model,
        artifacts.sample_data["instruction_tokens"],
        artifacts.bundle.vocabulary,
        artifacts.dfa,
        max_new_tokens=4,
    )

    dog_label = artifacts.bundle.vocabulary.label_for_token(" dog")
    assert result.accepted
    assert dog_label not in result.labels
    assert artifacts.dfa.accepts(result.labels)


def test_automata_demo_runs_short_mock_loop(capsys):
    automata_demo = import_task_module("automata_demo")

    assert automata_demo.main(["--kind", "hmm", "--steps", "1", "--pad-size", "4"]) == 0
    captured = capsys.readouterr().out

    assert "Automata learning path" in captured
    assert "Step 1:" in captured
    assert "After DFA-constrained:" in captured


def test_automata_demo_wfa_runs_short_mock_loop(capsys):
    automata_demo = import_task_module("automata_demo")

    assert automata_demo.main(["--kind", "wfa", "--steps", "1", "--pad-size", "4"]) == 0
    captured = capsys.readouterr().out

    assert "WFA head" in captured
    assert "Step 1:" in captured
    assert "After DFA-constrained:" in captured
