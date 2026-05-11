from __future__ import annotations

import importlib

import torch


def import_task_module(name: str):
    return importlib.import_module(f"Tasks.hf_generation.{name}")


def test_wfa_factor_program_builds_and_compiles_dfa():
    wfa_factor_program = import_task_module("wfa_factor_program")

    artifacts = wfa_factor_program.build_wfa_factor_program(pad_size=4)

    assert artifacts.bundle.state_names == ("A", "B", "C")
    assert artifacts.bundle.include_transition_pairs
    assert artifacts.bundle.wfa_transition_pair is not None
    assert artifacts.dfa.accepts([1, 2, 3, 0])
    assert not artifacts.dfa.accepts([1, 4, 2, 0])


def test_wfa_factor_is_next_relation_contains_only_adjacent_pairs():
    wfa_factor_program = import_task_module("wfa_factor_program")
    artifacts = wfa_factor_program.build_wfa_factor_program(pad_size=4)

    _model_loss, _, *output = artifacts.program.model(artifacts.sample_data)
    builder = output[1]

    assert len(builder["DataNodesConcepts"]["is_before_rel"]) == 6
    assert len(builder["DataNodesConcepts"]["is_next_rel"]) == 3


def test_wfa_factor_program_returns_finite_losses():
    wfa_factor_program = import_task_module("wfa_factor_program")
    artifacts = wfa_factor_program.build_wfa_factor_program(pad_size=4)

    model_loss, _, *output = artifacts.program.model(artifacts.sample_data)
    constraint_loss, *_ = artifacts.program.cmodel(output[1])
    labels = wfa_factor_program.target_labels_for_sample(artifacts)
    wfa_loss = wfa_factor_program.wfa_factor_sequence_energy_loss(artifacts.head, labels)
    factor_loss = wfa_factor_program.wfa_factor_consistency_loss(artifacts.head, labels)

    assert torch.isfinite(model_loss)
    assert torch.isfinite(constraint_loss)
    assert torch.isfinite(wfa_loss)
    assert torch.isfinite(factor_loss)


def test_wfa_factor_training_step_updates_shared_parameters():
    wfa_factor_program = import_task_module("wfa_factor_program")
    artifacts = wfa_factor_program.build_wfa_factor_program(pad_size=4)
    before = artifacts.head.initial.detach().clone()

    losses = wfa_factor_program.run_one_wfa_factor_step(artifacts, lr=0.5)

    assert set(losses) == {
        "model_loss",
        "constraint_loss",
        "wfa_factor_energy",
        "wfa_factor_consistency",
        "total_loss",
    }
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    assert not torch.allclose(before, artifacts.head.initial.detach())


def test_wfa_factor_dfa_constrained_decode_is_accepted():
    wfa_factor_program = import_task_module("wfa_factor_program")
    artifacts = wfa_factor_program.build_wfa_factor_program(pad_size=4)

    result = wfa_factor_program.constrained_decode(artifacts)

    dog_label = artifacts.bundle.vocabulary.label_for_token(" dog")
    assert result.accepted
    assert dog_label not in result.labels
    assert artifacts.dfa.accepts(result.labels)


def test_wfa_factor_demo_runs_short_loop(capsys):
    wfa_factor_demo = import_task_module("wfa_factor_demo")

    assert wfa_factor_demo.main(["--steps", "1", "--pad-size", "4"]) == 0
    captured = capsys.readouterr().out

    assert "Spectral WFA factor graph path" in captured
    assert "Transition-pair factor DataNodes: enabled" in captured
    assert "wfa_state_preds=" in captured
    assert "wfa_factor_energy" in captured
    assert "Step 1:" in captured
    assert "After DFA-constrained:" in captured
