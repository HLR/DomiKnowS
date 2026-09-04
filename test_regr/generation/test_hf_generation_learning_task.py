from __future__ import annotations

import importlib

import torch

from domiknows.generation import (
    constrained_label_beam_search_decode,
    constrained_label_greedy_decode,
    constrained_label_sample_decode,
    discover_generation_enforcement,
)


def import_task_module(name: str):
    return importlib.import_module(f"Tasks.hf_generation.{name}")


def test_learning_program_builds_in_mock_mode():
    learning_program = import_task_module("learning_program")

    artifacts = learning_program.build_learning_program(pad_size=4)

    assert artifacts.bundle.vocabulary.label_count == 6
    assert artifacts.program is not None
    assert artifacts.sample_data["instruction_tokens"].shape[0] == 1
    assert artifacts.sample_data["target_token_ids"].shape[1] == 4


def test_learning_graph_constraints_still_compile_to_dfa():
    learning_program = import_task_module("learning_program")

    artifacts = learning_program.build_learning_program(pad_size=4)
    enforcement = discover_generation_enforcement(artifacts.graph, artifacts.bundle, on_unsupported="error")
    dfa = enforcement.dfa

    assert dfa.accepts([1, 2, 3, 0])
    assert not dfa.accepts([1, 4, 2, 0])


def test_learning_step_returns_finite_losses():
    learning_program = import_task_module("learning_program")

    artifacts = learning_program.build_learning_program(pad_size=4)
    losses = learning_program.run_one_training_step(artifacts, lr=1e-2)

    assert set(losses) == {"model_loss", "constraint_loss", "total_loss"}
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    assert losses["model_loss"] > 0.0


def test_learning_step_can_include_latent_and_allowed_mass_losses():
    learning_program = import_task_module("learning_program")

    artifacts = learning_program.build_learning_program(pad_size=4, latent_mode="marked")
    losses = learning_program.run_one_training_step(
        artifacts,
        lr=1e-2,
        latent_weight=0.5,
        allowed_mass_weight=0.1,
        latent_diagnostics=True,
    )

    assert losses["latent_loss"] >= 0.0
    assert losses["allowed_mass_loss"] >= 0.0
    assert losses["latent_terms"] >= 0.0


def test_learning_model_trains_only_compact_head_by_default():
    learning_program = import_task_module("learning_program")

    artifacts = learning_program.build_learning_program(pad_size=4)
    trainable = artifacts.model.trainable_parameter_names()

    assert trainable == ["head.weight", "head.bias"]
    assert all(not parameter.requires_grad for parameter in artifacts.model.backbone.parameters())


def test_learn_demo_runs_short_mock_loop(capsys):
    learn_demo = import_task_module("learn_demo")

    assert learn_demo.main(["--steps", "1", "--pad-size", "4"]) == 0
    captured = capsys.readouterr().out

    assert "Trainable parameters" in captured
    assert "Before:" in captured
    assert "Step 1:" in captured
    assert "After unconstrained:" in captured
    assert "After DFA-constrained:" in captured


def test_learn_demo_short_loop_reaches_accepted_mock_sequence(capsys):
    learn_demo = import_task_module("learn_demo")

    assert learn_demo.main(["--steps", "3", "--pad-size", "4"]) == 0
    captured = capsys.readouterr().out

    assert "After DFA-constrained:" in captured
    assert "labels=[1, 2, 3, 0]" in captured
    assert "accepted=True" in captured


def test_trained_head_decodes_with_graph_discovered_dfa():
    learning_program = import_task_module("learning_program")

    artifacts = learning_program.build_learning_program(pad_size=4)
    optimizers = learning_program.make_optimizers(artifacts, lr=0.5)
    for _step in range(3):
        learning_program.run_one_training_step(
            artifacts,
            lr=0.5,
            optimizers=optimizers,
            supervised_weight=3.0,
            constraint_weight=1.0,
        )

    result = constrained_label_greedy_decode(
        artifacts.model,
        artifacts.sample_data["instruction_tokens"],
        artifacts.bundle.vocabulary,
        artifacts.dfa,
        max_new_tokens=4,
    )

    eos_label = artifacts.bundle.vocabulary.eos_label
    dog_label = artifacts.bundle.vocabulary.label_for_token(" dog")
    assert result.accepted
    assert dog_label not in result.labels
    assert result.labels[-1] == eos_label
    assert sum(1 for label in result.labels if label != eos_label) <= 3


def test_trained_head_beam_and_sample_decode_with_graph_discovered_dfa():
    learning_program = import_task_module("learning_program")

    artifacts = learning_program.build_learning_program(pad_size=4)
    optimizers = learning_program.make_optimizers(artifacts, lr=0.5)
    for _step in range(3):
        learning_program.run_one_training_step(
            artifacts,
            lr=0.5,
            optimizers=optimizers,
            supervised_weight=3.0,
            constraint_weight=1.0,
        )

    beam = constrained_label_beam_search_decode(
        artifacts.model,
        artifacts.sample_data["instruction_tokens"],
        artifacts.bundle.vocabulary,
        artifacts.dfa,
        max_new_tokens=4,
        beam_size=3,
    )
    sample = constrained_label_sample_decode(
        artifacts.model,
        artifacts.sample_data["instruction_tokens"],
        artifacts.bundle.vocabulary,
        artifacts.dfa,
        max_new_tokens=4,
        generator=torch.Generator().manual_seed(7),
    )

    eos_label = artifacts.bundle.vocabulary.eos_label
    dog_label = artifacts.bundle.vocabulary.label_for_token(" dog")
    for result in (beam, sample):
        assert result.accepted
        assert dog_label not in result.labels
        assert sum(1 for label in result.labels if label != eos_label) <= 3
        if eos_label in result.labels:
            eos_index = result.labels.index(eos_label)
            assert all(label == eos_label for label in result.labels[eos_index:])


def test_learning_head_teacher_forcing_uses_raw_token_ids():
    learning_model = import_task_module("learning_model")

    class RecordingBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(40, 4)
            self.seen = []

        def forward(self, input_ids):
            self.seen.append(input_ids.detach().clone())
            return self.embedding(input_ids.long())

    backbone = RecordingBackbone()
    model = learning_model.FrozenBackboneGenerationHead(
        backbone=backbone,
        label_count=3,
        pad_size=3,
        label_to_token_id=(10, 20, 30),
    )

    model(None, torch.tensor([[7]]), torch.tensor([1, 2, 0]))

    assert backbone.seen[0].tolist() == [[7]]
    assert backbone.seen[1].tolist() == [[7, 20]]
    assert backbone.seen[2].tolist() == [[7, 20, 30]]
