from __future__ import annotations

import importlib
from functools import partial

import pytest
import torch

from domiknows.generation import EnergyCompactLabelGenerationHead, constraints_to_dfa_from_graph, discover_generation_constraints
from domiknows.generation.learners import GraphHMMGenerationHead, PromptConditionedHMMGenerationHead


def import_task_module(name: str):
    return importlib.import_module(f"Tasks.real_hmm_pmd_learning.{name}")


def _labels(bundle, symbols):
    return [bundle.vocabulary.label_for_token(symbol) for symbol in symbols]


def test_real_hmm_pmd_learning_graph_is_local_and_has_one_constraint():
    graph_module = import_task_module("graph")

    graph, parts = graph_module.build_graph()
    generated_symbol = parts[4]
    logical_constraints = getattr(graph, "logicalConstrains", getattr(graph, "_logicalConstrains", {}))

    assert graph.findConcept("string") is not None
    assert graph.findConcept("generated_symbol") is generated_symbol
    assert tuple(generated_symbol.enum) == graph_module.ENUM_VALUES
    assert len(logical_constraints) == 1


def test_real_hmm_pmd_learning_documents_artifact_and_bundle_fields():
    graph_module = import_task_module("graph")
    learning_program = import_task_module("learning_program")

    artifact_fields = learning_program.RealHMMPMDArtifacts.__dataclass_fields__
    for name in (
        "program",
        "graph",
        "bundle",
        "model",
        "learner_name",
        "training_source",
        "stream_examples",
        "dfa",
        "enforcement",
        "stream_seed",
        "inference_prompt_name",
        "inference_prompt_text",
        "inference_prompt_token_id",
    ):
        assert artifact_fields[name].metadata["description"]
    for name in ("learner_name", "training_source", "stream_examples", "enforcement", "stream_seed", "inference_prompt_name"):
        assert artifact_fields[name].metadata["purpose"]

    schema = graph_module.GENERATION_SCHEMA
    assert schema["sequence"]["graph_name"] == "string"
    assert schema["sequence"]["bundle_field"] == "text"
    assert schema["sequence"]["builder_arg"] == "text_name"
    assert schema["label"]["graph_name"] == "generated_symbol"
    assert schema["label"]["bundle_field"] == "generated_token"
    assert schema["label"]["builder_arg"] == "generated_token_name"
    assert schema["vocabulary"]["graph_name"] is None
    assert schema["vocabulary"]["bundle_field"] == "vocabulary"
    assert schema["vocabulary"]["builder_arg"] == "vocab"
    assert all(value["description"] for value in schema.values())


def test_real_hmm_pmd_learning_builds_self_contained_bundle_and_dfa():
    graph_module = import_task_module("graph")

    graph, bundle = graph_module.build_bundle()
    constraints = discover_generation_constraints(graph, bundle, on_unsupported="error")
    dfa = constraints_to_dfa_from_graph(graph, bundle, on_unsupported="error")

    assert bundle.vocabulary.labels == ("A", "B", "C", "D", "END", "_other")
    assert len(constraints) == 1
    assert "at most 1" in constraints[0].name
    assert dfa.accepts(_labels(bundle, graph_module.CANDIDATES["valid"]))
    assert not dfa.accepts(_labels(bundle, graph_module.CANDIDATES["invalid"]))


def test_real_hmm_pmd_learning_stream_generator_uses_valid_and_invalid_outputs():
    stream_generator = import_task_module("stream_generator")
    learning_program = import_task_module("learning_program")

    artifacts = learning_program.build_learning_program(stream_count=4, stream_seed=0, pad_size=100, random_seed=0)

    assert isinstance(artifacts.training_source, stream_generator.GeneratorTrainingSource)
    assert artifacts.model.pad_size == 100
    assert artifacts.training_source.max_length == 100
    assert [example.name for example in artifacts.stream_examples] == ["valid", "invalid", "valid", "invalid"]
    assert [example.prompt_name for example in artifacts.stream_examples] == ["AB", "CD", "short", "AB"]
    assert artifacts.stream_examples[0].prompt_text == "prefer A and B"
    assert artifacts.stream_examples[0].prompt_token_id == stream_generator.PROMPTS["AB"]["token_id"]
    ab_symbols = [symbol for example in artifacts.stream_examples if example.prompt_name == "AB" for symbol in example.symbols if symbol != "END"]
    cd_symbols = [symbol for example in artifacts.stream_examples if example.prompt_name == "CD" for symbol in example.symbols if symbol != "END"]
    assert sum(symbol in {"A", "B"} for symbol in ab_symbols) >= sum(symbol in {"C", "D"} for symbol in ab_symbols)
    assert sum(symbol in {"C", "D"} for symbol in cd_symbols) >= sum(symbol in {"A", "B"} for symbol in cd_symbols)
    assert [example.accepted for example in artifacts.stream_examples] == [True, False, True, False]
    assert all(1 <= len(example.symbols) <= 100 for example in artifacts.stream_examples)
    assert len({len(example.symbols) for example in artifacts.stream_examples}) > 1
    assert any("D" in example.symbols for example in artifacts.stream_examples)
    for example in artifacts.stream_examples:
        if "END" in example.symbols:
            assert example.symbols[-1] == "END"
            assert example.symbols.count("END") == 1
    assert all(example.sample_data["sequence_labels_input"].shape[0] == 1 for example in artifacts.stream_examples)
    assert all(example.sample_data["sequence_labels_input"].shape[1] == len(example.symbols) for example in artifacts.stream_examples)
    assert all("target_labels_input" not in example.sample_data for example in artifacts.stream_examples)
    assert any(example.rejection for example in artifacts.stream_examples if not example.accepted)

    next_batch = artifacts.training_source.next_batch(step=1)
    assert [example.name for example in next_batch] == ["invalid", "valid", "invalid", "valid"]
    assert artifacts.training_source.training_data(next_batch) == [example.sample_data for example in next_batch]
    assert len(artifacts.training_source.training_data(next_batch)) == 4


def test_real_hmm_pmd_learning_default_pad_size_is_100():
    learning_program = import_task_module("learning_program")

    artifacts = learning_program.build_learning_program(learner="energy", stream_count=2, stream_seed=0, random_seed=0)

    assert artifacts.model.pad_size == 100
    assert artifacts.training_source.max_length == 100


def test_real_hmm_pmd_learning_program_builds_with_graph_hmm_learner():
    learning_program = import_task_module("learning_program")

    artifacts = learning_program.build_learning_program(learner="graph-hmm", stream_count=4, pad_size=12, random_seed=0)

    assert artifacts.program is not None
    assert len(artifacts.enforcement.dfa_constraints) == 1
    assert artifacts.learner_name == "graph-hmm"
    assert isinstance(artifacts.model, GraphHMMGenerationHead)
    assert any(isinstance(module, GraphHMMGenerationHead) for module in artifacts.program.model.modules())


def test_real_hmm_pmd_learning_program_builds_with_default_discrete_hmm_learner():
    learning_program = import_task_module("learning_program")

    artifacts = learning_program.build_learning_program(stream_count=4, pad_size=12, random_seed=0)

    assert artifacts.program is not None
    assert artifacts.learner_name == "discrete-hmm"
    assert isinstance(artifacts.model, PromptConditionedHMMGenerationHead)
    assert any(isinstance(module, PromptConditionedHMMGenerationHead) for module in artifacts.program.model.modules())


def test_real_hmm_pmd_learning_program_builds_with_energy_learner():
    learning_program = import_task_module("learning_program")

    artifacts = learning_program.build_learning_program(learner="energy", stream_count=4, pad_size=12, random_seed=0)

    assert artifacts.program is not None
    assert artifacts.learner_name == "energy"
    assert isinstance(artifacts.model, EnergyCompactLabelGenerationHead)
    assert any(isinstance(module, EnergyCompactLabelGenerationHead) for module in artifacts.program.model.modules())
    assert artifacts.model.vocab_size >= 100


@pytest.mark.parametrize("learner", ["discrete-hmm", "graph-hmm", "energy"])
def test_real_hmm_pmd_learning_uses_standard_program_train_and_updates_learner(learner):
    learning_program = import_task_module("learning_program")
    artifacts = learning_program.build_learning_program(learner=learner, stream_count=4, pad_size=12, random_seed=0)
    before = [parameter.detach().clone() for parameter in artifacts.model.parameters() if parameter.requires_grad]

    assert not hasattr(learning_program, "make_optimizers")
    assert not hasattr(learning_program, "run_one_training_step")
    assert not hasattr(learning_program, "run_training_loop")
    for step in range(2):
        artifacts.stream_examples = artifacts.training_source.next_batch(step)
        artifacts.program.train(
            artifacts.training_source.training_data(artifacts.stream_examples),
            train_epoch_num=1,
            Optim=partial(torch.optim.Adam, lr=0.05),
            c_lr=0.05,
            print_loss=False,
        )

    after = [parameter.detach() for parameter in artifacts.model.parameters() if parameter.requires_grad]
    assert any(not torch.allclose(old, new) for old, new in zip(before, after))


@pytest.mark.parametrize("learner", ["discrete-hmm", "graph-hmm", "energy"])
def test_real_hmm_pmd_learning_scoring_and_greedy_inference(learner):
    graph_module = import_task_module("graph")
    learning_program = import_task_module("learning_program")
    utils = import_task_module("utils")
    artifacts = learning_program.build_learning_program(learner=learner, stream_count=4, pad_size=12, random_seed=0)

    assert not hasattr(learning_program, "score_candidate_with_learner")
    assert not hasattr(learning_program, "predictions_for_sample")
    assert not hasattr(learning_program, "constrained_greedy_inference")
    valid_score = utils.score_candidate_with_learner(artifacts, graph_module.CANDIDATES["valid"])
    invalid_score = utils.score_candidate_with_learner(artifacts, graph_module.CANDIDATES["invalid"])
    decoded = utils.constrained_greedy_inference(artifacts)

    assert valid_score["accepted"] is True
    assert invalid_score["accepted"] is False
    assert invalid_score["rejection"]
    assert torch.isfinite(torch.tensor(valid_score["score"]))
    assert torch.isfinite(torch.tensor(invalid_score["score"]))
    assert decoded.accepted is True
    assert decoded.symbols


def test_real_hmm_pmd_learning_run_demo_main_runs_offline(capsys):
    run_demo = import_task_module("run_demo")
    utils = import_task_module("utils")

    assert not hasattr(run_demo, "_print_scores")
    assert not hasattr(run_demo, "_print_inference")
    assert not hasattr(run_demo, "_print_stream")
    assert hasattr(utils, "print_candidate_scores")
    assert hasattr(utils, "print_learning_snapshot")
    assert hasattr(utils, "print_stream_batch")

    assert run_demo.main(["--steps", "2", "--stream-count", "4", "--pad-size", "12"]) == 0
    captured = capsys.readouterr()
    output = captured.out
    assert "One-constraint DomiKnowS PMD learning demo" in output
    assert "Active compact-label learner: discrete-hmm" in output
    assert "Parameter meaning:" in output
    assert "prompt_encoder / initial_projector" in output
    assert "transition_logits: learns how hidden states move" in output
    assert "emission_logits: learns which symbols" in output
    assert "hidden-state example: one state can mean 'B has not appeared yet'" in output
    assert "emission example: the 'B already appeared' state" in output
    assert "prompt-conditioned DiscreteHMM-backed learner" in output
    assert "Rule: token B may appear at most once" in output
    assert "Generator stream: prompt-conditioned outputs are used for PMD training" in output
    assert "Prompt meanings: AB prefers A/B tokens; CD prefers C/D tokens; short prefers early END" in output
    assert "Inference prompt: AB (prefer A and B)" in output
    assert "This table is the DFA pre-check" in output
    assert "Snapshot guide:" in output
    assert "Predictions      = inspect learner on one generator-produced training sequence" in output
    assert "Candidate scores = score/rerank fixed diagnostic candidates for the inference prompt" in output
    assert "Greedy inference = let the learner generate a DFA-constrained sequence for the inference prompt" in output
    assert "prompt       generator_label" in output
    assert "AB" in output
    assert "CD" in output
    assert "generator_label" in output
    assert "stream_item" not in output
    assert "sequence_labels" in output
    assert "target_labels:" not in output
    assert "dfa_verdict" in output
    assert "sequence" in output
    assert "length=" in output
    assert "dfa_rejection:" in output
    assert "rejected" in output
    assert "Training uses PrimalDualProgram.train(...)" in output
    assert "GeneratorTrainingSource.next_batch(step)" in output
    assert "trained on batch 2: 4 generated samples" in output
    assert "Learned discrete-hmm greedy inference:" in output
    assert "labels:" in output
    assert "symbols:" in output
    assert "DFA note: the decoder masks illegal next labels while generating" in output
    assert "DFA-constrained greedy inference:" in output
    assert "decoder_call: constrained_label_greedy_decode(...)" in output
    assert "dfa_accepted:" in output
    assert "learner_log_score=" in output
    assert "learner_log_score:" in output
    assert "relative_preference=" in output
    assert "relative_preference: selected_path" not in output
    assert "beam:" not in output
    assert "sample:" not in output
    assert "Epoch 1 Training" not in output
    assert "Epoch 1 Training" not in captured.err


def test_real_hmm_pmd_learning_run_demo_main_runs_energy_learner(capsys):
    run_demo = import_task_module("run_demo")

    assert run_demo.main(["--learner", "energy", "--steps", "2", "--stream-count", "4", "--pad-size", "12"]) == 0
    output = capsys.readouterr().out
    assert "Active compact-label learner: energy" in output
    assert "energy_mlp: learns a compatibility cost" in output
    assert "Learned energy greedy inference:" in output
    assert "Generator stream: prompt-conditioned outputs are used for PMD training" in output
    assert "Prompt meanings: AB prefers A/B tokens" in output


def test_real_hmm_pmd_learning_run_demo_help_lists_only_simple_options(capsys):
    run_demo = import_task_module("run_demo")

    with pytest.raises(SystemExit) as excinfo:
        run_demo.main(["--help"])

    assert excinfo.value.code == 0
    output = capsys.readouterr().out
    assert "--steps" in output
    assert "--learner" in output
    assert "--head" not in output
    assert "--stream-count" in output
    assert "--inference-prompt" in output
    assert "--pad-size" in output
    assert "--seed" in output
    assert "--lr" in output
    assert "--target" not in output
    assert "--allowed-mass-weight" not in output
    assert "--demo" not in output
    assert "--beam-size" not in output
    assert "--sample-seed" not in output
    assert "--temperature" not in output
