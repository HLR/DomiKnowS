from __future__ import annotations

import importlib

from domiknows.generation import AnyOfGenerationConstraint, discover_generation_enforcement


def import_task_module(name: str):
    return importlib.import_module(f"Tasks.hf_generation.{name}")


def test_hf_generation_graph_builds_and_discovers_constraints():
    graph_module = import_task_module("graph")
    mock_hf = import_task_module("mock_hf")

    graph, bundle = graph_module.build_generation_graph(mock_hf.MockTokenizer())
    enforcement = discover_generation_enforcement(graph, bundle, on_unsupported="error")

    assert bundle.vocabulary.tokens == tuple(graph_module.VOCAB)
    assert len(enforcement.dfa_constraints) >= 5
    assert any(isinstance(constraint, AnyOfGenerationConstraint) for constraint in enforcement.dfa_constraints)


def test_hf_generation_graph_constraints_compile_to_dfa():
    graph_module = import_task_module("graph")
    mock_hf = import_task_module("mock_hf")

    graph, bundle = graph_module.build_generation_graph(mock_hf.MockTokenizer())
    enforcement = discover_generation_enforcement(graph, bundle, on_unsupported="error")
    dfa = enforcement.dfa

    labels = bundle.vocabulary.labels_for_token_ids
    tokenizer = mock_hf.MockTokenizer()
    valid = tokenizer.encode(" The") + tokenizer.encode(" cat") + tokenizer.encode("<eos>")
    missing_cat = tokenizer.encode(" The") + tokenizer.encode(" mat") + tokenizer.encode("<eos>")
    forbidden_dog = tokenizer.encode(" The") + tokenizer.encode(" dog") + tokenizer.encode(" cat")

    assert dfa.accepts(labels(valid))
    assert not dfa.accepts(labels(missing_cat))
    assert not dfa.accepts(labels(forbidden_dog))


def test_hf_generation_graph_accepts_real_hf_eos_token():
    graph_module = import_task_module("graph")
    mock_hf = import_task_module("mock_hf")

    tokenizer = mock_hf.MockTokenizer()
    tokenizer.token_to_id["<|endoftext|>"] = 6
    tokenizer.id_to_token[6] = "<|endoftext|>"
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.eos_token_id = 6

    graph, bundle = graph_module.build_generation_graph(
        tokenizer,
        ["<|endoftext|>", " The", " cat", " mat", " dog"],
        eos_token="<|endoftext|>",
    )
    enforcement = discover_generation_enforcement(graph, bundle, on_unsupported="error")
    dfa = enforcement.dfa

    assert bundle.vocabulary.eos_token == "<|endoftext|>"
    assert bundle.vocabulary.token_id_for_token("<|endoftext|>") == 6
    assert dfa.accepts(bundle.vocabulary.labels_for_token_ids([1, 2, 6]))


def test_hf_generation_demo_runs_mock_modes_without_downloads():
    run_demo = import_task_module("run_demo")

    results = run_demo.run_all_modes(prompt="Once", max_new_tokens=4)
    dfa = constraints_to_dfa(results["constraints"], results["vocabulary"])

    for mode in ("greedy", "beam", "sample"):
        result = results[mode]
        assert result.accepted
        assert dfa.accepts(result.labels)
    assert len({tuple(results[mode].labels) for mode in ("greedy", "beam", "sample")}) == 3


def test_hf_generation_cli_main_runs_in_mock_mode(capsys):
    run_demo = import_task_module("run_demo")

    assert run_demo.main(["--prompt", "Once", "--max-new-tokens", "4"]) == 0
    captured = capsys.readouterr().out

    assert "Discovered DFA constraints" in captured
    assert "Initial prompt" in captured
    assert "text: 'Once'" in captured
    assert "max_new_tokens: 4" in captured
    assert "greedy" in captured
    assert "beam" in captured
    assert "sample" in captured
    assert "intentionally branchy" in captured
    assert "Decoder spaces" in captured
    assert "greedy space" in captured
    assert "beam space" in captured
    assert "sample space" in captured
    assert "blocked=" in captured
    assert "accepted: True" in captured
