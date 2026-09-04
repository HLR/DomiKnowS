from __future__ import annotations

import importlib

from domiknows.generation import HybridController


def import_task_module(name: str):
    return importlib.import_module(f"Tasks.openai_generation.{name}")


def test_openai_generation_graph_constraints_compile_to_dfa():
    graph_module = import_task_module("graph")
    mock_openai = import_task_module("mock_openai")
    run_demo = import_task_module("run_demo")

    graph, bundle = graph_module.build_generation_graph(mock_openai.MockTokenizer())
    enforcement = run_demo.discover_generation_enforcement(graph, bundle, on_unsupported="error")
    dfa = enforcement.dfa

    assert graph.name == "openai_generation"
    assert dfa.accepts([1, 2, 3, 0])
    assert not dfa.accepts([1, 4, 2, 0])


def test_openai_generation_mock_accepted_output_verifies():
    run_demo = import_task_module("run_demo")

    summary = run_demo.run_generation(mock_output="accepted")

    assert summary["text"] == " The cat mat<eos>"
    assert summary["labels"] == [1, 2, 3, 0]
    assert summary["accepted"]
    assert summary["rejection"] is None


def test_openai_generation_mock_rejected_output_reports_reason():
    run_demo = import_task_module("run_demo")

    summary = run_demo.run_generation(mock_output="rejected")

    assert summary["text"] == " The dog<eos>"
    assert summary["labels"] == [1, 4, 0]
    assert not summary["accepted"]
    assert summary["rejection"]


def test_openai_generation_main_runs_offline(capsys):
    run_demo = import_task_module("run_demo")

    assert run_demo.main([]) == 0
    captured = capsys.readouterr().out

    assert "Backend: mock" in captured
    assert "Accepted: True" in captured
    assert "Discovered DFA constraints" in captured


def test_openai_generation_backend_profiles_set_default_base_urls():
    run_demo = import_task_module("run_demo")

    assert run_demo.backend_profile("ollama").base_url == "http://localhost:11434/v1/"
    assert run_demo.backend_profile("vllm").base_url == "http://localhost:8000/v1/"
    assert run_demo.backend_profile("llamacpp").base_url == "http://localhost:8080/v1/"
    assert run_demo.backend_profile("openai").base_url is None


def test_openai_generation_request_logprobs_preserves_mock_metadata():
    run_demo = import_task_module("run_demo")

    summary = run_demo.run_generation(request_logprobs=True)

    assert summary["request"]["logprobs"] is True
    assert summary["logprobs"]
    assert set(summary["logprobs"][0]) == {"token", "logprob"}


def test_openai_generation_extra_params_are_forwarded():
    run_demo = import_task_module("run_demo")

    summary = run_demo.run_generation(extra_params={"temperature": 0.2, "stream": False})

    assert summary["request"]["temperature"] == 0.2
    assert summary["request"]["stream"] is False


def test_openai_generation_mock_hybrid_generate_verify_rerank():
    graph_module = import_task_module("graph")
    mock_openai = import_task_module("mock_openai")
    run_demo = import_task_module("run_demo")

    tokenizer = mock_openai.MockTokenizer()
    graph, bundle = graph_module.build_generation_graph(tokenizer)
    enforcement = run_demo.discover_generation_enforcement(graph, bundle, on_unsupported="error")
    dfa = enforcement.dfa
    client = mock_openai.MockOpenAIClient(" The cat mat<eos>")
    adapter = run_demo.OpenAIResponsesAdapter(client=client, model="mock", tokenizer=tokenizer)
    controller = HybridController(
        generator=adapter,
        vocabulary=bundle.vocabulary,
        dfa=dfa,
        scorer_head=None,
        enforcement=enforcement,
        tokenizer=tokenizer,
        constraints=enforcement.dfa_constraints,
    )

    ranked = controller.generate_verify_rerank("Once", 1, max_new_tokens=4, explain=True)

    assert ranked[0].score.accepted
    assert ranked[0].candidate.labels == [1, 2, 3, 0]
