import torch

import domiknows.generation.visual_server as visual_server
from domiknows.generation import (
    TokenVocabulary,
    WeightedFiniteAutomaton,
    constraints_to_dfa,
    create_generation_debug_app,
    dfa_to_dot,
    explain_dfa_rejection,
    forbidden_token,
    max_non_eos,
    no_token_after_eos,
    product_trace_to_dot,
    reachable_product_graph,
    required_token,
    trace_dfa,
    trace_product_automaton,
)


def _demo_dfa():
    vocab = TokenVocabulary(["<eos>", "A", "B"], eos_token="<eos>")
    dfa = constraints_to_dfa(
        [
            no_token_after_eos(),
            max_non_eos(2),
            required_token("A"),
            forbidden_token("B"),
        ],
        vocab,
    )
    return vocab, dfa


def _demo_wfa(symbols):
    transition = torch.zeros((len(symbols), 2, 2), dtype=torch.float32)
    transition[0] = torch.tensor([[0.8, 0.1], [0.0, 0.5]])
    transition[1] = torch.tensor([[0.2, 0.7], [0.1, 0.4]])
    transition[2] = torch.tensor([[0.1, 0.1], [0.1, 0.1]])
    return WeightedFiniteAutomaton(
        initial=torch.tensor([1.0, 0.0]),
        transitions=transition,
        final=torch.tensor([0.5, 1.0]),
        symbols=symbols,
    )


def test_trace_dfa_accepts_and_records_steps():
    vocab, dfa = _demo_dfa()
    eos = vocab.eos_label
    a = vocab.label_for_token("A")

    trace = trace_dfa(dfa, [a, eos])

    assert trace.accepted
    assert not trace.blocked
    assert trace.rejection_reason is None
    assert [step.symbol for step in trace.steps] == [a, eos]
    assert trace.state_path[0] == dfa.start_state
    assert trace.to_dict(lambda value: vocab.token_for_label(value) if isinstance(value, int) else str(value))["accepted"]


def test_trace_dfa_explains_blocked_transition():
    vocab, dfa = _demo_dfa()
    b = vocab.label_for_token("B")

    trace = trace_dfa(dfa, [b])

    assert not trace.accepted
    assert trace.blocked
    assert trace.steps[0].blocked
    assert "dead state" in trace.rejection_reason or "not allowed" in trace.rejection_reason
    assert explain_dfa_rejection(dfa, [b]) == trace.rejection_reason


def test_trace_dfa_explains_non_accepting_terminal_state():
    vocab, dfa = _demo_dfa()
    eos = vocab.eos_label

    trace = trace_dfa(dfa, [eos])

    assert not trace.accepted
    assert not trace.blocked
    assert "non-accepting state" in trace.rejection_reason


def test_dfa_to_dot_marks_core_state_kinds_and_highlight_path():
    vocab, dfa = _demo_dfa()
    a = vocab.label_for_token("A")
    eos = vocab.eos_label
    trace = trace_dfa(dfa, [a, eos])

    dot = dfa_to_dot(dfa, highlight_path=trace, title="debug")

    assert "digraph DFA" in dot
    assert "__start__" in dot
    assert "doublecircle" in dot
    assert "fillcolor" in dot
    assert "penwidth" in dot


def test_product_trace_advances_wfa_and_dfa_together():
    vocab, dfa = _demo_dfa()
    symbols = tuple(range(vocab.label_count))
    wfa = _demo_wfa(symbols)
    a = vocab.label_for_token("A")
    eos = vocab.eos_label

    trace = trace_product_automaton(wfa, dfa, [a, eos])

    assert trace.accepted
    assert len(trace.steps) == 2
    assert len(trace.score_path) == 3
    assert trace.steps[0].to_wfa_state is not None
    dot = product_trace_to_dot(trace)
    assert "digraph ProductTrace" in dot
    assert "score=" in dot


def test_product_trace_marks_blocked_transition():
    vocab, dfa = _demo_dfa()
    symbols = tuple(range(vocab.label_count))
    wfa = _demo_wfa(symbols)
    b = vocab.label_for_token("B")

    trace = trace_product_automaton(wfa, dfa, [b])

    assert not trace.accepted
    assert trace.blocked
    assert trace.steps[0].blocked


def test_reachable_product_graph_is_bounded():
    vocab, dfa = _demo_dfa()
    wfa = _demo_wfa(tuple(range(vocab.label_count)))

    graph = reachable_product_graph(wfa, dfa, max_depth=2, max_states=3)

    assert len(graph.nodes) <= 3
    assert graph.to_dict()["nodes"]


def test_flask_debug_app_returns_html_and_json():
    vocab, dfa = _demo_dfa()
    a = vocab.label_for_token("A")
    eos = vocab.eos_label
    labels = {idx: token for idx, token in enumerate(vocab.labels)}

    app = create_generation_debug_app(dfa, sequence=[a, eos], symbol_labels=labels)

    with app.test_client() as client:
        html = client.get("/")
        trace = client.get("/api/trace")
        summary = client.get("/api/summary")
        dot = client.get("/api/dot")

    assert html.status_code == 200
    assert b"accepted" in html.data
    assert trace.get_json()["accepted"] is True
    assert summary.get_json()["dfa_state_count"] == len(dfa.states)
    assert "svg_render_available" in summary.get_json()
    assert "digraph DFA" in dot.get_data(as_text=True)


def test_flask_debug_app_embeds_mocked_svg(monkeypatch):
    vocab, dfa = _demo_dfa()
    a = vocab.label_for_token("A")
    eos = vocab.eos_label

    monkeypatch.setattr(
        visual_server,
        "dot_to_svg",
        lambda dot: "<svg xmlns=\"http://www.w3.org/2000/svg\"><text>rendered</text></svg>",
    )
    app = create_generation_debug_app(dfa, sequence=[a, eos])

    with app.test_client() as client:
        html = client.get("/")
        svg = client.get("/api/svg")
        summary = client.get("/api/summary")

    assert html.status_code == 200
    assert b"Rendered Graph" in html.data
    assert b"<svg" in html.data
    assert svg.status_code == 200
    assert svg.mimetype == "image/svg+xml"
    assert "<svg" in svg.get_data(as_text=True)
    assert summary.get_json()["svg_render_available"] is True


def test_flask_debug_app_svg_failure_is_non_fatal(monkeypatch):
    vocab, dfa = _demo_dfa()
    a = vocab.label_for_token("A")

    def fail(_dot):
        raise RuntimeError("Graphviz SVG rendering requires the system 'dot' executable")

    monkeypatch.setattr(visual_server, "dot_to_svg", fail)
    app = create_generation_debug_app(dfa, sequence=[a])

    with app.test_client() as client:
        html = client.get("/")
        svg = client.get("/api/svg")
        summary = client.get("/api/summary")

    assert html.status_code == 200
    assert b"Graphviz SVG rendering requires" in html.data
    assert svg.status_code == 200
    assert svg.get_json()["available"] is False
    assert "Graphviz SVG rendering requires" in svg.get_json()["error"]
    assert summary.get_json()["svg_render_available"] is False
