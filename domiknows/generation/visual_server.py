"""Optional Flask debug viewer for generation automata.

Flask is imported lazily so the tracing and package imports remain usable in
minimal environments.  The server is meant for local inspection only.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .learners.dfa import DFA, Symbol
from .learners.wfa import WeightedFiniteAutomaton
from .learners.dfa.visualization import (
    dfa_to_dot,
    explain_dfa_rejection,
    product_trace_to_dot,
    trace_dfa,
    trace_product_automaton,
)


@dataclass(frozen=True)
class DotSvgRenderResult:
    """Result of rendering Graphviz DOT to SVG for the local debug viewer."""

    svg: str | None = None
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.svg is not None


def dot_to_svg(dot: str) -> str:
    """Render Graphviz DOT text to SVG.

    The Python ``graphviz`` package and the system ``dot`` executable are both
    resolved lazily.  A ``RuntimeError`` with a human-readable message is raised
    when SVG rendering is unavailable.
    """

    try:
        from graphviz import Source
        from graphviz.backend.execute import ExecutableNotFound
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("Graphviz SVG rendering requires the Python graphviz package") from exc

    try:
        rendered = Source(dot).pipe(format="svg")
    except ExecutableNotFound as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("Graphviz SVG rendering requires the system 'dot' executable") from exc
    except Exception as exc:  # pragma: no cover - graphviz errors are environment dependent
        raise RuntimeError(f"Graphviz SVG rendering failed: {exc}") from exc
    return rendered.decode("utf-8") if isinstance(rendered, bytes) else str(rendered)


def render_dot_svg(dot: str) -> DotSvgRenderResult:
    """Render DOT to SVG, returning an error object instead of raising."""

    try:
        return DotSvgRenderResult(svg=dot_to_svg(dot))
    except RuntimeError as exc:
        return DotSvgRenderResult(error=str(exc))


def create_generation_debug_app(
    dfa: DFA,
    *,
    wfa: WeightedFiniteAutomaton | None = None,
    sequence: Sequence[Symbol] = (),
    title: str | None = None,
    symbol_labels: Mapping[Symbol, str] | None = None,
):
    """Create a local Flask app for inspecting DFA or WFA x DFA traces.

    Args:
        dfa: Constraint DFA to inspect.
        wfa: Optional WFA for product-state score tracing.
        sequence: Sequence of DFA symbols to trace.
        title: Optional page title.
        symbol_labels: Optional display labels for compact symbols.

    Returns:
        A ``flask.Flask`` app.  Flask is imported only when this function is
        called.
    """

    try:
        from flask import Flask, Response, jsonify, render_template_string
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise ImportError("Flask is required for create_generation_debug_app; install flask to use the web viewer") from exc

    labels = dict(symbol_labels or {})

    def label(value: Any) -> str:
        return labels.get(value, str(value))

    def current_trace():
        if wfa is None:
            return trace_dfa(dfa, sequence)
        return trace_product_automaton(wfa, dfa, sequence)

    def current_dot() -> str:
        trace = current_trace()
        if wfa is None:
            return dfa_to_dot(dfa, labeler=label, highlight_path=trace, title=title or "DFA constraint trace")
        return product_trace_to_dot(trace, labeler=label, title=title or "WFA x DFA product trace")

    def current_svg() -> DotSvgRenderResult:
        return render_dot_svg(current_dot())

    app = Flask(__name__)

    @app.get("/")
    def index():
        trace = current_trace()
        payload = trace.to_dict(label)
        dot = current_dot()
        svg = render_dot_svg(dot)
        return render_template_string(
            _HTML_TEMPLATE,
            title=title or "Generation Constraint Debugger",
            mode="WFA x DFA product" if wfa is not None else "DFA",
            payload=json.dumps(payload, indent=2),
            dot=dot,
            svg=svg.svg,
            svg_error=svg.error,
            accepted=payload["accepted"],
            reason=payload.get("rejection_reason"),
            state_count=len(dfa.states),
            transition_count=len(dfa.transitions),
        )

    @app.get("/api/trace")
    def api_trace():
        return jsonify(current_trace().to_dict(label))

    @app.get("/api/summary")
    def api_summary():
        trace = current_trace()
        payload = trace.to_dict(label)
        svg = current_svg()
        return jsonify(
            {
                "mode": "product" if wfa is not None else "dfa",
                "accepted": payload["accepted"],
                "blocked": payload["blocked"],
                "rejection_reason": payload.get("rejection_reason"),
                "sequence": payload["sequence"],
                "dfa_state_count": len(dfa.states),
                "dfa_transition_count": len(dfa.transitions),
                "dfa_accepting_state_count": len(dfa.accepting_states),
                "svg_render_available": svg.available,
                "svg_render_error": svg.error,
            }
        )

    @app.get("/api/dot")
    def api_dot():
        return Response(current_dot(), mimetype="text/vnd.graphviz")

    @app.get("/api/svg")
    def api_svg():
        rendered = current_svg()
        if rendered.available:
            return Response(rendered.svg, mimetype="image/svg+xml")
        return jsonify({"available": False, "error": rendered.error}), 200

    return app


def run_generation_debug_server(
    dfa: DFA,
    *,
    wfa: WeightedFiniteAutomaton | None = None,
    sequence: Sequence[Symbol] = (),
    title: str | None = None,
    symbol_labels: Mapping[Symbol, str] | None = None,
    host: str = "127.0.0.1",
    port: int = 5055,
    debug: bool = False,
):
    """Run the local Flask debug server."""

    app = create_generation_debug_app(
        dfa,
        wfa=wfa,
        sequence=sequence,
        title=title,
        symbol_labels=symbol_labels,
    )
    return app.run(host=host, port=port, debug=debug)


_HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ title }}</title>
  <style>
    :root { color-scheme: light; font-family: Inter, Segoe UI, Arial, sans-serif; }
    body { margin: 0; background: #f7f7f8; color: #18181b; }
    header { padding: 20px 28px 12px; background: #ffffff; border-bottom: 1px solid #e4e4e7; }
    main { padding: 20px 28px 32px; display: grid; gap: 18px; }
    h1 { font-size: 22px; margin: 0 0 6px; }
    h2 { font-size: 16px; margin: 0 0 10px; }
    .meta { color: #52525b; font-size: 14px; }
    .status { display: inline-flex; align-items: center; min-height: 28px; padding: 0 10px; border-radius: 6px; font-weight: 650; }
    .ok { color: #065f46; background: #d1fae5; }
    .bad { color: #991b1b; background: #fee2e2; }
    section { background: #ffffff; border: 1px solid #e4e4e7; border-radius: 8px; padding: 16px; }
    .rendered-graph { overflow: auto; background: #ffffff; }
    .rendered-graph svg { max-width: 100%; height: auto; display: block; }
    .render-error { color: #991b1b; background: #fee2e2; border: 1px solid #fecaca; border-radius: 6px; padding: 10px; }
    pre { margin: 0; overflow: auto; white-space: pre; font-size: 13px; line-height: 1.45; }
    code { font-family: Consolas, ui-monospace, SFMono-Regular, monospace; }
    .grid { display: grid; gap: 18px; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }
    @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>{{ title }}</h1>
    <div class="meta">{{ mode }} | DFA states: {{ state_count }} | transitions: {{ transition_count }}</div>
  </header>
  <main>
    <section>
      <h2>Status</h2>
      {% if accepted %}
      <span class="status ok">accepted</span>
      {% else %}
      <span class="status bad">rejected</span>
      <p class="meta">{{ reason }}</p>
      {% endif %}
    </section>
    <section>
      <h2>Rendered Graph</h2>
      {% if svg %}
      <div class="rendered-graph">{{ svg|safe }}</div>
      {% else %}
      <div class="render-error">{{ svg_error }}</div>
      <p class="meta">Raw Graphviz DOT is still available below and at <code>/api/dot</code>.</p>
      {% endif %}
    </section>
    <div class="grid">
      <section>
        <h2>Trace JSON</h2>
        <pre><code>{{ payload }}</code></pre>
      </section>
      <section>
        <h2>Graphviz DOT</h2>
        <pre><code>{{ dot }}</code></pre>
      </section>
    </div>
  </main>
</body>
</html>
"""
