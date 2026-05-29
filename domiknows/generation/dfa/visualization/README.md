# Product-Automaton Visualization

This note explains the DFA and product-automaton visualization utilities that
live under `domiknows.generation.dfa`, while remaining re-exported from
`domiknows.generation` for convenience.

The ownership split is:

- core trace and DOT helpers live in `domiknows.generation.dfa.visualization`
- the optional Flask web viewer lives in `domiknows.generation.dfa.visualization.server`
- `domiknows.generation` re-exports both layers for easier user-facing imports

These tools help debug why a DFA constraint accepts, rejects, or blocks a
generated sequence.

## What It Shows

The tools can inspect:

- DFA-only constraint traces.
- WFA x DFA product traces.
- per-step allowed symbols from the active DFA state.
- the first rejection reason.
- Graphviz DOT text for the DFA or product trace.
- a local Flask web page for interactive inspection.

The core trace and DOT helpers do not require Flask. Flask is only imported
when the web viewer is created.

## Python API

Use package-level imports when you want the shortest path:

```python
from domiknows.generation import dfa_to_dot, explain_dfa_rejection, trace_dfa

trace = trace_dfa(dfa, [1, 2, 3, 0])

print(trace.accepted)
print(trace.rejection_reason)
print(trace.to_dict())

dot = dfa_to_dot(dfa, highlight_path=trace)
print(dot)
```

Use DFA visualization imports when you want the owning implementation surface:

```python
from domiknows.generation.dfa.visualization import dfa_to_dot, explain_dfa_rejection, trace_dfa
```

For WFA x DFA product-state debugging:

```python
from domiknows.generation import product_trace_to_dot, trace_product_automaton

trace = trace_product_automaton(wfa, dfa, [1, 2, 3, 0])

print(trace.accepted)
print(trace.score_path)
print(product_trace_to_dot(trace))
```

## Run the Flask Viewer

Use the generation-level helper, or import it directly from
`domiknows.generation.dfa.visualization`.

```python
from domiknows.generation import run_generation_debug_server

run_generation_debug_server(
    dfa,
    sequence=[1, 2, 3, 0],
    symbol_labels={0: "<eos>", 1: " The", 2: " cat", 3: " mat"},
    port=5055,
)
```

Then open:

```text
http://127.0.0.1:5055
```

The server exposes:

- `/` - HTML debug view.
- `/api/trace` - JSON trace.
- `/api/summary` - compact acceptance or rejection summary.
- `/api/dot` - Graphviz DOT text.
- `/api/svg` - rendered SVG when Graphviz support is available.

## HuggingFace Task Example

From the repository root:

```powershell
uv run --project Tasks/hf_generation python Tasks/hf_generation/visualize_constraints.py
```

This builds the `Tasks/hf_generation` graph, discovers raw DomiKnowS graph
constraints, compiles them to a DFA, generates a mock constrained sequence, and
starts the local viewer.

For a quick non-blocking smoke check:

```powershell
uv run --project Tasks/hf_generation python Tasks/hf_generation/visualize_constraints.py --no-server-smoke
```

To inspect a specific compact-label sequence:

```powershell
uv run --project Tasks/hf_generation python Tasks/hf_generation/visualize_constraints.py --sequence 1,2,3,0
```

## Reading the Output

Important fields in `/api/trace`:

- `accepted`: whether the full sequence satisfies the DFA.
- `blocked`: whether decoding hit a symbol that was not allowed from the current state.
- `rejection_reason`: the first clear reason for rejection.
- `steps`: one entry per consumed symbol.
- `allowed_symbols`: symbols that the decoder could legally emit at that step.
- `state_path`: DFA states visited by the sequence.

For WFA x DFA traces, the JSON also includes:

- `score_path`: WFA score after each accepted product step.
- `from_wfa_state` / `to_wfa_state`: compact WFA prefix-state vectors.

## Notes

- this is a local debugging utility, not a production web service;
- DOT text is generated, but PNG or SVG rendering depends on optional Graphviz support;
- if Flask is not installed, the trace and DOT helpers still work;
- the visualization does not change decoding behavior; it explains the same DFA constraints used by hard constrained decoding.
