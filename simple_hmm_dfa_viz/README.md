# Simple HMM + DFA Visualization Demo

This is a tiny teaching demo for DomiKnowS generation internals. A mock
generator proposes a string from a small vocabulary, a DFA checks hard
constraints, and a tiny HMM shows the hidden-state probability flow.

The default graph keeps the original single constraint:

```text
B appears at most once
```

Run the original demo:

```bash
uv run --project Tasks/simple_hmm_dfa_viz python Tasks/simple_hmm_dfa_viz/run_demo.py --candidate invalid
```

There is also a second demo in the same task folder with two constraints:

```text
B appears at most once
C appears at least once
```

Run the two-constraint variant:

```bash
uv run --project Tasks/simple_hmm_dfa_viz python Tasks/simple_hmm_dfa_viz/run_demo.py --demo two --candidate two_b
```

Then open:

```text
Tasks/simple_hmm_dfa_viz/demo_output/index.html
```

Mental model:

```text
mock generator = proposes symbols
DomiKnowS graph = declares the rule
DFA = exact hard verifier
HMM = probabilistic explanation and scorer
JSON = all trace data consumed by the HTML page
```

The HMM is intentionally tiny and readable. It is not trying to be a language
model; it exists so a user can click through the same string and see how masks,
beliefs, emissions, and DFA states evolve step by step.
