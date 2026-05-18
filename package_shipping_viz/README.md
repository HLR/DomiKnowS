# Package Shipping HMM + DFA Visualization

This demo turns the existing `Tasks/package_shipping` declarative planning graph into a single-page teaching view.

It shows one proposed shipping plan moving through two execution layers:

```text
DomiKnowS planning graph + DFA = hard validity
Graph-HMM = learned phase/action likelihood over legal graph paths
```

The graph declares tasks, actions, phases, reference plans, required actions, count limits, and logical constraints. This demo reads that graph, derives the package-shipping DFA and graph-HMM masks, fits the graph-HMM on graph-declared reference plans, and writes a strict `flow.json` plus a static `index.html`.

Run:

```bash
uv run --project Tasks/package_shipping_viz python Tasks/package_shipping_viz/run_demo.py --task ship_fragile_vase --candidate-source invalid_drop_before_seal
```

Open the printed `index.html` link. Click any action step to see:

- DFA state transition and allowed next actions.
- Graph-HMM phase belief and likely phase.
- Remaining required actions and count pressure.
- Transition and emission masks derived from the graph.

The default invalid plan tries to deliver the package before sealing it, which makes the hard DFA reject and also leaves the graph-HMM with no legal phase path at that step.
