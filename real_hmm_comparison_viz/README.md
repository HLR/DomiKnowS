# Real HMM vs DomiKnowS-Aware HMM Visualization

This demo compares three layers on the same generated string:

```text
DFA = hard graph rule checker
DiscreteHMM = plain probability model
DomiKnowSAwareHMM = probability model projected through graph masks
```

## What Learns, What Enforces

The DFA and the DomiKnowS-aware HMM are related, but they do different jobs.

The DFA is the hard verifier:

```text
sequence -> accepted / rejected
```

It is compiled from DomiKnowS graph constraints and does not learn
probabilities.

The DomiKnowS-aware HMM is the probabilistic model:

```text
sequence -> hidden path beliefs + log-likelihood
```

It has initial, transition, and emission probabilities. During fitting, those
parameters can learn which graph-compatible paths are common or preferred.
The graph/DFA-derived support tells the HMM which paths are impossible, while
learning decides how probability mass is distributed across the possible paths.

Together:

```text
DFA = hard correctness
DomiKnowS-aware HMM = learned constrained dynamics and explanation
```

In this teaching demo, the graph-HMM support is initialized from the DFA so an
invalid string gets both DFA rejection and `-inf` HMM likelihood. In a training
setting, the DFA still supplies hard validity, while the HMM learns to rank,
score, decode, or explain legal sequences.

By default the graph keeps the original single rule:

```text
B appears at most once
```

Run:

```bash
uv run --project Tasks/real_hmm_comparison_viz python Tasks/real_hmm_comparison_viz/run_demo.py --candidate invalid
```

There is also an opt-in two-constraint variant:

```text
B appears at most once
C appears at least once
```

Run:

```bash
uv run --project Tasks/real_hmm_comparison_viz python Tasks/real_hmm_comparison_viz/run_demo.py --demo two --candidate two_b
```

or:

```bash
uv run --project Tasks/real_hmm_comparison_viz python Tasks/real_hmm_comparison_viz/run_demo.py --demo two --candidate missing_c
```

Then open the generated clickable HTML link, or open:

```text
Tasks/real_hmm_comparison_viz/demo_output/index.html
```

The invalid candidate `A B C B` shows the intended contrast: the DFA rejects
it, the permissive `DiscreteHMM` still assigns a finite likelihood, and the
`DomiKnowSAwareHMM` reaches a blocked support step because its emission mask
forbids a second `B`.

The plain `DiscreteHMM` intentionally uses neutral latent state names:

```text
S0, S1, S2
```

Those states are generic probability clusters, not rule phases. They all have
broad positive emission support over `A`, `B`, `C`, and `END`.

The mask panels are model-specific:

```text
Plain HMM support = what the DiscreteHMM parameters currently allow
DomiKnowS-aware masks = what the graph-constrained HMM is permitted to use
```

In this demo the plain HMM support is intentionally all positive, while the
DomiKnowS-aware HMM now uses automatically compiled DFA-edge states. For the
two-constraint variant, names such as `need_C_no_B__emit_B__to_need_C_seen_B`
are created from the graph-discovered DFA support, not hand-written masks.
There is no productive state like `seen_C_seen_B__emit_B`, so a second `B`
has no legal graph-HMM support.

Because the plain HMM support is all positive, the page summarizes it instead
of showing two all-ones tables. The meaningful plain-HMM internal state is the
changing belief over neutral clusters:

```text
S0 = a generic hidden pattern for common symbols
S1 = another statistical pattern, often useful around B-like observations
S2 = another statistical pattern for later or different contexts
```

The factor panels are also per-model. `Plain DiscreteHMM Factors` show alpha,
beta, and gamma for the unconstrained HMM. `DomiKnowS-Aware HMM Factors` show
the corresponding graph-projected flow; when the second `B` is impossible, the
graph-aware factor row collapses to zero support for display.
