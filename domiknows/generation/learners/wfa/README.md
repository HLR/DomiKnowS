# Weighted Finite Automata (WFA)

This package provides Torch-backed weighted finite automata for scoring
sequences, spectral learning from finite Hankel matrices, and WFA-based heads
for compact-label generation. WFA scores can be signed; they are not
necessarily normalized probabilities.

## What is here

| Module | Purpose |
| --- | --- |
| `hankel.py` | `WeightedFiniteAutomaton`, finite Hankel matrices, DFA-constrained Hankel helpers, and WFA × DFA product decoding |
| `spectral_learning.py` | Spectral/Hankel-SVD learning from a score oracle, samples, or pre-aggregated counts |
| `graph.py` | Spectral WFA fitting from graph-constrained Hankel blocks, with optional masks and DFA constraints |
| `head.py` | Trainable compact-label generation head initialized from a WFA or random parameters |
| `graph_head.py` | Compact-label head initialized from graph spectral learning |
| `prompt_conditioned_head.py` | Compact-label WFA head with prompt-conditioned initial state and optional gated dynamics |
| `factors.py` | DomiKnowS generation-graph encoder, head, and consistency/sequence-energy losses for explicit WFA factors |

## Spectral learning

Spectral learning evaluates sequence scores over a finite prefix/suffix basis
to form a Hankel matrix, truncates its singular-value decomposition to the
requested rank, and recovers the WFA parameters. The basis size grows with the
alphabet and maximum prefix/suffix lengths, so choose bounds and rank
appropriate to the problem.

```python
from domiknows.generation.learners.wfa import (
    build_spectral_basis,
    spectral_learn_from_oracle,
)

symbols = ("a", "b")
basis = build_spectral_basis(
    symbols,
    max_prefix_len=1,
    max_suffix_len=1,
)

# Replace this example score with an oracle for the target sequence model.
def score(sequence):
    return 1.0 if not sequence else 0.0

result = spectral_learn_from_oracle(
    score,
    symbols,
    rank=1,
    basis=basis,
)
automaton = result.model
```

`SpectralLearningResult` also includes the basis, retained rank, singular
values, and learning diagnostics. Use `spectral_learn_from_samples` for raw
sequences or `spectral_learn_from_counts` for pre-aggregated sequence counts.

## Graph and generation integration

`GraphSpectralAutomaton` fits a signed WFA from observed sequences and supplied
prefix/suffix bases. Graph transition/emission masks and an optional DFA can
constrain which Hankel queries contribute to fitting. The fitted model exposes
diagnostics through `fit_result_`; use `score(..., enforce_constraints=True)`
when completed-sequence scores must also reject strings that violate those
constraints.

`SpectralWFAGenerationHead`, `GraphSpectralGenerationHead`, and
`PromptConditionedSpectralWFAGenerationHead` implement the compact-label
generation-head interface. They return next-label scores for decoding and
teacher-forced log-probabilities for training. Since WFA scores may be signed,
next-label scores are treated as logits rather than as normalized WFA
probabilities.

For explicit automaton-state factors in a DomiKnowS generation graph, use
`SpectralWFAFactorGraphEncoder` and the helpers in `factors.py`. This path is
separate from ordinary compact-label decoding and can optionally represent
transition pairs as graph factors.

## Public imports

The package lazily exports its main WFA, spectral-learning, graph-learning,
head, and factor-graph APIs. They can be imported from
`domiknows.generation.learners.wfa` or, for APIs re-exported by the learners
package, from `domiknows.generation.learners`.
