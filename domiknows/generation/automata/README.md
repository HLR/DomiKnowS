# `domiknows.generation.automata`

Finite-automaton and probabilistic-model primitives that underpin the constrained
token generation pipeline of DomiKnowS.

This package provides four complementary tools — a Deterministic Finite Automaton
(DFA), a Weighted Finite Automaton (WFA), a Hidden Markov Model (HMM), and a
Spectral / Hankel learning algorithm — that together let you define hard
token-level rules, learn soft sequence distributions, and enforce both during
text generation.

If you are new to any of these concepts, read the
[Background: Key Concepts](#background-key-concepts) section first.

---

## Layers of a Generative Language Model — and Where These Tools Operate

Modern autoregressive language models (GPT-style, LLaMA, etc.) process text
through several distinct internal layers.  Understanding which layer each tool
in this package touches is essential for knowing *when* and *how* to apply it.

### The Five Layers (Numbered by Execution Order)

```
┌──────────────────────────────────────────────────────────────────────┐
│  Layer 0 — PROMPT           (fixed input, before generation starts)  │
│  The fixed input context; not generated, not modified.               │
│  ← DFA is initialised at start_state before the prompt is consumed   │
│  ← TokenVocabulary encodes prompt tokens into the label space        │
├──────────────────────────────────────────────────────────────────────┤
│  Layer 1 — HIDDEN STATE     (transformer forward pass)               │
│  Internal representation produced by the transformer blocks.         │
│  ← WFA scoring can approximate distributions at this level           │
│  ← Soft / latent constraints (latent_constraints.py) operate here    │
│  ← Hankel matrix is built over hidden-state-derived probabilities    │
│  ← Spectral learning recovers WFA parameters via SVD of Hankel       │
├──────────────────────────────────────────────────────────────────────┤
│  Layer 2 — LOGITS           (LM head projection + DFA mask)          │
│  Raw pre-softmax scores over the full tokenizer vocabulary.          │
│  ← DFA masking operates here (mask_logits_for_dfa)                   │
├──────────────────────────────────────────────────────────────────────┤
│  Layer 3 — GENERATED TOKEN  (argmax/sample → the response token)     │
│  New tokens produced by the model, one per decoding step.            │
│  ← DFA state is advanced one step per generated (response) token ID  │
│  ← HMM learns a probabilistic model from observed token sequences    │
├──────────────────────────────────────────────────────────────────────┤
│  Layer 4 — OUTPUT TEXT      (tokenizer decode)                       │
│  Plain decoded string.  Post-hoc verification only.                  │
└──────────────────────────────────────────────────────────────────────┘
```

#### Layer 0 — Prompt

The prompt is the fixed input text (or token-ID sequence) supplied by the
caller before generation begins.  It is never modified by the constrained
decoder — it is only read to initialise the model's context window.

**What operates here:**

- The DFA is reset to its **`start_state`** before any prompt token is
  consumed.  Prompt tokens themselves are *not* stepped through the DFA; the
  DFA starts constraining only from the first *generated* token onward.
- `TokenVocabulary` can encode prompt text into label IDs for external
  inspection, but the constrained decoder treats the prompt as an opaque
  prefix and does not validate it against the constraint automaton.
- The prompt length directly affects `remaining_steps` budgeting: every
  decoding call receives `max_new_tokens` as a separate argument that counts
  *generated* tokens only, not prompt tokens.

---

#### Layer 1 — Hidden States / Latent Representations

Between the embedding layer and the final linear projection sit the transformer
blocks.  Their output at each position is the *hidden state* — a
high-dimensional float vector (`d_model` floats, typically 768–8192 depending
on model size).  The hidden state encodes semantic context but is not yet a
probability distribution.

**What operates here:**

- **WFA scoring** — a `WeightedFiniteAutomaton` learned via spectral methods
  can approximate the marginal probability of a sequence suffix given a prefix
  by operating over a compact state vector that mirrors the structure of the
  hidden-state manifold.
- **Soft / latent constraints** (`latent_constraints.py`) — instead of hard
  masking, these compute a differentiable loss directly over token-probability
  sequences derived from the hidden states, allowing gradient-based training to
  *push* representations towards constraint satisfaction without blocking any
  token outright.

#### Layer 2 — Logits

The transformer's final linear layer (the *language-model head*) projects each
hidden state to a vector of `vocab_size` raw scores called **logits**.  Applying
softmax converts logits to a probability distribution over the next token.

**What operates here:**

- **`mask_logits_for_dfa`** — sets logits of DFA-forbidden tokens to a large
  negative value (default `-1e9`) *before* softmax, so those tokens receive
  effectively zero probability.  This is the hard-constraint enforcement
  mechanism used by every constrained decoder in `decoder.py`.
- **Temperature / top-k / top-p filtering** — applied *after* the DFA mask and
  *before* sampling in `constrained_sample_decode`.

#### Layer 3 — Generated Token IDs (Response Tokens)

Think of Layer 0 and Layer 3 as the two sides of a conversation:

- **Layer 0 — Prompt** = what *you* send to the model (the request).
- **Layer 3 — Generated token IDs** = what the *model sends back* (the response), emitted one integer at a time.

In a chat example:

```
Layer 0  →  "Translate to French: Hello"   (prompt / request,  supplied by caller)
Layer 3  ←  "Bon" | "jour"                 (response tokens,  produced by model)
```

Both layers are sequences of the same kind of integer (a tokenizer vocabulary index),
but only Layer 3 tokens are *decided* during decoding — and therefore only they are
subject to DFA masking and `max_new_tokens` budget counting.

The `TokenVocabulary` class (in `vocabulary.py`) maps these raw IDs to the
compact *label* space that the DFA understands, collapsing the large tokenizer
vocabulary (~50 k entries) into a small alphabet of semantically meaningful
labels plus a catch-all `other` label.

#### Layer 4 — Output Text

The decoded string produced by converting token IDs back through the tokenizer.
At this point token choices are irrevocable.

**What operates here:**

- **`OpenAIResponsesAdapter`** — hosted APIs only return text; DFA constraints
  can only be checked *post-hoc* by encoding the output back to label IDs and
  running `dfa.accepts(labels)`.
- **`encode_output`** — converts the string back to label IDs for verification.

---

### How Layers 1, 2, and 3 Relate Inside One Decoding Step

The layers are numbered in **execution order** — L1 runs first and feeds L2,
L2 feeds L3, and L3 is the result of the step:

```
  Within one decoding step
  ─────────────────────────────────────────────────────────────────────
  L1  Transformer blocks run a forward pass over the full context
      → produce the hidden state  (d_model float vector)
         │
         ▼
  L2  LM head projects hidden state → logits  (vocab_size float vector)
      DFA mask  zeroes out forbidden tokens   (mask_logits_for_dfa)
      Temperature / top-k / top-p applied
         │
         ▼
  L3  argmax or sample picks ONE token ID  ← this is the response token
         │
         ▼
      Token appended to context; loop repeats for next token
```

So L1 and L2 do **not** act on an already-generated token — they are the
machinery that **decides which token to generate next**:

| Layer | Role within one step |
|---|---|
| L1 — Hidden state | The model "thinks" — encodes all context into a float vector |
| L2 — Logits | Converts that thought into a score for every possible next word; DFA masks illegal ones |
| L3 — Token ID | The winning token is *selected* (argmax / sample) from the masked logits |

Layer 3 is therefore the **output** of layers 1 and 2 for that step, not their
input.

---

### Tool-to-Layer Mapping

| Tool / Module | Layer | Hard or Soft? | When applied |
|---|---|---|---|
| `TokenVocabulary.labels_for_token_ids` | Prompt (L0) | Reference only | Encoding prompt for inspection |
| `DFA` initialised to `start_state` | Prompt (L0) | Hard | Before first generated token |
| `WeightedFiniteAutomaton.score` | Hidden / latent (L1) | Soft | Scoring during beam search |
| `latent_constraints.py` losses | Hidden / latent (L1) | Soft | Training-time loss |
| `spectral_learn_*` | Hidden / latent (L1) | Soft | One-shot learning pass |
| `mask_logits_for_dfa` | Logits (L2) | Hard | Before argmax / sampling |
| `mask_label_logits_for_dfa` | Logits (L2) | Hard | Compact-label head variant |
| `DFA.step` / `DFA.allowed_tokens` | Generated Token IDs (L3) | Hard | During generation, per step |
| `HMM` / `baum_welch_train` | Generated Token IDs (L3) | Soft (probabilistic) | Learning from sequences |
| `OpenAIResponsesAdapter.encode_output` | Output text (L4) | Post-hoc check | After generation completes |

---

## Background: Key Concepts

### What is a Deterministic Finite Automaton (DFA)?

A **Deterministic Finite Automaton (DFA)** is the simplest kind of rule machine
for sequences.  Think of it as a flowchart with a fixed set of *states* (nodes)
and *transitions* (arrows labelled with tokens).

- You start at a designated **start state**.
- Each time you read a token from your sequence, you follow the corresponding
  arrow to a new state.
- When the sequence ends, you check whether you are in one of the **accepting
  states**.  If yes, the sequence is *accepted* (it satisfies the rule); if no,
  it is *rejected*.

**Concrete example — "no token may follow an end-of-sequence marker":**

```
State A (start)  --any token-->  State A
State A          --<eos>      -->  State B
State B (dead)   --any token-->  State B   (no escape)

Accepting states: {A, B}  (but State B after consuming more tokens is dead)
```

The DFA in this package goes a step further: during generation it answers the
question *"given that I am in state X and have N tokens left, which tokens are
still worth emitting?"* via `allowed_tokens()`.  This is used to mask the
language-model's output logits so only rule-compliant tokens are considered.

Multiple DFAs (one per constraint) are combined into a single DFA via
`product_dfa` (all constraints must hold simultaneously) or `union_dfa` (any
constraint may hold).

---

### What is a Weighted Finite Automaton (WFA)?

A **Weighted Finite Automaton (WFA)** generalises a DFA by replacing binary
accept/reject decisions with real-valued *scores*.  Instead of arrows that simply
move between states, each transition multiplies the current state vector by a
*weight matrix*.  The score of a complete sequence is a dot product between the
result and a final weight vector.

Concretely, for a sequence `x₁ x₂ … xₜ` the WFA computes:

```
score(x) = α · A_{x₁} · A_{x₂} · … · A_{xₜ} · ω
```

where `α` (alpha) is a start vector, each `A_σ` is the transition matrix for
symbol `σ`, and `ω` (omega) is a final vector.

**Why is this useful?**  A WFA can approximate the probability distribution of
an arbitrary language model over short sequences, compactly and without running
the full model.  Once learned, it is fast to evaluate and easy to intersect with
a hard-constraint DFA (see `hankel.py`).

---

### What is a Hankel Matrix?

The **[Hankel matrix](https://en.wikipedia.org/wiki/Hankel_matrix)** is the bridge between observed sequence probabilities and
the mathematical machinery of WFAs.

Given a set of short *prefixes* (beginnings of sequences) and *suffixes*
(endings), the Hankel matrix `H` is a 2-D table where each cell `H[i, j]`
stores the probability (or score) of the concatenation `prefix_i + suffix_j`:

```
          suffix_1   suffix_2   suffix_3 …
prefix_1 [ P(p1s1)   P(p1s2)   P(p1s3) … ]
prefix_2 [ P(p2s1)   P(p2s2)   P(p2s3) … ]
prefix_3 [ P(p3s1)   P(p3s2)   P(p3s3) … ]
…
```

**Key insight:** for any WFA, the Hankel matrix has *low rank* — its rows lie in
a low-dimensional subspace determined by the number of WFA states.  This means
you can **recover the WFA parameters** from the Hankel matrix by finding its
principal components (SVD).  That is exactly what `spectral.py` does.

A *constrained* Hankel matrix zeroes out cells where the constraint DFA rejects
`prefix + suffix`, effectively projecting the distribution onto sequences that
satisfy the hard rules before learning begins.

---

### What is a Hidden Markov Model (HMM)?

A **Hidden Markov Model (HMM)** is a probabilistic model for sequences where the
observed tokens are generated by an unobserved (hidden) sequence of states.

Imagine a machine with `S` internal gears (states) you cannot see directly.  At
each step:

1. The gear changes according to a *transition probability* matrix — from gear
   *i*, you move to gear *j* with probability `T[i][j]`.
2. The current gear *emits* an observable token — from gear *i*, token *k* is
   produced with probability `E[i][k]`.

Given a training corpus of observed sequences, the **Baum–Welch algorithm**
(a specialised Expectation-Maximisation procedure) adjusts `T` and `E` to
maximise the likelihood of the data.  No labels or annotations are needed —
the hidden states are inferred purely from the sequence statistics.

Once trained, the HMM assigns a probability to any new sequence.  It can also
be converted to a DFA by treating high-probability paths as accepted sequences,
which is useful for comparing what a learned model accepts against a
hand-written constraint DFA.

---

### What is Spectral Learning?

**Spectral learning** is an alternative to Expectation-Maximisation (EM)-based training (like Baum–Welch)
that recovers a WFA directly from the Hankel matrix using linear algebra —
specifically a **Singular Value Decomposition (SVD)**.

SVD decomposes any matrix `H` into `H ≈ U Σ Vᵀ` where `U` and `V` are
orthonormal bases and `Σ` contains the singular values in decreasing order.
Truncating to the top `k` singular values gives the best rank-`k` approximation
of `H`.  The WFA parameters (`α`, `{A_σ}`, `ω`) are then read off from the
factor matrices via closed-form equations.

**Advantages over Baum–Welch:**
- **Single pass, no iterations** — no convergence issues or local optima.
- **Consistent estimator** — with enough data it provably recovers the true
  distribution.
- **Works with negative weights** — unlike HMMs, WFAs can have negative
  entries, giving them greater expressive power.

**Trade-off:** the rank `k` must be chosen in advance, and quality degrades if
the chosen basis (set of prefixes and suffixes) is too small to capture the
structure of the distribution.

---

## Module Overview

| Module | Key classes / functions | Purpose |
|---|---|---|
| [`dfa.py`](#dfapy) | `DFA`, `product_dfa`, `union_dfa` | Core Deterministic Finite Automaton (DFA) primitives — stepping, acceptance, reachability, allowed-token masking, intersection and union |
| [`hankel.py`](#hankelpy) | `WeightedFiniteAutomaton`, `hankel_matrix`, `ProductDecoderState` | Weighted Finite Automaton (WFA) scoring and Hankel matrix construction for learning |
| [`hmm.py`](#hmmpy) | `ProbabilisticAutomaton`, `baum_welch_train`, `compare_hmm_dfa` | Hidden Markov Model (HMM) training (Baum–Welch EM) and acceptance comparison against a DFA |
| [`spectral.py`](#spectralpy) | `SpectralBasis`, `spectral_learn_from_oracle`, `spectral_learn_from_samples` | Spectral / Hankel-SVD learning of WFAs from probability oracles or sample corpora |

All four modules are re-exported from `__init__.py`, so you can import
everything from the package root:

```python
from domiknows.generation.automata import (
    DFA, product_dfa, union_dfa,
    WeightedFiniteAutomaton, hankel_matrix,
    ProbabilisticAutomaton, baum_welch_train,
    SpectralBasis, spectral_learn_from_samples,
)
```

---

## Module Details

### `dfa.py`

The foundation of the entire package.  Every other module either produces or
consumes a `DFA`.  See [What is a DFA?](#what-is-a-deterministic-finite-automaton-dfa)
for a conceptual introduction.

**`DFA`** is a frozen (immutable) dataclass with the following fields:

| Field | Type | Description |
|---|---|---|
| `states` | `frozenset` | All states in the automaton |
| `alphabet` | `frozenset` | All symbols the automaton can consume |
| `transitions` | `dict[(state, symbol) → state]` | Explicit transition table; missing pairs are implicit rejections |
| `start_state` | `Hashable` | Initial state (must be in `states`) |
| `accepting_states` | `frozenset` | States that mark a successful / rule-compliant sequence |
| `dead_states` | `frozenset` | Optional sink states from which no accepting state is reachable; used as a fast-reject hint |

Key methods:

| Method | Description |
|---|---|
| `step(state, symbol)` | Follow one transition; returns `None` if no transition is defined |
| `is_accepting(state)` | `True` if `state` is an accepting state |
| `accepts(sequence)` | Run the full sequence and return whether it is accepted |
| `can_reach_accepting(state, max_steps)` | Breadth-first search: is any accepting state reachable within `max_steps` transitions? |
| `allowed_tokens(state, remaining_steps)` | Return the subset of the alphabet that still leads to an accepting state within the budget |

**`product_dfa(dfas)`** — *intersection*: the result accepts a sequence only when
**all** input DFAs accept it.  Each state in the product is a tuple of component
states, and the construction explores only the reachable portion via breadth-first
search.

**`union_dfa(dfas)`** — *union*: the result accepts a sequence when **any** input
DFA accepts it.  Component branches that run out of transitions become `None`
but do not invalidate the others.

---

### `hankel.py`

Builds on `DFA` to provide the data structures needed for Weighted Finite
Automaton (WFA) scoring and spectral learning.  See
[What is a WFA?](#what-is-a-weighted-finite-automaton-wfa) and
[What is a Hankel Matrix?](#what-is-a-hankel-matrix) for background.

**`WeightedFiniteAutomaton`** represents a linear WFA that scores a sequence
`x₁ … xₜ` as:

```
score(x) = α · A_{x₁} · A_{x₂} · … · A_{xₜ} · ω
```

where `α` is the initial (row) weight vector, each `A_σ` is the per-symbol
square transition matrix, and `ω` is the final weight vector.  Unlike a DFA it
produces a real-valued score rather than a binary accept/reject decision.

**Hankel utilities**:

| Function | Description |
|---|---|
| `hankel_matrix(oracle, prefixes, suffixes)` | Build the Hankel matrix: `H[i,j] = oracle(prefix_i + suffix_j)` |
| `constrained_hankel_matrix(oracle, prefixes, suffixes, dfa)` | Same, but zero out any cell whose `prefix + suffix` is rejected by the DFA |
| `projection_summary(oracle, prefixes, suffixes, dfa)` | Report total probability mass before and after zeroing rejected cells |

**Product-decoder utilities** — used by `spectral.py` to decode with both a WFA
(for scoring) and a DFA (for hard constraints) simultaneously:

| Name | Description |
|---|---|
| `ProductDecoderState` | Frozen dataclass holding the current WFA weight vector and the current DFA state |
| `start_product_state(wfa, dfa)` | Initialise the combined WFA × DFA state |
| `step_product_state(state, symbol, wfa, dfa)` | Advance both the WFA and the DFA by one symbol |
| `allowed_product_symbols(state, wfa, dfa)` | Symbols that the DFA currently permits (to be used as a mask during decoding) |

---

### `hmm.py`

A self-contained Hidden Markov Model (HMM) implementation for learning a
probability distribution over short token sequences.  See
[What is an HMM?](#what-is-a-hidden-markov-model-hmm) for a conceptual
introduction.

**`ProbabilisticAutomaton`** — frozen dataclass that stores the learned HMM
parameters (`transition`, `emission`, `initial`, `symbols`) as nested tuples.
It can:
- Score any sequence via the forward algorithm.
- Be converted to a DFA by treating states whose forward probability exceeds a
  threshold as accepting — useful for comparing learned behaviour against a
  hand-written constraint DFA.

**Training and evaluation**:

| Function | Description |
|---|---|
| `baum_welch_train(sequences, params)` | Train an HMM with the Baum–Welch Expectation–Maximisation algorithm; uses scaled forward–backward passes to avoid numerical underflow |
| `compare_hmm_dfa(automaton, dfa, sequences)` | Compute agreement metrics between what the HMM accepts and what a reference DFA accepts over a shared corpus |
| `all_sequences(symbols, max_len)` | Generate every possible symbol sequence up to length `max_len` — useful for exhaustive evaluation on small alphabets |

Data containers:

| Class | Fields |
|---|---|
| `HMMParameters` | `n_states`, `n_symbols`, `max_iter`, `tol`, `random_seed` — hyperparameters for training |
| `BaumWelchResult` | `automaton`, `log_likelihood_history`, `converged` — output of a training run |

---

### `spectral.py`

Implements the **spectral / Hankel-SVD learning** algorithm to recover a Weighted
Finite Automaton (WFA) from probability data — no iterative EM required.  See
[What is Spectral Learning?](#what-is-spectral-learning) for background.

**Algorithm overview**:
1. Pick a set of short prefixes and suffixes (the *basis*).
2. Evaluate the target distribution over every `prefix + suffix` pair to fill
   the Hankel matrix `H` and per-symbol shifted matrices `H_σ`.
3. Compute a rank-`k` truncated Singular Value Decomposition (SVD) of `H`:
   `H ≈ U Σ Vᵀ`.
4. Recover WFA parameters `(α, {A_σ}, ω)` from `U`, `Σ`, `V`, and the shifted
   matrices using closed-form basis-change equations.

**Key types**:

| Name | Description |
|---|---|
| `SpectralBasis` | Immutable `(prefixes, suffixes, symbols)` triple defining which rows and columns appear in the Hankel matrix |
| `SpectralLearningResult` | Output container holding `wfa`, `basis`, `singular_values`, and `rank_used` |

**Learning functions**:

| Function | Signature | Description |
|---|---|---|
| `build_spectral_basis` | `(symbols, max_prefix_len, max_suffix_len)` | Convenience factory — includes the empty sequence `()` plus all sequences up to the specified lengths |
| `spectral_learn_from_oracle` | `(oracle, basis, rank)` | Learn from a callable `P(sequence) → float` that returns exact probabilities |
| `spectral_learn_from_samples` | `(samples, basis, rank)` | Learn from a list of observed sequences; converts them to empirical frequencies before learning |

---

## Module Dependency Graph

```
spectral.py
    └── hankel.py
            └── dfa.py

hmm.py
    └── dfa.py
```

`dfa.py` has **no internal dependencies** — it is the bedrock of the package.
`hankel.py` imports `DFA` and `State` from `dfa.py`.
`hmm.py` imports `DFA` from `dfa.py` for threshold-based extraction.
`spectral.py` imports `WeightedFiniteAutomaton` from `hankel.py`.

---

## How the Pieces Fit Together

### Typical constrained-generation workflow

```
 User-defined constraints
         │
         ▼
  constraints.py           ← builds DFA objects from high-level rules
         │
         ▼ product_dfa / union_dfa
    dfa.py (DFA)            ← single combined constraint DFA
         │
         ├──► decoder.py   ← mask_logits_for_dfa, constrained_greedy_decode
         │       uses DFA.allowed_tokens at each generation step
         │
         └──► hankel.py    ← WFA × DFA product decoding
                 │
                 └──► spectral.py  ← learn a WFA from data, then decode
                                      with the constraint DFA applied
```

### Learning a WFA and decoding with constraints

```python
from domiknows.generation.automata import (
    DFA, build_spectral_basis, spectral_learn_from_oracle, product_dfa,
    start_product_state, step_product_state, allowed_product_symbols,
)

# 1. Define an alphabet and a target probability oracle.
symbols = ("a", "b", "<eos>")
oracle = lambda seq: my_language_model_probability(seq)

# 2. Build a Hankel basis covering sequences up to length 3.
#    The basis determines the resolution of the learned WFA.
basis = build_spectral_basis(symbols, max_prefix_len=3, max_suffix_len=3)

# 3. Recover a WFA of rank 4 from the oracle (one-shot, no iterations).
result = spectral_learn_from_oracle(oracle, basis, rank=4)
wfa = result.wfa

# 4. Define a hard constraint DFA (e.g. from constraints.py).
constraint_dfa: DFA = ...

# 5. Decode greedily — at each step the DFA masks out illegal tokens
#    and the WFA scores the remaining ones.
state = start_product_state(wfa, constraint_dfa)
generated = []
for _ in range(20):
    allowed = allowed_product_symbols(state, wfa, constraint_dfa)
    if not allowed:
        break
    # Pick the symbol with highest WFA score among allowed ones.
    best = max(allowed, key=lambda sym: wfa.score(generated + [sym]))
    generated.append(best)
    state = step_product_state(state, best, wfa, constraint_dfa)
    if best == "<eos>":
        break
```

### Training an HMM from sequences and comparing it to a DFA

```python
from domiknows.generation.automata import (
    baum_welch_train, compare_hmm_dfa, all_sequences, HMMParameters,
)

# Train an HMM with 4 hidden states on observed sequences.
sequences = [("a", "b", "<eos>"), ("a", "<eos>"), ...]
params = HMMParameters(n_states=4, n_symbols=3, max_iter=200)
result = baum_welch_train(sequences, params)

# Exhaustively compare HMM acceptance to a reference DFA
# over every sequence up to length 5.
symbols = ("a", "b", "<eos>")
corpus = list(all_sequences(symbols, max_len=5))
metrics = compare_hmm_dfa(result.automaton, reference_dfa, corpus)
print(metrics)
```

### Intersecting multiple constraint DFAs

```python
from domiknows.generation.automata import DFA, product_dfa, union_dfa

# All three constraints must be satisfied simultaneously (intersection).
combined = product_dfa([dfa_no_repetition, dfa_eos_closure, dfa_max_length])

# At least one alternative must be satisfied (union).
either = union_dfa([dfa_option_a, dfa_option_b])
```

---

## Design Principles

- **Immutability** — all core types (`DFA`, `WeightedFiniteAutomaton`,
  `ProbabilisticAutomaton`, `SpectralBasis`) are frozen dataclasses.  This
  makes them safe to cache, share across threads, and use as dict keys.
- **No hidden state** — every algorithm is a pure function; side effects are
  limited to optional random seeds passed explicitly.
- **Lazy materialisation** — `product_dfa` and `union_dfa` only expand
  reachable states via breadth-first search, keeping memory proportional to
  the parts of the state space that are actually visited rather than the full
  Cartesian product.
- **No mandatory heavy dependencies** — `dfa.py` and `hmm.py` use only the
  Python standard library.  `hankel.py` adds no extra dependencies.
  `spectral.py` requires PyTorch solely for its `torch.linalg.svd`.
