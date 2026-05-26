domiknows.generation.learners

Finite-state, probabilistic, and compact-label learner primitives that underpin the constrained generation pipeline of DomiKnowS.

This package is the current home of the automata and structured learner stack that used to be documented under `domiknows.generation.automata`. The code has since been reorganized into subpackages under `domiknows.generation.learners`, but the core ideas remain the same: combine hard symbolic constraints with compact probabilistic sequence models and reusable constrained decoders.

At the top level, `domiknows.generation.learners` lazily re-exports the most important public APIs from its `dfa`, `wfa`, `hmm`, `compact`, and `common` subpackages, so most user code can still import from the package root.

This package provides four core modeling families:

- a Deterministic Finite Automaton (DFA) layer for hard sequence constraints
- a Weighted Finite Automaton (WFA) and Hankel layer for soft finite-state scoring
- a Hidden Markov Model (HMM) layer for probabilistic sequence modeling and constraint compilation
- a compact-label learner layer for lightweight GRU, Transformer, energy-based, n-gram, and CRF-style heads

Production API Scope

The probabilistic finite-state layer is Torch-backed. `DiscreteHMM` supports batched integer-label scoring, forward/backward factors, Viterbi paths, sampling, Baum-Welch fitting, DFA extraction, and pretrained-style serialization patterns. `WeightedFiniteAutomaton` stores Torch tensors, scores batched integer-label sequences, returns Torch Hankel matrices, supports WFA x DFA product decoding, and interoperates with the spectral-learning and graph-conditioned heads in the `wfa` package.

Spectral learning remains finite-basis learning, but the production path now lives in `wfa/spectral_learning.py` and uses Torch SVD. It can learn from exact oracles, raw samples, or pre-aggregated sequence counts. These utilities are production library components for compact alphabets and mid-scale CPU/GPU workflows; they are not standalone large language model trainers.

If you are new to any of these concepts, read the Background: Key Concepts section first.

Focused guides:

- `README_learning.md` covers training compact heads, PMD/latent losses, and post-training generation or reranking flows.
- `README_graph_hmm.md` covers graph-aware HMM and spectral automata, constraint compilation, dynamic masks, and factorized state spaces.
- `README_visualization.md` covers DFA and WFA x DFA tracing, DOT export, and the optional Flask debug viewer.

Current Package Layout

The code is now split by responsibility instead of living in four flat files:

| Location | Main purpose |
| --- | --- |
| `learners/__init__.py` | Canonical lazy-export surface for public learner APIs |
| `learners/dfa/core.py` | DFA primitives, products, unions, and complements |
| `learners/dfa/visualization.py` | DFA tracing, rejection explanations, and DOT export |
| `learners/wfa/hankel.py` | WFA scoring, Hankel matrices, and WFA x DFA product decoding |
| `learners/wfa/spectral_learning.py` | Spectral / Hankel-SVD learning |
| `learners/wfa/*.py` | WFA graph-conditioned, factor, and prompt-conditioned heads |
| `learners/hmm/core.py` | Core HMM training, scoring, and comparison utilities |
| `learners/hmm/*.py` | HMM constraint compilation, graph adapters, dynamic constraints, and generation heads |
| `learners/compact/*.py` | Compact-label GRU, Transformer, energy, neural n-gram, and CRF heads |
| `learners/common/*.py` | Shared generation-head interfaces, prompt encoders, losses, and utilities |

If you only need the public APIs, import from `domiknows.generation.learners`.

Layers of a Generative Language Model - and Where These Tools Operate

Modern autoregressive language models (GPT-style, LLaMA, and similar systems) process text through several distinct internal layers. Understanding which layer each tool in this package touches is essential for knowing when and how to apply it.

The Five Layers (Numbered by Execution Order)

```
┌──────────────────────────────────────────────────────────────────────┐
│  Layer 0 - PROMPT           (fixed input, before generation starts) │
│  The fixed input context; not generated, not modified.              │
│  ← DFA is initialized at start_state before the prompt is consumed  │
│  ← TokenVocabulary encodes prompt tokens into the label space       │
├──────────────────────────────────────────────────────────────────────┤
│  Layer 1 - HIDDEN STATE     (transformer forward pass)              │
│  Internal representation produced by the transformer blocks.        │
│  ← WFA scoring can approximate distributions at this level          │
│  ← Soft / latent losses in learners/common/losses.py operate here   │
│  ← Hankel matrix is built over hidden-state-derived probabilities   │
│  ← Spectral learning recovers WFA parameters via SVD of Hankel      │
├──────────────────────────────────────────────────────────────────────┤
│  Layer 2 - LOGITS           (LM head projection + DFA mask)         │
│  Raw pre-softmax scores over the full tokenizer vocabulary.         │
│  ← DFA masking operates here (mask_logits_for_dfa)                  │
│  ← Compact-label heads emit logits directly in label space          │
├──────────────────────────────────────────────────────────────────────┤
│  Layer 3 - GENERATED TOKEN  (argmax/sample → the response token)    │
│  New tokens produced by the model, one per decoding step.           │
│  ← DFA state advances one step per generated token or label         │
│  ← HMMs and compact heads learn distributions over these sequences  │
├──────────────────────────────────────────────────────────────────────┤
│  Layer 4 - OUTPUT TEXT      (tokenizer decode)                      │
│  Plain decoded string. Post-hoc verification only.                  │
└──────────────────────────────────────────────────────────────────────┘
```

Layer 0 - Prompt

The prompt is the fixed input text, or token-ID sequence, supplied by the caller before generation begins. It is never modified by the constrained decoder; it is only read to initialize the model's context window.

What operates here:

- the DFA is reset to its `start_state` before any prompt token is consumed
- prompt tokens themselves are not stepped through the DFA; the DFA starts constraining only from the first generated token onward
- `TokenVocabulary` can encode prompt text into label IDs for external inspection, but the constrained decoder treats the prompt as an opaque prefix and does not validate it against the constraint automaton
- prompt length directly affects `remaining_steps` budgeting because every decoding call receives `max_new_tokens` as a separate argument that counts generated tokens only, not prompt tokens

Layer 1 - Hidden States / Latent Representations

Between the embedding layer and the final linear projection sit the transformer blocks. Their output at each position is the hidden state: a high-dimensional float vector. The hidden state encodes semantic context but is not yet a probability distribution.

What operates here:

- WFA scoring: a `WeightedFiniteAutomaton` learned via spectral methods can approximate the marginal probability of a sequence suffix given a prefix by operating over a compact state vector that mirrors the structure of the hidden-state manifold
- soft or latent constraints: losses in `learners/common/losses.py`, plus factor-consistency losses in `learners/hmm/factors.py` and `learners/wfa/factors.py`, push representations and sequence factors toward legal structure without hard-masking every token

Layer 2 - Logits

The model's final projection produces logits. Applying softmax converts logits to a probability distribution over the next token or compact label.

What operates here:

- `mask_logits_for_dfa` and `mask_label_logits_for_dfa` set DFA-forbidden options to a large negative value before softmax so illegal options receive effectively zero probability
- temperature, top-k, and top-p filtering are applied after DFA masking and before stochastic decoding
- compact-label heads in `learners/compact` often emit logits directly over the compact label alphabet rather than over the full tokenizer vocabulary

Layer 3 - Generated Token IDs (Response Tokens)

Think of Layer 0 and Layer 3 as the two sides of a conversation:

- Layer 0 - Prompt = what you send to the model
- Layer 3 - Generated token IDs = what the model sends back, one integer at a time

In a chat example:

- Layer 0 → `Translate to French: Hello`
- Layer 3 ← `Bon`, `jour`

Both layers are sequences of the same kind of integer, a tokenizer vocabulary index, but only Layer 3 tokens are decided during decoding. Therefore only they are subject to DFA masking and `max_new_tokens` budget counting.

The `TokenVocabulary` class maps these raw IDs to the compact label space that the DFA understands, collapsing a large tokenizer vocabulary into a small alphabet of semantically meaningful labels plus a catch-all `other` label when needed.

Layer 4 - Output Text

The decoded string produced by converting token IDs back through the tokenizer. At this point token choices are irrevocable.

What operates here:

- post-hoc verification utilities can encode output text back to label IDs and check whether the generated label sequence is accepted by a DFA
- hosted text-only APIs can often be constrained only after decoding, not during the token selection step

How Layers 1, 2, and 3 Relate Inside One Decoding Step

The layers are numbered in execution order. Layer 1 runs first and feeds Layer 2, and Layer 2 feeds Layer 3.

```
Within one decoding step
─────────────────────────────────────────────────────────────────────
L1  Transformer blocks run a forward pass over the full context
    → produce the hidden state  (d_model float vector)
       │
       ▼
L2  LM head projects hidden state → logits  (vocab_size float vector)
    DFA mask zeroes out forbidden tokens or labels
    Temperature / top-k / top-p may be applied
       │
       ▼
L3  argmax or sample picks ONE token ID or compact label
       │
       ▼
    Token appended to context; loop repeats for next token
```

So Layers 1 and 2 do not act on an already-generated token; they are the machinery that decides which token to generate next.

| Layer | Role within one step |
| --- | --- |
| L1 - Hidden state | The model "thinks" by encoding all context into a float vector |
| L2 - Logits | Converts that state into a score for every possible next token or label, with DFA masking applied here |
| L3 - Token or label | The winning option is selected by argmax, beam expansion, or sampling |

Layer 3 is therefore the output of Layers 1 and 2 for that step, not their input.

Beam Search and Sampling in the Layer Pipeline

Beam search and sampling are two alternative strategies for selecting which token to emit at Layer 3. Both operate within the same L1→L2→L3 pipeline, but they differ in how they explore the candidate space.

Greedy Decoding (baseline)

At Layer 3, pick the single highest-logit option:

```
L1 → L2 (logits)
       ↓
L3: argmax(masked_logits)
```

Single sequence, single DFA state.

Beam Search

Maintain `k` parallel candidates, each with its own sequence history and DFA state. At Layer 3, instead of picking one option, expand each beam by taking its top candidates, then rank all resulting sequences and keep only the best `k`.

Because each beam carries its own DFA state, no beam can violate the constraint: the DFA mask is applied before candidate selection at every step.

Sampling (stochastic decoding)

Instead of deterministic ranking, apply temperature and filtering at Layer 2, then randomly draw one token or label at Layer 3:

```
L1 → L2 (logits)
     ↓
  Apply DFA mask
     ↓
  Apply temperature scaling
     ↓
  Optionally apply top-k or top-p filtering
     ↓
  L3: sample one token via softmax + multinomial
```

Temperature tuning:

- temperature < 1: sharper distribution, favors high-logit options
- temperature > 1: flatter distribution, more exploration
- temperature = 1: standard softmax

Top-k and top-p limit the effective candidate space before sampling, preventing very unlikely options from being selected.

Summary: Where DFA Constraints Apply

| Strategy | L1 | L2 | L3 | Constraint timing |
| --- | --- | --- | --- | --- |
| Greedy | 1 forward pass | mask logits | pick argmax | before argmax |
| Beam search | k forward passes | mask logits per beam | expand top candidates | before ranking candidates |
| Sampling | 1 forward pass | mask + temperature + filtering | sample from softmax | before softmax |

All three strategies enforce DFA constraints at Layer 2 by masking logits before the token decision is made at Layer 3.

Tool-to-Layer Mapping

| Tool / Module | Layer | Hard or Soft? | When applied |
| --- | --- | --- | --- |
| `TokenVocabulary.labels_for_token_ids` | Prompt (L0) | reference only | encoding prompt for inspection |
| DFA initialized to `start_state` | Prompt (L0) | hard | before first generated token |
| `WeightedFiniteAutomaton.score` | Hidden / latent (L1) | soft | scoring during search or reranking |
| `allowed_mass_loss`, `hmm_sequence_nll`, `wfa_sequence_energy_loss` | Hidden / latent (L1) | soft | training-time loss |
| `spectral_learn_*` | Hidden / latent (L1) | soft | one-shot learning pass |
| `mask_logits_for_dfa` | Logits (L2) | hard | before argmax / sampling |
| `mask_label_logits_for_dfa` | Logits (L2) | hard | compact-label head variant |
| `DFA.step` / `DFA.allowed_tokens` | Generated token IDs (L3) | hard | during generation, per step |
| `DiscreteHMM`, `baum_welch_train` | Generated token IDs (L3) | soft | learning from sequences |
| output re-encoding and DFA verification | Output text (L4) | post-hoc check | after generation completes |

Background: Key Concepts

What is a Deterministic Finite Automaton (DFA)?

A Deterministic Finite Automaton (DFA) is the simplest kind of rule machine for sequences. Think of it as a flowchart with a fixed set of states and transitions labeled by symbols.

- you start at a designated start state
- each time you read a token from your sequence, you follow the corresponding transition to a new state
- when the sequence ends, you check whether you are in one of the accepting states

Concrete example: "no token may follow an end-of-sequence marker"

```text
State A (start)  --any token-->  State A
State A          --<eos>      -->  State B
State B (dead)   --any token-->  State B
```

Accepting states: `{A, B}` in the simplest formulation, with `State B` treated as a sink once additional tokens are consumed.

The DFA in this package goes a step further. During generation it answers the question "given that I am in state X and have N tokens left, which tokens are still worth emitting?" via `allowed_tokens()`. This is used to mask the model's output logits so only rule-compliant tokens are considered.

Multiple DFAs can be combined into a single DFA via `product_dfa` for intersection, where all constraints must hold simultaneously, or `union_dfa`, where any constraint may hold.

What is a Weighted Finite Automaton (WFA)?

A Weighted Finite Automaton (WFA) generalizes a DFA by replacing binary accept/reject decisions with real-valued scores. Instead of arrows that simply move between states, each transition multiplies the current state vector by a weight matrix. The score of a complete sequence is a dot product between the result and a final weight vector.

For a sequence $x_1 x_2 \ldots x_t$, the WFA computes:

$$
\mathrm{score}(x) = \alpha \cdot A_{x_1} \cdot A_{x_2} \cdot \ldots \cdot A_{x_t} \cdot \omega
$$

where $\alpha$ is a start vector, each $A_\sigma$ is the transition matrix for symbol $\sigma$, and $\omega$ is a final vector.

Why is this useful? A WFA can approximate the probability distribution of an arbitrary language model over short sequences, compactly and without running the full model. Once learned, it is fast to evaluate and easy to intersect with a hard-constraint DFA.

What is a Hankel Matrix?

The Hankel matrix is the bridge between observed sequence probabilities and the mathematical machinery of WFAs.

Given a set of short prefixes and suffixes, the Hankel matrix $H$ is a two-dimensional table where each cell $H[i, j]$ stores the probability or score of the concatenation `prefix_i + suffix_j`.

Key insight: for any WFA, the Hankel matrix has low rank. Its rows lie in a low-dimensional subspace determined by the number of WFA states. This means you can recover WFA parameters from the Hankel matrix by finding its principal components with SVD. That is exactly what `learners/wfa/spectral_learning.py` does.

A constrained Hankel matrix zeroes out cells where the constraint DFA rejects `prefix + suffix`, effectively projecting the distribution onto sequences that satisfy the hard rules before learning begins.

What is a Hidden Markov Model (HMM)?

A Hidden Markov Model (HMM) is a probabilistic model for sequences where the observed tokens are generated by an unobserved sequence of hidden states.

Imagine a machine with $S$ internal states you cannot see directly. At each step:

- the state changes according to a transition probability matrix
- the current state emits an observable token according to an emission distribution

Given a training corpus of observed sequences, the Baum-Welch algorithm adjusts the transition and emission parameters to maximize the likelihood of the data. No labels or annotations are needed; the hidden states are inferred from sequence statistics.

Once trained, the HMM assigns a probability to any new sequence. It can also be converted to a DFA or aligned against a DFA, which is useful for comparing what a learned model accepts against a hand-written constraint automaton. In the current package, the HMM layer also includes constraint compilation and generation-head variants under `learners/hmm`.

What is Spectral Learning?

Spectral learning is an alternative to EM-based training such as Baum-Welch. It recovers a WFA directly from the Hankel matrix using linear algebra, specifically a Singular Value Decomposition (SVD).

SVD decomposes any matrix $H$ into approximately $U \Sigma V^\top$, where $U$ and $V$ are orthonormal bases and $\Sigma$ contains singular values in decreasing order. Truncating to the top $k$ singular values gives the best rank-$k$ approximation of $H$. The WFA parameters are then read off from the factor matrices via closed-form equations.

Advantages over Baum-Welch:

- single pass, no iterative convergence loop
- consistent estimator under the usual assumptions
- can represent negative weights, giving WFAs more expressive power than strictly probabilistic HMMs

Trade-off: the rank $k$ must be chosen in advance, and quality degrades if the basis of prefixes and suffixes is too small to capture the structure of the target distribution.

Module Overview

| Location | Key classes / functions | Purpose |
| --- | --- | --- |
| `dfa/core.py` | `DFA`, `product_dfa`, `union_dfa`, `complement_dfa` | core deterministic automaton primitives, acceptance, reachability, masking, and set-like composition |
| `dfa/visualization.py` | `trace_dfa`, `trace_product_automaton`, `dfa_to_dot`, `explain_dfa_rejection` | tracing, diagnostics, and graph export |
| `wfa/hankel.py` | `WeightedFiniteAutomaton`, `hankel_matrix`, `constrained_hankel_matrix`, `ProductDecoderState` | WFA scoring, Hankel construction, and WFA x DFA product decoding |
| `wfa/spectral_learning.py` | `SpectralBasis`, `build_spectral_basis`, `spectral_learn_from_oracle`, `spectral_learn_from_samples`, `spectral_learn_from_counts` | spectral / Hankel-SVD learning |
| `hmm/core.py` | `DiscreteHMM`, `baum_welch_train`, `compare_hmm_dfa`, `all_sequences`, `HMMParameters`, `BaumWelchResult` | HMM training, inference, and evaluation |
| `hmm/constraints.py` | mask specs and projection helpers | declarative transition and emission constraints for HMM-style models |
| `hmm/constraint_compiler.py` | `ConstraintHMMCompilation`, `compile_generation_constraints_to_hmm_support` | compile generation constraints into HMM support structures |
| `hmm/head.py` and related files | `HMMGenerationHead`, prompt-conditioned and graph-conditioned heads | deploy HMMs as generation heads |
| `compact/*.py` | GRU, Transformer, energy, CRF, neural n-gram heads | lightweight compact-label generation and scoring |
| `common/*.py` | base interfaces, prompt encoders, losses, shared utilities | common infrastructure for learner heads |

Canonical Imports

The public root import path is now:

```python
from domiknows.generation.learners import (
    DFA,
    product_dfa,
    union_dfa,
    WeightedFiniteAutomaton,
    hankel_matrix,
    SpectralBasis,
    spectral_learn_from_samples,
    DiscreteHMM,
    baum_welch_train,
)
```

Direct subpackage imports are also valid when you want a more explicit dependency:

```python
from domiknows.generation.learners.dfa.core import DFA, product_dfa
from domiknows.generation.learners.wfa.hankel import WeightedFiniteAutomaton
from domiknows.generation.learners.wfa.spectral_learning import build_spectral_basis
from domiknows.generation.learners.hmm.core import DiscreteHMM, baum_welch_train
```

Module Details

`dfa/core.py`

The foundation of the structured constraint stack. Every other finite-state module either produces or consumes a DFA.

`DFA` is an immutable dataclass with fields such as:

- `states`: all states in the automaton
- `alphabet`: all symbols the automaton can consume
- `transitions`: explicit transition table; missing pairs are implicit rejections
- `start_state`: initial state
- `accepting_states`: states that mark successful sequences
- `dead_states`: optional sink states from which no accepting state is reachable

Key methods:

| Method | Description |
| --- | --- |
| `step(state, symbol)` | follow one transition; returns `None` if no transition is defined |
| `is_accepting(state)` | whether the state is accepting |
| `accepts(sequence)` | run the full sequence and return whether it is accepted |
| `can_reach_accepting(state, max_steps)` | breadth-first search reachability test |
| `allowed_tokens(state, remaining_steps)` | return symbols that can still lead to acceptance within the remaining budget |

`product_dfa(dfas)` implements intersection: the result accepts a sequence only when all input DFAs accept it. The construction explores only the reachable portion of the product state space.

`union_dfa(dfas)` implements union: the result accepts a sequence when any input DFA accepts it.

`wfa/hankel.py`

Builds on the DFA layer to provide the data structures needed for WFA scoring and spectral learning.

`WeightedFiniteAutomaton` represents a linear WFA that scores a sequence via:

$$
\mathrm{score}(x) = \alpha \cdot A_{x_1} \cdot A_{x_2} \cdot \ldots \cdot A_{x_t} \cdot \omega
$$

Unlike a DFA it produces a real-valued score rather than a binary accept/reject decision.

Hankel utilities:

| Function | Description |
| --- | --- |
| `hankel_matrix(oracle, prefixes, suffixes)` | build the Hankel matrix $H[i,j] = oracle(prefix_i + suffix_j)$ |
| `constrained_hankel_matrix(oracle, prefixes, suffixes, dfa)` | same, but zero out cells rejected by the DFA |
| `projection_summary(oracle, prefixes, suffixes, dfa)` | report probability mass before and after projection |

Product-decoder utilities, used to decode with both a WFA and a DFA simultaneously:

| Name | Description |
| --- | --- |
| `ProductDecoderState` | frozen dataclass holding the current WFA weight vector and current DFA state |
| `start_product_state(wfa, dfa)` | initialize the combined WFA x DFA state |
| `step_product_state(state, symbol, wfa, dfa)` | advance both models by one symbol |
| `allowed_product_symbols(state, wfa, dfa)` | symbols currently permitted by the DFA |

`hmm/core.py`

The core HMM implementation for learning a probability distribution over short token sequences.

`DiscreteHMM` is the production HMM API. The current `learners/hmm` package also adds prompt-conditioned heads, graph-conditioned variants, dynamic constraints, factor-graph utilities, and compilation from declarative generation constraints.

Training and evaluation utilities:

| Function | Description |
| --- | --- |
| `baum_welch_train(sequences, params)` | train an HMM with Baum-Welch EM using numerically stable forward-backward passes |
| `compare_hmm_dfa(automaton, dfa, sequences)` | compare HMM acceptance behavior against a reference DFA |
| `all_sequences(symbols, max_len)` | generate every symbol sequence up to a maximum length |

Core containers:

| Class | Fields |
| --- | --- |
| `HMMParameters` | `n_states`, `n_symbols`, `max_iter`, `tol`, `random_seed` |
| `BaumWelchResult` | fitted automaton, log-likelihood history, convergence flag |

`wfa/spectral_learning.py`

Implements spectral / Hankel-SVD learning to recover a `WeightedFiniteAutomaton` from probability data without iterative EM.

Algorithm overview:

1. pick a set of short prefixes and suffixes, the basis
2. evaluate the target distribution over every prefix + suffix pair to fill the Hankel matrix and shifted matrices
3. compute a rank-$k$ truncated SVD of the Hankel matrix
4. recover WFA parameters from the resulting factorization

Key types:

| Name | Description |
| --- | --- |
| `SpectralBasis` | immutable `(prefixes, suffixes, symbols)` triple defining the basis |
| `SpectralLearningResult` | output container holding `wfa`, `basis`, `singular_values`, and `rank_used` |

Learning functions:

| Function | Signature | Description |
| --- | --- | --- |
| `build_spectral_basis` | `(symbols, max_prefix_len, max_suffix_len)` | include the empty sequence plus all sequences up to the specified lengths |
| `spectral_learn_from_oracle` | `(oracle, basis, rank)` | learn from an exact callable probability oracle |
| `spectral_learn_from_samples` | `(samples, basis, rank)` | learn from observed sequences by converting them to empirical frequencies |
| `spectral_learn_from_counts` | `(counts, basis, rank)` | learn from pre-aggregated sequence counts |

Compact-Label Learner Heads

The package now also includes a `compact` subpackage for lightweight label-space models used by constrained decoding and reranking.

Available heads include:

- `NeuralNGramCompactLabelGenerationHead`
- `EnergyCompactLabelGenerationHead`
- `GRUCompactLabelGenerationHead`
- `TransformerCompactLabelGenerationHead`
- `CRFCompactLabelScorer`

These heads emit logits or scores over compact labels instead of the full tokenizer vocabulary. They are especially useful when the output space can be abstracted into a small label alphabet and then intersected with a DFA.

Module Dependency Sketch

```text
learners/wfa/spectral_learning.py
    └── learners/wfa/hankel.py
            └── learners/dfa/core.py

learners/hmm/core.py
    └── learners/dfa/core.py

learners/compact/*.py
    └── learners/common/base.py
```

`dfa/core.py` has no internal learner dependency below it; it is the bedrock of the finite-state stack. `wfa/hankel.py` depends on DFA concepts. `hmm/core.py` interoperates with DFAs for comparison and extraction workflows. The `compact` heads depend on shared interfaces in `common`, and are then consumed by the decoding and hybrid scoring layers elsewhere in `domiknows.generation`.

How the Pieces Fit Together

Typical constrained-generation workflow:

```text
User-defined constraints
        │
        ▼
high-level constraint builders
        │
        ▼ product_dfa / union_dfa
learners/dfa/core.py
        │
        ├──► generation/decoder.py
        │       uses DFA.allowed_tokens at each generation step
        │
        ├──► learners/wfa/hankel.py
        │       supports WFA x DFA product decoding
        │
        └──► learners/hmm/constraint_compiler.py
                compiles compatible constraints into HMM support structures
```

Learning a WFA and decoding with constraints

```python
from domiknows.generation.learners import (
    DFA,
    build_spectral_basis,
    spectral_learn_from_oracle,
    start_product_state,
    step_product_state,
    allowed_product_symbols,
)

symbols = ("a", "b", "<eos>")
oracle = lambda seq: my_language_model_probability(seq)

basis = build_spectral_basis(symbols, max_prefix_len=3, max_suffix_len=3)
result = spectral_learn_from_oracle(oracle, basis, rank=4)
wfa = result.wfa

constraint_dfa: DFA = ...

state = start_product_state(wfa, constraint_dfa)
generated = []
for _ in range(20):
    allowed = allowed_product_symbols(state, wfa, constraint_dfa)
    if not allowed:
        break
    best = max(allowed, key=lambda sym: wfa.score(generated + [sym]))
    generated.append(best)
    state = step_product_state(state, best, wfa, constraint_dfa)
    if best == "<eos>":
        break
```

Training an HMM from sequences and comparing it to a DFA

```python
from domiknows.generation.learners import (
    baum_welch_train,
    compare_hmm_dfa,
    all_sequences,
    HMMParameters,
)

sequences = [("a", "b", "<eos>"), ("a", "<eos>")]
params = HMMParameters(n_states=4, n_symbols=3, max_iter=200)
result = baum_welch_train(sequences, params)

symbols = ("a", "b", "<eos>")
corpus = list(all_sequences(symbols, max_len=5))
metrics = compare_hmm_dfa(result.automaton, reference_dfa, corpus)
print(metrics)
```

Intersecting multiple constraint DFAs

```python
from domiknows.generation.learners import DFA, product_dfa, union_dfa

combined = product_dfa([dfa_no_repetition, dfa_eos_closure, dfa_max_length])
either = union_dfa([dfa_option_a, dfa_option_b])
```

Design Principles

Immutability: the core automata containers are designed to be safe to cache, share across threads, and reuse as stable modeling objects.

No hidden state: the main scoring and learning algorithms expose explicit inputs and outputs; side effects are limited to explicit random seeds and learnable Torch parameters.

Lazy materialization: product DFAs and related constructions expand only reachable states, keeping memory proportional to the explored portion of the state space rather than the full Cartesian product.

Layered specialization: DFA, WFA, HMM, compact, and common code now live in separate subpackages so the public API can grow without turning a single file into a monolith.

No mandatory heavy dependencies for the symbolic core: the DFA layer stays lightweight, while spectral learning and neural heads bring in Torch only where it is actually needed.
