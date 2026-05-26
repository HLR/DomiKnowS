# Compact Learners

This package contains lightweight sequence models that operate on the compact-label space used by the constrained generation stack.

Instead of predicting over the full tokenizer vocabulary, each learner predicts over a small label alphabet managed by the compact vocabulary layer. That makes these heads cheap to train, easy to inspect, and straightforward to combine with DFA-based constraints during decoding.

## What Problem These Modules Solve

The compact generation pipeline splits generation into two layers:

1. The prompt stays in raw tokenizer IDs.
2. The generated continuation is modeled in a compact label space.

Each learner in this folder answers the same question:

"Given the prompt and the compact labels generated so far, what should the next compact label be?"

That interface is consumed by the constrained decoders in the generation stack:

- `constrained_label_greedy_decode`
- `constrained_label_beam_search_decode`
- `constrained_label_sample_decode`

Those decoders ask the model for `next_label_logits(...)`, apply DFA masking in label space, choose the next label, then map that label back to a concrete token ID.

## Shared Interface

All heads in this folder derive from `CompactLabelGenerationHead` and follow the same core contract.

### Required inputs

- `label_count`: number of compact labels the head predicts.
- `pad_size`: maximum teacher-forced sequence length handled in one pass.
- `label_to_token_id`: mapping from compact labels back to concrete tokenizer IDs.
- `vocab_size`: size of the prompt-token embedding table used to embed instruction tokens.

### Main methods

- `next_label_logits(input_ids)`: returns logits over compact labels for one autoregressive step.
- `sequence_log_probs(target_labels, instruction_tokens=...)`: returns teacher-forced per-step log-probabilities over compact labels.
- `token_id_for_label(label)`: resolves the concrete token ID appended during decoding.

### Common prompt/prefix handling

All implementations use the same basic split:

- tokens before the first generated compact token are treated as prompt tokens
- tokens after that point are treated as the generated compact-label prefix

Prompt tokens are embedded with a learned prompt embedding table. Generated labels are represented in compact space, either directly or via a compact-label embedding table.

## Module Overview

| Module | Main class | Modeling idea | Best fit |
| --- | --- | --- | --- |
| `neural_ngram.py` | `NeuralNGramCompactLabelGenerationHead` | Fixed-width Markov context + MLP | fast local dependencies |
| `energy.py` | `EnergyCompactLabelGenerationHead` | Local energy model over context and candidate label | scoring and preference modeling |
| `gru.py` | `GRUCompactLabelGenerationHead` | Recurrent autoregressive model | ordered dependencies beyond fixed context |
| `transformer.py` | `TransformerCompactLabelGenerationHead` | Causal self-attention over compact labels | richer prefix interactions |
| `crf.py` | `CRFCompactLabelScorer` | Linear-chain CRF with exact normalization | structured sequence scoring and exact marginals |
| `utils.py` | helper functions | shared validation, shaping, and mapping | internal support |

## How Each Module Works

### `neural_ngram.py`

Class: `NeuralNGramCompactLabelGenerationHead`

This is the simplest learned autoregressive head in the package. It behaves like a neural n-gram model over compact labels.

How it works:

1. Embed the prompt tokens and average them into one prompt feature vector.
2. Take the last `context_size` generated compact labels.
3. Pad missing history with a dedicated start/pad label equal to `label_count`.
4. Embed those context labels.
5. Concatenate prompt features with context-label embeddings.
6. Run the concatenated vector through an MLP to produce next-label logits.

Key characteristics:

- Only the most recent `context_size` labels affect the next prediction.
- Cheap to evaluate and easy to reason about.
- Good baseline when rules are mostly local.
- Cannot model long-range dependencies beyond the fixed context window.

Teacher-forced training path:

- `sequence_log_probs(...)` builds a context tensor for every time step.
- The model produces logits for each step.
- A log-softmax turns those logits into per-label log-probabilities.

Use this when:

- compact sequences are short
- recent label history matters more than distant history
- you want a lightweight head for constrained decoding

### `energy.py`

Class: `EnergyCompactLabelGenerationHead`

This model uses the same fixed-width context idea as the neural n-gram head, but it scores each candidate label with an energy instead of directly emitting logits.

How it works:

1. Embed and average the prompt tokens.
2. Build a fixed-width context from the last `context_size` labels.
3. For every possible next label, concatenate:
   - prompt features
   - context-label features
   - the candidate label embedding
4. Feed those features through an MLP that outputs a scalar energy.
5. Negate the energies to obtain next-label logits.

Interpretation:

- lower energy means the model prefers that label
- higher negative energy means a larger decoding logit

Important methods:

- `step_energy(...)`: score one specific next label for one prompt and prefix
- `sequence_energy(...)`: sum teacher-forced local energies across a sequence
- `sequence_score(...)`: returns `-sequence_energy(...)`
- `sequence_log_probs(...)`: normalizes `-energy` values with log-softmax for probabilistic use

Why use an energy model:

- separates compatibility scoring from normalized probability computation
- useful when you want ranking behavior first and probabilities second
- aligns well with reranking or hybrid verification workflows

Limitation:

- like the neural n-gram model, it only sees a bounded local context window

### `gru.py`

Class: `GRUCompactLabelGenerationHead`

This is a recurrent autoregressive head over compact labels.

How it works:

1. Embed the prompt tokens and average them.
2. Project the prompt summary into the initial hidden state of a GRU.
3. Feed the generated compact-label prefix through a label embedding table.
4. Run the embedded prefix through the GRU.
5. Project the final GRU output at each step to label logits.

Decoding behavior:

- the model begins from a learned start symbol represented by `label_count`
- each generated compact label is fed back in as the next autoregressive input

Why this differs from the n-gram and energy heads:

- it carries a recurrent hidden state instead of using only a fixed context window
- it can preserve information from earlier labels, at least in compressed form
- it is still much smaller than a full token-level language model

Use this when:

- ordering matters
- dependencies extend beyond the last few labels
- you want a simple sequential model without Transformer cost

Trade-offs:

- more expressive than fixed-window MLPs
- usually simpler and cheaper than the Transformer head
- less parallel and potentially weaker at modeling complex long-range interactions than attention-based models

### `transformer.py`

Class: `TransformerCompactLabelGenerationHead`

This is the most expressive autoregressive model in the compact folder.

How it works:

1. Embed the prompt tokens and average them into a prompt summary.
2. Embed the previous compact labels, including a start symbol for the first step.
3. Add learned positional embeddings.
4. Add a projected prompt bias to every decoding position.
5. Pass the sequence through a causal `TransformerEncoder`.
6. Project the encoded states to next-label logits.

Why it is causal:

- it builds an upper-triangular attention mask
- each position can only attend to earlier positions and itself
- this preserves autoregressive decoding semantics

Why use it:

- better prefix modeling than fixed-window approaches
- can represent interactions across the whole compact-label prefix up to `pad_size`
- fits tasks where label dependencies are structured and non-local

Trade-offs:

- heavier than GRU, neural n-gram, and energy heads
- bounded by `pad_size` for the teacher-forced sequence window and position embeddings
- more parameters and more tuning surface

### `crf.py`

Class: `CRFCompactLabelScorer`

This module is structurally different from the other four. It is a linear-chain Conditional Random Field over compact labels.

Instead of only modeling the next label locally, it assigns a score to the whole label sequence and can normalize exactly with dynamic programming.

Model components:

- `prompt_embedding`: embeds instruction tokens
- `unary_projector`: converts the prompt summary into per-label unary scores
- `start_logits`: score for starting in each label
- `transition_logits`: score for moving from one label to the next
- `end_logits`: optional score for ending on a label

How scoring works:

For a target sequence, the CRF sums:

- a prompt-conditioned unary score for each label
- a start score for the first label
- transition scores between adjacent labels
- an optional end score on the last active label

What makes it special:

- `sequence_score(...)` returns the unnormalized path score
- `log_partition(...)` computes the exact log normalizer with the forward algorithm
- `crf_nll(...)` computes exact negative log-likelihood
- `marginal_log_probs(...)` returns exact label marginals per time step using forward-backward

Why `next_label_logits(...)` still exists:

The existing constrained decoders expect a per-step proposal interface. The CRF provides one by combining prompt-conditioned unary scores with either start or transition scores from the last label. That is useful for local proposal decoding, but it is not the same as exact globally optimal CRF decoding under DFA constraints.

In other words:

- exact CRF training is global
- exact CRF marginals are global
- `next_label_logits(...)` is a local approximation interface for compatibility with the rest of the decoding stack

Use this when:

- transition structure matters strongly
- you need exact normalization or marginals
- you want a structured scorer for reranking, analysis, or probabilistic diagnostics

## Shared Utilities

`utils.py` contains the internal helpers that keep all heads consistent.

The main groups are:

- shape normalization: `_normalise_flat_ids`, `_normalise_prompt_ids`, `_target_label_batch`
- mapping helpers: `_coerce_label_to_token_id`, `_invert_label_to_token_id`, `_resolve_vocab_size`
- validation: `_positive_int`, `_validate_label`, `_validate_labels`, `_validate_token_ids`
- prompt/prefix handling: `_empty_prompt`, `_first_generated_index`

These helpers are important because every compact head needs to solve the same bookkeeping problems:

- separate prompt tokens from generated compact tokens
- support both Python sequences and Torch tensors
- handle single-example and batched teacher-forced calls
- validate label and token ranges early

## How Compact Heads Integrate With Constrained Decoding

The compact decoding path looks like this:

1. Start with prompt token IDs.
2. Ask the head for `next_label_logits(...)`.
3. Ask the DFA which compact labels are allowed.
4. Mask out disallowed labels.
5. Pick the next label with greedy, beam, or sampling logic.
6. Convert that label back to a concrete token ID.
7. Append that token ID to the running input.
8. Repeat.

This separation is the main reason these models are useful. They do not need to model the full tokenizer vocabulary. They only need to rank the compact labels that matter for the constrained generation problem.

## How Compact Heads Integrate With Hybrid Verification

The hybrid controller can also use a compact head as a scorer or reranker.

Typical uses include:

- scoring candidate label sequences with `sequence_log_probs(...)`
- estimating failure risk from the probability mass outside DFA-allowed labels
- suggesting repairs by combining the current DFA state with the head's next-label preferences

This makes the compact heads useful even when they are not the primary generator.

## Choosing the Right Head

Use `NeuralNGramCompactLabelGenerationHead` when you want the smallest and simplest baseline.

Use `EnergyCompactLabelGenerationHead` when you want compatibility scoring and ranking behavior, especially for reranking-oriented workflows.

Use `GRUCompactLabelGenerationHead` when you need sequential memory beyond a fixed context window but still want a compact recurrent model.

Use `TransformerCompactLabelGenerationHead` when compact-label dependencies are global enough that self-attention is worth the extra cost.

Use `CRFCompactLabelScorer` when sequence structure and transition consistency matter more than purely local next-step prediction, or when you need exact structured probabilities.

## Minimal Example

```python
from domiknows.generation.learners.compact import GRUCompactLabelGenerationHead

head = GRUCompactLabelGenerationHead(
    label_count=4,
    pad_size=8,
    label_to_token_id=(101, 102, 103, 104),
    vocab_size=5000,
    embedding_dim=32,
    hidden_size=64,
)

# Teacher-forced training or scoring
log_probs = head.sequence_log_probs(
    target_labels=[1, 2, 3],
    instruction_tokens=[11, 12, 13, 14],
)

# One autoregressive decoding step in compact-label space
next_logits = head.next_label_logits([11, 12, 13, 14])
```

In practice, these heads are usually paired with a `TokenVocabulary` and a DFA-backed decoder rather than called in isolation.

## Exported Symbols

The package exports:

- `CompactLabelGenerationHead`
- `CompactLabelSequenceModel`
- `CRFCompactLabelScorer`
- `EnergyCompactLabelGenerationHead`
- `GRUCompactLabelGenerationHead`
- `NeuralNGramCompactLabelGenerationHead`
- `TransformerCompactLabelGenerationHead`

## Summary

The compact learner package is the lightweight modeling layer for structured generation over compact labels.

- `neural_ngram.py` and `energy.py` are local-context models
- `gru.py` and `transformer.py` are autoregressive sequence models
- `crf.py` is a globally normalized structured scorer
- `utils.py` provides the shared bookkeeping that makes them interchangeable at the interface level

That common interface is what lets the broader DomiKnowS generation stack swap heads, apply DFA constraints, and reuse the same decoding and hybrid-scoring machinery.