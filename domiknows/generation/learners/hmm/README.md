# HMM Learners

This folder contains the Hidden Markov Model (HMM) implementation used by the
compact-label generation stack.

This README focuses on the pure HMM path (no graph adapter, no graph masks),
then shows how the same HMM head plugs into DomiKnowS `ModuleLearner`.

## What Is In This Folder

- `core.py`: production `DiscreteHMM`, forward/backward factors, Viterbi,
  sampling, and Baum-Welch training.
- `head.py`: `HMMGenerationHead`, a compact-label Torch head compatible with
  constrained decoding and DomiKnowS concept learning.
- `factors.py`: optional explicit factor-graph view (`HMMFactorGraphHead`) for
  exposing latent/factor concepts to DomiKnowS.
- `graph.py`, `graph_adapter.py`, `graph_head.py`: graph-aware extensions.

If you want plain HMM training only, start with `core.py`.

## Pure HMM Training Flow (Forward and Backward)

The pure HMM object is `DiscreteHMM` in `core.py`.

Parameters:

- initial distribution: `pi` with shape `[state_count]`
- transition matrix: `A` with shape `[state_count, state_count]`
- emission matrix: `B` with shape `[state_count, symbol_count]`

Observations are integer labels shaped `[batch, seq]` plus optional
`lengths`.

### 1) Forward pass (scaled alpha)

Implementation entry point:

- `DiscreteHMM.forward_backward(...)`

For each time step, alpha is normalized by a scale factor to avoid underflow.

$$
\alpha_0(i) = \pi_i B_{i,o_0}, \qquad c_0 = \sum_i \alpha_0(i), \qquad \hat{\alpha}_0(i)=\alpha_0(i)/c_0
$$

$$
\alpha_t(j) = \left(\sum_i \hat{\alpha}_{t-1}(i) A_{ij}\right) B_{j,o_t}, \qquad c_t = \sum_j \alpha_t(j), \qquad \hat{\alpha}_t(j)=\alpha_t(j)/c_t
$$

In code, these correspond to the stacked tensor `alpha` and per-step
normalizers `scales`.

### 2) Backward pass (scaled beta)

Backward recursion uses the same scales from the forward pass:

$$
\hat{\beta}_{T-1}(i)=1
$$

$$
\hat{\beta}_t(i)=\frac{\sum_j A_{ij} B_{j,o_{t+1}} \hat{\beta}_{t+1}(j)}{c_{t+1}}
$$

In code, this is returned as `beta`.

### 3) Posterior factors

`forward_backward(...)` also returns:

- `gamma` with shape `[batch, seq, state_count]`
- `xi` with shape `[batch, seq-1, state_count, state_count]`

These are computed from alpha/beta and normalized per step.

### 4) Log-likelihood

Because forward probabilities are scaled, sequence log-likelihood is:

$$
\log P(o_{1:T}) = \sum_t \log c_t
$$

In code, this is returned as `log_likelihood`.

## Baum-Welch (EM) In This Implementation

Use either:

- `DiscreteHMM.baum_welch(...)`
- `baum_welch_train(...)` (wrapper)

Flow per EM iteration:

1. E-step: run `forward_backward(...)` over the batch.
2. Collect expected counts:
   - initial counts from `gamma[:, 0, :]`
   - transition counts from `xi.sum(dim=(0, 1))`
   - emission counts by summing `gamma` where observation equals each symbol
3. M-step: normalize counts (with smoothing) to update `pi`, `A`, and `B`.
4. Track total log-likelihood and stop when improvement is below `tol`.

Returned object: `BaumWelchResult(model, log_likelihoods, iterations, converged)`.

## Minimal Pure HMM Example (No Graph)

```python
from domiknows.generation.learners.hmm.core import baum_welch_train

result = baum_welch_train(
    sequences=[
        ["a", "a", "b"],
        ["a", "b", "b"],
        ["b", "a", "b"],
    ],
    symbols=["a", "b"],
    state_count=2,
    max_iter=50,
    tol=1e-6,
)

hmm = result.model
print(result.converged, result.iterations)
print(hmm.log_prob([[0, 1, 1]]))
```

## How It Integrates With DomiKnowS ModuleLearner

The bridge class is `HMMGenerationHead` in `head.py`.

Why this works:

- It implements the compact head contract expected by generation decoders:
  - `next_label_logits(...)`
  - `sequence_log_probs(...)`
  - `token_id_for_label(...)`
- It also implements `forward(_contains, instruction_tokens, target_labels, ...)`
  in a `torch.nn.Module` style, so it can be attached to a DomiKnowS
  `ModuleLearner` on `token[generated_token]`.

### Training path with ModuleLearner

```python
from domiknows.generation import GenerationEncoder
from domiknows.generation.learners import HMMGenerationHead, hmm_sequence_nll
from domiknows.sensor.pytorch.learners import ModuleLearner

encoder = GenerationEncoder(
    vocab=["<eos>", " A", " B"],
    eos_token="<eos>",
    tokenizer=tokenizer,
)
graph, bundle = encoder.build_graph(constraints=[])

head = HMMGenerationHead(
    label_count=bundle.vocabulary.label_count,
    state_count=3,
    pad_size=8,
    trainable=True,
)

token[bundle.generated_token] = ModuleLearner(
    token[bundle.contains],
    text["instruction_tokens"],
    "target_labels",
    module=head,
)

# Typical objective composition in a training step:
# model_loss, _, *output = program.model(sample)
# constraint_loss, *_ = program.cmodel(output[1])
# aux_nll = hmm_sequence_nll(head, sample["target_labels"])
# total = model_loss + constraint_loss + aux_nll
```

What gets optimized:

- `initial_logits`
- `transition_logits`
- `emission_logits`

Those logits are softmaxed into valid HMM probabilities on each forward pass.

## Inference and Decoding

After training, the same `HMMGenerationHead` can be used with compact
DFA-constrained decoders:

- `constrained_label_greedy_decode`
- `constrained_label_beam_search_decode`
- `constrained_label_sample_decode`

This keeps the division of responsibilities clear:

- HMM head proposes compact-label probabilities
- DFA enforces hard sequence validity

## Related Optional Components

If you later want graph-visible HMM factors, use `HMMFactorGraphHead` from
`factors.py`. It exposes generated token probabilities plus latent/factor
projections (`gamma`, optional alpha/beta/xi views) for richer PMD constraints.

For graph-aware constraints and dynamic masks, move to `graph.py` and
`graph_adapter.py`.
