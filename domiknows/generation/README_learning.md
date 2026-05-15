# Learning and Using Learned Generation Models

This note explains the learned-model side of `domiknows.generation`.

The short version:

```text
large generator = fluent open-vocabulary text producer
DFA             = hard rule enforcer / verifier
compact head    = learned DomiKnowS-aware scorer, proposal model, or controller
PMD / latent loss = training signals for symbolic and soft preferences
```

The compact learned model does not need to mimic the full size or behavior of a
large generative model. It operates over the compact `TokenVocabulary` label
space used by the DomiKnowS graph and DFA.

## What Is Learned?

Depending on the path, the learned component can learn one or more of these:

- `prompt -> compact output labels`, as in the frozen-backbone compact head;
- compact sequence dynamics, as in HMM/WFA generation heads;
- prompt-conditioned automata dynamics, including gated and step-adaptive
  prompt/prefix conditioning;
- latent-state or factor-graph projections for PMD-visible constraints;
- soft preferences that are awkward or impossible to encode as a DFA;
- candidate scoring, repair/risk estimation, or constraint-bundle selection.

Hard validity still comes from the DFA. Learned losses and learned heads improve
proposal quality, ranking, and task preference.

## Training Signals

The usual DomiKnowS generation training loop combines:

```python
total = (
    supervised_weight * supervised_loss
    + pmd_weight * pmd_constraint_loss
    + latent_weight * latent_loss
    + allowed_mass_weight * allowed_mass_loss
    + automata_weight * automata_aux_loss
)
```

Use `compute_generation_training_loss(...)` to combine these terms in a
consistent way.

Common terms:

- `supervised_loss`: fits known target compact labels.
- `pmd_constraint_loss`: comes from `PrimalDualProgram.cmodel(...)` over graph
  DataNodes.
- `latent_loss`: comes from graph-marked or graph-discovered
  `LatentWindowSpec` rules.
- `allowed_mass_loss`: softly encourages model probability mass on
  DFA-productive labels.
- `automata_aux_loss`: HMM NLL, WFA energy loss, or factor-graph diagnostics.

In the HF task demos, `constraint_loss` is printed as
`pmd_constraint_objective` because it is a signed primal-dual objective from
`PrimalDualProgram.cmodel(...)`. The combined `optimization_objective` can be
negative even while the normal positive terms are learning. Use
`positive_training_terms`, the individual NLL/cross-entropy terms, and final
DFA acceptance for human-readable progress.

## Main Training Paths

### Frozen-Backbone Compact Head

This is the Collie-style path used by `Tasks/hf_generation/learn_demo.py`.

```text
prompt ids -> frozen HF/mock backbone -> trainable compact-label head
```

The head populates `token[generated_token]` through `ModuleLearner`, so
`PrimalDualProgram` can compute DomiKnowS graph-constraint loss. After training,
the same head can be decoded with `constrained_label_*` or used by
`HybridController`.

Run:

```powershell
uv run --project Tasks/hf_generation python Tasks/hf_generation/learn_demo.py --steps 3
```

### HMM / WFA Compact Automata Heads

`HMMGenerationHead` and `SpectralWFAGenerationHead` are Torch compact-label
modules. They can be attached to `token[generated_token]` and trained by normal
Torch optimizers together with PMD constraint loss.

Use this path when you want inspectable compact sequence dynamics rather than a
black-box head.

Run:

```powershell
uv run --project Tasks/hf_generation python Tasks/hf_generation/automata_demo.py --kind hmm --steps 3
uv run --project Tasks/hf_generation python Tasks/hf_generation/automata_demo.py --kind wfa --steps 3
```

### Prompt-Conditioned Automata Heads

Prompt-conditioned HMM/WFA heads learn:

- prompt-conditioned initial state;
- optional prompt-gated transition/emission dynamics;
- optional step-adaptive dynamics from generated compact-label prefix.

Use this when prompts should influence compact automata behavior without turning
the automaton into a full language model.

Run:

```powershell
uv run --project Tasks/hf_generation python Tasks/hf_generation/prompt_automata_demo.py --kind hmm --steps 1
uv run --project Tasks/hf_generation python Tasks/hf_generation/prompt_automata_demo.py --kind wfa --steps 1
```

### HMM / WFA Factor Graphs

Factor-graph paths expose more internal structure to DomiKnowS:

- HMM: `latent_state`, adjacency, optional `forward_state`, `backward_state`,
  and `transition_pair` DP factors.
- WFA: normalized projections of signed `wfa_state` and optional
  `wfa_transition_pair` factors.

Torch still owns exact numeric HMM/WFA recurrences. DomiKnowS sees graph
concepts and symbolic consistency constraints over their projections.

Run:

```powershell
uv run --project Tasks/hf_generation python Tasks/hf_generation/hmm_factor_demo.py --steps 1
uv run --project Tasks/hf_generation python Tasks/hf_generation/wfa_factor_demo.py --steps 1
```

## Using a Learned Head After Training

There are two different post-training modes.

### Mode A: Compact Head as Generator

Use this when the compact head itself should emit the output labels.

```python
from domiknows.generation import constrained_label_beam_search_decode

result = constrained_label_beam_search_decode(
    trained_head,
    prompt_ids,
    bundle.vocabulary,
    dfa,
    max_new_tokens=8,
    beam_size=4,
)
```

Flow:

```text
prompt ids -> compact head -> compact-label logits -> DFA mask -> token ids
```

This is useful for small controlled generators, diagnostics, and automata-based
experiments.

### Mode B: Compact Head as Hybrid Controller/Scorer

Use this when a large generator remains responsible for fluent text.

```python
from domiknows.generation import HybridController

controller = HybridController(
    generator=adapter,
    vocabulary=bundle.vocabulary,
    dfa=dfa,
    scorer_head=trained_head,
    enforcement=enforcement,
    tokenizer=tokenizer,
    constraints=enforcement.dfa_constraints,
)

ranked = controller.generate_verify_rerank(
    "Once",
    num_candidates=4,
    max_new_tokens=8,
)

best = ranked[0]
```

Flow:

```text
prompt
  -> large generator proposes candidates
  -> DFA verifies hard validity
  -> compact head scores valid candidates
  -> best candidate returned
```

This is the path for:

- reranking several valid candidates;
- learning domain style or task-specific preferences;
- predicting constraint-satisfaction risk before retries;
- guiding repairs after failed verification;
- learning soft latent preferences not represented by a DFA;
- selecting which constraint bundle applies to a prompt.

Run:

```powershell
uv run --project Tasks/hf_generation python Tasks/hf_generation/hybrid_demo.py --steps 3
```

## OpenAI-Compatible Usage

Hosted OpenAI and generic OpenAI-compatible APIs are generate-then-verify from
the DomiKnowS side:

```text
OpenAI-compatible API -> text -> tokenizer -> compact labels -> DFA verify
```

With a trained compact head, use `HybridController` to rerank or diagnose these
outputs after generation. This does not become hard per-token DFA decoding
unless the backend exposes a native logits/guided-decoding hook.

## Choosing the Right Path

| Goal | Use |
| --- | --- |
| Guaranteed valid local HF decoding | `HuggingFaceGenerationAdapter.constrained_*` |
| Learned compact generator | `constrained_label_*` with a trained head |
| Use GPT/HF for fluency, learned head for preferences | `HybridController` |
| Train graph constraints into a head | `PrimalDualProgram` + `compute_generation_training_loss(...)` |
| Inspectable automata dynamics | HMM/WFA heads or factor-graph heads |
| Hosted OpenAI verification | `OpenAIResponsesAdapter.generate_and_verify(...)` |

## Current Limits

- Learned heads operate over compact `TokenVocabulary` labels, not arbitrary
  open-vocabulary text.
- DFA constraints remain the hard guarantee mechanism; PMD and latent losses are
  soft training signals.
- Generic OpenAI-compatible APIs remain generate-then-verify/rerank unless a
  backend-specific guided-decoding adapter is added.
- Batched compact-head decoding and batched HF cache reordering remain future
  work.
