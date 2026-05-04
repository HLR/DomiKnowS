# Constraint-Aware Generation

`domiknows.generation` adds a small generation layer on top of DomiKnowS. It
lets one constraint specification connect three places:

- DomiKnowS graph constraints for verification and loss.
- DFA/FSM constraints for hard token-level decoding.
- Soft latent losses for differentiable internal bias.

The package is intentionally lightweight. Core DFA/graph tooling is
dependency-light; HuggingFace decoding uses `torch`; spectral learning uses
`torch`; OpenAI support is generate-then-verify.

## Main Pieces

| Tool | Purpose |
| --- | --- |
| `GenerationEncoder` | Builds the common `text -> token -> generated_token` DomiKnowS graph shape. |
| `GenerationConstraint` classes | Declarative constraints such as EOS closure, max length, required tokens, forbidden tokens, ordered tokens, and conditional max length. |
| `constraints_to_dfa(...)` | Compiles constraint objects into one product DFA. |
| `discover_generation_constraints(...)` | Reads supported raw DomiKnowS graph constraints back into generation constraints. |
| `discover_generation_enforcement(...)` | Routes graph constraints into hard DFA constraints and soft latent specs. |
| `HuggingFaceGenerationAdapter` | Runs true hard constrained decoding by masking logits with a DFA. |
| `OpenAIResponsesAdapter` | Calls the OpenAI Responses API, then encodes and verifies output post hoc. |
| `latent_constraints.py` | Product t-norm soft losses over token/latent probability sequences. |
| `automata` | DFA, HMM/PFA checker, Hankel projection, WFA, and spectral WFA learning utilities. |

## Basic DFA Constraints

Start with a generation vocabulary. The vocabulary is a compact label space
used by DomiKnowS and the DFA. It also includes an `_other` label for tokenizer
IDs outside the listed tokens.

```python
from domiknows.generation import (
    TokenVocabulary,
    constraints_to_dfa,
    max_non_eos,
    no_token_after_eos,
    required_token,
)

vocabulary = TokenVocabulary(
    ["<|endoftext|>", " The", " slide"],
    eos_token="<|endoftext|>",
    tokenizer=tokenizer,
)

constraints = (
    no_token_after_eos(),
    max_non_eos(4),
    required_token(" The"),
    required_token(" slide"),
)

dfa = constraints_to_dfa(constraints, vocabulary)
```

The resulting DFA accepts label sequences satisfying all constraints. For
example, if `The`, `slide`, and `EOS` are labels:

```python
the = vocabulary.label_for_token(" The")
slide = vocabulary.label_for_token(" slide")
eos = vocabulary.eos_label

assert dfa.accepts([the, slide, eos, eos])
assert not dfa.accepts([the, eos, slide])  # EOS closure violation
```

## Building a DomiKnowS Generation Graph

Use `GenerationEncoder` to build the common graph shape used by generation
tasks:

```python
from domiknows.generation import (
    GenerationEncoder,
    max_non_eos,
    no_token_after_eos,
    required_token,
)

encoder = GenerationEncoder(
    vocab=["<|endoftext|>", " The", " slide"],
    eos_token="<|endoftext|>",
    tokenizer=tokenizer,
    graph_name="main",
)

graph, bundle = encoder.build_graph(
    constraints=[
        no_token_after_eos(),
        max_non_eos(4),
        required_token(" The"),
        required_token(" slide"),
    ]
)
```

`bundle` contains the graph concepts and the vocabulary:

```python
bundle.text
bundle.token
bundle.contains
bundle.generated_token
bundle.is_before_rel
bundle.vocabulary
```

Constraints that support DomiKnowS compilation are also written into
`graph.logicalConstrains`, so the same rule can participate in DomiKnowS
verification/loss and DFA decoding.

## Auto-Picking Graph Constraints

If a task writes supported constraints directly in the graph, the generation
module can discover them:

```python
from domiknows.generation import constraints_to_dfa_from_graph
from domiknows.graph.logicalConstrain import atMostAL, notL

with graph:
    # at most 4 non-EOS tokens
    atMostAL(
        notL(bundle.context.token_value("<|endoftext|>", "x")),
        4,
    )

dfa = constraints_to_dfa_from_graph(graph, bundle, on_unsupported="warn")
```

Supported graph-discovery patterns in V1:

| DomiKnowS shape | Generation constraint |
| --- | --- |
| `ifL(is_before_rel, ifL(EOS(first), EOS(second)))` | `EosClosureConstraint` |
| `atMostAL(notL(EOS("x")), n)` | `MaxNonEosConstraint(n)` |
| `atLeastAL(token("x"), n)` | `RequiredTokenConstraint(token, n)` |
| `existsAL(token("x"))` | `RequiredTokenConstraint(token, 1)` |
| `atMostAL(token("x"), 0)` | `ForbiddenTokenConstraint(token)` |
| `ifL(existsAL(token("x")), atMostAL(notL(EOS("y")), n))` | `ConditionalMaxNonEosConstraint(token, n)` |
| `andL(supported_constraint, ...)` | all child constraints must hold |
| `orL(supported_branch, ...)` | at least one branch must hold |

Unsupported generation-relevant graph constraints are skipped, warned, or
raised depending on `on_unsupported="ignore" | "warn" | "error"`.
Boolean discovery supports nested `andL` / `orL` formulas over the supported
generation leaves above.  `andL` is compiled as DFA intersection.  `orL` is
compiled as DFA union, so every branch must be fully generation-supported;
mixed non-generation or unsupported `orL` branches are not approximated.

## Routing DFA vs Latent Enforcement

DFA constraints are hard: they restrict which tokens can be generated.
Latent constraints are soft: they add differentiable losses over probability
sequences.

Use explicit markers when a graph constraint should route to latent loss or to
both hard and soft enforcement:

```python
from domiknows.generation import (
    LatentWindowSpec,
    discover_generation_enforcement,
    mark_for_latent,
)
from domiknows.graph.logicalConstrain import existsAL, ifL

with graph:
    lc = ifL(
        existsAL(bundle.context.token_value(" The", "x")),
        existsAL(bundle.context.token_value(" slide", "y")),
    )

    mark_for_latent(
        lc,
        LatentWindowSpec(
            if_label=bundle.vocabulary.label_for_token(" The"),
            formula=bundle.vocabulary.label_for_token(" slide"),
            window=2,
            weight=0.5,
        ),
    )

enforcement = discover_generation_enforcement(graph, bundle)
latent_loss = enforcement.latent_loss(token_probs)
```

`token_probs` may be shaped `[seq_len, label_count]` or
`[batch, seq_len, label_count]`.

## HuggingFace Hard Constrained Decoding

HuggingFace support can enforce DFA constraints during decoding because the
decoder can inspect and mask logits at every token step.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from domiknows.generation import (
    GenerationEncoder,
    HuggingFaceGenerationAdapter,
    constraints_to_dfa_from_graph,
    max_non_eos,
    no_token_after_eos,
    required_token,
)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")

vocab = ["<|endoftext|>", " The", " slide"]
encoder = GenerationEncoder(vocab, eos_token="<|endoftext|>", tokenizer=tokenizer)
graph, bundle = encoder.build_graph([
    no_token_after_eos(),
    max_non_eos(4),
    required_token(" The"),
    required_token(" slide"),
])

dfa = constraints_to_dfa_from_graph(graph, bundle)
adapter = HuggingFaceGenerationAdapter(model, tokenizer, bundle.vocabulary)

prompt_ids = tokenizer("Once upon a time", return_tensors="pt").input_ids
result = adapter.constrained_greedy(
    input_ids=prompt_ids,
    dfa=dfa,
    max_new_tokens=8,
)

print(tokenizer.decode(result.token_ids))
print(result.labels)
print(result.accepted)
```

The HuggingFace adapter also exposes constrained beam search and constrained
sampling. Both modes apply the DFA mask before choosing tokens, so beam
expansion and stochastic sampling remain inside the hard constraint language.
For HuggingFace-style models, these decoders use `past_key_values` KV-cache
calls when the model supports them and transparently fall back to full-prefix
calls for simple mocks or unsupported models. Pass `use_cache=False` to force
the old full-prefix path while debugging.

```python
beam_result = adapter.constrained_beam_search(
    input_ids=prompt_ids,
    dfa=dfa,
    max_new_tokens=8,
    beam_size=4,
)

sample_result = adapter.constrained_sample(
    input_ids=prompt_ids,
    dfa=dfa,
    max_new_tokens=8,
    temperature=0.8,
    top_p=0.95,
)
```

For custom generation loops, use the lower-level masking helper:

```python
from domiknows.generation import mask_logits_for_dfa

allowed = dfa.allowed_tokens(state, remaining_steps=remaining_steps)
masked_logits = mask_logits_for_dfa(logits, set(allowed), bundle.vocabulary)
```

## OpenAI Generate-Then-Verify

OpenAI support in V1 does not provide hard token-level decoding. Hosted APIs do
not expose a per-token decoder hook or gradients, so the adapter generates text,
maps the output into the DomiKnowS vocabulary, and verifies it against the DFA.

```python
from openai import OpenAI

from domiknows.generation import (
    OpenAIResponsesAdapter,
    constraints_to_dfa_from_graph,
)

client = OpenAI()
adapter = OpenAIResponsesAdapter(
    client=client,
    model="gpt-4.1-mini",
    tokenizer=tokenizer,
)

generation = adapter.generate(
    "Write a short sentence containing The and slide.",
    max_output_tokens=40,
)

labels = adapter.encode_output(generation.text, bundle.vocabulary)
accepted = dfa.accepts(labels)

print(generation.text)
print(labels)
print(accepted)
```

Use this path for API-based models when verification is enough. Use
HuggingFace or another local decoder when hard per-token guarantees are
required.

## Do You Need to Train a Small Model?

Not necessarily. The current generation package can already impose constraints
without training a new model:

```text
OpenAI/GPT output -> encode -> DFA/DomiKnowS verification
local HF logits   -> DFA mask -> hard constrained decoding
```

So a trained small model is not required for basic constraint enforcement. The
DFA and graph-verification layers already handle "valid vs invalid" rules.

Training a small model becomes useful when the problem is no longer only:

```text
Is this output valid?
```

but also:

```text
Among valid outputs, which one is best?
Which soft preference should be favored?
Which candidate is likely to need repair?
Which constraints should be active for this prompt?
```

A useful architecture is:

```text
prompt
  -> GPT / HF model generates candidates
  -> DFA or DomiKnowS filters invalid candidates
  -> small trained model scores/reranks valid candidates
  -> return the best valid output
```

The trained small model can be much smaller than the generator because it does
not need to reproduce the whole language model. It can learn one focused role:

- rerank several valid candidates;
- learn domain style or task-specific preferences;
- predict constraint-satisfaction risk before expensive retries;
- guide repairs after a failed verification;
- learn soft latent preferences that are not naturally represented as a DFA;
- select which constraints should apply to a prompt.

In short:

```text
DFA / DomiKnowS constraints = rule-based controller
trained small model         = learned scorer / controller
large generator             = fluent text producer
```

The small model is optional. It is most useful when many outputs are valid but
only some are good.

## Latent Constraint Losses

`latent_constraints.py` implements product t-norm soft logic:

- `soft_not(x) = 1 - x`
- `soft_and(a, b) = a * b`
- `soft_or(a, b) = 1 - (1 - a) * (1 - b)`
- `soft_exists(...) = 1 - product(1 - p_i)`
- `implication_loss(lhs, rhs) = lhs * (1 - rhs)`

Example multi-step rule:

```python
from domiknows.generation import window_formula_loss

PER, ORG, LOC, DATE, O = range(5)

# PER_t => (ORG and DATE in next 4) or LOC in next 4
loss = window_formula_loss(
    probs,
    if_label=PER,
    formula=("or", ("and", ORG, DATE), LOC),
    window=4,
)
```

Every integer label inside a window formula means "that label exists somewhere
in the next window." This is useful for soft internal bias, not exact decoding.

## HMM, Hankel, and Spectral Tools

The automata subpackage also contains experimental tools for the HMM/spectral
side of constrained generation research.

### HMM/PFA Checker

```python
from domiknows.generation.automata import (
    ProbabilisticAutomaton,
    all_sequences,
    compare_hmm_dfa,
)

hmm = ProbabilisticAutomaton(
    transition=[[0.8, 0.2], [0.3, 0.7]],
    emission=[[0.9, 0.1], [0.2, 0.8]],
    initial=[0.9, 0.1],
    symbols=["a", "b"],
)

dfa = hmm.extract_argmax_dfa()
summary = compare_hmm_dfa(hmm, dfa, all_sequences(["a", "b"], max_length=3))
```

Full discrete Baum-Welch training is available:

```python
from domiknows.generation.automata import baum_welch_train

result = baum_welch_train(
    [["a", "a", "b"], ["b", "b", "a"]],
    symbols=["a", "b"],
    state_count=2,
)

dfa = result.model.extract_argmax_dfa()
```

### Hankel Projection

For a weighted finite automaton, a finite Hankel table is:

```text
H(u, v) = P(uv)
```

A regular/logical constraint projects it to:

```text
H_C(u, v) = 1[uv accepted by DFA] * P(uv)
```

```python
from domiknows.generation.automata import (
    WeightedFiniteAutomaton,
    constrained_hankel_matrix,
    hankel_matrix,
    start_product_state,
    step_product_state,
)

wfa = WeightedFiniteAutomaton(...)
H = hankel_matrix(wfa, prefixes, suffixes)
H_c = constrained_hankel_matrix(wfa, dfa, prefixes, suffixes)

state = start_product_state(wfa, dfa)
state = step_product_state(wfa, dfa, state, "energy")
```

### Spectral WFA Learning

Spectral learning reconstructs a signed WFA from finite Hankel blocks using
Torch SVD.

```python
from domiknows.generation.automata import (
    build_spectral_basis,
    spectral_learn_from_oracle,
    spectral_learn_from_samples,
)

basis = build_spectral_basis(["a", "b"], max_prefix_len=2, max_suffix_len=2)

oracle_result = spectral_learn_from_oracle(
    probability_fn,
    symbols=["a", "b"],
    rank=2,
    basis=basis,
)

sample_result = spectral_learn_from_samples(
    sequences=[("a",), ("a",), ("b", "a")],
    symbols=["a", "b"],
    rank=2,
    basis=basis,
)
```

The learned WFA is signed. Negative transition weights or negative sequence
scores are preserved and reported in `result.diagnostics`; they are not
clamped into probabilities.

## Collie Example

`Tasks/collie` demonstrates the full bridge:

- It builds the generation graph with `GenerationEncoder`.
- It writes raw DomiKnowS constraints in `graph.py`.
- It compiles those graph constraints to a DFA for constrained decoding.
- It includes a toy latent-marked constraint and `latent_example.py`.

Typical Collie commands from the repo root:

```powershell
uv run --project Tasks/collie python Tasks/collie/build_vocab.py
uv run --project Tasks/collie python Tasks/collie/program.py --vocab_file Tasks/collie/data/vocab_val_10k.pkl --constrained_decoding
uv run --project Tasks/collie python Tasks/collie/latent_example.py
```

## Current Limits

- Graph discovery auto-compiles only the supported generation-shaped subset:
  EOS closure, max non-EOS length, required/forbidden tokens, conditional max
  length, and nested `andL` / `orL` over those leaves. It does not reverse
  compile arbitrary DomiKnowS logic, arbitrary `notL`, custom predicates, or
  relation-heavy formulas.
- HuggingFace hard decoding supports constrained greedy, beam search, and
  sampling. It uses `past_key_values` KV-cache calls when available and falls
  back to full-prefix calls for simple mocks or unsupported models. V1 is still
  single-prompt decoding; batched cache reordering is not implemented.
- Learned compact-label heads can be trained through the DomiKnowS-style
  learning path and decoded with DFA-constrained greedy decoding. Learned-head
  beam search and sampling are not implemented yet.
- OpenAI integration is generate-then-verify only; hosted APIs do not expose a
  per-token decoder hook for DFA masking.
- Token constraints operate over tokenizer-token labels, not arbitrary
  detokenized words or multi-token phrases.
- Latent constraints are explicit marked specs and soft losses. The module does
  not infer latent loss formulas from arbitrary DomiKnowS logic, and latent loss
  alone does not guarantee hard validity.
- HMM/PFA, Hankel, and spectral WFA tools are finite toy/research utilities, not
  production language model trainers.

## TODOs and Enhancements

- Add phrase-level constraints that compile multi-token phrases into DFA states.
- Add regex/grammar/JSON constraints through an optional extra such as
  `generation-guided`.
- Add broader DomiKnowS-to-DFA compilation beyond the current generation-shaped
  boolean subset.
- Add constrained beam search and sampling for learned compact-label heads.
- Add batched HuggingFace constrained decoding, including batched KV-cache
  reordering for beam search.
- Add adapter-level verification helpers so OpenAI outputs can return
  `GenerationResult(accepted=...)` directly.
- Add OpenAI structured-output examples where the constraint is schema-level
  rather than token-level.
- Add latent-potential APIs for HMM/WFA transition reweighting.
- Add product-automaton visualization utilities for debugging constraints.
- Add benchmark scripts for CTRL-G-style lexical constraints.
- Add documentation showing how to register custom `GenerationConstraint`
  classes.
