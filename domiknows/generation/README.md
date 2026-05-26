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

For a focused guide to training compact heads and using them after training,
see [`README_learning.md`](README_learning.md).

For graph-aware HMM and graph-constrained spectral automata learning, see
[`README_graph_hmm.md`](README_graph_hmm.md).

| Tool | Purpose |
| --- | --- |
| `GenerationEncoder` | Builds the common `text -> token -> generated_token` DomiKnowS graph shape. |
| `HMMFactorGraphEncoder` | Builds the opt-in HMM factor graph with generated labels, latent states, adjacency, and optional DP factor concepts. |
| `GenerationConstraint` classes | Declarative constraints such as EOS closure, max length, required tokens, forbidden tokens, ordered tokens, and conditional max length. |
| `constraints_to_dfa(...)` | Compiles constraint objects into one product DFA. |
| `discover_generation_constraints(...)` | Reads supported raw DomiKnowS graph constraints back into generation constraints. |
| `discover_generation_enforcement(...)` | Routes graph constraints into hard DFA constraints and soft latent specs. |
| `HuggingFaceGenerationAdapter` | Runs true hard constrained decoding by masking logits with a DFA. |
| `OpenAIResponsesAdapter` | Calls the OpenAI Responses API, then encodes and verifies output post hoc. |
| `latent_constraints.py` | Product t-norm soft losses over token/latent probability sequences. |
| `learners/compact/` | Shared compact-label head protocol plus GRU, Transformer, neural n-gram, local energy, and CRF heads for PMD, DFA-constrained label decoding, and hybrid scoring. |
| `automata` | DFA plus Torch-backed HMM/PFA, Hankel projection, WFA, and spectral WFA learning utilities. |
| `graph_hmm` | DomiKnowS-aware constrained HMM/spectral learners plus PMD-compatible Torch heads that compile graph structure, static specs, dynamic hooks, and graph-valid Hankel projection into compact-symbol automata. |
| `learners/automata/` | Production Torch HMM/WFA compact-label heads, prompt-conditioned HMM/WFA heads, and auxiliary losses for `PrimalDualProgram` task loops. |
| `hybrid.py` | Large-generator plus compact-head controller/scorer for candidate reranking, risk, repair, soft preferences, and constraint selection. |
| `learners/factors/` | Explicit HMM and WFA factor-graph encoders, shared learner heads, DP/factor projections, and NLL/diagnostic helpers. |

The package has three related automata layers:

- **Hard DFA decoding** masks invalid local HuggingFace tokens or compact labels during generation.
- **Core automata** in `domiknows.generation.automata` provide reusable DFA, HMM/PFA, WFA, Hankel, spectral, and visualization primitives.
- **Graph-aware HMM/spectral learning** in `domiknows.generation.graph_hmm` learns `P(x_1:T | G, C)` from compact sequences typed and restricted by DomiKnowS graph structure.

For automata focused guide,
see ['automata/README.md'](automata/README.md) 

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
breakdown = enforcement.latent_breakdown(
    token_probs,
    eos_label=bundle.vocabulary.eos_label,
)
latent_loss = breakdown.total
```

`token_probs` may be shaped `[seq_len, label_count]` or
`[batch, seq_len, label_count]`. Latent loss helpers also accept masks,
lengths, EOS-aware clipping, and cross-concept dictionaries such as
`{"generated_token": generated_probs, "latent_state": state_probs}`. Use
`latent_mode="auto"` or `"marked_and_auto"` with
`discover_generation_enforcement(...)` to compile conservative adjacent
`is_next_rel` rules over generated-token, HMM, or WFA factor concepts into
window losses. The default remains marked-only for source compatibility.

Project-specific graph fragments can opt in through explicit custom compilers.
Built-in adjacent/windowable discovery runs first; custom compilers are called
only for head logical constraints that the built-ins do not support.

```python
from domiknows.generation import (
    LatentWindowSpec,
    discover_generation_enforcement,
    graph_latent_compiler_result,
)

def my_project_latent_compiler(lc, bundle):
    if lc.__class__.__name__ != "MyProjectRule":
        return None
    return graph_latent_compiler_result(
        latent_specs=LatentWindowSpec(
            if_label=bundle.vocabulary.label_for_token(" The"),
            formula=bundle.vocabulary.label_for_token(" cat"),
            window=3,
            name="project_the_then_cat",
        ),
        compiler_name="my_project",
    )

enforcement = discover_generation_enforcement(
    graph,
    bundle,
    extra_latent_compilers=[my_project_latent_compiler],
)
```

Custom compilers can also return `LatentTransitionPotential` objects or report
`relevant=True, supported=False, reason="..."`; those reasons follow the same
`on_unsupported="ignore" | "warn" | "error"` policy as built-in discovery.

### Packaged Compiler Recipes

`latent_compiler_recipes.py` provides opt-in factory helpers for common
downstream custom compilers. They are convenience wrappers over
`GraphLatentCompiler`; they do not run unless passed explicitly.

```python
from domiknows.generation import (
    common_latent_compiler_recipes,
    discover_generation_enforcement,
)

recipes = common_latent_compiler_recipes(
    adjacent_lc_class_name="MyProjectAdjacentRule",
    adjacent_if_token=" The",
    adjacent_then_token=" cat",
)

enforcement = discover_generation_enforcement(
    graph,
    bundle,
    extra_latent_compilers=recipes,
)
```

Available recipes include adjacent implication, bounded lookahead,
co-occurrence AND/OR formulas, and forbidden latent-state transition
potentials. Each recipe can also take a matcher callback for project-specific
LC shapes.

For training loops, `compute_generation_training_loss(...)` combines
supervised model loss, `program.cmodel(...)` PMD loss, latent loss,
allowed-DFA-mass loss, and automata auxiliary losses into one named
breakdown. The HF and Collie examples expose `--latent-weight`;
`Tasks/hf_generation` also exposes `--allowed-mass-weight`,
`--latent-mode`, and `--latent-diagnostics`.

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

Learned compact-label heads use the same DFA language but decode over
`TokenVocabulary` labels directly. Greedy, beam search, and sampling all append
the concrete tokenizer id for the chosen label before asking the head for the
next label logits.

The compact-label decoder is model-agnostic. Any head implementing the
`CompactLabelSequenceModel` contract can be used here: HMM, WFA/spectral,
graph-HMM, GRU, Transformer, neural n-gram, local energy scorer, exact CRF scorer, or a
project-specific compact sequence model. The head supplies logits; the DFA
supplies the hard validity mask.

`EnergyCompactLabelGenerationHead` is a local energy model: lower sequence
energy means a better candidate, and `next_label_logits(...)` returns negative
next-step energy for decoder compatibility.

`CRFCompactLabelScorer` trains with exact global normalization, but
`constrained_label_*` still uses its local next-label proposal interface.
Exact constrained CRF decoding would require product-state Viterbi over
`(CRF previous label, DFA state)`, for example a future
`constrained_crf_viterbi_decode(...)`.

```python
import torch

from domiknows.generation import (
    constrained_label_beam_search_decode,
    constrained_label_greedy_decode,
    constrained_label_sample_decode,
)

greedy = constrained_label_greedy_decode(head, prompt_ids, bundle.vocabulary, dfa, 8)
beam = constrained_label_beam_search_decode(
    head,
    prompt_ids,
    bundle.vocabulary,
    dfa,
    max_new_tokens=8,
    beam_size=4,
)
sample = constrained_label_sample_decode(
    head,
    prompt_ids,
    bundle.vocabulary,
    dfa,
    max_new_tokens=8,
    temperature=0.9,
    generator=torch.Generator().manual_seed(7),
)
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

generation = adapter.generate_and_verify(
    "Write a short sentence containing The and slide.",
    bundle.vocabulary,
    dfa,
    max_output_tokens=40,
    explain=True,
)

print(generation.text)
print(generation.labels)
print(generation.accepted)
print(generation.rejection)
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

The package-level `HybridController` implements this pattern. It consumes
HuggingFace, OpenAI-compatible, or precomputed candidates, verifies them through
the graph-discovered DFA, and scores them with a compact DomiKnowS head:

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
    keep_rejected=False,
)
best = ranked[0]

risk = controller.predict_failure_risk(prompt_ids, prefix_labels=[])
repair = controller.suggest_repair(" The dog<eos>", prompt_ids=prompt_ids)
```

Use `constrained_label_*` when the compact head itself is the generator. Use
`HybridController` when a large model remains the generator and the compact head
acts as learned scorer/controller.

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

## Latent Transition Potentials

Latent potentials reweight HMM/WFA transition dynamics before scoring or
decoding. For HMMs the potential is a non-negative compatibility factor over
hidden-state transitions and the row is normalized back to a probability
distribution:

```python
from domiknows.generation import (
    apply_hmm_transition_potential,
    forbid_hmm_transition,
    penalize_hmm_transition,
)

PER, O, LOC = 0, 1, 2

# Hard latent mask: PER -> LOC gets zero probability, then the PER row is
# renormalized.
hard = forbid_hmm_transition(PER, LOC, state_count=3)
hmm_masked = hmm.with_transition_potential(hard)

# Soft latent penalty: PER -> LOC is discouraged but still possible.
soft = penalize_hmm_transition(PER, LOC, state_count=3, penalty=0.1)
transition = apply_hmm_transition_potential(hmm.transition, soft)
```

For signed WFAs the same idea reweights transition tensors without stochastic
normalization:

```python
from domiknows.generation import transition_potential_matrix

potential = transition_potential_matrix([
    [1.0, 0.0],
    [1.0, 1.0],
])

wfa_biased = wfa.with_transition_potential(potential)
score = wfa.sequence_probability(("a", "b"), transition_potential=potential)
```

Generation heads accept the same potential for auxiliary scoring paths:

```python
logits = head.next_label_logits(input_ids, transition_potential=soft)
log_probs = head.sequence_log_probs(target_labels, transition_potential=soft)
loss = hmm_sequence_nll(head, target_labels, transition_potential=soft)
```

Prompt-conditioned heads apply transition potentials after prompt-gated
dynamics are mixed. This means the prompt can choose dynamics experts first,
and the symbolic latent potential can still bias or block transitions before
the probabilities are used.

## HMM, Hankel, and Spectral Tools

The automata subpackage also contains Torch-backed tools for the HMM/spectral
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

### Product-Automaton Debugging

Use the visualization helpers when a constraint accepts or rejects something
unexpectedly. The core trace and DOT utilities do not require Flask or
Graphviz:

```python
from domiknows.generation import (
    dfa_to_dot,
    explain_dfa_rejection,
    trace_dfa,
    trace_product_automaton,
)

trace = trace_dfa(dfa, [the_label, eos_label])
print(trace.to_dict())
print(explain_dfa_rejection(dfa, [eos_label]))
print(dfa_to_dot(dfa, highlight_path=trace))

product_trace = trace_product_automaton(wfa, dfa, [the_label, eos_label])
print(product_trace.score_path)
```

For an interactive local view, create the optional Flask app. Flask is imported
lazily, so importing `domiknows.generation` does not start or require a web
server:

```python
from domiknows.generation import run_generation_debug_server

run_generation_debug_server(
    dfa,
    wfa=wfa,                    # optional
    sequence=[the_label, eos_label],
    symbol_labels={0: "<eos>", 1: " The"},
    port=5055,
)
```

The viewer exposes `/`, `/api/trace`, `/api/summary`, `/api/dot`, and
`/api/svg`. It is a local debugging utility, not a production service.
See `README_vizualization.md` for a shorter runbook focused only on this
debug viewer.

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

### HMM/WFA Heads in PrimalDualProgram

`HMMGenerationHead` and `SpectralWFAGenerationHead` make the automata tools
usable as DomiKnowS learning modules. They output log-probabilities shaped
`[seq_len, label_count]` for the `generated_token` enum concept, so they can be
attached with `ModuleLearner` the same way as the compact HuggingFace head.

```python
from domiknows.generation import constrained_label_greedy_decode
from domiknows.generation.learners import HMMGenerationHead, hmm_sequence_nll
from domiknows.sensor.pytorch.learners import ModuleLearner

head = HMMGenerationHead(
    label_count=bundle.vocabulary.label_count,
    state_count=3,
    label_to_token_id=label_token_id_map(bundle.vocabulary),
    trainable=True,
)

token[bundle.generated_token] = ModuleLearner(
    token[bundle.contains],
    text["instruction_tokens"],
    "target_labels",
    module=head,
)

model_loss, _, *output = program.model(sample)
constraint_loss, *_ = program.cmodel(output[1])
aux_loss = hmm_sequence_nll(head, target_labels)
total = model_loss + constraint_loss + aux_loss

result = constrained_label_greedy_decode(
    head,
    prompt_ids,
    bundle.vocabulary,
    dfa,
    max_new_tokens=4,
)
```

For `SpectralWFAGenerationHead`, signed WFA next-symbol scores are treated as
logits. Use `wfa_sequence_energy_loss(...)` as an auxiliary supervised signal.
The optional `allowed_mass_loss(...)` can softly reward probability mass on
DFA-allowed labels, but it is not a replacement for hard DFA decoding.

### Prompt-Conditioned Automata Heads

`PromptConditionedHMMGenerationHead` and
`PromptConditionedSpectralWFAGenerationHead` make the small automata models
learn a prompt-conditioned proposal:

```text
prompt -> initial automaton state
prompt -> gated transition/emission dynamics
prompt + generated prefix -> step-adaptive dynamics
automaton state + dynamics -> output labels
```

By default, current callers still get initial-state conditioning only. Pass
`dynamics_conditioning="gated"` to let the prompt choose a mixture over several
global dynamics experts. Expert 0 is the familiar base transition/emission
parameter, so the automaton remains inspectable while becoming prompt-aware.
Pass `step_dynamics_conditioning="prefix_gated"` with gated dynamics to
recompute expert weights at each generated step from prompt features plus a
mean-pooled learned embedding of generated labels seen so far.

```python
from domiknows.generation.learners import PromptConditionedHMMGenerationHead, hmm_sequence_nll

head = PromptConditionedHMMGenerationHead(
    label_count=bundle.vocabulary.label_count,
    state_count=3,
    label_to_token_id=label_token_id_map(bundle.vocabulary),
    prompt_encoder_type="embedding",
    prompt_vocab_size=1024,
    dynamics_conditioning="gated",
    dynamics_expert_count=3,
    step_dynamics_conditioning="prefix_gated",
    trainable=True,
)

weights = head.prompt_dynamics_weights(sample["instruction_tokens"])
transition = head.prompt_transition_probs(sample["instruction_tokens"])
emission = head.prompt_emission_probs(sample["instruction_tokens"])
step_weights = head.step_dynamics_weights(sample["instruction_tokens"], prefix_labels=[1, 2])

token[generated_token] = ModuleLearner(
    token[contains],
    text["instruction_tokens"],
    "target_labels",
    module=head,
)

aux_loss = hmm_sequence_nll(
    head,
    target_labels,
    instruction_tokens=sample["instruction_tokens"],
)

result = constrained_label_greedy_decode(
    head,
    sample["instruction_tokens"],
    bundle.vocabulary,
    dfa,
    max_new_tokens=4,
)
```

For offline tests and small tasks, use `prompt_encoder_type="embedding"`. For
HuggingFace-style tasks, use `prompt_encoder_type="frozen_backbone"` with a
frozen backbone; only the prompt-conditioning projector and automata parameters
train by default. For WFA heads, transition experts and final-vector experts are
gated in the same way; the mixed tensors remain signed WFA scores, not
probability matrices.

### Explicit HMM Factor Graph

`HMMFactorGraphEncoder` makes the HMM structure visible in the DomiKnowS graph:

```text
text -> token
token -> generated_token
token -> latent_state
is_next_rel(token_t, token_t+1)
```

The shared `HMMFactorGraphHead` exposes `ModuleLearner` projections for
generated-token marginals and latent-state marginals. If
`include_dp_factors=True`, the graph also gets:

```text
token -> forward_state
token -> backward_state
is_next_rel -> transition_pair
```

`forward_state` contains normalized scaled-alpha factors, `backward_state`
contains normalized beta factors, and `transition_pair` contains flattened xi
factors for adjacent `(state_i, state_j)` pairs. DomiKnowS can apply weak
logical consistency rules over these visible concepts, while Torch still owns
the exact numeric forward/backward recurrence and likelihood.

```python
from domiknows.generation.learners import (
    HMMFactorGraphEncoder,
    HMMFactorGraphHead,
    apply_hmm_dp_consistency_constraints,
    hmm_dp_factor_consistency_loss,
)

encoder = HMMFactorGraphEncoder(
    vocab=["<eos>", " cat", " mat"],
    eos_token="<eos>",
    state_names=["PER", "O", "LOC"],
    include_dp_factors=True,
)
graph, bundle = encoder.build_graph()
with graph:
    apply_hmm_dp_consistency_constraints(bundle)

head = HMMFactorGraphHead(
    label_count=bundle.vocabulary.label_count,
    state_names=bundle.state_names,
    trainable=True,
)

token[bundle.generated_token] = ModuleLearner(..., module=head.generated_module())
token[bundle.latent_state] = ModuleLearner(..., module=head.latent_module())
token[bundle.forward_state] = ModuleLearner(..., module=head.forward_module())
token[bundle.backward_state] = ModuleLearner(..., module=head.backward_module())
is_next_rel[bundle.transition_pair] = ModuleLearner(..., module=head.transition_pair_module())

dp_loss = hmm_dp_factor_consistency_loss(head, labels)
```

Use `hmm_factor_sequence_nll(head, labels)` as the probabilistic HMM loss and
`program.cmodel(...)` for DomiKnowS symbolic constraints such as
`PER(t) => not LOC(t+1)`. The DP consistency helper adds rules such as
`forward_state_i(t) AND backward_state_i(t) => latent_state_i(t)` and
`transition_pair_i_j(t,t+1) => latent_state_i(t) AND latent_state_j(t+1)`.
Those rules guide PMD over graph DataNodes; they do not replace the Torch
dynamic program.

### Explicit Spectral-WFA Factor Graph

`SpectralWFAFactorGraphEncoder` exposes the signed WFA structure in the same
generation graph style:

```text
text -> token
token -> generated_token
token -> wfa_state
is_next_rel(token_t, token_t+1)
```

If `include_transition_pairs=True`, adjacent relations also get
`wfa_transition_pair` values for flattened `(state_i, state_j)` factors. WFA
states are signed linear features, not probabilities, so PMD-visible concepts
use normalized `log_softmax(...)` projections of signed state and pair scores.
Exact WFA recurrence and energy scoring remain in Torch.

```python
from domiknows.generation.learners import (
    SpectralWFAFactorGraphEncoder,
    SpectralWFAFactorGraphHead,
    apply_wfa_factor_consistency_constraints,
    wfa_factor_consistency_loss,
    wfa_factor_sequence_energy_loss,
)

encoder = SpectralWFAFactorGraphEncoder(
    vocab=["<eos>", " cat", " mat"],
    eos_token="<eos>",
    state_names=["A", "B", "C"],
    include_transition_pairs=True,
)
graph, bundle = encoder.build_graph()
with graph:
    apply_wfa_factor_consistency_constraints(bundle)

head = SpectralWFAFactorGraphHead(
    label_count=bundle.vocabulary.label_count,
    state_names=bundle.state_names,
    trainable=True,
)

token[bundle.generated_token] = ModuleLearner(..., module=head.generated_module())
token[bundle.wfa_state] = ModuleLearner(..., module=head.state_module())
is_next_rel[bundle.wfa_transition_pair] = ModuleLearner(..., module=head.transition_pair_module())

energy_loss = wfa_factor_sequence_energy_loss(head, labels)
factor_loss = wfa_factor_consistency_loss(head, labels)
```

Use `program.cmodel(...)` for symbolic constraints over `wfa_state` and
`wfa_transition_pair`, and decode the generated-token projection with the same
DFA-constrained compact-label decoder used by other learned heads.

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

## Production Scope

The HMM/PFA, Hankel, spectral WFA, and automata-head stack is Torch-backed for
mid-scale CPU/GPU library use. Production-facing APIs accept batched integer
label tensors with optional lengths, preserve device/dtype, expose diagnostics,
and support `save_pretrained(...)` / `from_pretrained(...)` serialization for
HMM and WFA cores. Spectral learning can train from an oracle, raw samples, or
pre-aggregated counts, so streamed datasets do not need to remain in memory as
expanded sequence lists.

These models are production-quality constrained automata and DomiKnowS
integration tools. They are not intended to replace a large language model as a
general open-domain text generator; they provide compact learned proposal
models, scoring models, latent-state structure, and hard/soft constraint
bridges around generation workflows.

## Current Limits

- Graph discovery auto-compiles the regular token-sequence fragment of
  DomiKnowS constraints: boolean nesting over supported leaves (`andL`, `orL`,
  `notL`, `nandL`, `norL`, `xorL`, `iffL`/`equivalenceL`, and regular
  `ifL(A, B)`), accumulated token/token-set counts (`existsAL`, `atLeastAL`,
  `atMostAL`, `exactAL`), EOS closure, conditional max length, generalized
  before-path implications, and simple ordered-token existence. Arbitrary
  DataNode traversal, `eqL` path filters, query/selection/sum constraints,
  comparative counts, custom predicates, and non-generation relations remain
  solver-side unless explicitly marked with a supported `GenerationConstraint`.
- HuggingFace hard decoding with constrained greedy, beam search, sampling,
  KV-cache use, and full-prefix fallback is implemented for single-prompt
  decoding. Batched constrained decoding and batched cache reordering are not
  implemented.
- Learned compact-label heads can be trained through the DomiKnowS-style
  learning path and decoded with DFA-constrained greedy, beam search, and
  sampling. Batched learned-head decoding remains out of scope.
- `CRFCompactLabelScorer` supports exact globally normalized CRF training and
  exact token marginals for PMD/DataNodes. Exact constrained CRF decoding is
  future work because it needs CRF x DFA product-state Viterbi, not only local
  next-label masking.
- Hybrid controller/scorer support is implemented for HuggingFace,
  OpenAI-compatible, and precomputed candidates. It reranks and diagnoses
  candidates with compact heads, but does not turn hosted APIs into hard
  per-token decoders.
- HMM/WFA generation heads are production Torch compact-label automata modules
  for DomiKnowS PMD learning and constrained decoding. Prompt-conditioned
  variants can gate initial state, prompt-level dynamics, and optional
  step-adaptive dynamics from the generated compact-label prefix. They operate
  over compact `TokenVocabulary` labels and are not standalone open-domain
  language models.
- Explicit HMM and spectral-WFA factor graphs are opt-in. HMM factor graphs can
  expose forward/backward/xi DP factors; WFA factor graphs expose normalized
  projections of signed WFA states and adjacent pair scores. Exact numeric
  recurrences still run in Torch, not as logical equality constraints.
- OpenAI integration can return verified `GenerationResult(accepted=...)`, but
  it is still generate-then-verify only; hosted APIs do not expose a per-token
  decoder hook for DFA masking.
- Token constraints operate over tokenizer-token labels, not arbitrary
  detokenized words or multi-token phrases.
- Latent constraints support marked specs, conservative auto-discovery for
  adjacent `is_next_rel` rules over generation/HMM/WFA factor concepts, masks,
  lengths, EOS-aware clipping, cross-concept `LabelRef` formulas, and per-spec
  diagnostics. Per-call custom graph-to-latent compiler hooks can extend this
  for project-specific regular/windowable fragments. They are still soft
  training signals; latent loss alone does not guarantee hard validity and
  arbitrary non-windowable DomiKnowS logic remains PMD/solver-side unless a
  project compiler supplies a faithful latent loss or transition potential.
- Latent transition potentials are transition-level HMM/WFA reweighting tools.
  Adjacent latent-state graph rules such as `state_i(t) => not state_j(t+1)`
  can be discovered as potentials. Potentials do not yet cover emissions,
  initial-state priors, or arbitrary learned logical formulas.
- Automata cores are production library components for compact alphabets and
  mid-scale CPU/GPU workloads, but they are not distributed LLM trainers or
  standalone replacements for full generative language models.
- Product-automaton visualization is implemented as structured traces, DOT
  text, rendered SVG when Graphviz is available, and a local Flask debug page.
  It does not provide a hosted production UI.

## TODOs and Enhancements

- Add phrase-level constraints that compile multi-token phrases into DFA states.
- Add regex/grammar/JSON constraints through an optional extra such as
  `generation-guided`.
- Add batched HuggingFace constrained decoding, including cache reordering for
  batched beam search.
- Add distributed/multi-GPU automata training loops if workloads grow beyond
  the current mid-scale single-process production target.
- Add OpenAI structured-output examples where the constraint is schema-level
  rather than token-level.
- Add richer backend-native hybrid adapters for vLLM/llama.cpp/Ollama guided
  decoding and server-side logprob/candidate APIs.
- Add benchmark scripts for CTRL-G-style lexical constraints.
- Add documentation showing how to register custom `GenerationConstraint`
  classes.
