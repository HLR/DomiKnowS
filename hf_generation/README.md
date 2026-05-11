# HuggingFace Generation Example

This task is a small, offline-friendly example for `domiknows.generation`.
It mirrors the generation parts of Collie without the full DomiKnowS training
program: build a graph, write constraints, discover them, compile a DFA, and
decode with a HuggingFace-style model.

## What It Shows

- `GenerationEncoder` builds the `text -> token -> generated_token` graph.
- Raw DomiKnowS constraints in `graph.py` are discovered for hard decoding.
- `andL` / `orL` graph formulas compile into exact DFA intersection / union.
- `HuggingFaceGenerationAdapter` runs constrained greedy, beam search, and
  sampling. Real HuggingFace models use KV-cache decoding when available;
  simple mocks fall back to full-prefix calls.
- The default mock model is deterministic and needs no network access.

The mock graph vocabulary uses `<eos>`. In `--real-hf` mode the demo switches
to the loaded tokenizer's actual EOS token, usually `<|endoftext|>` for
GPT-style tokenizers.

The graph constraints are:

- once `<eos>` appears, only `<eos>` can follow;
- at most 3 non-EOS tokens;
- token `" cat"` must appear;
- token `" dog"` is forbidden;
- either `" The"` or `" mat"` must appear, expressed through `orL`.

The mock model strongly prefers `" dog"`, so successful output demonstrates
that the DFA mask is actually controlling generation.

## Learning vs Enforcement Paths

This example keeps the two paths separate and visible:

```text
graph logical constraints
    |-> PrimalDualProgram / cmodel: soft DomiKnowS constraint loss
    |-> DFA compiler: hard token masking during decoding
```

`run_demo.py` is the hard-enforcement path only: graph constraints are
discovered, compiled to a DFA, and used to mask HuggingFace logits. No
PrimalDual learning happens there.

`learn_demo.py` is the learning path plus final enforcement: it trains a small
compact-label head with supervised loss and DomiKnowS PMD constraint loss, then
shows the taught head both unconstrained and decoded again with graph-discovered
DFA masking.

`automata_demo.py` is the automata learning path: it swaps the frozen-backbone
head for either an HMM head or a signed spectral-WFA head. The head still
populates `token[generated_token]`, so `PrimalDualProgram.cmodel(...)` receives
normal DomiKnowS DataNodes and computes the same graph-constraint loss. The
task loop also adds an automata auxiliary loss: HMM sequence NLL or WFA
energy-style cross-entropy over compact labels.

`prompt_automata_demo.py` is the prompt-conditioned automata path. It compares
the non-prompt HMM/WFA head with a prompt-conditioned variant. The default demo
uses gated dynamics, so the prompt chooses mixture weights over inspectable
transition/emission experts as well as the initial automaton state. It also
uses step-adaptive prefix gating by default, so expert weights can change as
generated compact labels accumulate.

`hmm_factor_demo.py` is the explicit HMM factor-graph path. It adds a
`latent_state` enum concept on each token and an adjacent-token `is_next_rel`
relation, then attaches two projections of one shared HMM: generated-token
marginals and latent-state marginals. By default it also exposes
`forward_state`, `backward_state`, and adjacent `transition_pair` DP factors as
graph concepts, adds weak DP consistency constraints, and prints the DP
diagnostic loss. PMD can now evaluate graph constraints over visible output
labels, HMM hidden states, and graph-visible DP factors.

`wfa_factor_demo.py` is the explicit spectral-WFA factor-graph path. It adds a
`wfa_state` enum concept on each token and can expose adjacent
`wfa_transition_pair` factors. WFA features are signed internally, so the graph
receives normalized projections while the auxiliary energy loss keeps raw WFA
scoring in Torch.

`hybrid_demo.py` is the controller/scorer path. A HuggingFace-style generator
proposes candidates, the graph-discovered DFA verifies them, and a trained
compact-label DomiKnowS head reranks valid candidates and diagnoses rejected
ones. This is the path to use when the compact head should guide a larger
generator rather than replace it.

## Simple Hard-Decoding Demo

From the repository root:

```powershell
uv run --project Tasks/hf_generation python Tasks/hf_generation/run_demo.py
```

Expected behavior: greedy, beam, and sampling all produce accepted sequences
that satisfy the graph constraints.

## Hybrid Controller Demo

```powershell
uv run --project Tasks/hf_generation python Tasks/hf_generation/hybrid_demo.py --steps 3
```

The output shows PMD/head training losses, ranked generator candidates, a
rejected-candidate repair diagnostic, and a next-step risk estimate.

## Constraint Debug Viewer

`visualize_constraints.py` is a thin task wrapper around
`domiknows.generation.run_generation_debug_server(...)`. It builds the same
graph-discovered DFA as `run_demo.py`, traces a generated compact-label
sequence, and opens a local Flask page with the step-by-step DFA trace and DOT
text.

```powershell
uv run --project Tasks/hf_generation python Tasks/hf_generation/visualize_constraints.py
```

For a non-blocking smoke check:

```powershell
uv run --project Tasks/hf_generation python Tasks/hf_generation/visualize_constraints.py --no-server-smoke
```

To inspect a specific compact-label sequence:

```powershell
uv run --project Tasks/hf_generation python Tasks/hf_generation/visualize_constraints.py --sequence 1,2,3,0
```

## Optional Real HuggingFace Model

Real model execution is optional because it may need network/model-cache access:

```powershell
uv run --project Tasks/hf_generation python Tasks/hf_generation/run_demo.py --real-hf --model roneneldan/TinyStories-1M --prompt "Once upon a time"
```

The demo suppresses benign Transformers checkpoint load reports by default.
Pass `--show-transformers-load-report` if you want to inspect them.

The example vocabulary is intentionally tiny and token-level. For real models,
make sure every vocabulary string encodes to exactly one tokenizer id.

## Collie-Style Learning Demo

The separate learning path mirrors Collie at a smaller scale. It builds a
DomiKnowS `PrimalDualProgram`, attaches sensors, uses a `ModuleLearner`, and
trains only a compact generation head on top of a frozen backbone.
The loop optimizes the supervised compact-label loss together with the PMD
constraint loss. The mock defaults use a short-demo learning rate and a
supervised weight of `3.0`, so the three-step run visibly moves from an
invalid prediction to a DFA-accepted sequence.

Loss logs use `pmd_constraint_objective` and `optimization_objective` on
purpose. The PMD term is a signed primal-dual objective, so the combined
optimization objective can become negative. Watch `model_loss`, automata NLL or
energy terms, `positive_training_terms`, and final DFA acceptance when judging
whether the small model is learning.

After training, the demo prints two inference views:

- `After unconstrained`: the compact head's raw argmax labels.
- `After DFA-constrained`: the same taught head decoded with graph-discovered
  DFA masking at every step.

Offline mock mode:

```powershell
uv run --project Tasks/hf_generation python Tasks/hf_generation/learn_demo.py --steps 3
```

Useful knobs:

- `--lr`: head and PMD optimizer learning rate; the mock default is `0.5`.
- `--supervised-weight`: weight for fitting the target labels.
- `--constraint-weight`: weight for the DomiKnowS constraint loss.

Optional real frozen HuggingFace backbone:

```powershell
uv run --project Tasks/hf_generation python Tasks/hf_generation/learn_demo.py --real-hf --model roneneldan/TinyStories-1M --steps 3
```

This learning demo is intentionally separate from `run_demo.py`. The learning
path shows DomiKnowS PMD-style training with constraint loss; the simple path
shows hard DFA enforcement during generation.

## HMM / Spectral WFA Learning Demo

The automata path answers a slightly different question: "can a production
Torch automaton-shaped model participate in DomiKnowS learning?" In this demo,
yes: the HMM or WFA head is attached as a `ModuleLearner` for `generated_token`,
PMD computes constraint loss from the graph, and final inference still uses DFA
masking for hard enforcement.

```powershell
uv run --project Tasks/hf_generation python Tasks/hf_generation/automata_demo.py --kind hmm --steps 3
uv run --project Tasks/hf_generation python Tasks/hf_generation/automata_demo.py --kind wfa --steps 3
```

The important separation is the same as the HF-head path:

- HMM/WFA head: learned soft generator over compact labels.
- `PrimalDualProgram`: soft graph-constraint learning signal.
- DFA decoder: hard guarantee at inference time.

## Prompt-Conditioned Automata Demo

This path answers the next question: "can the small automaton learn prompt-aware
dynamics?" The answer is yes, within the small V1 scope. The prompt encoder
produces the initial HMM distribution or WFA vector and, by default in this
demo, gates a mixture of global transition/emission experts. The demo also
enables step-adaptive dynamics, where the gate is recomputed from prompt
features plus a mean-pooled embedding of generated labels seen so far.

```powershell
uv run --project Tasks/hf_generation python Tasks/hf_generation/prompt_automata_demo.py --kind hmm --steps 3
uv run --project Tasks/hf_generation python Tasks/hf_generation/prompt_automata_demo.py --kind wfa --steps 3
```

Use `--dynamics-conditioning none --step-dynamics-conditioning none` to recover
the initial-state-only behavior, tune `--dynamics-experts` to change the number
of global experts, or pass `--step-dynamics-conditioning none` to keep
prompt-level dynamics without generated-prefix adaptation.

The default encoder is a small trainable embedding/pooling encoder and remains
offline-safe. To use the mock frozen-backbone prompt encoder:

```powershell
uv run --project Tasks/hf_generation python Tasks/hf_generation/prompt_automata_demo.py --encoder frozen-backbone --steps 3
```

The demo prints dynamics weights, a baseline non-prompt prediction, the
prompt-conditioned prediction before/after PMD training, and the final
DFA-constrained decode.

## Explicit HMM Factor-Graph Demo

This path makes the HMM structure visible to DomiKnowS:

```text
text -> token
token -> generated_token
token -> latent_state
token -> forward_state      # optional scaled alpha, enabled by default here
token -> backward_state     # optional normalized beta, enabled by default here
is_next_rel(token_t, token_t+1)
is_next_rel -> transition_pair
```

The demo includes toy latent constraints such as `PER(t) => not LOC(t+1)` and
`LOC(t) => generated_token(t) == " mat"`. The HMM still computes the
probabilistic forward/backward math in Torch; DomiKnowS sees the resulting
DataNode probabilities and computes symbolic PMD loss. With DP factors enabled,
the graph also receives weak consistency constraints like
`forward_state_i(t) AND backward_state_i(t) => latent_state_i(t)` and
`transition_pair_i_j(t,t+1) => latent_state_i(t) AND latent_state_j(t+1)`.

```powershell
uv run --project Tasks/hf_generation python Tasks/hf_generation/hmm_factor_demo.py --steps 3
```

Use `--no-dp-factors` to run only the generated/latent state version. This path
is separate from `automata_demo.py`: the simpler automata demo only predicts
`generated_token`, while the factor demo exposes HMM structure for graph-level
constraints and diagnostics.

## Explicit Spectral-WFA Factor-Graph Demo

This path makes signed WFA structure visible to DomiKnowS:

```text
text -> token
token -> generated_token
token -> wfa_state
is_next_rel(token_t, token_t+1)
is_next_rel -> wfa_transition_pair
```

```powershell
uv run --project Tasks/hf_generation python Tasks/hf_generation/wfa_factor_demo.py --steps 3
```

Use `--no-transition-pairs` to run only generated-token and WFA-state
projections. The graph-visible WFA concepts are normalized projections of signed
features; exact WFA recurrence and energy-style auxiliary loss stay in Torch.

## Files

- `graph.py`: generation graph and raw DomiKnowS constraints.
- `automata_program.py`: PMD builder for HMM and spectral-WFA heads.
- `automata_demo.py`: tiny HMM/WFA training loop and DFA-constrained decode.
- `hmm_factor_program.py`: PMD builder for explicit HMM latent-state and DP-factor graphs.
- `hmm_factor_demo.py`: tiny factor-graph HMM loop and DFA-constrained decode.
- `wfa_factor_program.py`: PMD builder for explicit spectral-WFA state and transition-pair graphs.
- `wfa_factor_demo.py`: tiny factor-graph WFA loop and DFA-constrained decode.
- `prompt_automata_program.py`: PMD builder for prompt-conditioned HMM/WFA heads.
- `prompt_automata_demo.py`: prompt-conditioned automata loop and DFA decode.
- `learning_model.py`: frozen-backbone compact-label generation head.
- `learning_program.py`: Collie-style DomiKnowS Program builder.
- `learn_demo.py`: tiny PMD training loop.
- `mock_hf.py`: tiny tokenizer/model with HuggingFace-compatible behavior.
- `run_demo.py`: CLI and importable helpers used by tests.
- `visualize_constraints.py`: local Flask debug viewer for graph-discovered DFA traces.
- `pyproject.toml`: task-local uv project.
