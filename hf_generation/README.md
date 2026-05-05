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

`hmm_factor_demo.py` is the explicit HMM factor-graph path. It adds a
`latent_state` enum concept on each token and an adjacent-token `is_next_rel`
relation, then attaches two projections of one shared HMM: generated-token
marginals and latent-state marginals. PMD can now evaluate graph constraints
over both visible output labels and HMM hidden states.

## Simple Hard-Decoding Demo

From the repository root:

```powershell
uv run --project Tasks/hf_generation python Tasks/hf_generation/run_demo.py
```

Expected behavior: greedy, beam, and sampling all produce accepted sequences
that satisfy the graph constraints.

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

The automata path answers a slightly different question: "can a small
automaton-shaped model participate in DomiKnowS learning?" In this demo, yes:
the HMM or WFA head is attached as a `ModuleLearner` for `generated_token`, PMD
computes constraint loss from the graph, and final inference still uses DFA
masking for hard enforcement.

```powershell
uv run --project Tasks/hf_generation python Tasks/hf_generation/automata_demo.py --kind hmm --steps 3
uv run --project Tasks/hf_generation python Tasks/hf_generation/automata_demo.py --kind wfa --steps 3
```

The important separation is the same as the HF-head path:

- HMM/WFA head: learned soft generator over compact labels.
- `PrimalDualProgram`: soft graph-constraint learning signal.
- DFA decoder: hard guarantee at inference time.

## Explicit HMM Factor-Graph Demo

This path makes the HMM structure visible to DomiKnowS:

```text
text -> token
token -> generated_token
token -> latent_state
is_next_rel(token_t, token_t+1)
```

The demo includes toy latent constraints such as `PER(t) => not LOC(t+1)` and
`LOC(t) => generated_token(t) == " mat"`. The HMM still computes the
probabilistic forward/backward math in Torch; DomiKnowS sees the resulting
DataNode probabilities and computes symbolic PMD loss.

```powershell
uv run --project Tasks/hf_generation python Tasks/hf_generation/hmm_factor_demo.py --steps 3
```

This is opt-in and separate from `automata_demo.py`: the simpler automata demo
only predicts `generated_token`, while the factor demo also exposes
`latent_state`.

## Files

- `graph.py`: generation graph and raw DomiKnowS constraints.
- `automata_program.py`: PMD builder for HMM and spectral-WFA heads.
- `automata_demo.py`: tiny HMM/WFA training loop and DFA-constrained decode.
- `hmm_factor_program.py`: PMD builder for explicit HMM latent-state graphs.
- `hmm_factor_demo.py`: tiny factor-graph HMM loop and DFA-constrained decode.
- `learning_model.py`: frozen-backbone compact-label generation head.
- `learning_program.py`: Collie-style DomiKnowS Program builder.
- `learn_demo.py`: tiny PMD training loop.
- `mock_hf.py`: tiny tokenizer/model with HuggingFace-compatible behavior.
- `run_demo.py`: CLI and importable helpers used by tests.
- `pyproject.toml`: task-local uv project.
