# Real HMM PMD Learning Demo

This is the smallest PMD learning demo in the generation examples.

It is self-contained and has one graph rule:

```text
token B may appear at most once
```

The graph lives in `graph.py`. It declares:

- a `string` made of `position` nodes;
- a `generated_symbol` enum with `A`, `B`, `C`, `D`, `END`, and `_other`;
- one DomiKnowS logical constraint: `atMostAL(generated_symbol.B("x"), 1)`.

The demo then adapts that graph into the generation tooling:

```text
DomiKnowS graph       -> declares the rule
graph-discovered DFA -> hard verifier
compact-label learner -> trainable model
PMD loss             -> symbolic pressure over DataNodes
```

Training uses the standard DomiKnowS call:

```python
program.train(...)
```

There is no task-specific optimizer loop in this demo. The selected compact
learner is a normal `ModuleLearner` module.

Two compact learners are available:

```text
Graph-HMM learner = graph-shaped probabilistic automaton learner
Energy learner    = neural local energy scorer over compact labels
DFA               = hard validity checker for both
PMD               = logical constraint pressure for both
```

The mock generator stream emits deterministic prompt-conditioned valid and
invalid strings, up to the current pad size. The default pad size is `100`, and
shorter strings are padded with `END` by the DomiKnowS sensor path.

The prompts are intentionally tiny:

```text
AB    -> prefer A and B over C and D
CD    -> prefer C and D over A and B
short -> prefer short strings that reach END quickly
```

For example, prompt `AB` makes the generator assign higher probability to
symbols `A` and `B`. The DFA still enforces the hard rule, so a second `B` is
rejected even when the prompt likes `B`.

The named diagnostic candidates are still:

```text
valid:   A B C D END
invalid: A B C B END
```

Every generated string is used as a PMD training example. The DFA reports
whether the string violates the rule, but invalid strings are not hidden from
training. This is intentional: it lets the demo show DomiKnowS constraint
pressure pushing back when the generator proposes bad behavior.

The generator is wrapped by a small source object:

```text
GeneratorTrainingSource.next_batch(step) -> generated examples
GeneratorTrainingSource.training_data(batch) -> DomiKnowS PMD samples
PrimalDualProgram.train(...) -> one standard training epoch
```

Run:

```bash
uv run --project Tasks/real_hmm_pmd_learning python Tasks/real_hmm_pmd_learning/run_demo.py --steps 2 --stream-count 4 --inference-prompt AB
uv run --project Tasks/real_hmm_pmd_learning python Tasks/real_hmm_pmd_learning/run_demo.py --learner energy --steps 2 --stream-count 4 --inference-prompt CD
uv run --project Tasks/real_hmm_pmd_learning python Tasks/real_hmm_pmd_learning/run_demo.py --steps 2 --stream-count 4 --pad-size 100
```

Use `--help` to see the small CLI:

```bash
uv run --project Tasks/real_hmm_pmd_learning python Tasks/real_hmm_pmd_learning/run_demo.py --help
```

After training, the demo runs one learned greedy inference path under the DFA
mask. The DFA is still the hard rule checker; the selected compact learner is the
learned compact-label model.

Mental model:

```text
mock generator stream -> proposes prompt-conditioned strings
DFA                   -> reports hard validity
PMD training          -> learns from all streamed strings with graph constraints
compact-label learner -> learned model used for inference
```
