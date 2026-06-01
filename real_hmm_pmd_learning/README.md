# Real HMM PMD Learning Demo

This is the small PMD learning demo in the generation examples.

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

Three compact learners are available:

```text
Discrete-HMM learner = prompt-conditioned DiscreteHMM-backed compact-label learner, the default
Graph-HMM learner    = graph-shaped probabilistic automaton learner
Energy learner       = neural local energy scorer over compact labels
DFA                  = hard verifier discovered from the same graph rule
PMD                  = logical constraint pressure for all learners
```

The mock generator stream emits deterministic prompt-conditioned valid and
invalid strings, up to the current pad size. The default pad size is `6`, and
shorter strings are padded with `END` by the DomiKnowS sensor path.

The prompts are intentionally tiny:

```text
AB    -> prefer A and B over C and D
CD    -> prefer C and D over A and B
short -> prefer short strings that reach END quickly
```

For example, prompt `AB` makes the generator assign higher probability to
symbols `A` and `B`. Every generated string is used directly as a PMD training
example so the demo stays focused on the standard training call and the learned
compact-label model.

The generator is wrapped by a small source object:

```text
GeneratorTrainingSource.next_batch(step) -> generated examples
GeneratorTrainingSource.training_data(batch) -> DomiKnowS PMD samples
PrimalDualProgram.train(...) -> one standard training epoch
```

Run:

```bash
uv run --project Tasks/real_hmm_pmd_learning python Tasks/real_hmm_pmd_learning/run_demo.py --steps 3 --stream-count 4 --inference-prompt AB
uv run --project Tasks/real_hmm_pmd_learning python Tasks/real_hmm_pmd_learning/run_demo.py --learner graph-hmm --steps 3 --stream-count 4 --inference-prompt AB
uv run --project Tasks/real_hmm_pmd_learning python Tasks/real_hmm_pmd_learning/run_demo.py --learner energy --steps 3 --stream-count 4 --inference-prompt CD
uv run --project Tasks/real_hmm_pmd_learning python Tasks/real_hmm_pmd_learning/run_demo.py --steps 3 --stream-count 4 --pad-size 6
```

Run with remote debugging enabled:

```bash
uv run --project Tasks/real_hmm_pmd_learning python Tasks/real_hmm_pmd_learning/run_demo.py --steps 3 --remote-debug --debug-host 127.0.0.1 --debug-port 5678 --debug-wait
```

Then attach your debugger to `127.0.0.1:5678`. Drop `--debug-wait` if you want
the demo to start immediately and attach later.

Use `--help` to see the small CLI:

```bash
uv run --project Tasks/real_hmm_pmd_learning python Tasks/real_hmm_pmd_learning/run_demo.py --help
```

After training, `run_demo.py` prints an explicit inference step. The greedy
search queries the learned compact-label model for next-label scores and emits
the best learned continuation for the selected prompt.

Mental model:

```text
mock generator stream -> proposes prompt-conditioned strings
PMD training          -> learns from streamed strings with graph constraints
compact-label learner -> learned model used for inference
```
