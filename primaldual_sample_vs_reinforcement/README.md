# Primal-Dual Sample Loss vs ReinforcementProgram

This task compares two ways to learn the same global counting constraint:

```python
exactL(answer_b.zero, 3)
```

Both learners see the same single data item with eight `b` instances. Each `b`
gets a two-class prediction (`zero` or `one`), and the learning objective is to
make sampled joint decodings contain exactly three `zero` assignments.

The sampling-based constraint learner is DomiKnowS `SampleLossProgram`, which is
the sample-loss variant of the primal-dual constraint-loss stack. It is compared
with `ReinforcementProgram` using an explicit generated-output reward. The RL
decoder turns sampled `answer_b` assignments into generated labels such as
`"zero"` and `"one"`; `reward.py` then scores whether the generated labels
contain exactly the requested number of `"zero"` outputs.

Run:

```bash
uv run --project Tasks/primaldual_sample_vs_reinforcement python Tasks/primaldual_sample_vs_reinforcement/main.py
```

Useful options:

```bash
uv run --project Tasks/primaldual_sample_vs_reinforcement python Tasks/primaldual_sample_vs_reinforcement/main.py --epochs 200 --num-samples 64
uv run --project Tasks/primaldual_sample_vs_reinforcement python Tasks/primaldual_sample_vs_reinforcement/main.py --expected-zeros 4 --num-b 10
uv run --project Tasks/primaldual_sample_vs_reinforcement python Tasks/primaldual_sample_vs_reinforcement/main.py --rl-estimator reinforce
uv run --project Tasks/primaldual_sample_vs_reinforcement python Tasks/primaldual_sample_vs_reinforcement/main.py --sample-size 64 --num-samples 64
```

The printed comparison reports:

- sampled exact-count reward before and after training
- expected number of `zero` predictions from the learned probabilities
- argmax count, which can differ from sampled reward on exact-count tasks
- per-instance `P(zero)`
- sample-loss and reinforcement gradient diagnostics before and after training
- `ReinforcementProgram` generated-output reward before and after training
- which program was better on that run and why

With the default seed and hyperparameters, `SampleLossProgram` is usually better
on final sampled exact-count reward. The reason is task-specific: the
sample-loss path gets a direct graph constraint-loss signal, while
`ReinforcementProgram` receives sparse sampled 0/1 rewards from generated
labels. The reinforcement path is more general because it can optimize an
arbitrary decoder/reward, but on this small exact-count example its gradient
estimate is noisier. Different seeds, sample counts, or learning rates can
change the winner, so the script reports the measured winner for each run.

`reward.py` follows the same pattern as the reinforcement regression examples:
`reward_from_count(...)` scores generated output, and
`make_count_reward_function(...)` creates a per-item reward closure with
attached metadata (`logic_str`, `logic_label`, expected value, expected count,
and mode).
