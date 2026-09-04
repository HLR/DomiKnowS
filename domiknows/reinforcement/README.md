# `domiknows.reinforcement`

Reward-driven (reinforcement-learning) training for DomiKnowS.

`ReinforcementProgram` optimizes a graph against a **reward** instead of supervised
labels or differentiable constraint loss. At each step it samples discrete
decodings from the model's predicted distributions, scores each decoding with a
reward, and builds a differentiable loss that pushes probability mass toward the
high-reward decodings.

---

## Quick start

### Reward from a Python function

```python
import torch
from domiknows.program import ReinforcementProgram

def reward_fn(generator_output):
    return torch.tensor([1.0 if generator_output == "yes" else 0.0])

program = ReinforcementProgram(
    graph,
    targets=[b_answer],              # the discrete decision concept(s) to sample
    reward_function=reward_fn,       # reward_fn(generator_output) -> Tensor / float
    num_samples=16,
    estimator="importance_weighted", # or "reinforce"
)
program.train(dataset, train_epoch_num=200,
              Optim=lambda p: torch.optim.Adam(p, lr=5e-3))

print(program.evaluate_reward(dataset))   # mean reward of sampled decodings
```

Reward functions can also accept optional context keywords when they need the
current data item or sampled assignment:

```python
def reward_fn(generator_output, *, data_item=None, samples=None, targets=None, **_):
    expected = data_item["logic_label"]
    return torch.tensor([1.0 if generator_output == expected else 0.0])
```

Both forms are valid. This keeps the reward method a single plain Python
function: simple rewards close over metadata or take one argument; richer rewards
accept context.

### Reward from the graph's declared constraints

If your graph declares logical constraints (`ifL`, `atLeastL`, `atMostL`,
`exactL`, …), the reward can be how well a sampled decoding satisfies them — no
reward function required:

```python
program = ReinforcementProgram(
    graph,
    targets=[b_answer],
    reward_from_constraints=True,
    num_samples=48,
    estimator="importance_weighted",
)
```

Both sources can be combined; see [Reward sources](#reward-sources).

---

## How it works internally

For one data item, `train_epoch` does the following (see
[`reinforcement_program.py`](reinforcement_program.py)):

1. **Run the graph.** `self.model(data_item)` executes all sensors/learners and
   builds a `DataNode`. The model is a `ReinforcementModel` (a `PoiModel` whose
   `default_poi` runs *every* sensored property, since there are no labels). The
   model's own (supervised) loss is ignored.

2. **Collect predicted logits.** For each target concept the program reads the
   per-instance logits from the DataNode
   (`findDatanodes(select=<root concept>)` + `getAttribute(concept)`), producing
   one `[n_instances, n_classes]` tensor per target. These tensors are part of
   the autograd graph.

3. **Sample decodings.** `sample_assignments` draws `num_samples` joint
   assignments. With `weighted=True` (default) it samples from the model's own
   categorical distribution (`Categorical(logits=…)`, importance-weighted like a
   `WeightedSamplingSolver` style); with `weighted=False` it samples uniformly
   (`SamplingSolver` style). A "decoding" is one class index per instance of
   every target.

4. **Score each decoding.** Each sample is turned into a reward (a non-negative
   float). See [Reward sources](#reward-sources). Rewards are constants w.r.t.
   the model — no gradient flows through them.

5. **Compute the loss.** The log-probability of each decoding,
   `logprob_s = Σ log_softmax(logits)[chosen class]` (`decoding_logprob`), is
   combined with the per-sample rewards by the chosen estimator
   ([below](#estimators)). Gradients flow through `logprob`, so the learners are
   updated to make high-reward decodings more likely.

6. **Backprop.** `self._backward_and_step(loss)` (inherited from
   `LearningBasedProgram`) handles AMP, gradient clipping, and the optimizer step.

`test_epoch` runs the same scoring under `torch.no_grad()`; `evaluate_reward`
reports the mean reward of freshly sampled decodings.

---

## Estimators

Let `lp_s` be the log-probability of decoding `s` and `r_s ≥ 0` its reward
(`sampling.py`).

### `estimator="importance_weighted"` — `importance_weighted_loss`

```
log_weight_s = target_lp_s - stop_gradient(proposal_lp_s)
loss = -( logsumexp_s( log_weight_s + log r_s )
          - logsumexp_s( log_weight_s ) )
```

Callers that sample from a proposal policy should pass the proposal
log-probabilities. This removes proposal-frequency bias; for on-policy samples
the forward weights are equal while the target-policy gradient remains. If no
proposal probabilities are supplied, the helper retains the historical
sampled-mass objective for compatibility with generic graph samplers.
Both paths are verified in `test_regr/Reinforcement/test_sampling.py`.

### `estimator="reinforce"` — `reinforce_loss`

```
loss = -mean_s( (r_s - b) * lp_s )
```

The classic REINFORCE policy gradient. `b` is a baseline for variance reduction:
`baseline="mean"` (default) subtracts the batch-mean reward, `baseline=None`
uses no baseline. The advantage `(r_s - b)` is detached.

---

## Reward sources

A per-sample reward can come from a **reward function**, **graph constraints**, or
both (added together).

### 1. Reward function

A callable returning `Tensor | list | float`. Old-style rewards use
`reward_function(generator_output)`. Context-aware rewards may accept any of
`data_item`, `datanode`, `samples`, or `targets` as keyword-only or `**kwargs`
parameters. Outputs are normalized to a reward tensor and reduced to a scalar
with `.mean()`.

Provide a reward globally via `reward_function=...`, or per data item under
`reward_key` (default `"reward_function"`); the per-item function takes
precedence. Per-item closures are useful for examples with different labels or
constraints:

```python
from domiknows.reinforcement import make_binary_reward_function

data_item = {
    "logic_str": "question label",
    "logic_label": "yes",
    "reward_function": make_binary_reward_function("question label", "yes"),
}
```

The bridge from a sampled decoding to `generator_output` is the **decoder**:

- **Default decoder**: the flat list of sampled class indices across all targets,
  e.g. `[0, 1, 1, 0, …]`.
- **Custom decoder**: `decoder(samples, targets, datanode, data_item) -> generator_output`,
  where `samples` is `{concept: index_tensor[n_instances]}`. Use this when the
  reward expects something richer than raw indices (e.g. the hard NER example
  decodes a sample to a `"yes"/"no"` answer).

### 2. Graph constraints (`reward_from_constraints=True`)

The reward is how well a sampled decoding satisfies the constraints declared in
the graph, in `[0, 1]` (see [`constraint_reward.py`](constraint_reward.py)). For
each decoding the helper temporarily writes the sampled class into the DataNode
as a near-one-hot prediction, calls `DataNode.verifyResultsLC()` so DomiKnowS'
constraint verifier evaluates the constraints **on that decoding**, then restores
the original predictions (the model's logits/autograd graph are left untouched).

Per-constraint satisfaction rates (the conditional `ifSatisfied` rate for
`ifL`/`forAllL`, otherwise `satisfied`) are combined by
`constraint_reward_aggregate`:

- `"mean"` (default) — average satisfaction across constraints,
- `"min"` — the worst constraint,
- `"prod"` — product (≈ all-constraints-satisfied).

> Note: this re-runs the verifier once per sample, so it is heavier than a plain
> reward function. It assumes non-skeleton DataNodes (the default for
> `ReinforcementModel`).

### Combining

```
reward = function_reward + constraint_reward_weight * constraint_reward
```

At least one source must be enabled, or `reinforcement_loss` raises.

### Reusable reward helpers

`domiknows.reinforcement` exports small helpers for common reward plumbing:

- `flatten_generator_output(...)`
- `normalize_text(...)`
- `binary_label(...)`
- `binary_label_name(...)`
- `coerce_label_tensor(...)`
- `binary_match_reward(...)`
- `count_reward(...)`
- `make_binary_reward_function(...)`
- `make_count_reward_function(...)`

Domain-specific dense rewards should compose these helpers locally instead of
moving task-specific scoring formulas into the core reinforcement package.

---

## Options

| Argument | Default | Meaning |
|---|---|---|
| `targets` | `None` | Concepts to sample (the decision variables). `None` → auto-detect every learner-backed concept. |
| `reward_function` | `None` | Global `reward_fn(generator_output)`; fallback when a data item has no own reward. |
| `reward_key` | `"reward_function"` | Data-item key holding a per-item reward function (takes precedence). |
| `num_samples` | `8` | Decodings drawn per step. |
| `estimator` | `"importance_weighted"` | `"importance_weighted"` (log-ratio) or `"reinforce"` (policy gradient). |
| `weighted` | `True` | Sample from the model distribution (`True`) or uniformly (`False`). |
| `decoder` | `None` | Maps a decoding to `generator_output`; `None` → flat index list. |
| `baseline` | `"mean"` | REINFORCE baseline (`"mean"` or `None`). |
| `reward_from_constraints` | `False` | Use the graph's declared constraints as a reward source. |
| `constraint_reward_weight` | `1.0` | Scale on the constraint reward when combined with a function reward. |
| `constraint_reward_aggregate` | `"mean"` | Combine per-constraint rates: `"mean"`, `"min"`, `"prod"`. |
| `Model` | `ReinforcementModel` | Model class used to build the DataNode. |

Remaining `**kwargs` (e.g. `poi`, `device`) are forwarded to the model /
`LearningBasedProgram`.

### Useful methods

- `train(dataset, train_epoch_num=…, Optim=…, device=…)` — inherited training loop.
- `evaluate_reward(dataset, num_samples=None, device=None)` — mean reward of
  sampled decodings (no gradient); handy to confirm training improves the reward.
- `reinforcement_loss(datanode, reward_fn, data_item)` — the core
  sampling + reward loss for one item (returns `(loss, mean_reward)`).

---

## Visualization (Flask)

A built-in web visualizer shows the detail of **every training step** —
the *decoding* (predicted per-instance distribution per target), the *sampled
decodings*, the *generated sample* each decoding decodes to, the *applied reward*,
and the *calculated loss* — and, while active, **gates step progression**:
training pauses on each step until you click *Next step* (or *Play* to
auto-advance). It is generic — works for any `ReinforcementProgram` (easy or hard
example) with no other changes, because all per-step detail comes from the
program's `step_hook`.

Activate it inline:

```python
program = ReinforcementProgram(..., visualize=True, visualize_port=5000)
program.train(dataset, train_epoch_num=5, Optim=...)   # opens http://127.0.0.1:5000
```

or attach an explicit visualizer (configure host/port, start it yourself):

```python
from domiknows.reinforcement import ReinforcementVisualizer
viz = ReinforcementVisualizer(port=5000).attach(program).start()
program.train(...)
```

Both examples expose a `--visualize` flag (and `--port`), e.g.
`python test_regr/Reinforcement/easy_example/main.py --visualize --epoch 20`.

How it works: when activated, `ReinforcementProgram.reinforcement_loss` builds a
JSON-safe payload of the step and calls `step_hook(payload)`. The visualizer
serves a dashboard in a daemon thread and **blocks the training thread** inside
the hook until the browser advances it (a `threading.Condition`). Endpoints:
`GET /api/state`, `POST /api/next`, `POST /api/play`, `POST /api/pause`,
`POST /api/stop`. Flask is imported lazily, so it is only required when the
visualizer actually starts.

The **Stop** button aborts training (the hook raises `VisualizationStopped`) and,
by default, exits the whole program (`sys.exit`). Pass
`ReinforcementVisualizer(..., exit_on_stop=False)` to instead just stop training
and let `program.train` return.

---

## Files

| File | Contents |
|---|---|
| [`sampling.py`](sampling.py) | Pure-torch helpers: `sample_assignments`, `decoding_logprob`, `importance_weighted_loss`, `reinforce_loss`. |
| [`constraint_reward.py`](constraint_reward.py) | `constraint_satisfaction_reward` — reward derived from declared graph constraints. |
| [`reinforcement_program.py`](reinforcement_program.py) | `ReinforcementProgram` and `ReinforcementModel` (incl. the per-step `step_hook` payload). |
| [`visualization.py`](visualization.py) | `ReinforcementVisualizer` — Flask step-by-step dashboard that gates progression. |
| [`__init__.py`](__init__.py) | Public exports (also re-exported as `from domiknows.program import ReinforcementProgram`). |

Examples: `test_regr/Reinforcement/easy_example` (single EnumConcept, function
reward, both estimators) and `test_regr/Reinforcement/hard_example` (NER +
relations, per-question reward with a custom yes/no decoder). Tests:
`test_regr/Reinforcement/test_sampling.py`, `test_constraint_reward.py`, and
`test_visualization.py`.

---

## Notes & limitations

- Rewards are non-differentiable by design; gradients flow only through
  `decoding_logprob`. This is standard for sampling/policy-gradient estimators.
- Reward signal can be sparse (e.g. an exact-count target). Increase
  `num_samples`, prefer `weighted=True`, and/or use `reinforce` with the mean
  baseline to reduce variance.
- The constraint reward source requires non-skeleton DataNodes (the default) and
  re-runs the verifier per sample.
