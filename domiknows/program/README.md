# DomiKnows Program Components

This directory contains the training program implementations for the DomiKnows framework, providing high-level interfaces for training, testing, and evaluation.

---

## Program Hierarchy Overview

| Program Class | File | Parent Class | Primary Use Case |
|--------------|------|--------------|------------------|
| **Base Programs** | | | |
| `LearningBasedProgram` | `program.py` | - | Base training/testing program with epoch management |
| **Standard Programs** | | | |
| `POIProgram` | `model_program.py` | `LearningBasedProgram` | Train PoiModel on graph properties |
| `SolverPOIProgram` | `model_program.py` | `LearningBasedProgram` | Train SolverModel with inference (ILP/GBI) |
| `SolverPOIDictLossProgram` | `model_program.py` | `LearningBasedProgram` | SolverModel with custom per-property losses |
| `IMLProgram` | `model_program.py` | `LearningBasedProgram` | Train with Inference-Masked Loss |
| `POILossProgram` | `model_program.py` | `LearningBasedProgram` | POI with learner-integrated losses |
| **Constraint Learning Programs** | | | |
| `LossProgram` | `lossprogram.py` | `LearningBasedProgram` | Base for constraint-based training |
| `PrimalDualProgram` | `lossprogram.py` | `LossProgram` | Primal-dual optimization for constraints |
| `GumbelPrimalDualProgram` | `lossprogram.py` | `PrimalDualProgram` | + Gumbel-Softmax support |
| `InferenceProgram` | `lossprogram.py` | `LossProgram` | Learn to predict constraint satisfaction |
| `SampleLossProgram` | `lossprogram.py` | `LossProgram` | Constraint learning with sampling |
| `GumbelSampleLossProgram` | `lossprogram.py` | `SampleLossProgram` | + Gumbel-Softmax support |
| `SemanticLossProgram` | `lossprogram.py` | `SampleLossProgram`, `PrimalDualProgram` | Exact `-log P(constraint)` via circuit WMC |
| `StructuredProgram` | `lossprogram.py` | `PrimalDualProgram` | Constraint structure *inside* the model |
| `GBIProgram` | `lossprogram.py` | `LossProgram` | Gradient-Based Inference training |
| **Specialized Programs** | | | |
| `BatchProgram` | `batchprogram.py` | `LearningBasedProgram` | Mini-batch gradient accumulation |
| `CallbackProgram` | `callbackprogram.py` | `LearningBasedProgram` | Training with lifecycle callbacks |

### Constraint Models (the `CModel` slot)

A `LossProgram` pairs a **model** (the learners) with a **constraint model** that turns graph
logical constraints into a loss. The constraint model is selected via `CModel=` and configured
with kwargs forwarded by signature matching.

| Constraint Model | File | Constraint loss |
|---|---|---|
| `PrimalDualModel` | `model/lossModel.py` | t-norm violation, weighted by Lagrange multipliers |
| `SampleLossModel` | `model/lossModel.py` | sampled violation |
| `SemanticLossModel` | `model/lossModel.py` | exact `-log P(φ)` by weighted model counting |
| `InferenceModel` | `model/lossModel.py` | supervised constraint-satisfaction prediction |

### Structure Modules (the model side)

These change the **forward pass** rather than the loss, and are used through
`StructuredModel` / `StructuredProgram`.

| Module | File | Role |
|---|---|---|
| `StructuredModel` | `model/structured.py` | `SolverModel` that applies constraint structure before the loss |
| `ConstraintRefinement` | `model/refinement.py` | Iterative belief refinement by violation-gradient messages |
| `FactorGraphHead` | `model/factorGraphHead.py` | Exact constrained marginals `p(y \| x, φ)`; MAP decoding (most likely *joint* assignment) |
| `synthesize_model` / `analyze_exclusivity` | `model/synthesis.py` | Shared trunk + joint heads from the graph (advisory) |
| `DualCritic` | `model/dualCritic.py` | Per-grounding Lagrange multipliers |
| `gradSurgery` | `model/gradSurgery.py` | Supervised/constraint gradient conflict: diagnose, PCGrad, CAGrad |

### Quick Selection Guide

**Choose based on your needs:**

- **Standard supervised learning**: `POIProgram`
- **Learning + constraint inference**: `SolverPOIProgram`
- **Learning to satisfy constraints**: `PrimalDualProgram` or `InferenceProgram`
- **Probabilistically exact constraint loss**: `SemanticLossProgram`
- **Constraints shaping the representation, not just scoring the output**: `StructuredProgram`
- **Constraint-respecting predictions without ILP**: `StructuredProgram` with `inferTypes=['MAP']`
  (MAP = maximum a posteriori — the most likely *joint* assignment; see
  [Decoding: MAP vs MPM](#decoding-map-vs-mpm))
- **Gradient-based constraint satisfaction**: `GBIProgram`
- **Better discrete optimization**: `GumbelPrimalDualProgram` or `GumbelSampleLossProgram`
- **Mini-batch training**: `BatchProgram`
- **Custom training lifecycle**: `CallbackProgram`

### Two independent axes

Constraint mechanisms compose along two axes, which is why they combine rather than compete:

| Axis | What it changes | Selected by |
|---|---|---|
| **Constraint model** | how the constraint *loss* is computed and weighted | `compile_lc`, `dual_algorithm`, `dual_granularity`, `CModel` |
| **Model** | the *forward pass* — what the network computes | `refine`, `factor_graph` on `StructuredProgram` |

Because `Model` is a parameter of every program, a structured model composes with any
constraint model:

```python
program = StructuredProgram(
    graph, poi=[...],
    refine=True,                  # refinement inside the forward pass
    factor_graph=True,            # exact constrained marginals
    compile_lc=True,              # compiled (batched) constraint loss
    dual_algorithm='augmented',   # augmented-Lagrangian duals
)
```

---

## Core Components

### `program.py` - Base Program

#### `LearningBasedProgram`
Foundation class for all training programs.

**Key Features:**
- Automatic device management (CPU/GPU)
- Train/validation/test split handling
- Model checkpoint save/load
- Constraint verification utilities
- Flexible optimizer integration

**Key Methods:**
```python
program = LearningBasedProgram(graph, Model, **kwargs)

# Training
program.train(
    training_set, 
    valid_set=None, 
    test_set=None,
    device='auto',
    train_epoch_num=10,
    Optim=torch.optim.Adam
)

# Testing
program.test(test_set, device='cuda')

# Inference/Population
for datanode in program.populate(dataset):
    # Process datanode
    pass

# Save/Load
program.save('model.pt')
program.load('model.pt')

# Constraint Verification
program.verifyResultsLC(data, constraint_names=['LC1', 'LC2'])
```

---

## Standard Programs (`model_program.py`)

### `POIProgram`
Basic training program for `PoiModel`.

**Usage:**
```python
from domiknows.program import POIProgram

program = POIProgram(graph, loss=loss_fn, metric=metric_fn)
program.train(train_data, valid_data, train_epoch_num=50, 
              Optim=torch.optim.Adam)
```

### `SolverPOIProgram`
Training program for `SolverModel` with constraint inference.

**Features:**
- Automatic skeleton mode activation for GBI
- Supports multiple inference types (ILP, GBI, local)

**Usage:**
```python
program = SolverPOIProgram(
    graph, 
    inferTypes=['local/softmax', 'ILP'],
    loss=loss_fn
)
program.train(train_data, train_epoch_num=30)
```

### `SolverPOIDictLossProgram`
Enables custom loss functions per property.

**Usage:**
```python
custom_losses = {
    'property1': loss_fn1,
    'property2': loss_fn2,
    'default': default_loss
}
program = SolverPOIDictLossProgram(graph, dictloss=custom_losses)
```

---

## Constraint Learning Programs (`lossprogram.py`)

### `LossProgram`
Base class for constraint-based learning with dual optimization.

**Key Parameters:**
- `beta`: Weight for constraint loss (default: 1)
- `c_lr`: Constraint optimizer learning rate
- `c_warmup_iters`: Warmup iterations before constraints
- `c_freq`: Frequency of constraint optimizer updates
- `CModel`: constraint-model class (defaults to the program's `DEFAULTCMODEL`)

**How extra kwargs are routed:** anything else passed to the constructor is matched against
`CModel.__init__`'s signature and forwarded to the constraint model. That is why
`compile_lc`, `dual_algorithm`, `circuit_backend` and friends are given to the *program* even
though they configure the constraint model.

> **Gotcha — `Model` as a lambda.** Kwarg routing to the *model* uses
> `signature(Model.__init__)`. For a `model_helper(...)` lambda that resolves to
> `object.__init__`, which reports `**kwargs`, so **every** program kwarg is forwarded to the
> factory and constraint-model options raise `TypeError: unexpected keyword argument`. Pass a
> Model **class** when using constraint-model kwargs, or a factory that accepts and filters
> `**kwargs` (which is what `StructuredProgram` builds internally).

### `PrimalDualProgram`
Implements primal-dual optimization for constraint satisfaction.

**How It Works:**
1. **Primal update**: Minimize data loss + weighted constraint violations
2. **Dual update**: Adjust constraint weights (Lagrange multipliers)
3. Balances prediction accuracy with constraint satisfaction

**Advanced Training Phases:**
```python
program = PrimalDualProgram(graph, Model, beta=1.0)

# Phase-based training
program.train(
    training_set,
    valid_set=valid_set,
    warmup_epochs=10,           # Phase 1: Only data loss
    constraint_epochs=20,        # Phase 2: Combined training
    constraint_only=False,       # Use both losses in Phase 2
    train_epoch_num=30
)

# Constraint-only training (Phase 2)
program.train(
    training_set,
    warmup_epochs=10,
    constraint_epochs=20,
    constraint_only=True,        # Only constraint loss updates model
    constraint_loss_scale=2.0    # Scale constraint influence
)
```

**Training Parameters:**
```python
program.train(
    training_set,
    c_lr=0.05,                   # Dual learning rate
    c_warmup_iters=10,           # Warmup before constraints
    c_freq=10,                   # Dual update frequency
    c_freq_increase=5,           # Increase freq over time
    c_lr_decay=4,                # LR decay strategy
    batch_size=32,               # Gradient accumulation
    print_loss=True              # Log loss per update
)
```

#### Constraint-loss evaluation

```python
PrimalDualProgram(graph, Model, compile_lc=True)
```

`compile_lc=True` swaps the per-datanode Python interpreter for a compiled
(batched-gather) evaluator. Results are identical — parity against the interpreter is
asserted for every supported constraint type and t-norm — and unsupported types fall back
per constraint. Ignored when `sample=True`.

#### Dual variables

Two orthogonal knobs on the constraint model:

| `dual_algorithm` | Behaviour |
|---|---|
| `'ascent'` (default) | Lagrangian dual by sign-flipped gradient ascent through the constraint optimizer |
| `'augmented'` | Augmented Lagrangian: closed-form multiplier update plus a `(ρ/2)·Σv²` penalty, with ρ growing on stagnation |

| `dual_granularity` | Behaviour |
|---|---|
| `'constraint'` (default) | One multiplier per constraint template |
| `'amortized'` | A `DualCritic` predicts a multiplier **per grounding** from detached features, so hard and easy instances are not scaled identically |

All four combinations are supported. Under `amortized` + `augmented` the critic is
*regressed* onto the AL target `λ + ρ·v` rather than ascended, because an augmented
Lagrangian moves its multipliers in closed form and offers no ascent objective.

```python
PrimalDualProgram(graph, Model, dual_algorithm='augmented',
                  dual_granularity='amortized')
```

> **Checkpointing:** when the constraint model carries non-gradient dual state (multipliers,
> penalty coefficients), `program.save()` automatically writes a `{'model', 'cmodel'}` bundle
> so a save-best/reload cycle does not silently reset it. `load()` accepts both that bundle
> and the historical flat format.

#### Gradient surgery

The supervised and constraint gradients can point in opposing directions in parameters both
losses reach — and neither loss reports it: both totals fall while the shared parameters
receive a near-zero resultant.

```python
program = PrimalDualProgram(graph, Model, grad_surgery='diagnose')
program.train(...)
print(program.conflict_stats.render())
# gradient conflict: rate=1.000 (mean cos=-0.5786) over 48 step(s), ...
```

| `grad_surgery` | Behaviour |
|---|---|
| `'none'` (default) | One fused backward pass; unchanged |
| `'diagnose'` | Splits the two gradients and records their conflict **without changing the update** |
| `'pcgrad'` | Projects each gradient out of the other's conflicting direction |
| `'cagrad'` | Bounds the worst-case per-objective decrease (`cagrad_c` sets the trust region) |

**Run `'diagnose'` first.** Resolving requires the two gradients separately, which a fused
backward cannot provide, so it costs an extra backward pass on every step. If a task shows
little conflict, leave surgery off. A parameter counts as *shared* exactly when both losses
produce a gradient for it — parameters only one loss reaches are passed through untouched.

### `GumbelPrimalDualProgram`
Primal-dual training with Gumbel-Softmax for better discrete optimization.

**Features:**
- Temperature annealing schedules
- Backward compatible (use_gumbel=False → standard PMD)
- Automatic temperature management

**Usage:**
```python
program = GumbelPrimalDualProgram(
    graph, Model,
    use_gumbel=True,
    initial_temp=5.0,            # Start soft
    final_temp=0.1,              # End nearly discrete
    anneal_start_epoch=5,        # When to start annealing
    anneal_epochs=45             # Anneal over 45 epochs
)

program.train(train_data, train_epoch_num=50)
```

### `InferenceProgram`
Trains models to predict whether constraints are satisfied.

**Key Feature:**
```python
program = InferenceProgram(graph, Model, beta=0.5)
program.train(labeled_constraint_data, train_epoch_num=30)

# Evaluate constraint prediction accuracy
accuracy = program.evaluate_condition(eval_data, device='cuda')
print(f"Constraint prediction accuracy: {accuracy*100:.2f}%")
```

### `SampleLossProgram`
Constraint learning with optional sampling for large-scale problems.

**Usage:**
```python
program = SampleLossProgram(
    graph, Model,
    sample=True,
    sampleSize=100,              # Sample 100 groundings
    sampleGlobalLoss=False       # Per-constraint loss
)
```

### `GumbelSampleLossProgram`
Combines sampling with Gumbel-Softmax.

**Usage:**
```python
program = GumbelSampleLossProgram(
    graph, Model,
    use_gumbel=True,
    initial_temp=3.0,
    final_temp=0.5,
    hard_gumbel=False,           # Soft samples
    sample=True,
    sampleSize=100
)
```

### `SemanticLossProgram`
Trains against the **exact** probability that a constraint holds, rather than a fuzzy-logic
relaxation of it.

**Why it differs from a t-norm loss:** each constraint is compiled to a logical circuit (SDD
or BDD) and scored by differentiable weighted model counting, giving
`-log P(φ)` under the joint of all heads. The gradient each concept receives is then its true
marginal contribution to satisfaction probability — not an artifact of the chosen t-norm.
(Under Gödel, for instance, a violated implication gives its antecedent *exactly zero*
gradient, so half the available correction is structurally discarded.)

```python
from domiknows.program import SemanticLossProgram

program = SemanticLossProgram(
    graph, Model, beta=1.0,
    training_style='fixed',        # 'fixed': mloss + beta*closs
    circuit_backend='auto',        # 'auto' | 'pysdd' | 'bdd'
    circuit_max_nodes=100_000,
    circuit_aggregation='joint',   # or 'per_grounding'
)
```

**Parameters:**
- `training_style`: `'fixed'` (default) uses the classic `model_loss + beta * semantic_loss`;
  `'primal_dual'` routes the exact loss through the dual machinery so multipliers,
  augmented-Lagrangian updates or the amortized critic apply to it.
- `circuit_aggregation`: `'joint'` returns one `-log P(all groundings hold)` per constraint;
  `'per_grounding'` returns a per-grounding vector, required by per-grounding duals.
- `circuit_size_limit_action`: `'raise'` or `'warn'` when a constraint exceeds the node budget.

**Exactness reporting:** a constraint that exceeds the circuit budget falls back to the
product t-norm. `program.cmodel.exact_fraction` reports the share that really was exact — a
partially-exact run must never be described as exact.

---

### `StructuredProgram`
Puts constraint structure **inside the model** instead of only scoring the output.

Every other constraint program computes beliefs first and penalises them afterwards, so
constraints can correct the output layer but never the representation. `StructuredProgram`
supplies a `StructuredModel`, which applies structure between the learners and the loss.

```python
from domiknows.program import StructuredProgram

program = StructuredProgram(
    graph, poi=[...], loss=..., metric=...,
    refine=True,                   # R4B: iterative belief refinement
    factor_graph=False,            # R3: exact constrained marginals
    belief_flow='write_back',
    partition='auto',
    inferTypes=['local/softmax'],
)
```

**`refine=True` — constraint refinement.** Reads all concepts' beliefs, exchanges messages
along the constraint structure and returns corrected beliefs. The message a rule sends *is*
the gradient of its own violation, so one step is constraint-descent: it moves every
participating concept toward satisfaction **by construction**, before any training. Uses
Product semantics internally — under Gödel the antecedent would receive no message at all.

**`factor_graph=True` — factor-graph head.** Replaces beliefs with exact constrained
marginals `p(y | x, φ)` computed on the compiled circuit, batched across groundings. Exact but
costlier than refinement; groundings that exceed the budget keep their unrefined beliefs and
are reported.

**`belief_flow`:**

| Value | Effect |
|---|---|
| `'write_back'` (default) | Refined beliefs feed the supervised loss, constraint loss, metrics and inference — so constraints correct *representations* |
| `'constraint_only'` | Supervised loss keeps the raw head outputs; also the ablation for "did the structure do the work" |

**`partition` — what happens to constraints the structure enforces.** A structurally enforced
constraint has zero violation by construction, so a penalty term and its multiplier are dead
weight.

| Value | Multipliers | If a circuit falls back at runtime |
|---|---|---|
| `'auto'` (default) | none allocated for structural constraints | **gap** — constraint held by neither structure nor penalty; reported loudly |
| `'adaptive'` | one unused multiplier each | **self-closing** — the penalty returns automatically |
| `'none'` | all | n/a |

Only `factor_graph=True` licenses exclusion: refinement moves beliefs but guarantees nothing.
Prefer `'adaptive'` when the circuit budget is tight.

```python
print(program.report_partition())
# structural partition: 2 constraint(s) excluded from loss/duals, 1 still penalised
```

**Other options:** `structure_warmup` (defer structure until N *training* steps have run —
evaluation passes do not advance the counter), and `structure_kwargs` for the refinement's
`refine_steps` / `refine_step_size` / `refine_learn_gate`.

#### Decoding: MAP vs MPM

Once the head has a constrained distribution `p(y | x, φ)`, you still have to turn it into a
single prediction. There are two different ways to do that, and they do **not** agree.

| | Stands for | Question it answers | How |
|---|---|---|---|
| **MPM** | **M**aximum **P**osterior **M**arginals | "For each variable *separately*, which value is most likely?" | `argmax` each concept's marginal independently |
| **MAP** | **M**aximum **A** **P**osteriori | "Which *complete assignment* is most likely, all variables at once?" | max-product over the circuit, tracing back the winning branch |

MPM optimises each variable in isolation; MAP optimises the joint. When the variables are
independent these coincide — but a constraint is precisely a statement that they are *not*
independent, which is why they diverge here.

**Why MPM can violate a constraint even when the marginals are exact.** The marginals are
correct answers to the question MPM asks; the problem is that stitching per-variable winners
together is not guaranteed to produce an assignment the joint gives any mass to. Take
`exactly-one(A, B, C)` with constrained marginals:

```
P(A=true) = 0.45     P(B=true) = 0.35     P(C=true) = 0.20
```

Each marginal is exact — under `exactly-one` they must sum to 1, and they do. But MPM asks
each variable separately, and for every one of them "false" is the more likely value (0.55,
0.65, 0.80), so MPM decodes `A=B=C=false` — which `exactly-one` forbids, an assignment of
**zero** posterior probability. MAP instead compares *whole* assignments; here only three
have any mass — `(T,F,F)=0.45`, `(F,T,F)=0.35`, `(F,F,T)=0.20` — so it returns `A=true`, and
every candidate it considers satisfies the constraint.

**So: decode with MAP, never with `argmax` of the marginals.** Measured on an `exactly-one`
constraint over 3000 random inputs, MPM violated it 73 times; MAP violated it **zero** times.
The same caveat applies to sampling — sample the joint through the circuit, not the product of
marginals.

```python
program = StructuredProgram(graph, poi=[...], factor_graph=True,
                            inferTypes=['local/softmax', 'MAP'])
# after populate/test, each datanode carries a constraint-respecting one-hot:
datanode.getAttribute('<entity_label>/MAP')
```

`inferTypes=['MAP']` decodes with max-product on the compiled circuit and writes
`<concept>/MAP` one-hots (mirroring how ILP writes `<concept>/ILP`). It is
constraint-respecting *by construction* for anything that compiles, so it replaces ILP there;
ILP remains available as the cross-check, and for constraints that do not compile.

> **The marginals are still the right thing for *training*.** `forward` returns marginals
> because the loss needs a differentiable distribution; MAP is a discrete argmax used at
> inference. The rule is about decoding, not about which quantity is correct.

**Model synthesis is advisory.** `synthesize_model` builds a shared trunk with one joint
softmax head per `EnumConcept` group — under which an exclusivity constraint becomes
impossible to violate. `analyze_exclusivity(graph)` reports which binary sibling groups are
provably exclusive and prints the exact `EnumConcept` declaration to switch to, but never
rewrites the graph: silently turning declared constraints into architecture would change what
a graph means.

```python
from domiknows.program.model.synthesis import analyze_exclusivity, synthesize_model
print(analyze_exclusivity(graph).render())
```

---

### `GBIProgram`
Training with Gradient-Based Inference.

**Usage:**
```python
program = GBIProgram(
    graph, Model,
    poi=poi_list,
    gbi_iters=30,
    lr=0.1,
    beta=1.0
)
program.train(train_data)
```

---

## Specialized Programs

### `BatchProgram` (`batchprogram.py`)
Implements mini-batch gradient accumulation.

**Usage:**
```python
from domiknows.program import BatchProgram

program = BatchProgram(graph, Model, batch_size=32)
program.train(train_data, Optim=torch.optim.Adam)
```

**Key Feature:**
- Accumulates gradients over `batch_size` samples
- Single optimizer step per batch
- Memory efficient for large batches

### `CallbackProgram` (`callbackprogram.py`)
Provides lifecycle hooks for custom training logic.

**Available Callbacks:**
- `before_train` / `after_train`
- `before_train_epoch` / `after_train_epoch`
- `before_train_step` / `after_train_step`
- `before_test` / `after_test`
- `before_test_epoch` / `after_test_epoch`
- `before_test_step` / `after_test_step`

**Usage:**
```python
from domiknows.program import CallbackProgram

program = CallbackProgram(graph, Model)

# Add custom callbacks
def log_batch_loss(output):
    loss, metric, *_ = output
    print(f"Batch loss: {loss.item()}")

program.after_train_step = [
    program.default_after_train_step,
    log_batch_loss
]

program.train(train_data, train_epoch_num=20)
```

---

## Supporting Components

### Loss Functions (`loss.py`)

| Loss Class | Use Case |
|-----------|----------|
| `NBCrossEntropyLoss` | Cross-entropy with automatic reshaping |
| `BCEWithLogitsLoss` | Binary cross-entropy with logits |
| `BCEFocalLoss` | Focal loss for imbalanced data |
| `BCEWithLogitsIMLoss` | Inference-masked BCE loss |
| `NBCrossEntropyIMLoss` | Inference-masked cross-entropy |

### Metrics (`metric.py`)

| Metric Class | Computes |
|-------------|----------|
| `CMWithLogitsMetric` | Confusion matrix from logits |
| `DatanodeCMMetric` | Confusion matrix from datanode inference |
| `MetricTracker` | Tracks metrics across batches |
| `MacroAverageTracker` | Macro-averaged metrics |
| `PRF1Tracker` | Precision, Recall, F1, Accuracy |

**Usage:**
```python
from domiknows.program.metric import PRF1Tracker, CMWithLogitsMetric

metric = PRF1Tracker(CMWithLogitsMetric())
program = POIProgram(graph, loss=loss_fn, metric=metric)
```

### Trackers (`tracker.py`)

Simple metric tracking utilities:
- `MacroAverageTracker`: Average across batches
- `ConfusionMatrixTracker`: Aggregate confusion matrices

---

## Common Training Patterns

### Basic Supervised Learning
```python
program = POIProgram(graph, loss=nn.CrossEntropyLoss(), 
                     metric=PRF1Tracker())
program.train(train_data, valid_data, train_epoch_num=50,
              Optim=torch.optim.Adam)
```

### Constraint-Based Learning
```python
# Primal-dual training
program = PrimalDualProgram(graph, Model, beta=1.0)
program.train(train_data, valid_data, 
              c_lr=0.05, c_warmup_iters=10,
              train_epoch_num=100)
```

### Faster + Better-Weighted Constraints
```python
# Compiled constraint loss with augmented-Lagrangian, per-grounding duals
program = PrimalDualProgram(
    graph, Model, beta=1.0,
    compile_lc=True,
    dual_algorithm='augmented',
    dual_granularity='amortized',
)
program.train(train_data, valid_data, train_epoch_num=100)
```

### Exact Constraint Semantics
```python
from domiknows.program import SemanticLossProgram

program = SemanticLossProgram(graph, Model, beta=1.0)
program.train(train_data, valid_data, train_epoch_num=100)
print(f'exact fraction: {program.cmodel.exact_fraction:.2f}')
```

### Constraints Inside the Model
```python
from domiknows.program import StructuredProgram

program = StructuredProgram(
    graph, poi=[...], loss=..., metric=...,
    refine=True,                  # correct representations, not just outputs
    compile_lc=True,              # composes with any constraint-model option
    dual_algorithm='augmented',
    inferTypes=['local/softmax'],
)
program.train(train_data, valid_data, train_epoch_num=100)
print(program.report_partition())
```

### Diagnosing Gradient Conflict
```python
# Measure before paying for a resolver: surgery costs an extra backward pass
program = PrimalDualProgram(graph, Model, grad_surgery='diagnose')
program.train(train_data, train_epoch_num=5)
print(program.conflict_stats.render())

# If the conflict rate is high, resolve it
program = PrimalDualProgram(graph, Model, grad_surgery='pcgrad')
```

### Comparing Mechanisms
```python
from domiknows.program.training_comparison import (
    TrainingComparison, DEFAULT_VARIANTS, R3_VARIANTS, R4_VARIANTS,
)

comparison = TrainingComparison(
    build_program, dataset, evaluate=my_eval, epochs=20,
    variants=DEFAULT_VARIANTS + R4_VARIANTS + R3_VARIANTS,
)
print(comparison.run().render())
```

Trains a freshly built program per variant from the same seed and reports wall-clock,
constraint-loss time, before/after constraint violation and the caller's task metrics.
`build_program` should honour `variant.resolve_program_class(default)` and merge
`variant.program_kwargs`.

### Phased Training Strategy
```python
program = PrimalDualProgram(graph, Model, beta=1.0)
program.train(
    train_data,
    valid_data,
    warmup_epochs=20,            # Warmup on data only
    constraint_epochs=30,         # Then add constraints
    constraint_only=False,        # Combined training
    train_epoch_num=50
)
```

### Constraint-Only Fine-tuning
```python
# After pre-training, fine-tune only on constraints
program.train(
    pretrained_data,
    warmup_epochs=0,
    constraint_epochs=20,
    constraint_only=True,         # Only constraint gradients
    constraint_loss_scale=2.0     # Boost constraint influence
)
```

### Gumbel-Softmax Training
```python
program = GumbelPrimalDualProgram(
    graph, Model,
    use_gumbel=True,
    initial_temp=5.0,
    final_temp=0.1,
    anneal_start_epoch=10
)
program.train(train_data, train_epoch_num=100)
```

### Mini-Batch Accumulation
```python
program = BatchProgram(graph, Model, batch_size=32)
program.train(train_data, Optim=torch.optim.SGD, lr=0.01)
```

### Custom Training Lifecycle
```python
program = CallbackProgram(graph, Model)

def early_stopping(output):
    loss, metric, *_ = output
    if metric['F1'] > 0.95:
        program.stop = True

program.after_train_epoch = [early_stopping]
program.train(train_data, train_epoch_num=100)
```

---

## Device Management

All programs support automatic device placement:
```python
# Automatic detection
program.train(train_data, device='auto')

# Explicit GPU
program.train(train_data, device='cuda:0')

# CPU
program.train(train_data, device='cpu')

# Manual device setting
program.to('cuda:1')
```

---

## Model Persistence

```python
# Save trained model
program.save('checkpoints/model_epoch_50.pt')

# Load model
program.load('checkpoints/model_epoch_50.pt')

# Continue training
program.train(train_data, train_epoch_num=100)
```

---

## Constraint Verification

```python
# Verify all constraints
program.verifyResultsLC(test_data)

# Verify specific constraints
program.verifyResultsLC(test_data, 
                       constraint_names=['LC1', 'LC2'])
```

**Output:**
```
Constraint name: LC1 datanode accuracy: 95.2 total accuracy: 89.7
Constraint name: LC2 datanode accuracy: 87.3 total accuracy: 85.1
Results for all constraints:
datanode accuracy: 91.25
total accuracy: 87.4
```

---

## Advanced Features

### Gradient Clipping
Automatically applied in `LossProgram`:
- Standard mode: `max_norm=10.0`
- Constraint-only mode: `max_norm=5.0`

### Learning Rate Decay Strategies
Multiple strategies available in `PrimalDualProgram`:
- `c_lr_decay=0`: Inverse time decay
- `c_lr_decay=1`: Square root decay
- `c_lr_decay=2`: Linear decay
- `c_lr_decay=3`: Exponential decay
- `c_lr_decay=4`: Sqrt schedule

### Logging

Programs use Python's logging module:
```python
import logging
logging.basicConfig(level=logging.INFO)

program = PrimalDualProgram(graph, Model)
program.train(train_data)  # Logs epoch info, losses, metrics
```

---

## Best Practices

1. **Start simple**: Use `POIProgram` for baseline
2. **Add constraints gradually**: Use warmup phases
3. **Monitor metrics**: Track both data loss and constraint satisfaction
4. **Use Gumbel-Softmax**: For tasks with discrete decisions
5. **Tune beta**: Balance data fit vs. constraint satisfaction
6. **Verify constraints**: Check satisfaction rates on test set
7. **Save checkpoints**: Regularly save models during training
8. **Use callbacks**: For complex training logic without subclassing

---

## Performance Tips

- **Batch size**: Use `BatchProgram` for gradient accumulation
- **Sampling**: Use `SampleLossProgram` for large constraint sets
- **Device**: Always specify `device='cuda'` if available
- **Gradient clipping**: Already implemented in constraint programs
- **Early stopping**: Use `CallbackProgram` with custom logic