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
| `GBIProgram` | `lossprogram.py` | `LossProgram` | Gradient-Based Inference training |
| **Specialized Programs** | | | |
| `BatchProgram` | `batchprogram.py` | `LearningBasedProgram` | Mini-batch gradient accumulation |
| `CallbackProgram` | `callbackprogram.py` | `LearningBasedProgram` | Training with lifecycle callbacks |

### Quick Selection Guide

**Choose based on your needs:**

- **Standard supervised learning**: `POIProgram`
- **Learning + constraint inference**: `SolverPOIProgram`
- **Learning to satisfy constraints**: `PrimalDualProgram` or `InferenceProgram`
- **Gradient-based constraint satisfaction**: `GBIProgram`
- **Better discrete optimization**: `GumbelPrimalDualProgram` or `GumbelSampleLossProgram`
- **Mini-batch training**: `BatchProgram`
- **Custom training lifecycle**: `CallbackProgram`

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