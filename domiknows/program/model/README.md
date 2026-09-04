# DomiKnows Model Components

This directory contains the core model implementations for the DomiKnows framework, which integrates neural learning with logical constraints.

---

## Model Hierarchy Overview

| Model Class | File | Parent Class | Primary Use Case |
|------------|------|--------------|------------------|
| **Base Models** | | | |
| `Mode` | `base.py` | Enum | Training/testing mode definitions |
| `TorchModel` | `pytorch.py` | `nn.Module` | Base PyTorch model with device management |
| **Standard Training** | | | |
| `PoiModel` | `pytorch.py` | `TorchModel` | Train on specific graph properties (POIs) |
| `SolverModel` | `pytorch.py` | `PoiModel` | Add constraint-based inference (ILP/GBI) |
| `GumbelSolverModel` | `pytorch.py` | `SolverModel` | SolverModel + Gumbel-Softmax for discrete optimization |
| **Constraint Learning** | | | |
| `LossModel` | `lossModel.py` | `nn.Module` | Learn with weighted logical constraint losses |
| `PrimalDualModel` | `lossModel.py` | `LossModel` | Primal-dual optimization for constraints |
| `InferenceModel` | `lossModel.py` | `LossModel` | Learn to predict constraint satisfaction |
| `SampleLossModel` | `lossModel.py` | `nn.Module` | Constraint loss + sampling + Gumbel-Softmax |
| **Gradient-Based Inference** | | | |
| `GBIModel` | `gbi.py` | `nn.Module` | Iterative gradient optimization to satisfy constraints |
| **Specialized Losses** | | | |
| `IMLModel` | `iml.py` | `SolverModel` | Inference-masked loss (mask by ILP results) |
| `ILPUModel` | `ilpu.py` | `SolverModel` | Use ILP inference as soft targets |
| **Dictionary Loss Variants** | | | |
| `PoiModelDictLoss` | `pytorch.py` | `PoiModel` | Per-property custom loss functions |
| `SolverModelDictLoss` | `pytorch.py` | `PoiModelDictLoss` | SolverModel with custom losses |
| `GumbelSolverModelDictLoss` | `pytorch.py` | `SolverModelDictLoss` | + Gumbel-Softmax support |
| **Legacy** | | | |
| `BaseModel`, `TorchModel`, `PoiModel` | `torch.py` | `nn.Module` | Earlier implementations (use `pytorch.py` versions) |

### Quick Selection Guide

**Choose based on your needs:**

- **Standard neural training**: `PoiModel`
- **Neural + constraint inference**: `SolverModel` (with `inferTypes=['ILP']`)
- **Neural + gradient-based constraint satisfaction**: `GBIModel` wrapping `SolverModel`
- **Training with discrete decisions**: `GumbelSolverModel` or `SampleLossModel`
- **Learning to satisfy constraints**: `PrimalDualModel` or `InferenceModel`
- **Custom loss per property**: `*DictLoss` variants

---

## Core Files

### `base.py`
Defines the fundamental `Mode` enumeration used across all models:
- `TRAIN`: Training mode
- `TEST`: Testing/evaluation mode  
- `POPULATE`: Data population mode

---

## Main Model Types

### Loss-Based Models (`lossModel.py`)

#### `LossModel`
Base class for constraint-based learning using logical constraints from the knowledge graph.

**Key Features:**
- Computes weighted constraint losses using lambda parameters
- Supports multiple t-norms (Product, Łukasiewicz, etc.)
- Optional sampling for large-scale constraints
- Dedicated logging for debugging operations

**Parameters:**
- `tnorm`: T-norm type for constraint aggregation (default: 'P')
- `counting_tnorm`: Optional separate t-norm for counting quantifiers
- `sample`: Enable constraint sampling
- `sampleSize`: Number of samples per constraint
- `sampleGlobalLoss`: Use global vs. per-constraint loss

#### `PrimalDualModel`
Extends `LossModel` with primal-dual optimization for constraint satisfaction. Automatically balances data loss and constraint violations.

#### `InferenceModel`
Specialized for learning constraint satisfaction from labeled examples. Trains models to predict whether constraints are satisfied.

**Key Method:**
```python
forward(builder, build=None)
# Returns: (loss_scalar, datanode, builder)
```

#### `SampleLossModel`
Advanced loss model with optional **Gumbel-Softmax** support for discrete optimization.

**Gumbel-Softmax Features:**
- Differentiable discrete sampling during training
- Temperature annealing schedules (constant/exponential/linear)
- Hard mode with straight-through estimator
- Enables gradient flow through discrete decisions

**Parameters:**
- `use_gumbel`: Enable Gumbel-Softmax (default: False)
- `temperature`: Initial temperature (default: 1.0, lower = more discrete)
- `hard_gumbel`: Use straight-through estimator
- `temperature_schedule`: 'constant', 'exponential', or 'linear'
- `anneal_rate`: Temperature decay rate

---

### Neural Model Classes (`pytorch.py`)

#### `TorchModel`
Base PyTorch model with device management and data handling.

**Key Methods:**
- `mode(mode)`: Switch between TRAIN/TEST/POPULATE
- `move(value, device)`: Recursively move tensors to device
- `forward(data_item, build)`: Main computation pipeline

#### `PoiModel` (Point of Interest Model)
Trains on specific properties (POIs) in the knowledge graph.

**Features:**
- Automatic POI detection from graph structure
- Sensor-based loss computation
- Flexible metric tracking
- Supports both label and output sensors

#### `SolverModel`
Extends `PoiModel` with constraint-based inference capabilities.

**Inference Types:**
- `'local/argmax'` / `'local/softmax'`: Local inference per concept
- `'argmax'` / `'softmax'`: Global inference
- `'ILP'`: Integer Linear Programming inference
- `'GBI'`: Gradient-Based Inference (see below)

**Parameters:**
- `inferTypes`: List of inference methods to apply
- `inference_with`: Additional inference constraints
- `probKey`: Tuple specifying probability computation keys
- `probAcc`: Accuracy values for ILP variants

#### `GumbelSolverModel`
`SolverModel` with integrated Gumbel-Softmax support for better discrete optimization during training.

**When to Use:**
- Training with discrete constraints
- Improving gradient flow in categorical decisions
- Need differentiable sampling

---

### Gradient-Based Inference (`gbi.py`)

#### `GBIModel`
Implements gradient-based inference (GBI) to satisfy logical constraints at inference time through iterative optimization.

**How It Works:**
1. Takes model predictions as starting point
2. Iteratively updates predictions via gradient descent
3. Minimizes constraint violations + regularization
4. Stops when constraints satisfied or max iterations reached

**Key Features:**
- **Early stopping**: Stops when loss plateaus
- **Adaptive clipping**: Optional gradient norm clipping
- **Multiple optimizers**: SGD or Adam
- **Statistics tracking**: Monitors convergence patterns
- **NaN handling**: Automatic parameter reset on instability

**Parameters:**
```python
GBIModel(
    graph,
    solver_model=None,     # Underlying model
    gbi_iters=50,          # Max optimization steps
    lr=1e-1,               # Learning rate
    reg_weight=1,          # Regularization strength
    reset_params=True,     # Reset after optimization
    grad_clip=None,        # Gradient clipping threshold
    early_stop_patience=5, # Early stopping patience
    optimizer='sgd',       # 'sgd' or 'adam'
    momentum=0.0           # SGD momentum
)
```

**Usage:**
```python
gbi = GBIModel(graph, solver_model=my_model)
loss, updated_node, builder = gbi(datanode, verbose=True)
stats = gbi.get_stats()  # Get optimization statistics
```

**Statistics Available:**
- `total_iterations`: Total optimization steps
- `early_stops`: Times early stopping triggered
- `constraint_satisfied_stops`: Times all constraints satisfied
- `nan_resets`: Times NaN caused parameter reset
- `avg_iterations`: Average iterations per call

---

### Specialized Models

#### `IMLModel` (`iml.py`)
**Inference-Masked Loss**: Masks loss computation based on ILP inference results. Only computes loss where ILP suggests corrections.

#### `ILPUModel` (`ilpu.py`)  
**ILP-Updated Loss**: Uses ILP inference as soft targets instead of ground truth labels.

---

## Dictionary Loss Models

For advanced per-property loss customization:

- `PoiModelDictLoss`: PoiModel with property-specific loss functions
- `SolverModelDictLoss`: SolverModel variant
- `GumbelSolverModelDictLoss`: Gumbel-Softmax variant

**Usage:**
```python
dictloss = {
    'property1': custom_loss_fn1,
    'property2': custom_loss_fn2,
    'default': default_loss_fn
}
model = SolverModelDictLoss(graph, dictloss=dictloss)
```

---

## Common Patterns

### Basic Training Loop
```python
model = SolverModel(graph, loss=loss_fn, metric=metric_fn, 
                    inferTypes=['local/softmax', 'ILP'])
model.train()

for batch in dataloader:
    loss, metrics, datanode, builder = model(batch)
    loss.backward()
    optimizer.step()
```

### Using GBI for Inference
```python
# Train base model
base_model = SolverModel(graph, inferTypes=['local/softmax'])

# Wrap with GBI for constraint satisfaction
gbi_model = GBIModel(graph, solver_model=base_model, 
                     gbi_iters=30, lr=0.1)

# At inference time
loss, constrained_output, builder = gbi_model(data, verbose=True)
```

### Gumbel-Softmax Training
```python
model = GumbelSolverModel(
    graph, 
    use_gumbel=True,
    temperature=5.0,  # Start high
    temperature_schedule='exponential',
    anneal_rate=0.001
)

for epoch in range(epochs):
    for batch in dataloader:
        loss, metrics, datanode, builder = model(batch)
        loss.backward()
        optimizer.step()
    # Temperature anneals automatically during training
```

---

## Device Management

All models support automatic device placement:
```python
model = SolverModel(graph, device='cuda:0')  # Specific GPU
model = SolverModel(graph, device='auto')    # Auto-detect
```

---

## Logging

Models include dedicated loggers for operations tracking:
- `LossModel`: Logs constraint loss calculations
- `PrimalDualModel`: Logs primal-dual optimization
- `InferenceModel`: Logs constraint prediction training
- `SampleLossModel`: Logs sampling operations

Logs are stored in `logs/` directory and disabled in production mode.

---

## Legacy (`torch.py`)

Contains earlier model implementations (`BaseModel`, `TorchModel`, `PoiModel`). Use `pytorch.py` versions for new projects.