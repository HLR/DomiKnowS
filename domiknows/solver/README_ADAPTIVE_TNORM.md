# Adaptive T-Norm Selection System

## What Problem Does This Solve?

DomiKnowS translates logical constraints into differentiable losses using **t-norms**. Different constraint types have fundamentally different gradient properties under different t-norms:

| Constraint Type | What It Does | Best T-Norm | Why |
|---|---|---|---|
| `sumL` (counting) | "How many X?" | L (Łukasiewicz) | Linear gradients to all elements |
| `atLeastAL` / `atMostAL` | "At least/most N" | L or G | Stable bounded gradients |
| `exactAL` | "Exactly N" | L | Needs both directions |
| `andL` / `orL` (boolean) | Logical conjunction/disjunction | SP (Simplified Product) | Fast multiplicative gradients |
| `ifL` (implication) | "If X then Y" | P (Product) | Precise implication semantics |

Using a single global t-norm forces a compromise. The adaptive system eliminates this by selecting the best t-norm **per constraint type** and adjusting it during training based on observed performance.

## System Components

### 1. LossCalculator — Per-Type T-Norm Selection

`LossCalculator` reads from a class-level `TNORM_CONFIG` dictionary when computing loss. Instead of using one t-norm for all constraints, it looks up the constraint's type and applies the mapped t-norm.

```
datanode.calculateLcLoss(tnorm='L')
    └─► LossCalculator.calculateLoss()
            │
            ├─ LC0 (atMostAL)  → TNORM_CONFIG['atMostAL'] → 'L'
            ├─ LC1 (atMostAL)  → TNORM_CONFIG['atMostAL'] → 'L'
            ├─ LC2 (atMostAL)  → TNORM_CONFIG['atMostAL'] → 'L'
            ├─ LC3 (atLeastAL) → TNORM_CONFIG['atLeastAL'] → 'L'
            └─ ELC0 (sumL)     → TNORM_CONFIG['sumL'] → 'L'
```

You can set it manually at any time:

```python
from domiknows.solver.ilpOntSolverTools.lossCalculator import LossCalculator

# Set one type
LossCalculator.set_tnorm_for_type('atMostAL', 'P')

# Or replace the entire config
LossCalculator.set_global_config({
    'atMostAL': 'P',
    'atLeastAL': 'L',
    'sumL': 'L',
    'andL': 'SP',
    'default': 'L',
})
```

### 2. AdaptiveTNormLossCalculator — Monitoring + Auto-Switching

This component observes training dynamics and automatically updates `LossCalculator.TNORM_CONFIG` at epoch boundaries.

```
                        Training Loop
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                   ▼
     Step N              Step N+k           Epoch End
       │                    │                   │
  record_observation()  record_tnorm_        on_epoch_end()
  (loss, grad, tnorm)  comparison()            │
       │               (alt t-norm loss)   ┌────┴────┐
       ▼                    │              ▼         ▼
  epoch_metrics         epoch_metrics   get_recs  apply_recs
  cumulative_metrics    cumulative       │         │
                        _metrics     score each   write to
                                     t-norm    TNORM_CONFIG
```

**Three levels of tracking:**

| Level | Scope | Lifetime | Purpose |
|---|---|---|---|
| Per-constraint | Individual `LC0`, `ELC42`, etc. | Full run | Debugging |
| Per-epoch per-type | `atMostAL`, `sumL`, etc. | One epoch (reset) | Epoch display |
| Cumulative per-type | `atMostAL`, `sumL`, etc. | Full run | Stable recommendations |

**Scoring formula** (for `gradient_weighted` strategy):

```
score = -(avg_loss × 10)
      + (loss_improvement × 5)        # positive = loss decreasing over time
      + gradient_penalty               # -10 vanishing, -5 exploding, 0 healthy
```

**Constraint coverage** distinguishes:
- **Global constraints** — names `LC0`, `LC1`, ... (defined in `graph.py`, seen every sample)
- **Executable constraints** — names `ELC0`, `ELC1`, ... (compiled from per-sample queries, seen once)

## Integration Guide

### Step 1: Create the Tracker

```python
from domiknows.solver.adaptiveTNormLossCalculator import AdaptiveTNormLossCalculator

tracker = AdaptiveTNormLossCalculator(
    tnorms=["L", "P", "SP", "G"],       # t-norms to compare
    adaptation_interval=10,               # steps between comparison probes
    warmup_steps=5,                       # steps before probing starts
    selection_strategy="gradient_weighted",# or "loss_based"
    auto_apply=True,                      # push changes to LossCalculator
    min_observations=20,                  # minimum data before recommending
)
```

### Step 2: Wire into Training Callbacks

You need two callbacks: one per training step, one per epoch end.

**Per-step callback** — records observations and periodically probes alternative t-norms:

```python
import torch

def on_step_end(output):
    """Call after each training step. `output` comes from program's train loop."""
    # Extract the datanode from the training output
    datanode = None
    if isinstance(output, (tuple, list)):
        for item in output:
            if item is not None and hasattr(item, 'calculateLcLoss'):
                datanode = item
                break
    if datanode is None:
        return

    current_tnorm = 'L'  # whatever your default is

    # 1. Compute losses with current t-norm
    losses = datanode.calculateLcLoss(tnorm=current_tnorm)

    # 2. Compute gradient norm from your model parameters
    grad_norm = 0.0
    for p in your_model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item() ** 2
    grad_norm = grad_norm ** 0.5

    # 3. Record each constraint's loss
    for lc_name, loss_dict in losses.items():
        lc = loss_dict.get('lc')
        loss_tensor = loss_dict.get('loss')
        if loss_tensor is None or not torch.is_tensor(loss_tensor):
            continue
        loss_val = loss_tensor.item() if loss_tensor.numel() == 1 else loss_tensor.mean().item()
        tracker.record_observation(lc_name, lc, loss_val, grad_norm, current_tnorm)

    # 4. Periodically probe alternative t-norms
    if tracker.step_count % tracker.adaptation_interval == 0:
        for alt_tnorm in tracker.tnorms:
            if alt_tnorm == current_tnorm:
                continue
            try:
                alt_losses = datanode.calculateLcLoss(tnorm=alt_tnorm)
                for lc_name, loss_dict in alt_losses.items():
                    lc = loss_dict.get('lc')
                    loss_tensor = loss_dict.get('loss')
                    if loss_tensor is not None and torch.is_tensor(loss_tensor):
                        loss_val = loss_tensor.item() if loss_tensor.numel() == 1 else loss_tensor.mean().item()
                        tracker.record_tnorm_comparison(lc_name, lc, alt_tnorm, loss_val)
            except Exception:
                pass
    tracker.step_count += 1
```

**Per-epoch callback** — analyzes, prints, and optionally applies switches:

```python
def on_epoch_end():
    tracker.on_epoch_end()  # prints table, applies if auto_apply=True
```

### Step 3: Register Callbacks with Your Program

If using `CallbackProgram` (or a subclass like `InferenceProgramWithCallbacks`):

```python
program.after_train_step.append(on_step_end)
program.after_train_epoch.append(on_epoch_end)
```

If using a custom training loop:

```python
for epoch in range(num_epochs):
    for batch in dataset:
        output = train_step(batch)
        on_step_end(output)
    on_epoch_end()
```

### Step 4 (Optional): Manual Override

You can override the adaptive system at any point:

```python
# Force a specific t-norm for a type
LossCalculator.set_tnorm_for_type('exactAL', 'P')

# Inspect current active config
print(tracker.active_tnorms)
# {'atMostAL': 'L', 'atLeastAL': 'L', 'exactAL': 'L', 'sumL': 'L'}

# Get recommendations without applying
recs = tracker.get_recommendations()

# Check history across epochs
for epoch_recs in tracker.recommendation_history:
    print(epoch_recs)
```

## Available T-Norms

| Code | Name | Formula | Gradient | Best For |
|---|---|---|---|---|
| `L` | Łukasiewicz | max(a + b - 1, 0) | Linear, stable, all elements | Counting, exact counts |
| `P` | Product | a × b | Multiplicative, can vanish | Implication, precision |
| `SP` | Simplified Product | a × b (simplified) | Fast multiplicative | Boolean (and/or/not) |
| `G` | Gödel | min(a, b) | Sparse — only min element | Upper bounds (use carefully) |

## Reading the Epoch Output

```
📊 AGGREGATED BY CONSTRAINT TYPE:
---------------------------------------------------------------------------
Type          Count  AvgLoss  AvgGrad │       L       P      SP       G │ Best
---------------------------------------------------------------------------
atMostAL      15096   0.054     0.00 │  0.054✓  0.072   0.072   0.072  │ L
exactAL         667   0.934     2.74 │  0.934✓  0.998   0.998   0.998  │ L

📈 CONSTRAINT COVERAGE:
   Global constraints (graph-level):    4
   Executable constraints (per-sample): 4816

🔄 APPLIED T-NORM CHANGES:
   exactAL: 'G' → 'L'
```

| Column | Meaning |
|---|---|
| `Count` | Observations this epoch (reset each epoch) |
| `AvgLoss` | Mean loss under the currently active t-norm |
| `L/P/SP/G` | Mean loss when probed with each t-norm (✓ = best) |
| `Best` | Recommended t-norm (applied if `auto_apply=True`) |

## Configuration Reference

| Parameter | Default | Description |
|---|---|---|
| `tnorms` | `["L","P","SP","G"]` | T-norms to compare |
| `adaptation_interval` | `50` | Steps between probing alternative t-norms |
| `warmup_steps` | `100` | Steps before probing begins |
| `selection_strategy` | `"gradient_weighted"` | `"gradient_weighted"` or `"loss_based"` |
| `auto_apply` | `True` | Push recommendations to `LossCalculator.TNORM_CONFIG` |
| `min_observations` | `20` | Minimum observations per type before recommending |

## Design Decisions

**Why group by constraint type?** Executable constraints are each seen once per epoch — too few observations. Grouping all `atMostAL` constraints together gives thousands of data points for comparison.

**Why cumulative metrics for recommendations?** Single-epoch metrics are noisy. Cumulative history gives stable recommendations. Per-epoch metrics are still displayed for transparency.

**Why skip epoch 1 for switching?** The first epoch establishes baselines. Switching too early on noisy data can hurt convergence.

**Why reset per-epoch metrics?** The epoch display should reflect current-epoch behavior, not accumulate unboundedly. Cumulative metrics still retain full history.