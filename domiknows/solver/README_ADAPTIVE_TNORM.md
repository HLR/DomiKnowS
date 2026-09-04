# Adaptive T-Norm Selection System

## What Problem Does This Solve?

DomiKnowS translates logical constraints into differentiable losses using **t-norms**. Different constraint types have fundamentally different gradient properties under different t-norms:

| Constraint Type | What It Does | Default T-Norm | Why |
|---|---|---|---|
| `sumL` (counting) | "How many X?" | L (Łukasiewicz) | Linear gradients to all elements |
| `atLeastAL` / `atMostAL` | "At least/most N" | L / G | Stable bounded gradients |
| `exactAL` | "Exactly N" | L | Needs both directions |
| `andL` / `orL` (boolean) | Logical conjunction/disjunction | SP (Simplified Product) | Fast multiplicative gradients |
| `ifL` (implication) | "If X then Y" | P (Product) | Precise implication semantics |
| `existsL` | "There exists" | L | Stable counting semantics |
| `notL` | Negation | SP | Simple complement |

Using a single global t-norm forces a compromise. The adaptive system eliminates this by selecting the best t-norm **per constraint type** and optionally adjusting it during training.

## T-Norm Selection Modes

The system supports three modes, controlled by the `--counting_tnorm` argument:

### 1. Specific Mode (`G`, `P`, `L`, `SP`)

Use a **single fixed t-norm for all constraints** — no adaptation, no per-type defaults.

```bash
python main_new.py --counting_tnorm L    # Łukasiewicz for everything
python main_new.py --counting_tnorm SP   # Simplified Product for everything
```

Every constraint gets the same t-norm regardless of type.

```
TNormSelector.select(lc)  →  always returns "L"  (or whichever was specified)
```

### 2. Default Mode (`default`)

Use **per-type optimal defaults** from `DEFAULT_TNORM_BY_TYPE`. Each constraint type gets its mathematically optimal t-norm without any runtime adaptation.

```bash
python main_new.py --counting_tnorm default
```

The selector looks up the constraint type and returns the default mapping:

```
TNormSelector.select(lc)
    → get_constraint_type(lc) → "atMostAL"
    → DEFAULT_TNORM_BY_TYPE["atMostAL"] → "G"
```

The default mapping is:

```python
DEFAULT_TNORM_BY_TYPE = {
    'sumL': 'L',       'atLeastL': 'L',    'atLeastAL': 'L',
    'atMostL': 'G',    'atMostAL': 'G',
    'exactL': 'L',     'exactAL': 'L',
    'andL': 'SP',      'orL': 'SP',        'nandL': 'SP',    'norL': 'SP',
    'ifL': 'P',        'existsL': 'L',     'notL': 'SP',
    'default': 'L',
}
```

### 3. Auto Mode (`auto`)

**Dynamically adapt** the t-norm per constraint type during training. Selecting `auto` **automatically enables adaptation** with sensible defaults — no additional flags required.

```bash
# This is all you need — adaptation is enabled automatically
python main_new.py --counting_tnorm auto
```

The `AdaptiveTNormLossCalculator` tracks loss/gradient metrics per constraint type. At each epoch boundary it computes recommendations and applies them, which the `TNormSelector` reads on the next epoch:

```
TNormSelector.select(lc)
    → get_constraint_type(lc) → "exactAL"
    → check tracker.active_tnorms["exactAL"] → "P"  (adapted)
    → if not found, fall back to DEFAULT_TNORM_BY_TYPE
```

**Default adaptation parameters** (used when no overrides are specified):

| Parameter | Default | Description |
|---|---|---|
| `--tnorm_adaptation_interval` | `10` | Steps between probing alternative t-norms |
| `--tnorm_warmup_steps` | `5` | Steps before probing begins |
| `--tnorm_strategy` | `gradient_weighted` | Scoring strategy for selecting t-norms |
| `--tnorm_min_observations` | `20` | Minimum observations per type before recommending |

#### Customizing Adaptation Parameters

You can override any of the defaults to tune the adaptation behavior:

```bash
# Longer warmup and less frequent probing (for large datasets)
python main_new.py --counting_tnorm auto \
    --tnorm_warmup_steps 50 \
    --tnorm_adaptation_interval 100 \
    --tnorm_min_observations 100

# Aggressive adaptation (for small datasets or fast iteration)
python main_new.py --counting_tnorm auto \
    --tnorm_warmup_steps 2 \
    --tnorm_adaptation_interval 5 \
    --tnorm_min_observations 10

# Use loss-based scoring instead of gradient-weighted
python main_new.py --counting_tnorm auto \
    --tnorm_strategy loss_based
```

These parameters only take effect in `auto` mode. In `specific` or `default` mode they are ignored — the tracker still monitors metrics for logging, but no adaptation is applied.

## Architecture

### Unified T-Norm Selection via `TNormSelector`

`LossCalculator`  use the `TNormSelector` class for all t-norm decisions. This ensures consistent behavior regardless of which calculator processes a constraint.

```
┌─────────────────┐     ┌──────────────────┐
│LossCalculator │────▶│   TNormSelector │
└─────────────────┘     │                  │
                      │  mode: specific  │──▶ fixed t-norm
                        └────────┬─────────┘
                                 │
                        ┌────────▼──────────────────┐
                        │ AdaptiveTNormLossCalculator │
                        │  (tracker.active_tnorms)   │
                        └────────────────────────────┘
```

The calculators call `setTNorm(selected_tnorm)` on the boolean methods object **per constraint**, before that constraint is evaluated. The boolean methods (`lcLossBooleanMethods`) use `self.tnorm` for all operations — including counting operations (`countVar`, `compareCountsVar`). There is no separate `counting_tnorm` field; the single `self.tnorm` is set to the correct value for each constraint by the calculator.

### Per-Constraint Flow

```python
# Inside LossCalculator / SampleLossCalculator:
for lc in all_constraints:
    selected_tnorm = selector.select(lc=lc)   # TNormSelector decides
    myBooleanMethods.setTNorm(selected_tnorm)  # Set for this constraint
    # ... construct and evaluate constraint using self.tnorm ...
```

### Component Responsibilities

| Component | Role |
|---|---|
| `TNormSelector` | Stateless t-norm lookup. Three modes: specific, default, auto. |
| `LossCalculator` | Loss-based constraint evaluation. Calls `selector.select(lc)` per constraint. |
| `lcLossBooleanMethods` | Differentiable boolean ops. Uses `self.tnorm` (set per-constraint by calculator). |
| `AdaptiveTNormLossCalculator` | Tracks metrics, computes recommendations, stores `active_tnorms`. |
| `AdaptiveTNormPlugin` | Wires the tracker into training callbacks. |

## Adaptive Tracking Details (Auto Mode)

### Training Loop Integration

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
                                     t-norm    active_tnorms
```

### Three Levels of Tracking

| Level | Scope | Lifetime | Purpose |
|---|---|---|---|
| Per-constraint | Individual `LC0`, `ELC42`, etc. | Full run | Debugging, CSV export |
| Per-epoch per-type | `atMostAL`, `sumL`, etc. | One epoch (reset) | Epoch display |
| Cumulative per-type | `atMostAL`, `sumL`, etc. | Full run | Stable recommendations |

### Scoring Formula (gradient_weighted strategy)

```
score = -(avg_loss × 10)
      + (loss_improvement × 5)        # positive = loss decreasing over time
      + gradient_penalty               # -10 vanishing, -5 exploding, 0 healthy
```

### Constraint Coverage

- **Global constraints** — names `LC0`, `LC1`, ... (graph-level, seen every sample)
- **Executable constraints** — names `ELC0`, `ELC1`, ... (per-sample, seen once per epoch)

## Quick Start

### Fixed T-Norm (Simplest)

```bash
python main_new.py --counting_tnorm L
```

All constraints use Łukasiewicz. No per-type selection, no adaptation.

### Per-Type Defaults (Recommended Starting Point)

```bash
python main_new.py --counting_tnorm default
```

Each constraint type gets its mathematically optimal t-norm. No runtime adaptation.

### Adaptive (Full Auto)

```bash
python main_new.py --counting_tnorm auto
```

Starts with per-type defaults, then adapts at each epoch boundary based on observed training dynamics. Uses default adaptation parameters.

### Adaptive with Custom Parameters

```bash
python main_new.py --counting_tnorm auto \
    --tnorm_adaptation_interval 20 \
    --tnorm_warmup_steps 10 \
    --tnorm_strategy loss_based \
    --tnorm_min_observations 50
```

Same adaptive behavior, but with longer warmup and more conservative switching.

## Programmatic Usage

### Creating a Selector Manually

```python
from domiknows.solver.adaptiveTNormLossCalculator import TNormSelector

# Fixed t-norm
selector = TNormSelector(tnorm_arg="L")
print(selector.select(lc=my_constraint))  # Always "L"

# Per-type defaults
selector = TNormSelector(tnorm_arg="default")
print(selector.select(lc=my_counting_lc))  # Looks up type → "L"
print(selector.select(lc=my_if_lc))        # Looks up type → "P"

# Auto mode (requires tracker)
from domiknows.solver.adaptiveTNormLossCalculator import AdaptiveTNormLossCalculator

tracker = AdaptiveTNormLossCalculator(
    tnorms=["L", "P", "SP", "G"],
    adaptation_interval=10,
    warmup_steps=5,
    auto_apply=True,
    min_observations=20,
)
selector = TNormSelector(tnorm_arg="auto", tracker=tracker)
# selector.select(lc) reads from tracker.active_tnorms
```

### Injecting into Calculators

```python
from domiknows.solver.lossCalculator import LossCalculator
from domiknows.solver.sampleLossCalculator import SampleLossCalculator

# Both calculators accept an optional TNormSelector
loss_calc = LossCalculator(solver, tnorm_selector=selector)

# If no selector is injected, they create one from the tnorm argument:
loss_calc.calculateLoss(dn, tnorm='default')   # creates one-shot default selector
loss_calc.calculateLoss(dn, tnorm='L')          # creates one-shot specific selector
```

### Manual Override

```python
# Force a specific t-norm for a type via class-level config
LossCalculator.set_tnorm_for_type('exactAL', 'P')

# Replace entire config
LossCalculator.set_global_config({
    'atMostAL': 'P',
    'atLeastAL': 'L',
    'sumL': 'L',
    'andL': 'SP',
    'default': 'L',
})

# Inspect tracker state (auto mode)
print(tracker.active_tnorms)
recs = tracker.get_recommendations()
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
| `Best` | Recommended t-norm (applied automatically in auto mode) |

## Configuration Reference

### Command-Line Arguments

| Argument | Values | Default | Description |
|---|---|---|---|
| `--counting_tnorm` | `G`, `P`, `L`, `SP`, `default`, `auto` | `L` | T-norm selection mode. `auto` enables adaptation automatically. |
| `--tnorm_adaptation_interval` | int | `10` | Steps between probing alternative t-norms (auto mode only) |
| `--tnorm_warmup_steps` | int | `5` | Steps before probing begins (auto mode only) |
| `--tnorm_strategy` | `gradient_weighted`, `loss_based`, `rotating` | `gradient_weighted` | Scoring strategy (auto mode only) |
| `--tnorm_min_observations` | int | `20` | Minimum observations per type before recommending (auto mode only) |
| `--adaptive_tnorm` | flag | — | Deprecated. Auto mode now implies adaptation. Kept for backward compat. |

### AdaptiveTNormLossCalculator Parameters (Programmatic)

| Parameter | Default | Description |
|---|---|---|
| `tnorms` | `["L","P","SP","G"]` | T-norms to compare |
| `adaptation_interval` | `50` | Steps between probing alternative t-norms |
| `warmup_steps` | `100` | Steps before probing begins |
| `selection_strategy` | `"gradient_weighted"` | `"gradient_weighted"` or `"loss_based"` |
| `auto_apply` | `True` | Push recommendations to `active_tnorms` |
| `min_observations` | `20` | Minimum observations per type before recommending |
