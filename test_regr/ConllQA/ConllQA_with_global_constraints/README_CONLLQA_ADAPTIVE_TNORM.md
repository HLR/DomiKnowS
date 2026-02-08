# ConllQA Example: Adaptive T-Norm Selection

This document describes how the adaptive t-norm system is integrated into the ConllQA training pipeline (`main_new.py`).

## The ConllQA Setup

The ConllQA example trains a BERT-based NER model with logical constraints defined at two levels:

**Global constraints** (in `graph.py`) — applied to every training sample:

```python
atMostAL(people, 3)        # LC0
atMostAL(organization, 3)  # LC1
atMostAL(location, 3)      # LC2
atLeastAL(location, 1)     # LC3
```

**Executable constraints** (compiled from data) — one per sample, compiled from query strings like:

```
atMostAL(organization, 1)     # YN question
atLeastAL(people, 1)          # YN question
exactAL(people, 2)            # YN question
sumL(organization, 1)         # Counting question
```

This gives 4 global constraints + ~4816 executable constraints per epoch, spanning 4 constraint types: `atMostAL`, `atLeastAL`, `exactAL`, `sumL`.

## How It's Wired In main_new.py

### The Callback Factory

`create_adaptive_training_callback` creates the tracker and two callbacks:

```python
def create_adaptive_training_callback(program, models, args):
    adaptive_tracker = AdaptiveTNormLossCalculator(
        tnorms=["L", "P", "SP", "G"],
        adaptation_interval=args.tnorm_adaptation_interval,  # default: 10
        warmup_steps=args.tnorm_warmup_steps,                # default: 5
        selection_strategy=args.tnorm_strategy,               # default: gradient_weighted
        auto_apply=args.adaptive_tnorm,                       # False = track only
        min_observations=20,
    )
    # Returns: on_step_end, on_epoch_end, adaptive_tracker
```

**`on_step_end(output)`** — called after every training step:
1. Extracts the datanode from the training output tuple
2. Calls `datanode.calculateLcLoss(tnorm=current_tnorm)` to get losses
3. Computes classifier gradient norm across all classifier heads
4. Calls `tracker.record_observation()` for each constraint
5. Every `adaptation_interval` steps, probes alternative t-norms via additional `calculateLcLoss()` calls and records via `tracker.record_tnorm_comparison()`

**`on_epoch_end()`** — called after every epoch:
1. Delegates to `tracker.on_epoch_end(apply=auto_apply)`
2. Prints the analysis table
3. If `auto_apply=True`, pushes recommended t-norms into `LossCalculator.TNORM_CONFIG`
4. Resets per-epoch metrics

### Registration in main()

```python
# Single unified callback — replaces the old dual-tracker setup
on_step, on_epoch, tracker = create_adaptive_training_callback(program, _models, args)
program.after_train_step.append(on_step)
program.after_train_epoch.append(on_epoch)

if args.adaptive_tnorm:
    print("[Adaptive T-Norm] Enabled with auto-apply")
else:
    print("[Adaptive T-Norm] Tracking only")
```

There is no separate `_adaptive_tnorm` instance — `create_adaptive_training_callback` handles everything.

## Running

### Tracking Only (default)

```bash
uv run main_new.py --epochs 6 --device cuda:3
```

You get the analysis table each epoch but t-norms stay fixed at whatever `--counting_tnorm` specifies (default: `L`).

### With Auto-Switching

```bash
uv run main_new.py --epochs 6 --device cuda:3 --adaptive_tnorm
```

After epoch 1, the system compares t-norm performance and switches if beneficial. Example output:

```
🔄 APPLIED T-NORM CHANGES:
   atMostAL: 'L' → 'P'
```

From epoch 3 onward, all `atMostAL` constraints will use Product instead of Łukasiewicz.

### With Custom Intervals

```bash
uv run main_new.py --epochs 10 --device cuda:3 \
    --adaptive_tnorm \
    --tnorm_adaptation_interval 5 \
    --tnorm_warmup_steps 3 \
    --tnorm_strategy loss_based
```

Lower `tnorm_adaptation_interval` means more frequent probing (more accurate but slightly slower). `loss_based` strategy ignores gradient information and scores purely on loss value.

## CLI Arguments

| Argument | Default | Effect |
|---|---|---|
| `--counting_tnorm` | `L` | Initial t-norm used for all counting constraints |
| `--adaptive_tnorm` | `False` | Enable auto-switching at epoch boundaries |
| `--tnorm_strategy` | `gradient_weighted` | Scoring: `gradient_weighted` or `loss_based` |
| `--tnorm_adaptation_interval` | `10` | Steps between t-norm comparison probes |
| `--tnorm_warmup_steps` | `5` | Steps before first probe |

## What Changed vs. the Old Setup

| Before | After |
|---|---|
| Two separate tracker instances (`_adaptive_tnorm` + `create_adaptive_training_callback`) | Single `AdaptiveTNormLossCalculator` via `create_adaptive_training_callback` |
| Manual `type_metrics` dict with hand-rolled type detection | `tracker.record_observation()` with shared `get_constraint_type()` helper |
| Metrics accumulated across epochs without reset | Per-epoch metrics reset; cumulative metrics retained separately |
| Coverage counted by `len(m.losses) > 1` (wrong — counted samples not constraints) | Coverage counted by `LC*` vs `ELC*` prefix (correct — 4 global, ~4816 executable) |
| Recommendations printed but never applied | `auto_apply=True` pushes changes to `LossCalculator.TNORM_CONFIG` |

## Typical Training Output

```
[Epoch 3] Adaptive T-Norm Analysis
===========================================================================

📊 AGGREGATED BY CONSTRAINT TYPE (what the model actually learns):
---------------------------------------------------------------------------
Type          Count  AvgLoss  AvgGrad │       L       P      SP       G │ Best
---------------------------------------------------------------------------
atMostAL      15096   0.0740     0.00 │  0.071✓  0.195   0.195   0.195  │ L
atLeastAL      5448   0.0019     0.00 │  0.000✓  0.027   0.027   0.027  │ L
exactAL         667   0.9869     1.24 │  0.984✓  0.988   0.988   0.988  │ L
sumL           2869   0.0000     0.00 │  0.000✓  0.000   0.000   0.000  │ L
---------------------------------------------------------------------------

📈 CONSTRAINT COVERAGE:
   Global constraints (graph-level):    4
   Executable constraints (per-sample): 4816

💡 RECOMMENDATIONS:
   atLeastAL: Use t-norm 'L'
   atMostAL: Use t-norm 'L'
   exactAL: Use t-norm 'L'
   sumL: Use t-norm 'L'

   (no t-norm changes needed)

===========================================================================
```

The `Count` column shows observations **this epoch only** (resets between epochs). `Global constraints: 4` correctly reflects the 4 constraints in `graph.py`. `Executable constraints: 4816` reflects the per-sample compiled constraints.

## Accessing Results After Training

```python
# What the tracker recommends now
tracker.get_recommendations()
# {'atMostAL': 'L', 'atLeastAL': 'L', 'exactAL': 'L', 'sumL': 'L'}

# What was recommended each epoch
tracker.recommendation_history
# [{'atMostAL': 'L', ...}, {'atMostAL': 'P', ...}, ...]

# What's currently active in LossCalculator
from domiknows.solver.ilpOntSolverTools.lossCalculator import LossCalculator
print(LossCalculator.TNORM_CONFIG)

# Global vs executable count
global_n, exec_n = tracker.get_constraint_coverage()
```