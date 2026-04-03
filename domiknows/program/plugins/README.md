# DomiKnowS Training Callback Plugins

A modular, reusable library of callback plugins for DomiKnowS neural-symbolic training experiments. These plugins provide comprehensive monitoring, diagnostics, and adaptive training capabilities without cluttering your main training code.

## 📁 Directory Structure

```
domiknows/program/plugins/
├── callback_plugin_manager.py      # Central plugin coordinator
├── epoch_logging_plugin.py         # Comprehensive epoch metrics
├── adaptive_tnorm_plugin.py        # Adaptive t-norm selection
├── gradient_flow_plugin.py         # Gradient flow diagnostics
├── counting_schedule_plugin.py     # Counting constraint scheduling
├── gumbel_monitoring_plugin.py     # Gumbel-Softmax monitoring
├── bert_unfreezing_plugin.py       # BERT gradual unfreezing
└── README.md                        # This file
```

## 🚀 Quick Start

### 1. Basic Integration (3 steps)

```python
from domiknows.program.plugins.callback_plugin_manager import create_standard_plugin_manager

# Step 1: Create plugin manager
plugin_manager = create_standard_plugin_manager()

# Step 2: Add plugin arguments to your argument parser
def parse_arguments():
    parser = argparse.ArgumentParser()
    # ... your existing arguments ...
    
    plugin_manager.add_arguments_to_parser(parser)
    args = parser.parse_args()
    return args

# Step 3: Configure plugins before training
plugin_manager.configure_all(
    program=program,
    models=models,
    args=args,
    dataset=dataset,
    optimizer_factory=create_optimizer_factory
)

# Train normally - plugins work automatically via callbacks
program.train(dataset, ...)

# Display plugin summaries after training
plugin_manager.final_display_all(final_eval=final_eval)
```

### 2. Run with Default Plugins

```bash
# All plugins enabled with defaults
python main.py --epochs 10 --classifier_lr 1e-3

# Enable adaptive t-norm
python main.py --adaptive_tnorm --tnorm_strategy gradient_weighted

# Enable counting schedule
python main.py --use_counting_schedule --counting_warmup_epochs 4

# Enable Gumbel-Softmax (plugin monitors automatically)
python main.py --use_gumbel --gumbel_temp_start 5.0

# Enable BERT unfreezing (plugin manages automatically)
python main.py --freeze_bert false --warmup_epochs 2 --unfreeze_layers 2
```

## 📦 Available Plugins

### 1. **Epoch Logging Plugin** (`epoch_logging_plugin.py`)

Logs comprehensive training metrics after each epoch.

**Features:**
- Overall, boolean, and counting constraint accuracy
- Gradient norm tracking
- Learning progress assessment
- Performance delta calculations

**Arguments:**
```bash
--eval_fraction 0.2          # Fraction of data for epoch evaluation
--eval_min_samples 50        # Minimum samples for evaluation
--eval_seed 42               # Random seed for subset selection
```

**Output:**
```
[Epoch 5] Metrics:
  Overall Acc:    65.23 (Δ+2.15)
  Boolean Acc:    67.50 (Δ+1.80)
  Counting Acc:   58.30 (Δ+3.20)
  AvgGradNorm:    0.000234
  BERT:           4L unfrozen
```

---

### 2. **Adaptive T-Norm Plugin** (`adaptive_tnorm_plugin.py`)

Tracks per-constraint-type performance and optionally applies automatic t-norm selection.

**Features:**
- Tracks losses and gradients by constraint type
- Compares multiple t-norms (L, P, SP, G)
- Auto-applies best t-norm (optional)
- Exports detailed CSV statistics
- Monitors gradient health (vanishing/exploding)

**Arguments:**
```bash
--adaptive_tnorm             # Enable auto-apply (default: track-only)
--tnorm_strategy gradient_weighted  # Strategy: gradient_weighted, loss_based, rotating
--tnorm_adaptation_interval 10      # Steps between t-norm comparison
--tnorm_warmup_steps 5              # Steps before adaptation begins
--tnorm_min_observations 20         # Min observations for recommendations
```

**Output:**
```
[Adaptive T-Norm Analysis]
  Detailed constraint stats exported to: adaptive_tnorm_details_epoch10.csv (156 records)
  
  Final T-Norm Recommendations by Type:
    atLeastAL            -> L
    atMostAL             -> G
    exactlyAL            -> L
  
  Constraint Type Summary:
    LC0 (atLeastAL  ): 120 obs, loss=0.2341, tnorm=L
    LC1 (atMostAL   ):  98 obs, loss=0.1876, tnorm=G ⚠️
    LC2 (exactlyAL  ):  87 obs, loss=0.3012, tnorm=L
```

**CSV Export:**
Contains per-constraint metrics including observations, average loss, gradient norms, and recommended t-norms.

---

### 3. **Gradient Flow Plugin** (`gradient_flow_plugin.py`)

Diagnoses gradient flow from constraints (especially sumL) back to classifiers.

**Features:**
- Checks if constraints generate gradients
- Isolates sumL gradient contribution
- Detects vanishing/exploding gradients
- Validates backprop through constraint system

**Arguments:**
```bash
--gradient_check_interval 500  # Steps between diagnostic checks
```

**Output:**
```
[Gradient Flow Check - Step 1000]
============================================================
  ✓ LC_sumL_count_people: sumL loss=0.2341, requires_grad=True
  
  Constraint Summary:
    sumL constraints:   8 (avg loss: 0.2456)
    Other constraints:  12 (avg loss: 0.1823)
  
  Classifier Gradient Status:
    people         : grad_norm=  0.0234 (10/10 params)
    organization   : grad_norm=  0.0198 (10/10 params)
    location       : grad_norm=  0.0267 (10/10 params)
  
  Attempting to isolate sumL gradient contribution...
    ✓ sumL contributes grad_norm=0.0156 to classifiers
============================================================

[Gradient Flow Summary - Epoch Complete]
============================================================
  sumL Observations:     24
  Avg sumL Loss:         0.2398
  sumL with requires_grad: 24/24
  Avg sumL→Classifier Gradient: 0.0145
============================================================
```

**Use Cases:**
- Debugging constraint gradient issues
- Validating new constraint implementations
- Tuning t-norm selection for gradient flow

---

### 4. **Counting Schedule Plugin** (`counting_schedule_plugin.py`)

Gradually introduces counting constraints during training with adaptive loss weighting.

**Features:**
- Boolean-only warmup period
- Gradual counting weight increase
- Configurable weight range
- Per-step loss breakdown logging

**Arguments:**
```bash
--use_counting_schedule          # Enable counting schedule
--counting_warmup_epochs 4       # Epochs before introducing counting
--counting_weight_min 0.01       # Minimum counting weight after warmup
--counting_weight_max 0.1        # Maximum counting weight at end
```

**Output:**
```
[Counting Weight Schedule]
============================================================
  Epoch  1: Boolean Only
  Epoch  2: Boolean Only
  Epoch  3: Boolean Only
  Epoch  4: Boolean Only
  Epoch  5: Counting Weight: 0.010
  Epoch  6: Counting Weight: 0.028
  Epoch  7: Counting Weight: 0.046
  Epoch  8: Counting Weight: 0.064
  Epoch  9: Counting Weight: 0.082
  Epoch 10: Counting Weight: 0.100
============================================================

[Epoch 7] Loss Weighting:
  Counting weight: 0.046
  Boolean loss:    0.3421
  Counting loss:   0.2156
  Weighted total:  0.3520
```

**Rationale:**
Counting constraints often have sparse gradients initially. Starting with boolean-only training establishes better base representations before introducing counting.

---

### 5. **Gumbel Monitoring Plugin** (`gumbel_monitoring_plugin.py`)

Monitors Gumbel-Softmax temperature annealing during training.

**Features:**
- Tracks temperature progression
- Logs soft→sharp transition
- Records temperature history
- Identifies annealing phases

**Arguments:**
No plugin-specific arguments (monitors main program's Gumbel settings).

**Output:**
```
  [Gumbel] Temperature: 5.0000 (soft - gradients flow well)
  [Gumbel] Using hard (straight-through) mode

  [Gumbel] Temperature: 2.7500 (medium - balancing gradients and discreteness)
  
  [Gumbel] Temperature: 0.5000 (sharp - approaching discrete predictions)

[Gumbel-Softmax Temperature History]
============================================================
  Epoch  1: 5.0000
  Epoch  2: 4.5500
  Epoch  3: 4.1000
  ...
  Epoch 10: 0.5000

  Temperature change: 5.0000 → 0.5000
  Total annealing: 4.5000
============================================================
```

---

### 6. **BERT Unfreezing Plugin** (`bert_unfreezing_plugin.py`)

Gradually unfreezes BERT layers with differential learning rates.

**Features:**
- Layer-by-layer unfreezing after warmup
- Automatic optimizer recreation
- Differential learning rates (BERT vs classifiers)
- Unfreezing history tracking

**Arguments:**
No plugin-specific arguments (uses main program's BERT settings).

**Output:**
```
[Epoch 1] Warmup - BERT frozen
[Epoch 2] Warmup - BERT frozen
[Epoch 3] Unfroze 2 layers, optimizer updated
[Epoch 4] Unfroze 4 layers, optimizer updated
[Epoch 5] Unfroze 6 layers, optimizer updated

[BERT Unfreezing History]
============================================================
  Epoch  3: Unfroze 2 layers
  Epoch  4: Unfroze 4 layers
  Epoch  5: Unfroze 6 layers
  Epoch  6: Unfroze 8 layers
  Epoch  7: Unfroze 10 layers
  Epoch  8: Unfroze 12 layers

  Final state: 12/12 layers unfrozen
============================================================
```

**Helper Functions:**
```python
from domiknows.program.plugins.bert_unfreezing_plugin import (
    create_optimizer_with_differential_lr,
    create_optimizer_factory
)
```

---

## 🎛️ Plugin Manager (`callback_plugin_manager.py`)

Central coordinator for managing multiple plugins.

### Core Methods

```python
manager = CallbackPluginManager()

# Register plugins
manager.register(EpochLoggingPlugin(), name='EpochLogging')
manager.register(AdaptiveTNormPlugin(), name='AdaptiveTNorm')

# Add all plugin arguments
manager.add_arguments_to_parser(parser)

# Configure all plugins
manager.configure_all(
    program=program,
    models=models,
    args=args,
    dataset=dataset,
    optimizer_factory=create_optimizer_factory
)

# Log all configurations
manager.log_all_configs(args)

# Display all final summaries
manager.final_display_all(final_eval=final_eval)

# Access specific plugin
epoch_plugin = manager.get_plugin('EpochLogging')
metrics = epoch_plugin.metrics_history
```

### Standard Plugin Set

```python
from domiknows.program.plugins.callback_plugin_manager import create_standard_plugin_manager

# Creates manager with all 6 standard plugins
manager = create_standard_plugin_manager()
```

---

## 💡 Usage Examples

### Example 1: Full Feature Training

```bash
python main.py \
  --epochs 10 \
  --classifier_lr 1e-3 \
  --eval_fraction 0.2 \
  --adaptive_tnorm \
  --tnorm_strategy gradient_weighted \
  --use_counting_schedule \
  --counting_warmup_epochs 4 \
  --use_gumbel \
  --gumbel_temp_start 5.0 \
  --freeze_bert false \
  --warmup_epochs 2 \
  --unfreeze_layers 2 \
  --bert_lr 1e-5
```

### Example 2: Selective Plugin Usage

```python
from domiknows.program.plugins.callback_plugin_manager import CallbackPluginManager
from domiknows.program.plugins.epoch_logging_plugin import EpochLoggingPlugin
from domiknows.program.plugins.adaptive_tnorm_plugin import AdaptiveTNormPlugin

# Create custom manager with only specific plugins
manager = CallbackPluginManager()
manager.register(EpochLoggingPlugin())
manager.register(AdaptiveTNormPlugin())

# Use as normal
manager.add_arguments_to_parser(parser)
manager.configure_all(program=program, models=models, args=args)
```

### Example 3: Accessing Plugin Data

```python
# After training completes
epoch_plugin = plugin_manager.get_plugin('EpochLogging')
if epoch_plugin:
    metrics = epoch_plugin.metrics_history
    
    # Analyze training progression
    initial_acc = metrics['overall_acc'][0]
    final_acc = metrics['overall_acc'][-1]
    improvement = final_acc - initial_acc
    
    print(f"Total improvement: {improvement:.2f}%")
    
    # Plot metrics
    import matplotlib.pyplot as plt
    plt.plot(metrics['epoch'], metrics['overall_acc'], label='Overall')
    plt.plot(metrics['epoch'], metrics['bool_acc'], label='Boolean')
    plt.plot(metrics['epoch'], metrics['counting_acc'], label='Counting')
    plt.legend()
    plt.show()

# Export adaptive t-norm results
tnorm_plugin = plugin_manager.get_plugin('AdaptiveTNorm')
if tnorm_plugin and tnorm_plugin.tracker:
    tnorm_plugin.tracker.export_detailed_stats_to_csv('tnorm_results.csv')
    stats = tnorm_plugin.tracker.get_summary_stats()
    recommendations = stats.get('final_recommendations_by_type', {})
    print(f"Recommended t-norms: {recommendations}")
```

### Example 4: Custom Plugin

```python
class CustomMetricPlugin:
    """Example custom plugin following the standard interface."""
    
    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--custom_interval", type=int, default=100)
    
    def configure(self, program, args):
        self.program = program
        self.args = args
        self.metrics = []
        
        program.after_train_step.append(self._collect_metric)
    
    def _collect_metric(self, output):
        # Custom metric collection logic
        self.metrics.append(some_metric)
    
    @staticmethod
    def log_config(args):
        print(f"  Custom Metric:")
        print(f"    Interval: {args.custom_interval}")
    
    def final_display(self):
        print(f"\n[Custom Metric Summary]")
        print(f"  Collected {len(self.metrics)} metrics")

# Register custom plugin
manager.register(CustomMetricPlugin(), 'CustomMetric')
```

---

## 🔧 Plugin Interface Specification

All plugins should implement this interface:

```python
class PluginTemplate:
    """Template for creating new plugins."""
    
    @staticmethod
    def add_arguments(parser):
        """
        Add plugin-specific command-line arguments.
        
        Args:
            parser: argparse.ArgumentParser instance
        """
        parser.add_argument("--my_param", type=int, default=10)
    
    def configure(self, program, models=None, args=None, dataset=None, **kwargs):
        """
        Configure the plugin and register callbacks.
        
        Args:
            program: CallbackProgram instance
            models: Dict with 'bert' and 'classifiers' keys (optional)
            args: Parsed arguments (optional)
            dataset: Training dataset (optional)
            **kwargs: Additional context (optimizer_factory, etc.)
        """
        self.program = program
        self.args = args
        
        # Register callbacks
        program.after_train_epoch.append(self._my_callback)
    
    def _my_callback(self):
        """Internal callback method."""
        pass
    
    @staticmethod
    def log_config(args, models=None):
        """
        Log plugin configuration during training setup.
        
        Args:
            args: Parsed arguments
            models: Optional model dictionary
        """
        print(f"  My Plugin:")
        print(f"    Parameter: {args.my_param}")
    
    def final_display(self, final_eval=None):
        """
        Display final summary/recommendations after training.
        
        Args:
            final_eval: Optional final evaluation results
        """
        print("\n[My Plugin Summary]")
        print(f"  Results: ...")
```

---

## 📊 Output Files

Plugins may generate the following files:

| File | Plugin | Description |
|------|--------|-------------|
| `adaptive_tnorm_details_*.csv` | Adaptive T-Norm | Per-constraint metrics with t-norm recommendations |
| `result.txt` | (Main script) | Final accuracy results |
| `training_*.pth` | (Main script) | Model checkpoints |

---

## 🐛 Troubleshooting

### Issue: Plugin not receiving callbacks

**Solution:** Ensure `configure()` is called before `program.train()`:
```python
plugin_manager.configure_all(program=program, ...)
program.train(dataset, ...)  # Now plugins will receive callbacks
```

### Issue: Missing plugin arguments

**Solution:** Call `add_arguments_to_parser()` before `parse_args()`:
```python
plugin_manager.add_arguments_to_parser(parser)
args = parser.parse_args()  # Now plugin args are available
```

### Issue: Import errors

**Solution:** Add the plugins directory to your Python path:
```python
import sys
sys.path.append('path/to/domiknows/program/plugins')
```

Or use absolute imports:
```python
from domiknows.program.plugins.callback_plugin_manager import create_standard_plugin_manager
```

---

## 📈 Performance Considerations

- **Epoch Logging Plugin**: Evaluates on a subset (`--eval_fraction 0.2`) to minimize overhead
- **Adaptive T-Norm Plugin**: Only compares t-norms at intervals (`--tnorm_adaptation_interval 10`)
- **Gradient Flow Plugin**: Checks every N steps (`--gradient_check_interval 500`)
- All plugins are designed to minimize training slowdown

---

## 🤝 Contributing

When creating new plugins:

1. Follow the plugin interface specification
2. Add comprehensive docstrings
3. Use `@staticmethod` for methods that don't need instance state
4. Handle missing/optional arguments gracefully
5. Provide clear error messages
6. Include usage examples in docstrings
7. Update this README with your plugin
