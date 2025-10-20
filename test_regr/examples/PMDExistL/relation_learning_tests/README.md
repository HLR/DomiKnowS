# Relation Learning Tests

This module implements logical constraint learning for object relations using curriculum learning approaches. The experiments focus on learning complex relational patterns between objects in scenes while enforcing logical constraints through the DomiKnows framework.

## Overview

The relation learning experiment models scenes containing objects with properties and relations between them. The goal is to learn classifiers that can predict object properties and relations while satisfying logical constraints expressed using existential and universal quantifiers.

## Files

- **`main_rel.py`** - Main relation learning experiment runner
- **`graph_rel.py`** - Graph definition for relational constraints
- **`utils.py`** - Relation learning utilities (dataset creation, training, evaluation)
- **`run_curriculum_learning.py`** - Curriculum learning with progressive complexity
- **`run_process.py`** - Batch processing for hyperparameter exploration

## Problem Formulation

### Scene Structure
Each scene contains:
- **Objects**: M objects, each with K-dimensional embeddings
- **Properties**: 4 possible object conditions (is_cond1, is_cond2, is_cond3, is_cond4)
- **Relations**: 4 possible pairwise relations (is_relation1, is_relation2, is_relation3, is_relation4)

### Object Conditions
1. **is_cond1**: `sum(embedding) > 0`
2. **is_cond2**: `sum(abs(embedding)) > 0.2`
3. **is_cond3**: `sum(embedding) < 0`
4. **is_cond4**: `sum(abs(embedding)) < 0.5`

### Object Relations
1. **is_relation1**: `obj1[0] * obj2[0] >= 0`
2. **is_relation2**: `obj1[0] * obj2[0] < 0`
3. **is_relation3**: `obj1[-1] * obj2[-1] >= 0`
4. **is_relation4**: `obj1[-1] * obj2[-1] < 0`

## Usage

### Quick Test (Verify Installation)
Run a minimal test to ensure the framework is working correctly (1-2 minutes):

```bash
uv run python main_rel.py --N 100 --lr 1e-4 --epoch 2 --max_relation 1
```

This minimal test:
- Uses only 100 training samples
- Trains for 2 epochs
- Tests single relation type
- Completes quickly to verify setup

### Basic Execution
```bash
uv run python main_rel.py --N 1000 --lr 1e-4 --epoch 10 --max_relation 3
```

### Key Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--N` | Number of training samples | 1000 | int |
| `--lr` | Learning rate | 1e-4 | float |
| `--epoch` | Training epochs | 4 | int |
| `--max_property` | Max property complexity | 3 | 0-3 |
| `--max_relation` | Max relation complexity | 3 | 0-3 |
| `--constraint_2_existL` | Use constraint 2 ExistL | False | flag |
| `--use_andL` | Use AndL instead of ExistL | False | flag |
| `--evaluate` | Evaluation mode only | False | flag |
| `--load_save` | Load pretrained model | "" | path |
| `--save_file` | Save model path | "model.pth" | path |

## Logical Constraints

### Basic ExistL Constraint
For simple property-only learning:
```
existsL(is_cond_x('x'))
```

### Complex Relational Constraint
For full relational learning:
```
existsL(is_cond_x('x'), is_relation_y('rel1', path=('x', obj1.reversed)), is_cond_z('y', path=('rel1', obj2)))
```

### Alternative AndL Constraint
Using conjunction instead of existential:
```
andL(is_cond_x('x'), is_relation_y('rel1', path=('x', obj1.reversed)), existsL(is_cond_z('y', path=('rel1', obj2))))
```

## Neural Architecture

### Property Classifiers
```python
class Regular2Layer(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 2)
        )
        self.softmax = torch.nn.Softmax(dim=1)
```

### Relation Classifiers
```python
class RelationLayers(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(size * 2, 512),
            torch.nn.Sigmoid(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 2)
        )
```

## Curriculum Learning

### Progressive Complexity
The curriculum learning script (`run_curriculum_learning.py`) trains models with increasing complexity:

1. **Stage 0**: Property-only learning (no relations)
2. **Stage 1**: Single relation type
3. **Stage 2**: Two relation types
4. **Stage 3**: All relation types

### Usage
```bash
uv run python run_curriculum_learning.py
```

This automatically runs:
- Training with progressive relation complexity
- Evaluation after each stage
- Model saving and loading between stages

## Batch Processing

### Hyperparameter Exploration
```bash
uv run python run_process.py
```

Tests combinations of:
- Sample sizes: [1000, 10000]
- Learning rates: [1e-4, 1e-5]
- Epochs: [5, 10]
- With/without ExistL constraints

## Dataset Management

### Automatic Generation
By default, datasets are generated automatically with:
- Balanced positive/negative examples
- Random object embeddings
- Deterministic property/relation evaluation

### Custom Dataset Loading
Place datasets in `../dataset/` directory:
```
dataset/
├── train.json
└── test.json
```

Enable with:
```bash
uv run python main_rel.py --read_data
```

### Dataset Format
```json
{
  "all_obj": [0],
  "obj_index": [0, 1],
  "obj_emb": [[0.1, -0.2, ...], [0.3, 0.1, ...]],
  "condition_label": [true],
  "logic_str": "existsL(is_cond1('x'), ...)"
}
```

## Evaluation Metrics

### Accuracy Tracking
- Training accuracy on constraint satisfaction
- Comparison with majority vote baseline
- Detailed performance logging

### Results Logging
Results are automatically logged to `results_N_{N}.text` with:
- Timestamp and configuration
- Training accuracy
- Baseline comparisons
- Hyperparameter details

## Advanced Features

### Model Persistence
```bash
# Save model
uv run python main_rel.py --save_file my_model.pth

# Load and evaluate
uv run python main_rel.py --load_save my_model.pth --evaluate
```

### Constraint Variations
```bash
# Use AndL instead of ExistL
uv run python main_rel.py --use_andL

# Use alternative constraint formulation
uv run python main_rel.py --constraint_2_existL
```

## Troubleshooting

### Common Issues

1. **Low Accuracy**
   - Increase learning rate or epochs
   - Check constraint complexity vs. dataset size
   - Verify logical constraint formulation

2. **Memory Issues**
   - Reduce batch size (currently fixed at 1)
   - Decrease model dimensions
   - Use smaller datasets for debugging

3. **Constraint Violations**
   - Check logical constraint syntax
   - Verify graph construction in `graph_rel.py`
   - Ensure proper sensor configuration

### Debug Mode
```bash
# Enable verbose logging
uv run python main_rel.py --N 100 --epoch 1
```