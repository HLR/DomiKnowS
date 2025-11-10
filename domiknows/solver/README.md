# DomiKnows Solver Components

This directory contains the constraint solver implementations for the DomiKnows framework, enabling inference and learning with logical constraints.

---

## Solver Hierarchy Overview

| Solver Class | File | Parent Class | Primary Use Case |
|-------------|------|--------------|------------------|
| **Base Solvers** | | | |
| `ilpOntSolver` | `ilpOntSolver.py` | Abstract | Base interface for all solvers |
| `dummyILPOntSolver` | `dummyILPOntSolver.py` | `ilpOntSolver` | Pass-through solver (no inference) |
| **ILP Solvers** | | | |
| `gurobiILPOntSolver` | `gurobiILPOntSolver.py` | `ilpOntSolver` | Gurobi-based ILP inference |
| **Boolean Method Processors** | | | |
| `ilpBooleanProcessor` | `ilpBooleanMethods.py` | Abstract | Base interface for logical operations |
| `gurobiILPBooleanProcessor` | `gurobiILPBooleanMethods.py` | `ilpBooleanProcessor` | ILP encoding of logical operators |
| `lcLossBooleanMethods` | `lcLossBooleanMethods.py` | `ilpBooleanProcessor` | Differentiable t-norm logic |
| `lcLossSampleBooleanMethods` | `lcLossSampleBooleanMethods.py` | `ilpBooleanProcessor` | Sample-based logic evaluation |
| `booleanMethodsCalculator` | `ilpBooleanMethodsCalculator.py` | `ilpBooleanProcessor` | Numeric logic evaluation |
| **Factory** | | | |
| `ilpOntSolverFactory` | `ilpOntSolverFactory.py` | - | Solver instance management |

### Quick Selection Guide

**Choose based on your needs:**

- **Constraint-based inference**: `gurobiILPOntSolver` (requires Gurobi license)
- **Differentiable constraint learning**: Use `lcLossBooleanMethods` with loss models
- **Constraint verification**: Use `booleanMethodsCalculator` 
- **No inference needed**: `dummyILPOntSolver`

---

## Core Components

### Solver Factory (`ilpOntSolverFactory.py`)

#### `ilpOntSolverFactory`
Centralized factory for creating and managing solver instances.

**Key Features:**
- Automatic solver selection based on configuration
- Instance caching (singleton per graph+ontology combination)
- Support for multiple solver types

**Usage:**
```python
from domiknows.solver import ilpOntSolverFactory

# Create solver instance
solver = ilpOntSolverFactory.getOntSolverInstance(
    graph,
    _ilpConfig={
        'ilpSolver': 'Gurobi',
        'log_level': logging.INFO
    }
)

# Solver is cached - subsequent calls return same instance
solver2 = ilpOntSolverFactory.getOntSolverInstance(graph)
assert solver is solver2
```

---

## Base Classes

### `ilpOntSolver` (`ilpOntSolver.py`)
Abstract base class defining the solver interface.

**Key Methods:**
```python
class ilpOntSolver:
    def calculateILPSelection(self, dn, *conceptsRelations, **kwargs):
        """
        Main inference method - finds optimal variable assignments.
        
        Args:
            dn: Root DataNode
            *conceptsRelations: Concepts/relations to solve for
            key: Probability key (default: ("local", "softmax"))
            fun: Optional probability transformation
            epsilon: Probability clipping value
            minimizeObjective: Minimize vs maximize (default: False)
        
        Returns:
            Updated DataNode with ILP inference results
        """
        pass
    
    def calculateLcLoss(self, dn, tnorm='L', **kwargs):
        """
        Calculate differentiable constraint loss.
        
        Args:
            dn: DataNode
            tnorm: T-norm type ('L', 'G', 'P', 'SP')
            sample: Enable sampling
            sampleSize: Number of samples
        
        Returns:
            Dictionary mapping constraint names to loss values
        """
        pass
    
    def verifyResultsLC(self, dn, key="/argmax"):
        """
        Verify constraint satisfaction on predictions.
        
        Args:
            dn: DataNode with predictions
            key: Attribute key for predictions
        
        Returns:
            Dictionary with satisfaction rates per constraint
        """
        pass
```

---

## ILP Solver Implementation

### `gurobiILPOntSolver` (`gurobiILPOntSolver.py`)
Full-featured ILP solver using Gurobi optimizer.

**Key Features:**
- Creates ILP variables for concepts/relations
- Encodes logical constraints as linear inequalities
- Solves for optimal assignment maximizing probabilities
- Supports multiclass concepts with exclusivity constraints
- Model reuse for efficiency
- Comprehensive logging and debugging

**Constraint Types Encoded:**
1. **Graph constraints**: Subclass relations, disjointness, domain/range
2. **Logical constraints**: User-defined constraints from graph
3. **Multiclass exclusivity**: Exactly one label per instance

**Main Workflow:**
```python
solver = gurobiILPOntSolver(graph, ontologies, config)

# 1. Create ILP variables for each concept instance
# 2. Add graph-based constraints (subclass, disjoint, etc.)
# 3. Add user logical constraints
# 4. Set objective: maximize sum of (probability × variable)
# 5. Solve ILP model
# 6. Extract solution and update DataNode
```

**Advanced Features:**

#### Model Reuse
```python
solver = gurobiILPOntSolver(graph, ontologies, config, reuse_model=True)

# First call builds complete model
solver.calculateILPSelection(dn, *concepts)

# Subsequent calls with same structure reuse model
# (much faster - only updates objective coefficients)
solver.calculateILPSelection(dn2, *concepts)
```

#### Probability Transformation
```python
def clip_extreme(probs):
    """Custom probability transformation"""
    return torch.clamp(probs, min=0.01, max=0.99)

solver.calculateILPSelection(
    dn, 
    *concepts,
    fun=clip_extreme,
    epsilon=0.00001
)
```

#### Constraint Priority (P-values)
```python
# Constraints with priority p=100 enforced first
# Then try p=90, p=80, etc. until model feasible
# Returns solution with maximum priority satisfied

logicalConstraint.p = 90  # Set priority on constraint
solver.calculateILPSelection(dn, *concepts)
```

---

## Boolean Method Processors

### `ilpBooleanProcessor` (`ilpBooleanMethods.py`)
Abstract interface defining logical operations. All implementations support:

**Core Operations:**
- `notVar`: Logical negation
- `andVar`: N-ary conjunction
- `orVar`: N-ary disjunction
- `nandVar`: N-ary NAND
- `norVar`: N-ary NOR
- `xorVar`: Exclusive OR
- `ifVar`: Implication (→)
- `equivalenceVar`: Bi-conditional (↔)

**Quantifiers:**
- `countVar`: Count-based constraints (≥, ≤, ==)
- `compareCountsVar`: Compare counts between sets
- `summationVar`: Sum of binary variables

**Special:**
- `fixedVar`: Fix variables to ground truth

**Two Modes:**
1. **Reified** (`onlyConstrains=False`): Returns binary variable representing truth value
2. **Hard** (`onlyConstrains=True`): Adds constraints forcing truth

### `gurobiILPBooleanProcessor` (`gurobiILPBooleanMethods.py`)
Encodes logical operators as ILP constraints.

**Example Encodings:**

#### NOT
```python
# varNOT + var = 1
# Creates: varNOT ∈ {0,1}, varNOT = 1 - var
```

#### AND
```python
# varAND ≤ var_i  (for all i)
# Σ var_i ≤ varAND + N - 1
```

#### OR  
```python
# var_i ≤ varOR  (for all i)
# Σ var_i ≥ varOR
```

#### IF (Implication)
```python
# varIF ≥ 1 - var1
# varIF ≥ var2
# varIF ≤ 1 - var1 + var2
# Encodes: var1 → var2 ≡ ¬var1 ∨ var2
```

#### COUNT (At Least K)
```python
# For ">=" operator with limit k:
# Σ var_i ≥ k - M(1 - varCOUNT)
# Σ var_i ≤ k - 1 + M·varCOUNT
# varCOUNT = 1 iff count ≥ k
```

**Key Implementation Details:**
- Uses "None" → 1 for positive context, 0 for negative
- Handles mixed numeric/variable arguments
- Automatic constraint naming for debugging
- BigM method for indicator constraints

### `lcLossBooleanMethods` (`lcLossBooleanMethods.py`)
Differentiable logic using t-norms for gradient-based learning.

**Supported T-norms:**

| T-norm | Symbol | AND | OR | Key Feature |
|--------|--------|-----|-----|-------------|
| Łukasiewicz | 'L' | max(0, Σx - n + 1) | min(1, Σx) | Piece-wise linear |
| Gödel | 'G' | min(x₁,...,xₙ) | max(x₁,...,xₙ) | Idempotent |
| Product | 'P' | ∏x_i | Σx - ∏x_i | Smooth gradients |
| Simplified Product | 'SP' | ∏x_i | Similar to 'P' | Faster computation |

**Usage:**
```python
processor = lcLossBooleanMethods()
processor.setTNorm('P')  # Product t-norm
processor.setCountingTNorm('L')  # Different for counting

# Differentiable AND
success = processor.andVar(None, var1, var2)  # Returns tensor
loss = 1 - success  # Constraint violation

# With gradient tracking
loss.backward()  # Gradients flow through logical operations
```

**Advanced Counting:**

The counting operations use **Poisson-binomial PMF** for exact probability:

```python
# For count(vars) == k with Product t-norm
pmf = processor.calc_probabilities(probs, n)  # Full PMF over 0..n
loss = 1 - pmf[k]  # Differentiable loss
```

**Count Loss Computation:**
- **Łukasiewicz**: Selects top-k probabilities, sums with offset
- **Gödel**: Takes minimum of top-k (for ≥) or bottom-(n-k) (for ≤)
- **Product**: Exact Poisson-binomial distribution
- **Simplified Product**: Approximation using top-k products

**Logging:**
Dedicated count operations logger tracks:
- Input variables and their values
- Intermediate calculations
- Final loss/success values
- Stored in `logs/lc_loss_count_operations.log`

### `lcLossSampleBooleanMethods` (`lcLossSampleBooleanMethods.py`)
Sample-based constraint evaluation for large-scale problems.

**How It Works:**
1. Generate binary samples from probability distributions
2. Evaluate constraints on samples (Boolean logic)
3. Aggregate results: success rate = fraction of satisfied samples

**Usage:**
```python
processor = lcLossSampleBooleanMethods()
processor.sampleSize = 100

# Samples are generated once per concept
# Then reused across all constraints
success = processor.andVar(None, sampled_var1, sampled_var2)
# Returns: Boolean tensor [batch_size] indicating constraint satisfaction
```

**When to Use:**
- Large constraint groundings (>1000 instances)
- Memory constraints
- Approximate constraint satisfaction sufficient

### `booleanMethodsCalculator` (`ilpBooleanMethodsCalculator.py`)
Numeric evaluation for constraint verification.

**Usage:**
```python
calculator = booleanMethodsCalculator()

# Evaluates constraints on concrete 0/1 assignments
result = calculator.andVar(None, 1, 0, 1)  # Returns: 0
result = calculator.countVar(None, 1, 1, 0, limitOp='>=', limit=2)  # Returns: 1
```

**Use Cases:**
- Constraint verification after inference
- Testing constraint definitions
- Debugging logical expressions

---

## Configuration

### `ilpConfig` (`ilpConfig.py`)
Global solver configuration dictionary.

**Key Settings:**
```python
ilpConfig = {
    'ilpSolver': 'Gurobi',  # Solver type
    'ifLog': True,          # Enable logging
    'log_name': 'ilpOntSolver',
    'log_level': logging.INFO,
    'log_filename': 'logs/ilpOntSolver',
    'log_filesize': 5*1024*1024*1024,  # 5GB
    'log_backupCount': 5,
    'log_fileMode': 'a'
}
```

**Customization:**
```python
custom_config = ilpConfig.copy()
custom_config['log_level'] = logging.DEBUG
custom_config['ilpSolver'] = 'Gurobi'

solver = ilpOntSolverFactory.getOntSolverInstance(
    graph, 
    _ilpConfig=custom_config
)
```

---

## Common Workflows

### 1. ILP Inference
```python
from domiknows.solver import ilpOntSolverFactory

# Create solver
solver = ilpOntSolverFactory.getOntSolverInstance(graph)

# Run inference
solver.calculateILPSelection(
    root_datanode,
    *concepts_and_relations,
    key=("local", "softmax"),
    epsilon=0.00001
)

# Results stored in datanode attributes with key '<concept>/ILP'
```

### 2. Constraint Loss Calculation
```python
# During training
lcLosses = solver.calculateLcLoss(
    datanode,
    tnorm='P',              # Product t-norm
    counting_tnorm='L',     # Łukasiewicz for counts
    sample=False
)

# Extract losses
for lc_name, lc_info in lcLosses.items():
    loss = lc_info['loss']
    conversion = lc_info['conversion']  # 1 - loss
    print(f"{lc_name}: loss={loss:.4f}, satisfaction={conversion:.4f}")

# Aggregate loss
total_loss = sum(lc['loss'] for lc in lcLosses.values() if lc['loss'] is not None)
```

### 3. Sample-Based Loss (Large Scale)
```python
lcLosses = solver.calculateLcLoss(
    datanode,
    sample=True,
    sampleSize=100,          # 100 samples per constraint
    sampleGlobalLoss=False   # Per-constraint sampling
)
```

### 4. Constraint Verification
```python
# After inference/prediction
verification = solver.verifyResultsLC(
    datanode,
    key="/local/argmax"  # or "/ILP" for ILP results
)

for lc_name, results in verification.items():
    satisfied_pct = results['satisfied']
    print(f"{lc_name}: {satisfied_pct:.2f}% satisfied")
    
    # For implication constraints
    if 'ifSatisfied' in results:
        if_satisfied = results['ifSatisfied']
        print(f"  (when antecedent true: {if_satisfied:.2f}%)")
```

### 5. Model Reuse for Efficiency
```python
solver = ilpOntSolverFactory.getOntSolverInstance(
    graph,
    _ilpConfig={'ilpSolver': 'Gurobi'},
    reuse_model=True
)

# First call: builds full model (~1s)
solver.calculateILPSelection(batch1, *concepts)

# Subsequent calls: reuse model (~0.1s)
for batch in batches:
    solver.calculateILPSelection(batch, *concepts)
```

---

## Constraint Definition

### Graph-Based Constraints

Automatically derived from graph structure:

```python
from domiknows.graph import Concept, Relation

# Subclass constraint: Person ⊆ Entity
person = Concept('person')
entity = Concept('entity')
person.is_a(entity)
# Generates: IF(person(x), entity(x))

# Disjoint constraint: Person ⊥ Organization  
organization = Concept('organization')
person.not_a(organization)
# Generates: NAND(person(x), organization(x))

# Domain/range constraint
work_for = Relation('work_for')
work_for.has_a(person, organization)
# Generates: IF(work_for(x,y), AND(person(x), organization(y)))
```

### User-Defined Logical Constraints

```python
from domiknows.graph import V, ifL, andL, orL, countL

# Implication: work_for(x,y) → person(x) ∧ organization(y)
ifL(work_for(V.pair), andL(person(V.pair[0]), organization(V.pair[1])))

# Counting: At least 2 entities per sentence
countL(entity(V.x), V.x.from_sentence(V.s), '>=', 2)

# Complex: If someone works_for an org, they must have a job_title
ifL(
    work_for(V.x, V.y),
    countL(job_title(V.x, V.t), V.t.from_person(V.x), '>=', 1)
)
```

### Constraint Priority

```python
# Critical constraints (always satisfied if possible)
critical_lc.p = 100

# Important constraints
important_lc.p = 80

# Nice-to-have constraints  
optional_lc.p = 50

# Solver tries to satisfy highest priority first
# Falls back to lower priorities if infeasible
```

---

## T-norm Selection Guide

| Use Case | Recommended T-norm | Reason |
|----------|-------------------|---------|
| Smooth gradients | Product ('P') | Continuous derivatives |
| Sparse constraints | Łukasiewicz ('L') | Piece-wise linear, efficient |
| Hard constraints | Gödel ('G') | Idempotent (repeated AND doesn't change) |
| Counting operations | Łukasiewicz ('L') | Exact for discrete counts |
| Fast approximation | Simplified Product ('SP') | Faster than full Product |

**Mixed T-norms:**
```python
# Use Product for main logic, Łukasiewicz for counts
processor.setTNorm('P')
processor.setCountingTNorm('L')
```

---

## Performance Optimization

### 1. Model Reuse
```python
# Enable model caching
solver = ilpOntSolverFactory.getOntSolverInstance(
    graph, 
    reuse_model=True
)
# Subsequent calls ~10x faster
```

### 2. Constraint Sampling
```python
# For >1000 constraint groundings
lcLosses = solver.calculateLcLoss(
    datanode,
    sample=True,
    sampleSize=100  # Approximate with 100 samples
)
```

### 3. Semantic Sampling
```python
# Generate all valid assignments (for small domains)
lcLosses = solver.calculateLcLoss(
    datanode,
    sample=True,
    sampleSize=-1,  # Special: semantic complete sampling
    conceptsRelations=concepts
)
```

### 4. Batch Processing
```python
# Process multiple datanodes with same structure
for batch in batches:
    solver.calculateILPSelection(batch, *concepts)
    # Reuses model automatically
```

---

## Debugging

### Logging Levels
```python
import logging

# Detailed constraint processing
ilpConfig['log_level'] = logging.DEBUG

# Timing information  
ilpConfig['log_level'] = logging.INFO

# Errors only
ilpConfig['log_level'] = logging.ERROR
```

### Model Inspection
```python
# ILP model written to logs/GurobiModel.lp
# Infeasible models written to logs/GurobiInfeasible.ilp
# Solutions written to logs/GurobiSolution.sol

# After solving, inspect:
# 1. GurobiModel.lp - full model
# 2. GurobiSolution.sol - optimal values
# 3. ilpOntSolver.log - detailed execution log
```

### Count Operations Logging
```python
# Dedicated logger for count operations
# Logs every step of count constraint evaluation
# Stored in logs/lc_loss_count_operations.log

# Enable debug logging
custom_config = ilpConfig.copy()
custom_config['count_log_level'] = logging.DEBUG
```

### Constraint Verification
```python
# Verify constraints are satisfied
verification = solver.verifyResultsLC(datanode)

for lc_name, results in verification.items():
    if results['satisfied'] < 95.0:
        print(f"Warning: {lc_name} only {results['satisfied']:.1f}% satisfied")
```

---

## Advanced Features

### Custom Boolean Processors

Create custom logic implementations:

```python
class CustomBooleanProcessor(ilpBooleanProcessor):
    def andVar(self, m, *var, onlyConstrains=False):
        # Custom AND implementation
        pass
    
    def countVar(self, m, *var, limitOp='==', limit=1, **kwargs):
        # Custom counting logic
        pass

# Use with solver
solver.myIlpBooleanProcessor = CustomBooleanProcessor()
```

### Constraint-Specific T-norms

```python
# Use different t-norms for different constraint types
processor = lcLossBooleanMethods()

# Main logic: Product
processor.setTNorm('P')

# Counting: Łukasiewicz  
processor.setCountingTNorm('L')

# Automatically applies appropriate t-norm per operation
```

### Gumbel-Softmax Integration

```python
# For discrete optimization during training
from domiknows.model import SampleLossModel

model = SampleLossModel(
    graph,
    use_gumbel=True,
    temperature=1.0,
    hard_gumbel=False
)

# Solver automatically uses Gumbel-Softmax samples
lcLosses = solver.calculateLcLoss(datanode, sample=True, sampleSize=100)
```

---

## Best Practices

1. **Start with ILP**: Use `gurobiILPOntSolver` for initial inference
2. **Profile constraints**: Check satisfaction rates with `verifyResultsLC`
3. **Choose t-norms carefully**: Product for smooth gradients, Łukasiewicz for efficiency
4. **Use sampling**: For >1000 constraint groundings
5. **Enable model reuse**: 10x speedup for repeated inference
6. **Set priorities**: Critical constraints should have p=100
7. **Monitor logs**: Check `ilpOntSolver.log` for issues
8. **Verify results**: Always run `verifyResultsLC` on test set

---

## Common Issues

### Infeasible Models
```python
# Check logs/GurobiInfeasible.ilp
# Common causes:
# 1. Conflicting constraints
# 2. Over-constrained problem
# 3. Bug in constraint definition

# Solution: Lower priority of some constraints
optional_constraint.p = 50  # Instead of 100
```

### Slow Inference
```python
# Enable model reuse
solver = ilpOntSolverFactory.getOntSolverInstance(graph, reuse_model=True)

# Or use sampling
lcLosses = solver.calculateLcLoss(datanode, sample=True, sampleSize=100)
```

### NaN Losses
```python
# Usually caused by extreme probabilities
# Use epsilon clipping
solver.calculateILPSelection(datanode, *concepts, epsilon=0.001)

# Or probability transformation
def safe_probs(p):
    return torch.clamp(p, min=0.01, max=0.99)

solver.calculateILPSelection(datanode, *concepts, fun=safe_probs)
```

---

## Requirements

- **Gurobi Optimizer**: Required for `gurobiILPOntSolver` (free academic license available)
- **PyTorch**: For differentiable constraint learning
- **owlready2**: For ontology loading (optional)

---

## Example: Complete Workflow

```python
from domiknows.solver import ilpOntSolverFactory
from domiknows.graph import Concept, V, ifL, andL
import torch

# 1. Define graph with constraints
graph = Graph('my_graph')
person = Concept('person')
organization = Concept('organization') 
work_for = Relation('work_for')

# Add logical constraint
ifL(work_for(V.x, V.y), andL(person(V.x), organization(V.y)))

# 2. Create solver
solver = ilpOntSolverFactory.getOntSolverInstance(
    graph,
    _ilpConfig={'ilpSolver': 'Gurobi'},
    reuse_model=True
)

# 3. Training: compute constraint loss
for batch in train_loader:
    # Forward pass
    predictions = model(batch)
    
    # Compute constraint loss
    lcLosses = solver.calculateLcLoss(
        batch,
        tnorm='P',
        counting_tnorm='L'
    )
    
    # Aggregate losses
    data_loss = criterion(predictions, labels)
    constraint_loss = sum(lc['loss'] for lc in lcLosses.values())
    
    total_loss = data_loss + 0.5 * constraint_loss
    total_loss.backward()
    optimizer.step()

# 4. Inference: use ILP
for batch in test_loader:
    predictions = model(batch)
    
    # Apply ILP inference
    solver.calculateILPSelection(
        batch,
        person, organization, work_for,
        key=("local", "softmax")
    )
    
    # Results in batch attributes under '<concept>/ILP'

# 5. Verification
verification = solver.verifyResultsLC(test_data, key="/ILP")
for lc_name, results in verification.items():
    print(f"{lc_name}: {results['satisfied']:.2f}% satisfied")
```

---

## Migration Notes

### From Earlier Versions

If using legacy `torch.py` models:
- Import from `pytorch.py` instead
- Update `ilpConfig` structure if customized
- Check constraint definitions for new syntax

### Gurobi License

Academic license: https://www.gurobi.com/academia/academic-program-and-licenses/
- Free for academic use
- Requires institutional email
- Includes WLS (Web License Service)

---

## Further Reading

- **Logical Constraints**: See `domiknows/graph/logicalConstrain.py`
- **T-norms**: See papers on fuzzy logic and t-norm semantics
- **ILP Encoding**: Review `gurobiILPBooleanMethods.py` for details
- **Gurobi**: Official documentation at gurobi.com