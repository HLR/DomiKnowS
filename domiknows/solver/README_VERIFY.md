# LogicalConstraintVerifier

## Overview

The `LogicalConstraintVerifier` class provides functionality for verifying logical constraint satisfaction on model predictions. It evaluates how well predictions comply with domain knowledge encoded as logical constraints.

## Purpose

**`verifyResults()` does NOT enforce constraints** - it only checks if predictions already satisfy them.

- Use `calculateILPSelection()` to **enforce** constraints (changes predictions)
- Use `verifyResultsLC()` to **measure** constraint satisfaction (reports metrics only)

The verifier is essentially a **constraint compliance metric** for evaluating model quality beyond traditional accuracy measures.

## Key Features

- Operates on discrete predictions (e.g., argmax results) rather than probability distributions
- Computes overall satisfaction rates for each logical constraint
- For conditional constraints (`ifL`, `forAllL`), computes conditional satisfaction when the antecedent is true
- Provides detailed statistics including processing time and satisfaction percentages
- Works post-inference without requiring the ILP solver

## When to Use

### 1. After ML-Only Inference (No ILP)

Check if the neural network learned to respect the constraints during training.

```python
# Train model without ILP
model.train()

# Get predictions
predictions = model.predict(data)

# Verify how well predictions satisfy constraints
results = logicalConstraintVerifier.verifyResultsLC(dn, key="/local/argmax")
print(f"Constraint satisfaction: {results['my_constraint']['satisfied']:.1f}%")
```

### 2. Comparing ML vs ILP Results

Demonstrate that ILP inference improves constraint satisfaction. **ILP results should verify to 100%** (or very close) for all active constraints that were enforced during ILP inference.

```python
# Get ML predictions
ml_results = logicalConstraintVerifier.verifyResultsLC(dn, key="/local/argmax")

# Run ILP inference
solver.calculateILPSelection(dn, *concepts)

# Verify ILP results
ilp_results = logicalConstraintVerifier.verifyResultsLC(dn, key="/ILP/x")

# Compare
print(f"ML satisfaction: {ml_results['constraint1']['satisfied']:.1f}%")
# Expected: 65-95% (depends on training)

print(f"ILP satisfaction: {ilp_results['constraint1']['satisfied']:.1f}%")
# Expected: 100.0% (ILP enforces constraints)
```

**Note**: ILP satisfaction < 100% may indicate:
- Model is infeasible (no solution exists satisfying all constraints)
- Constraint has lower priority (`p` parameter) and was relaxed
- Constraint was inactive during ILP but active during verification
- New constraints added after ILP inference

### 3. Debugging Constraint Definitions

Identify problematic constraints that are frequently violated.

```python
# After defining constraints
results = logicalConstraintVerifier.verifyResultsLC(dn, key="/local/argmax")

for constraint_name, result in results.items():
    if result['satisfied'] < 50:  # Low satisfaction
        print(f"⚠️ {constraint_name} only {result['satisfied']:.1f}% satisfied")
        print(f"   Check if constraint is too strict or incorrectly defined")
```

### 4. Evaluating Different Training Strategies

Assess if constraint-aware training improves satisfaction rates.

```python
# Strategy 1: Train without loss constraints
model1.train()
results1 = logicalConstraintVerifier.verifyResultsLC(dn1, key="/local/argmax")

# Strategy 2: Train with loss constraints
model2.train_with_constraints()
results2 = logicalConstraintVerifier.verifyResultsLC(dn2, key="/local/argmax")

# Compare which strategy better respects constraints
```

### 5. Test/Validation Set Evaluation

Report constraint satisfaction as a metric alongside accuracy/F1.

```python
# After training, evaluate on test set
test_predictions = model.predict(test_data)
test_results = logicalConstraintVerifier.verifyResultsLC(test_dn, key="/local/argmax")

print(f"Test set constraint satisfaction:")
for name, result in test_results.items():
    print(f"  {name}: {result['satisfied']:.1f}%")
```

## Usage

### Basic Usage

```python
from domiknows.solver.logicalConstraintVerifier import LogicalConstraintVerifier

# Initialize verifier with solver instance
verifier = LogicalConstraintVerifier(solver)

# Verify results
results = verifier.verifyResults(dn, key="/local/argmax")

# Access results
for constraint_name, metrics in results.items():
    print(f"{constraint_name}:")
    print(f"  Overall satisfaction: {metrics['satisfied']:.1f}%")
    if 'ifSatisfied' in metrics:
        print(f"  Conditional satisfaction: {metrics['ifSatisfied']:.1f}%")
    print(f"  Processing time: {metrics['elapsedInMsLC']:.2f}ms")
```

### Parameters

- **`dn`**: Data node containing the predictions to verify
- **`key`**: Attribute key for accessing predictions in datanodes
  - Default: `"/argmax"` for discrete predicted labels
  - Common alternatives: `"/local/argmax"`, `"/ILP/x"`

### Return Structure

Returns an `OrderedDict` with the following structure:

```python
{
    'constraint_name': {
        'verifyList': [[bool, ...], ...],    # Satisfaction per instance
        'satisfied': float,                   # Overall satisfaction % (0-100)
        'ifVerifyList': [[bool, ...], ...],  # (ifL/forAllL only) Filtered list
        'ifSatisfied': float,                 # (ifL/forAllL only) Conditional satisfaction % (0-100)
        'elapsedInMsLC': float                # Processing time in milliseconds
    },
    ...
}
```

## Result Interpretation

### Overall Satisfaction (`satisfied`)

- **100%**: All instances satisfy the constraint (perfect compliance)
  - **For ILP results**: This is expected for all active constraints that were enforced
  - **For ML results**: Indicates excellent constraint learning
- **0%**: No instances satisfy the constraint (complete violation)
- **< 50%**: Low satisfaction - investigate constraint definition or model training

### Conditional Satisfaction (`ifSatisfied`)

For `ifL` and `forAllL` constraints only:

- Measures satisfaction rate **when the antecedent/condition is true**
- Often more meaningful than overall satisfaction
- Example: For constraint "if(person, adult)" → "Person is adult"
  - `satisfied`: % of all entities that satisfy the rule
  - `ifSatisfied`: % of persons that are adults (more relevant)

### ILP vs ML Satisfaction

| Method | Expected Satisfaction | Meaning |
|--------|----------------------|---------|
| **ML-only** | 60-95% | Depends on training quality and constraint complexity |
| **ILP** | ~100% | ILP enforces constraints during inference |

**If ILP < 100%**: Check for infeasibility, constraint priorities, or inactive constraints during ILP.

## Use Cases Summary

| Use Case | When | Purpose |
|----------|------|---------|
| **ML-Only Evaluation** | After training without ILP | Check if model learned constraints |
| **ML vs ILP Comparison** | After running both approaches | Show ILP improves compliance |
| **Constraint Debugging** | During development | Find problematic constraints |
| **Training Strategy Eval** | Comparing approaches | Assess constraint-aware training |
| **Test Set Metrics** | Final evaluation | Report compliance alongside accuracy |

## Notes

- Only **active head constraints** are verified (nested constraints are skipped)
- **Fixed constraints** (`fixedL`) are skipped as they don't need verification
- Verification operates on **discrete predictions**, not probabilities
- Does not require the ILP solver to run
- Useful for evaluating constraint compliance as a standalone metric

## Integration

The verifier is automatically initialized when creating a `gurobiILPOntSolver`:

```python
solver = gurobiILPOntSolver(graph, ontologies, config)

# Verifier is accessible via:
results = logicalConstraintVerifier.verifyResultsLC(dn, key="/local/argmax")
```

Or can be used standalone:

```python
from domiknows.solver.logicalConstraintVerifier import LogicalConstraintVerifier

verifier = LogicalConstraintVerifier(solver)
results = verifier.verifyResults(dn, key="/local/argmax")
```