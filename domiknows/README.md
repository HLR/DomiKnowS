# DomiKnows Framework

DomiKnows is a neural-symbolic AI framework that seamlessly integrates deep learning with logical reasoning. It enables you to build models that learn from data while respecting domain knowledge expressed as logical constraints.

---

## Overview

DomiKnows combines three key paradigms:

1. **Neural Learning**: Standard PyTorch-based deep learning
2. **Symbolic Reasoning**: First-order logic constraints over structured knowledge
3. **Constraint Satisfaction**: ILP and gradient-based inference to satisfy logical rules

This combination allows you to:
- Train models that respect domain knowledge and business rules
- Enforce logical consistency during inference
- Learn from both labeled data and logical constraints
- Achieve better generalization with less training data

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   DomiKnows Framework             │
├─────────────────────────────────────────────────────────┤
│                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Graph     │  │   Sensor   │  │   Program   │  │
│  │  (Ontology) │  │ (Data Flow)│  │ (Training)  │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘ │
│         │                  │                  │   │
│         └─────────┬────────┴────────┬─────────┘       │
│                   │                 │             │
│           ┌───────▼────────┐    ┌─────▼──────┐       │
│           │    DataNode       │   Model   │       │
│           │   (Runtime)       │  (Neural) │       │
│           └───────┬─────────┘   └─────┬──────┘       │
│                   │                 │             │
│                   └─────────┬─────────┘             │
│                             │                     │
│                     ┌───────▼────────┐              │
│                     │     Solver   │              │
│                     │ (Constraints)│              │
│                     └────────────────┘              │
│                                                   │
└─────────────────────────────────────────────────────────┘
```

---

## Core Components

### [Graph](graph/README_GRAPH.md) - Knowledge Representation
Define your domain knowledge as a structured graph with concepts, relations, and logical constraints.

```python
with Graph('knowledge_base') as graph:
    person = Concept('person')
    organization = Concept('organization')
    
    work_for = Concept('work_for')
    work_for.has_a(person, organization)
    
    # Add logical constraint
    ifL(work_for(V.x, V.y), 
        andL(person(V.x), organization(V.y)))
```

**Key Features:**
- Hierarchical concept definitions
- Relation types (IsA, HasA, Contains, Equal)
- First-order logic constraints
- Path-based queries

### [Sensor](sensor/README.md) - Data Pipeline
Connect your data to the knowledge graph through a flexible sensor system.

```python
with person:
    person['embedding'] = ModuleLearner(
        person['input'],
        module=bert_model
    )
    
    person['prediction'] = FullyConnectedLearner(
        person['embedding'],
        input_dim=768,
        output_dim=2,
        loss=nn.CrossEntropyLoss()
    )
```

**Key Features:**
- Automatic data flow management
- PyTorch integration
- Hierarchical aggregation
- Reusable sensor components

### [Program](program/README.md) - Training Interface
High-level programs for training and evaluation with various constraint learning strategies.

```python
program = SolverPOIProgram(
    graph, 
    inferTypes=['local/softmax', 'ILP']
)
program.train(train_data, valid_data, train_epoch_num=50)
```

**Available Programs:**
- `POIProgram`: Standard supervised learning
- `SolverPOIProgram`: Learning + ILP inference
- `PrimalDualProgram`: Constraint satisfaction learning
- `GBIProgram`: Gradient-based inference

### [Solver](solver/README.md) - Constraint Reasoning
Enforce logical constraints through various inference methods.

```python
solver = ilpOntSolverFactory.getOntSolverInstance(graph)

# ILP inference
solver.calculateILPSelection(datanode, person, organization)

# Constraint loss
lc_losses = solver.calculateLcLoss(datanode, tnorm='P')
```

**Inference Methods:**
- Integer Linear Programming (ILP)
- Gradient-Based Inference (GBI)
- Differentiable logic (t-norms)
- Constraint verification

### [DataNode](graph/README.md#datanode-and-data-graph-components) - Runtime Structure
Runtime data instances bound to ontological concepts.

```python
# Query instances
persons = root_dn.findDatanodes(select='person')

# Access predictions
for person_dn in persons:
    pred = person_dn.getAttribute('<person>/ILP')
    
# Compute metrics
metrics = root_dn.getInferMetrics(person, inferType='ILP')
```

---

## Quick Start

### Installation

```bash
pip install domiknows
```

### Basic Example

```python
from domiknows.graph import Graph, Concept, V, ifL, andL
from domiknows.program import SolverPOIProgram
from domiknows.sensor.pytorch.sensors import ReaderSensor, ModuleLearner
import torch.nn as nn

# 1. Define knowledge graph
with Graph('example') as graph:
    # Concepts
    entity = Concept('entity')
    person = Concept('person')
    organization = Concept('organization')
    
    # Hierarchy
    person.is_a(entity)
    organization.is_a(entity)
    
    # Relation
    work_for = Concept('work_for')
    work_for.has_a(person, organization)
    
    # Constraint: work_for implies person and organization
    ifL(work_for(V.x, V.y), 
        andL(person(V.x), organization(V.y)))

# 2. Attach sensors
with person:
    person['features'] = ReaderSensor(keyword='person_features')
    person['prediction'] = ModuleLearner(
        person['features'],
        module=nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        ),
        loss=nn.CrossEntropyLoss()
    )

# 3. Create training program
program = SolverPOIProgram(
    graph,
    inferTypes=['local/softmax', 'ILP']
)

# 4. Train
program.train(
    train_data,
    valid_data,
    train_epoch_num=50,
    Optim=torch.optim.Adam
)

# 5. Test with ILP inference
program.test(test_data, device='cuda')
```

---

## Key Features

### Neural-Symbolic Integration
- **Learn from constraints**: Use logical rules as differentiable loss functions
- **Constrained inference**: Guarantee predictions satisfy logical rules
- **Weak supervision**: Learn from partial labels + domain knowledge

### Flexible Constraint Types
- **Logical**: AND, OR, NOT, IF, equivalence
- **Quantifiers**: exists, forall, counting constraints
- **Paths**: Express constraints over graph structure
- **Priorities**: Soft vs. hard constraints

### Multiple Inference Methods
- **ILP**: Exact inference via integer programming
- **GBI**: Gradient-based iterative refinement
- **Local**: Fast neural predictions
- **Hybrid**: Combine methods for speed/accuracy tradeoff

### Production-Ready
- **PyTorch integration**: Standard deep learning workflow
- **Efficient**: Model reuse, skeleton mode, batching
- **Scalable**: Sampling for large constraint groundings
- **Debuggable**: Visualization, logging, verification tools

---

## Documentation

### Component READMEs
- **[Graph Components](graph/README_GRAPH.md)**: Concepts, relations, logical constraints
- **[DataNode Components](graph/README.md)**: Runtime data structures and queries
- **[Sensor Components](sensor/README.md)**: Data pipeline and feature computation
- **[Solver Components](solver/README.md)**: Constraint satisfaction methods
- **[Program Components](program/README.md)**: Training and evaluation programs
- **[Model Components](program/README.md#model-components)**: Neural model architectures

### Directory Structure
```
domiknows/
├── graph/           # Knowledge representation
│   ├── graph.py           # Graph container
│   ├── concept.py         # Concept definitions
│   ├── relation.py        # Relation types
│   ├── logicalConstrain.py # Logic constraints
│   └── dataNode.py        # Runtime instances
├── sensor/          # Data pipeline
│   ├── sensor.py          # Base sensors
│   ├── learner.py         # Learnable sensors
│   └── pytorch/           # PyTorch implementations
├── solver/          # Constraint reasoning
│   ├── ilpOntSolver.py         # Base solver
│   ├── gurobiILPOntSolver.py   # ILP solver
│   └── lcLossBooleanMethods.py # Differentiable logic
├── program/         # Training interface
│   ├── program.py         # Base program
│   ├── model_program.py   # Standard programs
│   └── lossprogram.py     # Constraint learning
└── utils.py         # Utilities
```

---

## Example Applications

### Named Entity Recognition with Constraints
```python
# Define entities and relations
with Graph('ner') as graph:
    entity = Concept('entity')
    person = Concept('person')
    organization = Concept('organization')
    
    person.is_a(entity)
    organization.is_a(entity)
    
    # Constraint: entities are disjoint
    person.not_a(organization)
    
    # Train with constraint satisfaction
    program = PrimalDualProgram(graph)
    program.train(train_data)
```

### Relation Extraction
```python
# Define relation with domain/range constraints
with Graph('relation_extraction') as graph:
    person = Concept('person')
    org = Concept('organization')
    work_for = Concept('work_for')
    
    work_for.has_a(person, org)
    
    # Automatic domain/range constraints
    # ifL(work_for(x,y), andL(person(x), org(y)))
    
    program = SolverPOIProgram(graph, inferTypes=['ILP'])
    program.train(train_data)
```

### Hierarchical Classification
```python
# Multi-level text classification
with Graph('classification') as graph:
    document = Concept('document')
    paragraph = Concept('paragraph')
    sentence = Concept('sentence')
    
    document.contains(paragraph)
    paragraph.contains(sentence)
    
    # Aggregate with constraints
    program = POIProgram(graph)
    program.train(train_data)
```

---

## Advanced Topics

### Constraint Learning Strategies
- **Primal-Dual**: Balance data fit and constraint satisfaction
- **Inference-Masked**: Focus learning on constraint violations
- **Gumbel-Softmax**: Better discrete optimization
- **Sampling**: Scale to large constraint groundings

### T-norm Selection
- **Product**: Smooth gradients for learning
- **Łukasiewicz**: Efficient for counting constraints
- **Gödel**: Idempotent logic operations

### Performance Optimization
- **Model reuse**: Cache ILP models across batches
- **Skeleton mode**: Fast DataNode construction
- **Constraint sampling**: Approximate large groundings
- **Batch processing**: Efficient gradient accumulation

---

## Requirements

- **Python**: 3.7+
- **PyTorch**: 1.7+
- **NumPy**: For array operations
- **scikit-learn**: For metrics
- **Gurobi** (optional): For ILP inference (free academic license)
- **Graphviz** (optional): For visualization


## Support

- **Documentation**: See component READMEs above
- **Issues**: [GitHub Issues](https://github.com/HLR/DomiKnows/issues)
- **Discussions**: [GitHub Discussions](https://github.com/HLR/DomiKnows/discussions)

---

## Summary

DomiKnows provides a complete framework for neural-symbolic AI:

1. **Define** domain knowledge as a graph with logical constraints
2. **Connect** data through sensors and neural networks
3. **Train** with constraint-aware learning programs
4. **Infer** predictions that satisfy logical rules
5. **Verify** constraint satisfaction and model quality

Start with the [Quick Start](#quick-start) above, then explore the component READMEs for detailed documentation.