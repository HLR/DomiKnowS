# NER with Relation Extraction Test

## Overview
This test validates a Named Entity Recognition (NER) and Relation Extraction system using the DomiKnows framework with logical constraints.

## Test Scenario
- **Input**: "John works for IBM"
- **Entities**: 3 phrases - `['John', 'works for', 'IBM']`
- **Entity Types**: people, organization, location, other, O
- **Relations**: work_for, located_in, live_in, orgbase_on, kill

## Key Components

### graph.py
Defines the ontology structure:
- **Linguistic concepts**: word, sentence, phrase, pair
- **Application concepts**: Entity types (people, organization, location, other)
- **Logical constraints**:
  - LC0: `nandL(people, organization)` - entities cannot be both person and organization
  - LC1: work_for relation requires (people, organization) arguments
  - LC2: Each sentence must contain at least one person phrase
  - LC3: Phrase boundaries validation using B/I/E/O tags

### config.py
Model configuration using PoiModel with all concepts and relations.

### test_main.py
Test fixture creating:
- 4 words with embeddings (2048-dim)
- 3 phrases combining words
- 9 pairs (all phrase combinations)
- Ground truth labels and predictions

## Running Tests

```bash
pytest test_main.py -v
```

For Gurobi-dependent tests:
```bash
pytest test_main.py -v -m gurobi
```

## Expected Results
- 4 words extracted from sentence
- 3 phrases identified
- Logical constraint losses validated against expected values
- Entity and relation predictions match ground truth structure