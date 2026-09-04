# Edge Sensor Tests - README

## Overview
This directory contains tests for EdgeSensor functionality in DomiKnowS, demonstrating how to create relationships between concepts and process structured data using edge sensors.

## Files

### `graph.py`
Defines the knowledge graph structure with:
- **Concepts**: `sentence`, `word`, `word1`, `pair`
- **Relations**: 
  - `sentence_con_word`: sentence contains words
  - `word_equal_word1`: word equality relation
  - `pair_word1`, `pair_word2`: pair has two word arguments

### `test.ipynb`
Interactive notebook demonstrating:
- Custom EdgeSensor implementation (`SampleEdge`, `SpacyGloveRep`)
- Tokenization with BertTokenizerFast
- Word embeddings using Spacy's GloVe vectors
- CandidateEqualSensor for finding matching words
- CandidateRelationSensor for creating word pairs

### `test_main.py`
Basic edge sensor test with:
- Simple text splitting EdgeSensor
- JointSensor for multiple properties
- Data validation

### `test_main_torch.py`
PyTorch-based test featuring:
- Spacy GloVe embeddings as tensors
- GPU/CPU device handling
- Vector representations for words

## Key Concepts

**EdgeSensor**: Creates child nodes from parent nodes based on relations (e.g., extracting words from sentences)

**CandidateEqualSensor**: Finds matching entities across concepts

**CandidateRelationSensor**: Creates candidate pairs of related entities

## Running Tests

```bash
# Run specific test
pytest test_main.py

# Run PyTorch test
pytest test_main_torch.py

# Run with notebook
jupyter notebook test.ipynb
```

## Dependencies
- DomiKnowS framework
- PyTorch
- Transformers (BertTokenizer)
- Spacy with en_core_web_lg model
- pytest