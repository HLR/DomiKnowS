"""
Regression tests for domiknows.generation.graph_hmm.

This package contains comprehensive test suites for the graph_hmm module, which provides:
- DomiKnowSAwareHMM: Hidden Markov Models integrated with DomiKnows knowledge graphs
- GraphSpectralAutomaton: Spectral learning for automata with graph constraints
- Dynamic constraints: Context-aware constraint application during inference
- Graph adapters: Tools to convert DomiKnows graphs to HMM/automaton constraints

Test modules:
    test_constraints.py: Tests for constraint utility functions
    test_dynamic_constraints.py: Tests for dynamic constraints in HMM
    test_dynamic_spectral.py: Tests for dynamic constraints in spectral automata
    test_graph_adapter.py: Tests for DomiKnows graph to constraint conversion
    test_graph_hmm.py: Tests for HMM model training and inference
    test_spectral.py: Tests for spectral automaton learning
    test_torch_learners.py: Tests for PyTorch generation heads integration
"""
