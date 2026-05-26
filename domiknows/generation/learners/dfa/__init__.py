"""DFA and product-automata core utilities."""

from .core import DFA, State, Symbol, complement_dfa, product_dfa, union_dfa

__all__ = [
    "DFA",
    "State",
    "Symbol",
    "complement_dfa",
    "product_dfa",
    "union_dfa",
]
