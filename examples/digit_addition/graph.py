import sys
sys.path.append("../..")

from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import orL, andL, existsL, notL, atLeastL, atMostL, ifL, V

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('global') as graph:
    # Single digits
    digit = Concept(name='digit')

    digits = {}
    for i in range(10):
        digits[i] = digit(name=f"digit_{i}")

    disjoint(*list(digits.values()))

    # Number
    number = Concept(name='number')
    number.has_a(arg1=digit)

    # Summation
    summation = Concept(name='summation')
    summation.has_a(arg1=number, arg2=number)