import sys
sys.path.append("../..")

from itertools import product
from regr.graph import Graph, Concept, Relation
from regr.graph.concept import EnumConcept
from regr.graph.logicalConstrain import orL, andL, existsL, notL, atLeastL, atMostL, ifL, V

Graph.clear()
Concept.clear()
Relation.clear()

digit_names =[
        "zero","one","two","three","four","five","six","seven","eight","nine"]

with Graph('global') as graph:
    # Single digits
    digit = Concept(name='digit')
    digit_labels = digit(name='tag', ConceptClass=EnumConcept, values=digit_names)

    summation = Concept(name='summation')
    summation.has_a(arg1=digit, arg2=digit)

    summation_labels = {}
    for i, j in product(range(10), range(10)):
        ifL(andL((digit_labels, i), V("x"), (digit_labels, j), V("y")), summation)
        digit_labels

        summation_labels[i] = summation_labels(name=f"summation_{i}")

    