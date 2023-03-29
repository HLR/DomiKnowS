'''
Task: MNIST Arithmetic

Train digit classifiers on pairs of digit images where the only source of supervision is the sum of the two digits.

A training data instance might include e.g., image of 1, image of 5, label = 6
'''
from domiknows.graph.concept import EnumConcept
from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import ifL, nandL, orL, notL, andL, atMostL, exactL, fixedL, eqL
from domiknows.graph.relation import disjoint
import config
from itertools import product

Graph.clear()
Concept.clear()
Relation.clear()
digits = []
summations = []

for digit_val in range(config.digitRange):
    digits.append(f'd0_{digit_val}')

for sum_val in range(config.summationRange):
    summations.append(f's_{sum_val}')

numbers = digits + summations

image = Concept(name='image')
with Graph(name='global') as graph:
    # digit classes 0-9
    digit = image(name='digits',
                ConceptClass=EnumConcept,
                values=digits)

    image_pair = Concept(name='pair')
    pair_d0, pair_d1 = image_pair.has_a(digit0=image, digit1=image)

    # sum value classes 0-18
    s = image_pair(name='summations',
                ConceptClass=EnumConcept,
                values=summations)

    exactL(*digit.attributes)
    exactL(*s.attributes)

    # fix summation value
    FIXED = True
    fixedL(s("x", eqL(image_pair, "summationEquality", {True})), active = FIXED)

    # for each possible summation value 0...18 (sum_val)
    for sum_val in range(0, 19):
        # stores every valid combination of digits that could sum to sum_val
        sum_combinations = set()

        # for each possible combinations that could sum to that value (d0_val + d1_val = sum_val)
        for d0_val in range(sum_val + 1):
            d1_val = sum_val - d0_val

            # skip if digit value is invalid
            if d0_val >= len(digits) or d1_val >= len(digits):
                continue

            # add digit pair to set of valid combinations
            sum_combinations.add(
                (getattr(digit, digits[d0_val]), getattr(digit, digits[d1_val]))
            )

        ifL(
            # if the given summation value is sum_val
            getattr(s, summations[sum_val])('x'),

            # pair_d0 and pair_d1 in image_pair has to match some pair of digits in the set of valid
            # digit combinations (sum_combinations)
            is_in(image_pair, sum_combinations)
        )
