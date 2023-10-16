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

with Graph(name='global') as graph:
    image_batch = Concept(name='image_batch')
    image = Concept(name='image')

    image_contains, = image_batch.contains(image)

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

    exactL(*[digit.__getattr__(d) for d in digits])
    exactL(*[s.__getattr__(d) for d in summations])

    #fixedL(s)
    FIXED = True
    fixedL(s("x", eqL(image_pair, "summationEquality", {True})), active = FIXED)

    for sum_val in range(config.summationRange):
        sum_combinations = []

        sum_nm = summations[sum_val]

        for d0_val in range(sum_val + 1):
            d1_val = sum_val - d0_val

            if d0_val >= len(digits) or d1_val >= len(digits):
                continue

            d0_nm = digits[d0_val]
            d1_nm = digits[d1_val]

            # for each combination of digits that sum to sum_val add constraint to list
            sum_combinations.append(andL(getattr(digit, d0_nm)(path=('x', pair_d0)),
                                         getattr(digit, d1_nm)(path=('x', pair_d1))
                                         ))

        print(sum_val, '-', sum_combinations)

        # if the given summation value is some value, then the digits must be one of a set of
        # digit pairs that add to that value
        # i.e. if sum val = s, d0 = 0 and d1 = s or d0 = 1 and d1 = s-1 ...
        # e.g. if sum val = 1, d0 = 0 and d1 = 1 or d0 = 1 and d0 = 0
        if len(sum_combinations) == 1:
            ifL(
                getattr(s, sum_nm)('x'),
                sum_combinations[0]
            )
        else:
            ifL(
                getattr(s, sum_nm)('x'),
                orL(*sum_combinations)
            )
