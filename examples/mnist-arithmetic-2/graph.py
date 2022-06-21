from regr.graph.concept import EnumConcept
from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import ifL, nandL, orL, notL, andL, atMostL, exactL, fixedL, eqL
from regr.graph.relation import disjoint
import config
from itertools import product

Graph.clear()
Concept.clear()
Relation.clear()

digits_0 = []
digits_1 = []
summations = []

for digit_val in range(config.digitRange):
    digits_0.append(f'd0_{digit_val}')

for digit_val in range(config.digitRange):
    digits_1.append(f'd1_{digit_val}')

for sum_val in range(config.summationRange):
    summations.append(f's_{sum_val}')

numbers = digits_0 + digits_1 + summations

with Graph(name='global') as graph:
    images = Concept(name='images')

    d0 = images(name='digits0',
                ConceptClass=EnumConcept,
                values=digits_0)

    d1 = images(name='digits1',
               ConceptClass=EnumConcept,
               values=digits_1)

    s = images(name='summations',
                ConceptClass=EnumConcept,
                values=summations)

    #exactL(*d0.attributes)
    #exactL(*d1.attributes)
    #exactL(*s.attributes)

    #fixedL(s)
    FIXED = True
    fixedL(s("x", eqL(images, "summationEquality", {True})), active = FIXED)

    for sum_val in range(config.summationRange):
        sum_combinations = []

        sum_nm = summations[sum_val]

        for d0_val in range(sum_val + 1):
            d1_val = sum_val - d0_val

            if d0_val >= len(digits_0) or d1_val >= len(digits_1):
                continue

            d0_nm = digits_0[d0_val]
            d1_nm = digits_1[d1_val]

            sum_combinations.append(andL(getattr(d0, d0_nm)(), getattr(d1, d1_nm)()))

        print(sum_val, '-', sum_combinations)

        if len(sum_combinations) == 1:
            ifL(
                getattr(s, sum_nm)(),
                sum_combinations[0]
            )
        else:
            ifL(
                getattr(s, sum_nm)(),
                orL(*sum_combinations)
            )
