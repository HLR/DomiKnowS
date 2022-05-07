from regr.graph.concept import EnumConcept
from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import ifL, nandL, orL, notL, andL, atMostL, exactL
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

def name_to_number(name):
    return int(name.split('_')[-1])

with Graph(name='global') as graph:
    images = Concept(name='images')

    def make(name):
        return images(name=name)

    digits_0_c = list(map(make, digits_0))
    digits_1_c = list(map(make, digits_1))
    summations_c = list(map(make, summations))

    exactL(*digits_0_c)
    exactL(*digits_1_c)
    exactL(*summations_c)

    exactL(*d0.attributes)
    exactL(*d1.attributes)
    exactL(*s.attributes)

    for d0_nm in digits_0:
        for d1_nm in digits_1:
            d0_val = int(d0_nm.split('_')[-1])
            d1_val = int(d1_nm.split('_')[-1])

    for d0_nm, d0_c in zip(digits_0, digits_0_c):
        for d1_nm, d1_c in zip(digits_1, digits_1_c):
            d0_number = name_to_number(d0_nm)
            d1_number = name_to_number(d1_nm)

            sum_val = d0_number + d1_number

            ifL(
                d0_c,
                ifL(
                    d1_c,
                    summations_c[sum_val]
                ),
                active=True
            )
