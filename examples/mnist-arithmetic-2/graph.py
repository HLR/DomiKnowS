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

    exactL(*d0.attributes)
    exactL(*d1.attributes)
    exactL(*s.attributes)

    for d0_nm in digits_0:
        for d1_nm in digits_1:
            d0_val = int(d0_nm.split('_')[-1])
            d1_val = int(d1_nm.split('_')[-1])

            sum_val = d0_val + d1_val
            sum_nm = f's_{sum_val}'

            #print(d0_nm, d1_nm, sum_nm)

            ifL(
                getattr(d0, d0_nm)(),
                ifL(
                    getattr(d1, d1_nm)(),
                    getattr(s, sum_nm)()
                ),
                active=True
            )
