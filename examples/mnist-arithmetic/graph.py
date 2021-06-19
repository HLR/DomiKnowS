from itertools import combinations, product
from regr.graph.concept import EnumConcept
from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import V, ifL, nandL, orL, notL, andL

Graph.clear()
Concept.clear()
Relation.clear()

with Graph(name='global') as graph:
    T1 = Concept(name='t1')
    image = Concept(name='image')
    (tci,)= T1.contains(image)
    digit = image(name='digit', ConceptClass=EnumConcept, values=list(map(lambda v: f'_{v}', range(10))))

    addition = Concept(name='addition')
    (operand1, operand2) = addition.has_a(operand1=image, operand2=image)
    summation = addition(name='summation', ConceptClass=EnumConcept, values=list(map(lambda v: f'_{v}', range(19))))

    for d1, d2 in combinations(digit.attributes, 2):
        nandL(d1, d2)

    for s1, s2 in combinations(summation.attributes, 2):
        nandL(s1, s2)

    for i, j in product(range(10), repeat=2):
        # rule: x is i and y is j => z is i+j
        # ifL(
        #     andL(
        #         getattr(digit, f'{i}'), V(name='x'),
        #         getattr(digit, f'{j}'), V(name='y', v=('x', operand1.reversed, operand2),
        #     ),
        #     getattr(summation, f'{i+j}'), V(name='z', v=('x', operand1.reversed))
        # )

        # rule: x is i and z is i+j => y is j
        ifL(
            andL(
                image('i'),
                getattr(digit, f'_{i}'), V(name='x', v='i'),
                addition('a', v=('i', operand1.reversed)),
                getattr(summation, f'_{i+j}'), V(name='z', v='a'),
            ),
            andL(
                image('j', v=('a', operand2)),
                getattr(digit, f'_{j}'), V(name='y', v='j')
            )
        )
