from itertools import product
from regr.graph.concept import EnumConcept
from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import V, ifL, orL, notL, andL

Graph.clear()
Concept.clear()
Relation.clear()

with Graph(name='global') as graph:
    image = Concept(name='image')
    digit = image(name='digit', ConceptClass=EnumConcept, values=list(map(lambda v: f'_{v}', range(10))))

    addition = Concept(name='addition')
    (operand1, operand2) = addition.has_a(operand1=digit, operand2=digit)
    summation = addition(name='sumation', ConceptClass=EnumConcept, values=list(map(lambda v: f'_{v}', range(19))))

    for i, j in product(range(10), repeat=2):
        # rule: x is i and y is j => z is i+j
        # ifL(
        #     andL(
        #         getattr(digit, f'{i}'), V(name='x'),
        #         getattr(digit, f'{j}'), V(name='y', v=('x', operand1.reversed, operand2),
        #     ),
        #     getattr(sumation, f'{i+j}'), V(name='z', v=('x', operand1.reversed))
        # )

        # rule: x is i and z is i+j => y is j
        ifL(
            andL(
                getattr(digit, f'_{i}')('x'),
                getattr(summation, f'_{i+j}')('z', path=('x', operand1.reversed.name)),
            ),
            getattr(digit, f'_{j}')('y', path=('z', operand2.name))
        )
