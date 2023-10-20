from itertools import combinations, product
from domiknows.graph.concept import EnumConcept
from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import ifL, nandL, orL, notL, andL, atMostL

Graph.clear()
Concept.clear()
Relation.clear()



with Graph(name='global', reuse_model=True) as graph:

    digitRange = 10
    summationVal = 9
    summationRange = 2 * digitRange - 1
    
    T1 = Concept(name='t1')
    image = Concept(name='image')
    (tci,)= T1.contains(image)
    
    digit = image(name='digit', ConceptClass=EnumConcept, values=list(map(lambda v: f'd_{v}', range(digitRange))))

    addition = Concept(name='addition')
    (operand1, operand2) = addition.has_a(operand1=image, operand2=image)
    
    summation = addition(name='summation', ConceptClass=EnumConcept, values=list(map(lambda v: f's_{v}', range(summationRange))))

    ifL(image,  atMostL(*digit.attributes))
    ifL(addition, atMostL(*summation.attributes))

    for i in range(digitRange):
        j = summationVal - i
        ifL(
            getattr(digit, f'd_{i}')('i'),
            getattr(digit, f'd_{j}')('j', path=('i', operand1.reversed, operand2)),
            active = True
        )
