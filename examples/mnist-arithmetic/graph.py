from itertools import combinations, product
from regr.graph.concept import EnumConcept
from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import ifL, nandL, orL, notL, andL, atMostL

Graph.clear()
Concept.clear()
Relation.clear()

with Graph(name='global') as graph:
    digitRange = 10
    summationRange = 2 * digitRange - 1
    
    T1 = Concept(name='t1')
    image = Concept(name='image')
    (tci,)= T1.contains(image)
    
    digit = image(name='digit', ConceptClass=EnumConcept, values=list(map(lambda v: f'd_{v}', range(digitRange))))

    addition = Concept(name='addition')
    (operand1, operand2) = addition.has_a(operand1=image, operand2=image)
    
    summation = addition(name='summation', ConceptClass=EnumConcept, values=list(map(lambda v: f's_{v}', range(summationRange))))

    atMostL(*digit.attributes)
    atMostL(*summation.attributes)
   
    for i, j in product(range(digitRange), repeat=2):
       
        ifL(
            getattr(digit, f'd_{i}')('i'),
            ifL(
                getattr(summation, f's_{i+j}')('a', path=('i', operand1.reversed)),
                getattr(digit, f'd_{j}')('j', path=('a', operand2)),
            ),
            active = True
        )
        
        n = (summationRange + i - j) % summationRange
        
        if n == i+j:
            continue
        
        ifL(
            getattr(digit, f'd_{i}')('i'),
            notL(getattr(summation, f's_{n}')('a', path=('i', operand1.reversed)))
        )