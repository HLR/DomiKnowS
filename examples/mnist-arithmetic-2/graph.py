from regr.graph.concept import EnumConcept
from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import ifL, nandL, orL, notL, andL, atMostL
import config

Graph.clear()
Concept.clear()
Relation.clear()

with Graph(name='global') as graph:
    image = Concept(name='image')
    
    digit = image(name='digit', ConceptClass=EnumConcept, values=list(map(lambda v: f'd_{v}', range(config.digitRange))))
    
    addition = Concept(name='addition')
    (operand1, operand2) = addition.has_a(operand1=image, operand2=image)
    
    summation = addition(name='summation', ConceptClass=EnumConcept, values=list(map(lambda v: f's_{v}', range(config.summationRange))))

    ifL(image,  atMostL(*digit.attributes))
    ifL(addition, atMostL(*summation.attributes))
    
    for i in range(config.digitRange):
        for j in range(i, config.digitRange):
            sumVal = i + j
            
            ifL(
                getattr(digit, f'd_{i}')('i'),
                ifL(
                    getattr(digit, f'd_{j}')('j', path=('i', operand1.reversed, operand2)),
                    getattr(summation, f's_{i+j}')('a', path=('j', operand2.reversed))
                ),
                active = True
            )