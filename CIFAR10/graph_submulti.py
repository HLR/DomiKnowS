from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.concept import EnumConcept
from domiknows.graph.logicalConstrain import nandL, ifL
from itertools import combinations

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('CIFAR10') as graph:
    Image = Concept(name='image')
    Category = Image(name="category", ConceptClass=EnumConcept, values=["animal", "vehicle"])
    ALabel = Image(name="ALabel", ConceptClass=EnumConcept, values=["dog", "bird", "cat", "deer", "frog", "horse"])
    VLabel = Image(name="VLabel", ConceptClass=EnumConcept, values=["airplane", "truck", "automobile", "ship"])
    
    ifL(Category.animal, ALabel)
    ifL(Category.vehicle, VLabel)
    
    for l1, l2 in combinations(Category.attributes, 2):
        nandL(l1, l2)
        
    for l1, l2 in combinations(ALabel.attributes, 2):
        nandL(l1, l2)
        
    for l1, l2 in combinations(VLabel.attributes, 2):
        nandL(l1, l2)
