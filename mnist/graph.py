from domiknows.graph.concept import EnumConcept
from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import orL, notL, andL

Graph.clear()
Concept.clear()
Relation.clear()

with Graph(name='global') as graph:
    image = Concept(name='image')
    digit = image(name='digit', ConceptClass=EnumConcept, values=list(map(str, range(10))))
