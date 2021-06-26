from regr.graph.concept import EnumConcept
from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import orL, notL, andL

Graph.clear()
Concept.clear()
Relation.clear()

with Graph(name='global') as graph:
    image = Concept(name='image')
    digit = image(name='digit', ConceptClass=EnumConcept, values=list(map(str, range(10))))
