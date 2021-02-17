from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import nandL, orL, andL, existsL, notL, atLeastL, atMostL, ifL
from regr.graph.relation import disjoint

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('CIFAR10') as graph:
    Image = Concept(name='image')
    Category = Image(name="catgeory", values=["animal", "thing", 'person'])
    Label = Image(name="label", values=["airplane", "dog", "truck", "automobile", "bird", "cat", "deer", "frog", "horse", "ship"])

    # Add a constraint here later
    # ifL((Category, "animal"), v('x'), notL((Label, "ship"), v('x')))
