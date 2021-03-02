from regr.graph import Graph, Concept, Relation
from regr.graph.concept import EnumConcept
from regr.graph.logicalConstrain import nandL, orL, andL, existsL, notL, atLeastL, atMostL, ifL
from regr.graph.relation import disjoint

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('CIFAR10') as graph:
    Image = Concept(name='image')
    Category = Image(name="catgeory", ConceptClass=EnumConcept, values=["animal", "vehicle"])
    Label = Image(name="label", ConceptClass=EnumConcept, values=["airplane", "dog", "truck", "automobile", "bird", "cat", "deer", "frog", "horse", "ship"])

    # Add a constraint here later
    # ifL((Category, "animal"), v('x'), notL((Label, "ship"), v('x')))
    ifL((Category, "animal"), notL((Label, "airplane")))
    ifL((Category, "animal"), notL((Label, "truck")))
    ifL((Category, "animal"), notL((Label, "automobile")))
    ifL((Category, "animal"), notL((Label, "ship")))


    ifL((Category, "vehicle"), notL((Label, "dog")))
    ifL((Category, "vehicle"), notL((Label, "bird")))
    ifL((Category, "vehicle"), notL((Label, "cat")))
    ifL((Category, "vehicle"), notL((Label, "deer")))
    ifL((Category, "vehicle"), notL((Label, "frog")))
    ifL((Category, "vehicle"), notL((Label, "horse")))





