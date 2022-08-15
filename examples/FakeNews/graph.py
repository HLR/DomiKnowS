import sys
sys.path.append("../")
sys.path.append("../../")

from regr.graph import Graph, Concept, Relation
from regr.graph.concept import EnumConcept
from regr.graph.logicalConstrain import nandL, orL, ifL
from regr.graph.relation import disjoint
from itertools import combinations

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('FakeNews') as graph:
    TextSequence = Concept(name='TextSequence')

    Category = TextSequence(name="Category", ConceptClass=EnumConcept, values=["HasAnno"])

    ParentTag = Category(name='ParentTag', ConceptClass=EnumConcept,
                             values=["Anno1", "Anno2", "Anno3", "Anno4", "Anno5", "Anno6", 
                                     "Anno7", "Anno8", "Anno9", "Anno10", "Anno11", "Anno12"])

    # ChildTag = TextSequence(name='ChildTag', ConceptClass=EnumConcept,
    #                        values=["Anno1a", "Anno1b", "Anno1c", "Anno3a", "Anno4a", "Anno4b", 
    #                                "Anno4c", "Anno4d", "Anno7a", "Anno7b", "Anno7c", "Anno8a",
    #                                "Anno9a", "Anno9b", "Anno9c", "Anno9d", "Anno9e", "Anno10a", 
    #                                "Anno10b", "Anno10c", "Anno10d", "Anno11a", "Anno11b", "Anno11c", 
    #                                "Anno11d", "Anno11e", "Anno12a", "Anno12b", "Anno12c", "Anno12d"])

    # for label in ParentTag.attributes:
    #     nandL(label, category.NoAnno, active=True)

    # disjoint(category.NoAnno, category.HasAnno)

    # nandL("NoAnno", "HasAnno")

    # ifL(orL(ChildTag.Anno1a, ChildTag.Anno1b, ChildTag.Anno1c), ParentTag.Anno1, active=True)
    # ifL(ChildTag.Anno3a, ParentTag.Anno3, active=True)
    # ifL(orL(ChildTag.Anno4a, ChildTag.Anno4b, ChildTag.Anno4c, ChildTag.Anno4d), ParentTag.Anno4, active=True)
    # ifL(orL(ChildTag.Anno7a, ChildTag.Anno7b, ChildTag.Anno7c), ParentTag.Anno7, active=True)
    # ifL(ChildTag.Anno8a, ParentTag.Anno8, active=True)
    # ifL(orL(ChildTag.Anno9a, ChildTag.Anno9b, ChildTag.Anno9c, ChildTag.Anno9d, ChildTag.Anno9e), ParentTag.Anno9, active=True)
    # ifL(orL(ChildTag.Anno10a, ChildTag.Anno10b, ChildTag.Anno10c, ChildTag.Anno10d), ParentTag.Anno10, active=True)
    # ifL(orL(ChildTag.Anno11a, ChildTag.Anno11b, ChildTag.Anno11c, ChildTag.Anno11d, ChildTag.Anno11e), ParentTag.Anno11, active=True)
    # ifL(orL(ChildTag.Anno12a, ChildTag.Anno12b, ChildTag.Anno12c, ChildTag.Anno12d), ParentTag.Anno12, active=True)

