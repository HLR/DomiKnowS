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
    def get_concept(attribute):
        return attribute[0]


    TextSequence = Concept(name='TextSequence')

    Category = TextSequence(name="Category", ConceptClass=EnumConcept, values=["NoAnno", "HasAnno"])

    categories = {"1": ["a", "b", "c"], "2": [], "3": ["a"], "4": ["a", "b", "c", "d"], "5": [], "6": [],
                  "7": ["a", "b", "c"], "8": ["a"], "9": ["a", "b", "c", "d", "e"], "10": ["a", "b", "c", "d"],
                  "11": ["a", "b", "c", "d", "e"], "12": ["a", "b", "c", "d"]}
    ParentCategory = Category(name='ParentCategory', ConceptClass=EnumConcept,
                              values=list(categories.keys()))

    for l1, l2 in combinations(Category.attributes, 2):
        nandL(l1, l2)

    for l1, l2 in combinations(ParentCategory.attributes, 2):
        nandL(l1, l2)

    ### This is the problem
    if True:
        SubCategory = Category(name='SubCategory', ConceptClass=EnumConcept,
                           values=['1a', '1b', '1c', '3a', '4a', '4b', '4c', '4d', '7a', '7b', '7c', '8a', '9a', '9b',
                                   '9c', '9d', '9e', '10a', '10b', '10c', '10d', '11a', '11b', '11c', '11d', '11e',
                                   '12a', '12b', '12c', '12d'])
    if False:
        SubCategory = EnumConcept(name='SubCategory',
                              values=['1a', '1b', '1c', '3a', '4a', '4b', '4c', '4d', '7a', '7b', '7c', '8a', '9a',
                                      '9b',
                                      '9c', '9d', '9e', '10a', '10b', '10c', '10d', '11a', '11b', '11c', '11d', '11e',
                                      '12a', '12b', '12c', '12d'])

    for l1, l2 in combinations(SubCategory.attributes, 2):
        nandL(l1, l2)

    for category, sub_categories_suffixes in categories.items():
        for sub_categories_suffix in sub_categories_suffixes:
            subCategory = category + sub_categories_suffix
            ifL(getattr(SubCategory, subCategory), getattr(ParentCategory, category))
