'''
    Current setting: Match predicate to color word
    
    EX: re1 -> red
    
    Concepts: predicate and word
'''


# import sys
# sys.path.append('../..')
# print("sys.path - %s"%(sys.path))

from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import ifL, andL, notL, nandL, V, orL, exactL
from regr.graph.relation import disjoint
from regr.graph.concept import EnumConcept
from itertools import combinations

Graph.clear()
Concept.clear()
Relation.clear()

# the list of predicates
P = [x for x in open("../data/vocabulary.txt") if x not in ['the','of','to']]

with Graph('global') as graph:
    
    predicate = Concept(name='predicate')
    
    category = predicate(name='category', ConceptClass=EnumConcept, values=['color','shape','size','position'])
    word = predicate(name='word', ConceptClass=EnumConcept, values=P)
    
    # Set the constraints about the predicates and its corresponding label
    
    # Separate the size
    nandL(category.size('x'), word.red)
    nandL(category.size('x'), word.blue)
    nandL(category.size('x'), word.yellow)
    nandL(category.size('x'), word.orange)
    nandL(category.size('x'), word.purple)
    nandL(category.size('x'), word.green)
    
    nandL(category.size('x'), word.square)
    nandL(category.size('x'), word.circle)
    nandL(category.size('x'), word.ellipse)
    nandL(category.size('x'), word.star)
    nandL(category.size('x'), word.hexagon)
    nandL(category.size('x'), word.triangle)
    
    nandL(category.size('x'), word.left)
    nandL(category.size('x'), word.right)
    nandL(category.size('x'), word.above)
    nandL(category.size('x'), word.below)
    
    
    # Separate the shapes
    nandL(category.shape('x'), word.red)
    nandL(category.shape('x'), word.blue)
    nandL(category.shape('x'), word.yellow)
    nandL(category.shape('x'), word.orange)
    nandL(category.shape('x'), word.purple)
    nandL(category.shape('x'), word.green)
    
    nandL(category.shape('x'), word.small)
    nandL(category.shape('x'), word.medium)
    nandL(category.shape('x'), word.big)
    
    nandL(category.shape('x'), word.left)
    nandL(category.shape('x'), word.right)
    nandL(category.shape('x'), word.above)
    nandL(category.shape('x'), word.below)
    
    # Separate the colors
    nandL(category.color('x'), word.square)
    nandL(category.color('x'), word.circle)
    nandL(category.color('x'), word.ellipse)
    nandL(category.color('x'), word.star)
    nandL(category.color('x'), word.hexagon)
    nandL(category.color('x'), word.triangle)
    
    nandL(category.color('x'), word.small)
    nandL(category.color('x'), word.medium)
    nandL(category.color('x'), word.big)
    
    nandL(category.color('x'), word.left)
    nandL(category.color('x'), word.right)
    nandL(category.color('x'), word.above)
    nandL(category.color('x'), word.below)
    
    # Separate the position
    nandL(category.position('x'), word.square)
    nandL(category.position('x'), word.circle)
    nandL(category.position('x'), word.ellipse)
    nandL(category.position('x'), word.star)
    nandL(category.position('x'), word.hexagon)
    nandL(category.position('x'), word.triangle)
    
    nandL(category.position('x'), word.red)
    nandL(category.position('x'), word.blue)
    nandL(category.position('x'), word.yellow)
    nandL(category.position('x'), word.orange)
    nandL(category.position('x'), word.purple)
    nandL(category.position('x'), word.green)
    
    nandL(category.position('x'), word.small)
    nandL(category.position('x'), word.medium)
    nandL(category.position('x'), word.big)
    
    for l1, l2 in combinations(category.attributes, 2):
        nandL(l1, l2)

    for l1, l2 in combinations(word.attributes, 2):
        nandL(l1, l2)