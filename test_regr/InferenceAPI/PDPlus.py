import logging
from domiknows.graph import Graph, Concept, Relation, andL, orL

logging.basicConfig(level=logging.INFO)
Graph.clear()
Concept.clear()
Relation.clear()

with Graph(name='global') as graph:

    people = Concept(name='people')
    p1 = people(name='p1')
    p2 = people(name='p2')
    p3 = people(name='p3')
    p1_is_real = p1(name='p1_is_real')
    p2_is_real = p1(name='p2_is_real')
    p3_is_real = p1(name='p3_is_real')

    location = Concept(name='location')
    l1 = location(name='l1')
    l2 = location(name='l2')
    l3 = location(name='l3')

    pair = Concept(name="pair")
    people_arg,location_arg = pair.has_a(arg1=people, arg2=location)
    work_in = pair(name="work_in")

    andL(andL(p1_is_real("x"),work_in("z",path=('x', people_arg.reversed))),andL(p2_is_real("y"),work_in("t",path=('y', people_arg.reversed))))
    orL(andL(p2_is_real("x"),work_in("z",path=('x', people_arg.reversed))),andL(p3_is_real("y"),work_in("t",path=('y', people_arg.reversed))))

