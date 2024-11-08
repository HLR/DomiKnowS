from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import orL, existsL, ifL, notL, andL, atMostL, atLeastL

Graph.clear()
Concept.clear()
Relation.clear()

def get_graph(constraint,atmost,atleast):

    with Graph('global') as graph:

        csp = Concept(name='csp')
        csp_range = Concept(name='csp_range')
        enforce_csp_range = csp_range(name='enforce_csp_range')

        orbs = Concept(name='orbs')

        (csp_contains_csp_range,) = csp.contains(csp_range)
        (csp_range_contains_orbs,) = csp_range.contains(orbs)

        colored_orbs=orbs(name='colored_orbs')
        if constraint=="simple_constraint":
            atMostL(colored_orbs,atmost)
            atLeastL(colored_orbs,atleast)
        elif constraint=="foreach_bag_atLeastAL":
            ifL(csp_range("x"), 
                atLeastL(colored_orbs("y",path=("x",csp_range_contains_orbs)),atleast)
            )
        elif constraint=="foreach_bag_atMostAL":
            ifL(csp_range("x"), 
                atMostL(colored_orbs("y",path=("x",csp_range_contains_orbs)),atmost)
            )
        elif constraint=="foreach_bag_existsL":
            ifL(csp_range("x"), 
                existsL(colored_orbs("y",path=("x",csp_range_contains_orbs)))
            )
        elif constraint=="foreach_bag_existsL_notL": # try with --colored
            ifL(csp_range("x"), 
                existsL(notL(colored_orbs("y",path=("x",csp_range_contains_orbs))))
            )
        else:
            print("no contraint")
    
    return graph, csp,csp_range,orbs,csp_contains_csp_range,csp_range_contains_orbs,colored_orbs,enforce_csp_range

