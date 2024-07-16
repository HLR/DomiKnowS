from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import orL, existsL, ifL, notL, andL, atMostL, atLeastL, atLeastAL, exactL

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
        elif constraint=="atleastAL_notatmostL":
            # 1) there must be at least 5 colored orbs total
            # 2) there can NOT be more than 1 colored orbs in each bag

            atLeastAL(colored_orbs, 5)

            ifL(csp_range("x"),
                notL(
                    atMostL(colored_orbs("y",path=("x",csp_range_contains_orbs)), 1)
                )
            )
        elif constraint=='atleastAL_notexistL':
            # 1) there must be at least 5 colored orbs total
            # 2) there can NOT be exactly 5 colored orbs in each bag

            atLeastAL(colored_orbs, 5)

            ifL(csp_range("x"),
                notL(
                    exactL(colored_orbs("y",path=("x",csp_range_contains_orbs)), 5)
                )
            )
        elif constraint=="ifL_atMostL_infeasible":
            print(constraint)
            # 1) there must be at least 5 colored orbs total
            # 2) if there is a bag with <= 2 colored orbs, then that bag must have >= 4 colored orbs
            # This should be infeasible.

            atLeastAL(colored_orbs, 5)

            ifL(csp_range("x"),
                ifL(
                    atMostL(colored_orbs("y",path=("x",csp_range_contains_orbs)), 2),
                    atLeastL(colored_orbs("z",path=("x",csp_range_contains_orbs)), 4)
                )
            )
        else:
            print("no contraint")
    
    return graph, csp,csp_range,orbs,csp_contains_csp_range,csp_range_contains_orbs,colored_orbs,enforce_csp_range

