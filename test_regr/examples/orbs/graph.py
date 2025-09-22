from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import orL, existsL, ifL, notL, andL, atMostL, atLeastL,atMostAL,atLeastAL, atMostAL, exactL

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
            atMostAL(colored_orbs,atmost)
            atLeastAL(colored_orbs,atleast)
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
        elif constraint=="foreach_IfL_atleastL_bag_existsL_notL": # try with --colored
            ifL(csp_range("x"),
                ifL(atLeastL(colored_orbs("y",path=("x",csp_range_contains_orbs)),atleast),
                    existsL(notL(colored_orbs("z",path=("x",csp_range_contains_orbs))))
                    )
            )
        elif constraint=="foreach_IfL_atleastL_bag_existsL":
            ifL(csp_range("x"),
                ifL(atLeastL(colored_orbs("y",path=("x",csp_range_contains_orbs)),atleast),
                    existsL(colored_orbs("z",path=("x",csp_range_contains_orbs)))
                    )
            )
        elif constraint=="ifL_atMostL_feasible":
            print(constraint)
            # 1) there must be at least 5 colored orbs total
            # 2) if there is a bag with <= 2 colored orbs, then that bag must have >= 4 colored orbs
            # This is feasible.

            atLeastAL(colored_orbs, 5)

            ifL(csp_range("x"),
                ifL(     
                    atMostL(colored_orbs("y",path=("x",csp_range_contains_orbs)), 2),
                    atLeastL(colored_orbs("z",path=("x",csp_range_contains_orbs)), 4)
                )
            )
            
            
        elif constraint=="ifL_atLeastL_feasible":
            print(constraint)
            # 1) there must be at least 5 colored orbs total
            # 2) if there is a bag with <= 2 colored orbs, then that bag must have >= 4 colored orbs
            # This is feasible.

            atLeastAL(colored_orbs, 5)

            ifL(csp_range("x"),
                ifL(     
                    atLeastL(colored_orbs("z",path=("x",csp_range_contains_orbs)), 4),
                    atMostL(colored_orbs("y",path=("x",csp_range_contains_orbs)), 2)
                )
            )
            
        elif constraint=="ifL_atLeastL_infeasible":
            print(constraint)
            # 1) there must be at least 16 colored orbs total
            # 2) if there is a bag with <= 2 colored orbs, then that bag must have >= 4 colored orbs
            # This is infeasible.

            atLeastAL(colored_orbs, 16)

            ifL(csp_range("x"),
                ifL(     
                    atLeastL(colored_orbs("z",path=("x",csp_range_contains_orbs)), 4),
                    atMostL(colored_orbs("y",path=("x",csp_range_contains_orbs)), 2)
                )
            )
            
        elif constraint=="ifL_atMostL_atLeastL_infeasible":
            print(constraint)
            # there must be at least 5 colored orbs total
            # if there is a bag then this bag has <= 2 colored orbs
            # if there is a bag then this bag has >= 3 colored orbs
            
            # This should be infeasible.
            
            atLeastAL(colored_orbs, 5)

            ifL(csp_range("x"),
                atMostL(colored_orbs("y",path=("x",csp_range_contains_orbs)), 2),
            )
            
            ifL(csp_range("x"),
                atLeastL(colored_orbs("y",path=("x",csp_range_contains_orbs)), 3),
            )
            
        elif constraint=="ifL_atMostAL_atLeastAL":
            print(constraint)
            # 1) there must be at least 5 colored orbs total
            # 2) there should be not more then 4 colored orbs
            # This should be infeasible.

            atLeastAL(colored_orbs, 5)

        elif constraint=="exactL":
            print(constraint)
            # 1) there is exactly 2 orbs in each bag
           

            ifL(csp_range("x"),
                exactL(colored_orbs("y",path=("x",csp_range_contains_orbs)), 2),
            )
            
        else:
            print("no constraint")
    
    return graph, csp,csp_range,orbs,csp_contains_csp_range,csp_range_contains_orbs,colored_orbs,enforce_csp_range

