from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import orL, andL, atMostL, ifL

Graph.clear()
Concept.clear()
Relation.clear()

with Graph(name='global') as graph:
    procedure = Concept(name="procedure")
    step = Concept(name="step")
    (procedure_contain_step, ) = procedure.contains(step)
#     entity = Concept(name="entity")
    non_existence = step(name="non_existence")
    unknown_loc = step(name="unknown_location")
    known_loc = step(name="known_location")
    before = Concept(name="before")
    (before_arg1, before_arg2) = before.has_a(arg1=step, arg2=step)
    action = Concept(name="action")
    (action_arg1, action_arg2) = action.has_a(arg1=step, arg2=step)
    create = action(name="create")
    destroy = action(name="destroy")
    other = action(name="other")

    atMostL(create, destroy, other)
    atMostL(known_loc, unknown_loc, non_existence)
    
    ifL(
        create("x"), 
        andL(
            non_existence(path=("x", "arg1")), 
            orL(
                known_loc(path=("x", "arg2")), 
                unknown_loc(path=("x", "arg2")))
            )
        )
    
    ifL(destroy("x")), andL(orL(known_loc(path=("x", "arg1")), unknown_loc(path=("x", "arg1"))), non_existence(path=("x", "arg2")))

#   atMostL(andL(entity, create))
    # No entity_step


