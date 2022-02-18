from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import orL, andL, atMostL, ifL, nandL

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('global') as graph:
    procedure = Concept("procedure")
    step = Concept("step")
    (procedure_contain_step, ) = procedure.contains(step)
#     entity = Concept("entity")
    non_existence = step("non_existence")
    unknown_loc = step("unknown_location")
    known_loc = step("known_location")
    before = Concept("before")
    (before_arg1, before_arg2) = before.has_a(arg1=step, arg2=step)
    action = Concept("action")
    (action_arg1, action_arg2) = action.has_a(arg1=step, arg2=step)
    create = action(name="create")
    destroy = action(name="destroy")
    other = action(name="other")

    nandL(create, destroy, other)
    nandL(known_loc, unknown_loc, non_existence)
    
    ifL(create("x"), andL(non_existence(path=("x", "arg1")), orL(known_loc(path=("x", "arg2")), unknown_loc(path=("x", "arg2")))))
    ifL(destroy("x")), andL(orL(known_loc(path=("x", "arg1")), unknown_loc(path=("x", "arg1"))), non_existence(path=("x", "arg2")))

#   atMostL(andL(entity, create))
    # No entity_step


