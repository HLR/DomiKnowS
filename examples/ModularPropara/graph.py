from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import orL, andL, existsL, notL, atLeastL, atMostL, ifL, nandL, V

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('global') as graph:
    procedure = Concept("procedure")
    
    context = Concept("context")
    entities = Concept("entities")
    
    (procedure_context, procedure_entities) = procedure.has_a(context, entities)
    
    entity = Concept('entity')
    (entity_rel, ) = entities.contains(entity)
    
    step = Concept("step")
    (procedure_contain_step, ) = procedure.contains(step)
    before = Concept("before")
    (before_arg1, before_arg2) = before.has_a(arg1=step, arg2=step)
    
    action = Concept(name='action')
    (action_step, action_entity) = action.has_a(step, entity)
    
    create = action(name="create")
    destroy = action(name="destroy")
    move = action(name="move")
    nochange = action('none')
    
    Tcreate = action(name="trips_create")
    Tdestroy = action(name="trips_destroy")
    Tmove = action(name="trips_move")
    Tnochange = action('trips_none')

    exactL(Tcreate, Tdestroy, Tmove, Tnochange)
    exactL(create, destroy, move, nochange)
    
    ifL(Tcreate, create)
    ifL(Tdestroy, destroy)
    ifL(Tmove, move)
    ifL(Tnochange, nochange)
    
#     ifL(
#         destroy('x'), 
#         notL(
#             andL(
#                 existsL(
#                     andL(destroy('y'), existsL(before('y', path=('x', arg1, 'before', arg1)))),
#                     notL(
#                         existsL(
#                             andL(
#                                 create('z'), 
#                                 eqL(before('z'))
#                             )
#                         )
#                     )
#                 )
#             )
#         )
#     )
#   atMostL(1, ("x"), andL(entity, "y", create()))
    # No entity_step


