from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import orL, andL, existsL, notL, atLeastL, atMostL, ifL, nandL, V
from regr.graph.concept import EnumConcept

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
    
    action_label = (name="action_label", ConceptClass=EnumConcept, values=["nochange", "destroy", "create", "move"])
#     create = action(name="create")
#     destroy = action(name="destroy")
#     move = action(name="move")
#     nochange = action('none')
    
#     Tcreate = action(name="trips_create")
#     Tdestroy = action(name="trips_destroy")
#     Tmove = action(name="trips_move")
#     Tnochange = action('trips_none')
    trips_action_label = (name="trips_action_label", ConceptClass=EnumConcept, values=["trips_none", "trips_destroy", "trips_create", "trips_move"])
    
#     exactL(Tcreate, Tdestroy, Tmove, Tnochange)
#     exactL(create, destroy, move, nochange)
    
    ifL(trips_action_label.trips_create, action_label.create)
    ifL(trips_action_label.trips_destroy, action_label.destroy)
    ifL(trips_action_label.trips_move, action_label.move)
    
    # x is destroy at step i and entity e then and (step i, (step j before step i), not(destroy ))
#     ifL(destroy('x'), andL(step('i', path=('x', arg2)), step('j', path=('i', before1)), notL(destroy('y', path=(’j’, action_step.reversed))), step('k', path=('i', before2)), notL(create('y', path=('j', inverse-arg2)))))
    
    
    ### Rules we need
    
    ## if x is destroyed at step i, it should exists at step i-1
    
    ## if x is moved at step i, it should exists at step i-1 
    
    ## if x is created at step i, it shouldnt exists at step i-1 --> Shouldn't exist means there is no move before that.


