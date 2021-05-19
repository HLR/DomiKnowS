from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import ifL, orL, andL, existsL, notL, exactL

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('global') as graph:
    
    context = Concept("context")
    entities = Concept("entities")
        
    procedure = Concept("procedure")
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
    
    exactL(create, destroy, move, nochange)

    # if 
    ifL(
        # action a1 is destroy, i is a1's step and e is action entity
        andL(
            destroy('a1'), 
            step('i', path=('a1', action_step)),
            entity('e', path=('a1', action_entity))
            ), 
        # then either
        orL(
            # step j associated with entity e, which is before step i cannot be associated with destroy action a2
            andL(
                step('j', path=(('e', action_entity.reversed, action_step), ('i', before_arg1))), 
                notL(destroy('a2', path=('j', action_step.reversed)))
                ), 
            # or if  
            ifL(
                # step j1 which is before step i is associated with destroy action a2
                andL(
                    step('j1', path=('i', before_arg1)), 
                    destroy('a2', path=('j', action_step.reversed))
                    ), 
                # then exists step k associated with entity e, which is between step i and j1 associated with create action a3
                andL(
                    step('k', path=(('e', action_entity.reversed, action_step), ('j1', before_arg2), ('i', before_arg1))), 
                    existsL(create('a3', path=('k', action_step.reversed)))
                    )
                )
            )
        ) 

    # ----------------------------
    
    Tcreate = action(name="trips_create")
    Tdestroy = action(name="trips_destroy")
    Tmove = action(name="trips_move")
    Tnochange = action('trips_none')
    
    #trips_action_label = (name="trips_action_label", ConceptClass=EnumConcept, values=["trips_none", "trips_destroy", "trips_create", "trips_move"])
    
    exactL(Tcreate, Tdestroy, Tmove, Tnochange)
    
    #ifL(trips_action_label.trips_create, action_label.create)
    #ifL(trips_action_label.trips_destroy, action_label.destroy)
    #ifL(trips_action_label.trips_move, action_label.move)
    
    # x is destroy at step i and entity e then and (step i, (step j before step i), not(destroy ))
#     ifL(destroy('x'), andL(step('i', path=('x', arg2)), step('j', path=('i', before1)), notL(destroy('y', path=(’j’, action_step.reversed))), step('k', path=('i', before2)), notL(create('y', path=('j', inverse-arg2)))))
    
    
    ### Rules we need
    
    ## if x is destroyed at step i, it should exists at step i-1
    
    ## if x is moved at step i, it should exists at step i-1 
    
    ## if x is created at step i, it shouldnt exists at step i-1 --> Shouldn't exist means there is no move before that.


