from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import orL, andL, existsL, notL, atLeastL, atMostL, ifL, nandL, V, exactL
from regr.graph import EnumConcept


Graph.clear()
Concept.clear()
Relation.clear()

with Graph('global') as graph:
    procedure = Concept(name="procedure")
    
    context = Concept(name="context")
    entities = Concept(name="entities")
    locations = Concept(name='locations')
    
    (procedure_context, procedure_entities, procedure_locations) = procedure.has_a(context, entities, locations)
    
    entity = Concept(name='entity')
    (entity_rel, ) = entities.contains(entity)
    
    step = Concept(name="step")
    (context_step, ) = context.contains(step)
    
    location = Concept(name="location")
    (loc_rel, ) = locations.contains(location)
    
    before = Concept(name="before")
    (before_arg1, before_arg2) = before.has_a(arg1=step, arg2=step)
    
    exact_before = Concept(name="exact_before")
    (ebefore_arg1, ebefore_arg2) = exact_before.has_a(earg1=step, earg2=step)
    
    action = Concept(name='action')
    (action_step, action_entity) = action.has_a(step=step, entity=entity)
    
    same_mention = Concept(name='same_mention')
    (same_entity, same_location) = same_mention.has_a(se1=entity, se2=location)
    
    entity_location = Concept(name='entity_location')
    (lentity, lstep, llocation) = entity_location.has_a(lentity=entity, lstep=step, llocation=location)
    
    entity_location_label = entity_location(name='entity_location_label')
    
    action_label = action(name="action_label", ConceptClass=EnumConcept, values=["nochange", "move", "create", "destroy"])
    
    #  ------------ Destroy

    ifL(
        # action a1 is destroy, i is a1's step and e is action entity
        andL(
            action_label.destroy('a1'), 
            step('i', path=('a1', action_step)),
            entity('e', path=('a1', action_entity))
            ), 
        # then either
        orL(
            # step j associated with entity e, which is before step i cannot be associated with destroy action a2
            ifL(
                step('j', path=(('i', before_arg2.reversed, before_arg1))), 
                notL(action_label.destroy('a2', path=(('j', action_step.reversed), ('e', action_entity.reversed))))
                ), 
            # or if  
            ifL(
                # step j1 which is before step i is associated with destroy action a2
                andL(
                    step('j1', path=('i', before_arg2.reversed, before_arg1)), 
                    action_label.destroy('a3', path=(('j1', action_step.reversed), ('e', action_entity.reversed)))
                    ), 
                # then exists step k associated with entity e, which is between step i and j1 associated with create action a3
                existsL(
                    andL(
                        step('k', path=(('j1', before_arg1.reversed, before_arg2), ('i', before_arg2.reversed, before_arg1))), 
                        action_label.create('a4', path=(('k', action_step.reversed), ('e', action_entity.reversed)))
                        )
                    )
                )
            ), active = True
        ) 

    #  ------------ Create
    
    ifL(
        # action a1 is create, i is a1's step and e is action entity
        andL(
            action_label.create('a1'), 
            step('i', path=('a1', action_step)),
            entity('e', path=('a1', action_entity))
            ), 
        # then either
        orL(
            # step j associated with entity e, which is before step i cannot be associated with create action a2
            ifL(
                step('j', path=(('i', before_arg2.reversed, before_arg1))), 
                notL(action_label.create('a2', path=(('j', action_step.reversed), ('e', action_entity.reversed))))
                ), 
            # or if  
            ifL(
                # step j1 which is before step i is associated with create action a2
                andL(
                    step('j1', path=('i', before_arg2.reversed, before_arg1)), 
                    action_label.create('a2', path=(('j1', action_step.reversed), ('e', action_entity.reversed)))
                    ), 
                # then exists step k associated with entity e, which is between step i and j1 associated with destroy action a3
                existsL(
                    andL(
                        step('k', path=(('j1', before_arg1.reversed, before_arg2), ('i', before_arg2.reversed, before_arg1))), 
                        action_label.destroy('a3', path=(('k', action_step.reversed), ('e', action_entity.reversed)))
                        )
                    )
                )
            ), active = True
        )
    
    #  ------------ Move
    
    ifL(
        # action a1 is move, i is a1's step and e is action entity
        andL(
            action_label.move('a1'), 
            step('i', path=('a1', action_step)),
            entity('e', path=('a1', action_entity))
            ), 
        # then either
        orL(
            # step j associated with entity e, which is before step i cannot be associated with destroy action a2
            step('j', path=(('i', before_arg2.reversed, before_arg1))), 
            notL(action_label.destroy('a2', path=(('j', action_step.reversed), ('e', action_entity.reversed)))), 
            # or if  
            ifL(
                # step j1 which is before step i is associated with destroy action a2
                andL(
                    step('j1', path=('i', before_arg2.reversed, before_arg1)), 
                    action_label.destroy('a3', path=(('j1', action_step.reversed), ('e', action_entity.reversed)))
                    ), 
                # then exists step k associated with entity e, which is between step i and j1 associated with create action a3
                existsL(
                    andL(
                        step('k', path=(('j1', before_arg1.reversed, before_arg2), ('i', before_arg2.reversed, before_arg1))), 
                        action_label.create('a4', path=(('k', action_step.reversed), ('e', action_entity.reversed)))
                        )
                    )
                )
            ), active = True
        )