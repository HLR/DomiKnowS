from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import orL, andL, existsL, existsAL, notL, atLeastL, atMostL, ifL, nandL, V, exactL, forAllL, eqL, atLeastAL, exactAL, atMostAL
from domiknows.graph import combinationC
from domiknows.graph import EnumConcept


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
    
    ### The action label
    action = Concept(name='action')
    (action_step, action_entity) = action.has_a(step=step, entity=entity)
    action_label = action(name="action_label", ConceptClass=EnumConcept, values=["create", "exists", "move", "destroy", "outside"])

    ### boolean action labels for create, destroy, and move
    action_create = action(name="action_create")
    action_destroy = action(name="action_destroy")
    action_move = action(name="action_move")

    ### boolean location change
    location_change = action(name="location_change")

    ### when_create and when_destroy actions
    when_create = action(name="when_create")
    when_destroy = action(name="when_destroy")

    ### before/after existence for entity per step, extend from action
    before_existence = action(name="before_existence")
    after_existence = action(name="after_existence")

    
    same_mention = Concept(name='same_mention')
    (same_entity, same_location) = same_mention.has_a(se1=entity, se2=location)
    
    ### entity, step, and after location
    entity_location = Concept(name='entity_location')
    (lentity, lstep, llocation) = entity_location.has_a(lentity=entity, lstep=step, llocation=location)
    entity_location_label = entity_location(name='entity_location_label')

    ### entity, step, and before location
    entity_location_before_label = entity_location(name='entity_location_before_label')

    ### input, output
    input_entity = entity(name='input_entity')
    output_entity = entity(name='output_entity')

    ### input, output alternatives
    input_entity_alt = entity(name='input_entity_alt')
    output_entity_alt = entity(name='output_entity_alt')
        
    # LC Active status
    All_LC = True
    Tested_Lc = True
    
    #  ------------ Destroy
    # No subsequent destroy action unless there is a create action between them
    
    ### the before action of step i+1 should match the after action of step i
    # ifL(
    #     andL(
    #         entity_location_label('x'),
    #         step('i', path=('x', lstep)),
    #         entity('e', path=('x', lentity))
    #     ),
    #     andL(
    #         step('j', path=('i', ebefore_arg1.reversed, ebefore_arg2)),
    #         entity_location_before_label('y', path=(('j', lstep.reversed), ('e', lentity.reversed), ('x', llocation, llocation.reversed)))
    #     )
    # )

    ### the alone candidate "3" is never the answer of a location
    forAllL(
        combinationC(entity, step, location(path=eqL(location, 'text', {"3"})))('e', 'i', 'l'),
        # combinationC(entity, location(path=eqL(location, 'text', {5839})))('e', 'l'),
        notL(
            entity_location_before_label('el1', path=(
                                ("e", lentity.reversed),
                                ("i", lstep.reversed),
                                ("l", llocation.reversed)
                )
            )
        ), active = Tested_Lc
    )

    ### if entity is not input, there should exists at least one create event associated with it
    ifL(
        notL(input_entity('e')),
        atLeastL(
            action_label.create('a', path=(('e', action_entity.reversed))), 1
        ), active = All_LC
    )

    ### if entity is input, the first state should not be `none`
    forAllL(
        combinationC(entity, location(path=eqL(location, 'text', {"5839", "150", "14794", "597", "1", "3"})))('e', 'l'),
        # combinationC(entity, location(path=eqL(location, 'text', {5839})))('e', 'l'),
        ifL(
            #input_entity(path=('e')),
            input_entity(path=('e',)),
            notL(
                entity_location_before_label('el1', path=(
                                    ("e", lentity.reversed),
                                    ("e", lentity.reversed, lstep, eqL(step, 'index', {0}), lstep.reversed),
                                    ("l", llocation.reversed)
                ))
            )
        ), active = All_LC
    )

    ### Prolog: input_entity(e) :- not(non-location(l) and step_index(i, 0) entity_location_before_label(e, i, l))


    forAllL(
        combinationC(entity, exact_before)('e', 'step_rel'),
        ifL(
            entity_location_label('x', path=(('e', lentity.reversed), ('step_rel', ebefore_arg1, lstep.reversed))),
            entity_location_before_label('y', path=(('e', lentity.reversed), ('step_rel', ebefore_arg2, lstep.reversed), ('x', llocation, llocation.reversed))),
        ), active = All_LC
    )

    #### Prolog: entity_location_label(e, i, l):- entity_location_before_label(e, i-1, l).

    forAllL(
        combinationC(entity, exact_before)('e', 'step_rel'),
        ifL(
            entity_location_before_label('x', path=(('e', lentity.reversed), ('step_rel', ebefore_arg2, lstep.reversed))),
            entity_location_label('y', path=(('e', lentity.reversed), ('step_rel', ebefore_arg1, lstep.reversed), ('x', llocation, llocation.reversed))),
        ), active = All_LC
    )

    

    ### if entity is input, and there is an action create for it, it should have been destroyed before that
    ifL(
        input_entity('e'),
        ifL(
            action_label.create('a', path=('e', action_entity.reversed)),
            existsL(
                andL(
                    step('k', path=(('a', action_step, before_arg2.reversed, before_arg1))), 
                    action_label.destroy('a64', path=(('k', action_step.reversed), ('e', action_entity.reversed)))
                )
            )
        ), active = All_LC
    )
    
    ### 

    # ifL(
    #     input_entity(path=('e',)),
    #     notL(
    #         entity_location_before_label('el1', path=(
    #                             ("e", lentity.reversed),
    #                             ("e", lentity.reversed, lstep, eqL(step, 'index', {0}), lstep.reversed),
    #                             ("e", lentity.reversed, llocation, eqL(location, 'text', {5839, 150, 14794, 597, 1}), llocation.reversed),
    #         ))
    #     )
    # )

    ### if entity does not exists before the step, its location should be not none
    ifL(
        andL(
            notL(before_existence('a56')),
            entity('e', path=('a56', action_entity)),
            step('i', path=('a56', action_step))
        ),
        entity_location_before_label('el1', path=(
                                ("e", lentity.reversed),
                                ("e", lentity.reversed, llocation, eqL(location, 'text', {"5839"}), llocation.reversed),
                                ("i", lstep.reversed)
                            )
        ), active = All_LC, name='checking_CL'
    )
    
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
            ), active = All_LC
        ) 

    #  ------------ Create
    # No subsequent create action unless there is a destroy action between them
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
                    action_label.create('a3', path=(('j1', action_step.reversed), ('e', action_entity.reversed)))
                    ), 
                # then exists step k associated with entity e, which is between step i and j1 associated with destroy action a3
                existsL(
                    andL(
                        step('k', path=(('j1', before_arg1.reversed, before_arg2), ('i', before_arg2.reversed, before_arg1))), 
                        action_label.destroy('a4', path=(('k', action_step.reversed), ('e', action_entity.reversed)))
                        )
                    )
                )
            ), active = All_LC
        )
    
    ### --- there cannot be a move action before the create event unless there is another create event before them
    ifL(
        andL(
            action_label.create('a100'),
            step('i', path=('a100', action_step)),
            entity('e', path=('a100', action_entity))
        ),
        ifL(
            step('j', path=(('i', before_arg2.reversed, before_arg1))),
            orL(
                notL(action_label.move('a101', path=(('j', action_step.reversed), ('e', action_entity.reversed)))),
                ifL(
                    action_label.move('a102', path=(('j', action_step.reversed), ('e', action_entity.reversed))),
                    existsL(
                        andL(
                                step('k', path=(('j', before_arg2.reversed, before_arg1))), 
                                action_label.create('a103', path=(('k', action_step.reversed), ('e', action_entity.reversed)))
                            )
                        )
                )
            )
        ), active = All_LC
    )
    
    ### --- there cannot be a exists action before the create event unless there is another create event before them
    ifL(
        andL(
            action_label.create('a100'),
            step('i', path=('a100', action_step)),
            entity('e', path=('a100', action_entity))
        ),
        ifL(
            step('j', path=(('i', before_arg2.reversed, before_arg1))),
            orL(
                notL(action_label.exists('a101', path=(('j', action_step.reversed), ('e', action_entity.reversed)))),
                ifL(
                    action_label.exists('a102', path=(('j', action_step.reversed), ('e', action_entity.reversed))),
                    existsL(
                        andL(
                                step('k', path=(('j', before_arg2.reversed, before_arg1))), 
                                action_label.create('a103', path=(('k', action_step.reversed), ('e', action_entity.reversed)))
                            )
                        )
                )
            )
        ), active = All_LC
    )

    ### If there is an outside action, there cannot be create, exist, move action before unless there is a destroy action after them
    ifL(
        andL(
            action_label.outside('a110'),
            step('i', path=('a110', action_step)),
            entity('e', path=('a110', action_entity))
        ),
        ifL(
            andL(
                step('j', path=(('i', before_arg2.reversed, before_arg1))),
                orL(
                    action_label.exists('a114', path=(('j', action_step.reversed), ('e', action_entity.reversed))),
                    action_label.move('a115', path=(('j', action_step.reversed), ('e', action_entity.reversed))),
                    action_label.create('a116', path=(('j', action_step.reversed), ('e', action_entity.reversed)))
                ),
            ),                
            existsL(
                andL(
                        step('k', path=(('j', before_arg1.reversed, before_arg2), ('i', before_arg2.reversed, before_arg1))),
                        action_label.destroy('a117', path=(('k', action_step.reversed), ('e', action_entity.reversed)))
                    )
            )
        ), active = All_LC
    )

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
                    action_label.create('a3', path=(('j1', action_step.reversed), ('e', action_entity.reversed)))
                    ), 
                # then exists step k associated with entity e, which is between step i and j1 associated with destroy action a3
                existsL(
                    andL(
                        step('k', path=(('j1', before_arg1.reversed, before_arg2), ('i', before_arg2.reversed, before_arg1))), 
                        action_label.destroy('a4', path=(('k', action_step.reversed), ('e', action_entity.reversed)))
                        )
                    )
            )
        ), active = All_LC
    )

    #### New constraints for the sequential order of actions
    # forAllL(
    #     combinationC(exact_before, entity)('ebstep', 'e'),
    #     ifL(
    #         action_label.outside('a1', path=(
    #                             ('ebstep', ebefore_arg2, action_step),
    #                             ('e', action_entity.reversed)
    #                             )
    #         ),
    #         orL(
    #             action_label.outside('a2', path=(
    #                             ('ebstep', ebefore_arg1, action_step),
    #                             ('e', action_entity.reversed)
    #                         )
    #             ),
    #             action_label.destroy('a3', path=(
    #                             ('ebstep', ebefore_arg1, action_step),
    #                             ('e', action_entity.reversed)
    #                         )
    #             ),
    #         ),
    #     )
    # )

    # forAllL(
    #     combinationC(exact_before, entity)('ebstep', 'e'),
    #     ifL(
    #         action_label.exists('a1', path=(
    #                             ('ebstep', ebefore_arg2, action_step),
    #                             ('e', action_entity.reversed)
    #                             )
    #         ),
    #         orL(
    #             action_label.exists('a2', path=(
    #                             ('ebstep', ebefore_arg1, action_step),
    #                             ('e', action_entity.reversed)
    #                         )
    #             ),
    #             action_label.move('a3', path=(
    #                             ('ebstep', ebefore_arg1, action_step),
    #                             ('e', action_entity.reversed)
    #                         )
    #             ),
    #             action_label.create('a4', path=(
    #                             ('ebstep', before_arg1, action_step),
    #                             ('e', action_entity.reversed)
    #                         )
    #             ),
    #         ),
    #     )
    # )

    # forAllL(
    #     combinationC(exact_before, entity)('ebstep', 'e'),
    #     ifL(
    #         action_label.create('a1', path=(
    #                             ('ebstep', ebefore_arg2, action_step),
    #                             ('e', action_entity.reversed)
    #                             )
    #         ),
    #         orL(
    #             action_label.outside('a2', path=(
    #                             ('ebstep', ebefore_arg1, action_step),
    #                             ('e', action_entity.reversed)
    #                         )
    #             ),
    #             action_label.destroy('a3', path=(
    #                             ('ebstep', ebefore_arg1, action_step),
    #                             ('e', action_entity.reversed)
    #                         )
    #             ),
    #         ),
    #     )
    # )

    # forAllL(
    #     combinationC(exact_before, entity)('ebstep', 'e'),
    #     ifL(
    #         action_label.move('a1', path=(
    #                             ('ebstep', ebefore_arg2, action_step),
    #                             ('e', action_entity.reversed)
    #                             )
    #         ),
    #         orL(
    #             action_label.exists('a2', path=(
    #                             ('ebstep', ebefore_arg1, action_step),
    #                             ('e', action_entity.reversed)
    #                         )
    #             ),
    #             action_label.move('a3', path=(
    #                             ('ebstep', ebefore_arg1, action_step),
    #                             ('e', action_entity.reversed)
    #                         )
    #             ),
    #             action_label.create('a4', path=(
    #                             ('ebstep', ebefore_arg1, action_step),
    #                             ('e', action_entity.reversed)
    #                         )
    #             ),
    #         ),
    #     )
    # )

    # forAllL(
    #     combinationC(exact_before, entity)('ebstep', 'e'),
    #     ifL(
    #         action_label.destroy('a1', path=(
    #                             ('ebstep', ebefore_arg2, action_step),
    #                             ('e', action_entity.reversed)
    #                             )
    #         ),
    #         orL(
    #             action_label.exists('a2', path=(
    #                             ('ebstep', ebefore_arg1, action_step),
    #                             ('e', action_entity.reversed)
    #                         )
    #             ),
    #             action_label.move('a3', path=(
    #                             ('ebstep', ebefore_arg1, action_step),
    #                             ('e', action_entity.reversed)
    #                         )
    #             ),
    #             action_label.create('a4', path=(
    #                             ('ebstep', ebefore_arg1, action_step),
    #                             ('e', action_entity.reversed)
    #                         )
    #             ),
    #         ),
    #     )
    # )

    ifL(
        entity('e'),
        atMostL(
            action_label.create(path=('e', action_entity.reversed)),
            2
        ), active = All_LC
    )

    ifL(
        entity('e'),
        atMostL(
            action_label.destroy(path=('e', action_entity.reversed)),
            2
        ), active = All_LC
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
            # step j associated with entity e, which is before step i cannot be associated with destroy action a2 or outside action a2
            ifL(
                step('j', path=(('i', before_arg2.reversed, before_arg1))), 
                andL(
                    notL(action_label.destroy('a2', path=(('j', action_step.reversed), ('e', action_entity.reversed)))),
                    notL(action_label.outside('a3', path=(('j', action_step.reversed), ('e', action_entity.reversed))))
                )
            ),
            # or if  
            ifL(
                # step j1 which is before step i is associated with destroy action a2
                andL(
                    step('j1', path=('i', before_arg2.reversed, before_arg1)), 
                    orL(
                        action_label.destroy('a4', path=(('j1', action_step.reversed), ('e', action_entity.reversed))),
                        action_label.outside('a5', path=(('j1', action_step.reversed), ('e', action_entity.reversed)))
                    )
                    ), 
                # then exists step k associated with entity e, which is between step i and j1 associated with create action a3
                existsL(
                    andL(
                        step('k', path=(('j1', before_arg1.reversed, before_arg2), ('i', before_arg2.reversed, before_arg1))), 
                        action_label.create('a6', path=(('k', action_step.reversed), ('e', action_entity.reversed)))
                        )
                    )
                )
            ), active = All_LC
        )
    
    ifL(
        # action a1 is exists, i is a1's step and e is action entity
        andL(
            action_label.exists('a1'), 
            step('i', path=('a1', action_step)),
            entity('e', path=('a1', action_entity))
            ), 
        # then either
        orL(
            # step j associated with entity e, which is before step i cannot be associated with destroy action a2 or outside action a2
            ifL(
                step('j', path=(('i', before_arg2.reversed, before_arg1))), 
                andL(
                    notL(action_label.destroy('a2', path=(('j', action_step.reversed), ('e', action_entity.reversed)))),
                    notL(action_label.outside('a3', path=(('j', action_step.reversed), ('e', action_entity.reversed))))
                )
            ),
            # or if  
            ifL(
                # step j1 which is before step i is associated with destroy action a2
                andL(
                    step('j1', path=('i', before_arg2.reversed, before_arg1)), 
                    orL(
                        action_label.destroy('a4', path=(('j1', action_step.reversed), ('e', action_entity.reversed))),
                        action_label.outside('a5', path=(('j1', action_step.reversed), ('e', action_entity.reversed)))
                    )
                ), 
                # then exists step k associated with entity e, which is between step i and j1 associated with create action a3
                existsL(
                    andL(
                        step('k', path=(('j1', before_arg1.reversed, before_arg2), ('i', before_arg2.reversed, before_arg1))), 
                        action_label.create('a6', path=(('k', action_step.reversed), ('e', action_entity.reversed)))
                        )
                    )
                )
            ), active = All_LC
        )
    

    ### ------ Move cannot happen if the item is not created or exist from before
    # ifL(
    #     # action a1 is move, i is a1's step and e is action entity
    #     andL(
    #         action_label.move('a1'),
    #         step('i', path=('a1', action_step)),
    #         entity('e', path=('a1', action_entity))
    #         ),
    #     # then either
    #     orL(
    #         ## the entity exists from before
    #         ifL(
    #             # step j associated with entity e, which is before step i cannot be associated with destroy action a2

    ### If the action is move, then the location from step before should be different from the current step
    ifL(
        # action a1 is move, i is a1's step and e is action entity
        andL(
            action_label.move('a1'), 
            step('i', path=('a1', action_step)),
            step('j', path=(('i', before_arg2.reversed, before_arg1))),
            entity('e', path=('a1', action_entity)),
            entity_location_before_label('x', path=(('i', lstep.reversed), ('e', lentity.reversed)))
            ), 
        notL(entity_location_label('y', path=(('j', lstep.reversed), ('e', lentity.reversed), ('x', llocation, llocation.reversed))))
    , active = All_LC
    )
    # forAllL(
    #     combinationC(step, entity, location)('i', 'e', 'l'),
    #     ifL(
    #         andL(
    #             action_label.move('a1', path=(
    #                 ('i', action_step),
    #                 ('e', action_entity)
    #                 )),
    #             entity_location_before_label('x', path=(
    #                 ('i', lstep.reversed),
    #                 ('e', lentity.reversed),
    #                 ('l', llocation.reversed)
    #                 ))
    #         ),
    #         notL(
    #             entity_location_label('y', path=(
    #                 ('i', lstep.reversed),
    #                 ('e', lentity.reversed),
    #                 ('l', llocation.reversed)
    #                 ))
    #         )
    #     ), active = All_LC
    # )

    ### if action is exists, the location should not change
    ifL(
        andL(
            action_label.exists('a1'),
            step('i', path=('a1', action_step)),
            entity('e', path=('a1', action_entity)),
            entity_location_before_label('x', path=(('i', lstep.reversed), ('e', lentity.reversed)))
            ),
        entity_location_label('y', path=(('i', lstep.reversed), ('e', lentity.reversed), ('x', llocation, llocation.reversed)))
    )

    
    
    ### There can only be one location for each entity at each step
    forAllL(
         combinationC(step, entity)('i', 'e'), #this is the search space, cartesian product is expected between options
         exactL(
             entity_location_label('x', path=(('i', lstep.reversed), ('e', lentity.reversed))), 1
         ), active = All_LC# this is the condition that should hold for every assignment
    )
    
    forAllL(
         combinationC(step, entity)('i', 'e'), #this is the search space, cartesian product is expected between options
         exactL(
             entity_location_before_label('x', path=(('i', lstep.reversed), ('e', lentity.reversed))), 1
         ), active = All_LC # this is the condition that should hold for every assignment
    )
    
    ### for each entity, only one when_creat and when_destroy can be correct
    ifL(
        entity('e'),
        atMostL(
            when_create('x', path=('e', action_entity.reversed)),
        ), active = Tested_Lc
    )

    ifL(
        entity('e'),
        atMostL(
            when_destroy('x', path=('e', action_entity.reversed)),
        ), active = All_LC
    )

    ### the input/output and alternative should match each other
    ifL(input_entity('x'), input_entity_alt(path=('x')), active = All_LC)
    ifL(output_entity('x'), output_entity_alt(path=('x')), active = All_LC)
    ifL(input_entity_alt('x'), input_entity(path=('x')), active = All_LC)
    ifL(output_entity_alt('x'), output_entity(path=('x')), active = All_LC)

    ### for each step and entity at most one action is applicable
    # forAllL(
    #     combinationC(step, entity)('i', 'e'),
    #     ifL(
    #         action(path=(('i', action_step.reversed), ('e', action_entity.reversed))),
    #         atMostL(action_create, action_destroy, action_move)
    #     ), active = All_LC
    # )

    ### for each step and entity at most one action is applicable
    forAllL(
         combinationC(step, entity)('i', 'e'),
         ifL(
             action('x', path=(('i', action_step.reversed), ('e', action_entity.reversed))),
             atMostL(action_create(path='x'), action_destroy(path='x'), action_move(path='x'))
         ), active = All_LC
    )
    
    ### if action is create, the location should not be `none` and before location should be none
    forAllL(
        combinationC(step, entity)('i', 'e'),
        ifL(
            action_label.create('a1', path=(('i', action_step.reversed), ('e', action_entity.reversed))),
            andL(
                notL(
                    entity_location_label('el1', path=(
                                    ("e", lentity.reversed),
                                    ("e", lentity.reversed, llocation, eqL(location, 'text', {"5839", "150", "14794", "597", "1", "3"}), llocation.reversed), 
                                    ("i", lstep.reversed)
                                )
                    )
                ),
                entity_location_before_label('el2', path=(
                                    ("e", lentity.reversed),
                                    ("e", lentity.reversed, llocation, eqL(location, 'text', {"5839"}), llocation.reversed), 
                                    ("i", lstep.reversed)
                                )
                    )
            )
        ), active = All_LC
    )

    ### if action is destroy, the location should be `none`
    forAllL(
        combinationC(step, entity)('i', 'e'),
        ifL(
            action_label.destroy('a1', path=(('i', action_step.reversed), ('e', action_entity.reversed))),
            andL(
                notL(
                    entity_location_before_label('el1', path=(
                                    ("e", lentity.reversed),
                                    ("e", lentity.reversed, llocation, eqL(location, 'text', {"5839", "150", "14794", "597", "1", "3"}), llocation.reversed), 
                                    ("i", lstep.reversed)
                                )
                    )
                ),
                entity_location_label('el2', path=(
                                    ("e", lentity.reversed),
                                    ("e", lentity.reversed, llocation, eqL(location, 'text', {"5839"}), llocation.reversed), 
                                    ("i", lstep.reversed)
                                )
                    )
            )
        ), active = All_LC
    )

    ### if action is move, the location should not be `none`
    forAllL(
        combinationC(step, entity)('i', 'e'),
        ifL(
            action_label.move('a1', path=(('i', action_step.reversed), ('e', action_entity.reversed))),
            andL(
                notL(
                    entity_location_label('el1', path=(
                                    ("e", lentity.reversed),
                                    ("e", lentity.reversed, llocation, eqL(location, 'text', {"5839", "150", "14794", "597", "1", "3"}), llocation.reversed), 
                                    ("i", lstep.reversed)
                                )
                    )
                ),
                notL(entity_location_before_label('el2', path=(
                                    ("e", lentity.reversed),
                                    ("e", lentity.reversed, llocation, eqL(location, 'text', {"5839", "150", "14794", "597", "1", "3"}), llocation.reversed), 
                                    ("i", lstep.reversed)
                                )
                    )
                )
            )
        ), active = All_LC
    )

    ### if action is move, the location should be different from the previous step(before/after location)
    forAllL(
        combinationC(step, entity)('i', 'e'),
        ifL(
            action_label.move('a1', path=(('i', action_step.reversed), ('e', action_entity.reversed))),
            ifL(
                entity_location_before_label('el1', path=(
                                ("e", lentity.reversed),
                                ("i", lstep.reversed)
                            )
                ),
                notL(
                    entity_location_label('el2', path=(
                                        ("e", lentity.reversed),
                                        ("i", lstep.reversed),
                                        ('e', llocation, llocation.reversed)
                                    )
                        )
                )
            )
        ), active = All_LC
    )

    ### if action is exists, the location should not be `none` and before location should not be none
    forAllL(
        combinationC(step, entity)('i', 'e'),
        ifL(
            action_label.exists('a1', path=(('i', action_step.reversed), ('e', action_entity.reversed))),
            andL(
                notL(
                    entity_location_label('el1', path=(
                                    ("e", lentity.reversed),
                                    ("e", lentity.reversed, llocation, eqL(location, 'text', {"5839", "150", "14794", "597", "1", "3"}), llocation.reversed), 
                                    ("i", lstep.reversed)
                                )
                    )
                ),
                notL(entity_location_before_label('el2', path=(
                                    ("e", lentity.reversed),
                                    ("e", lentity.reversed, llocation, eqL(location, 'text', {"5839", "150", "14794", "597", "1", "3"}), llocation.reversed), 
                                    ("i", lstep.reversed)
                                )
                    )
                )
            )
        ), active = All_LC
    )

    ### if action is outside, the location should  be `none` and before location should be none
    forAllL(
        combinationC(step, entity)('i', 'e'),
        ifL(
            action_label.outside('a1', path=(('i', action_step.reversed), ('e', action_entity.reversed))),
            andL(
                entity_location_label('el1', path=(
                                ("e", lentity.reversed),
                                ("e", lentity.reversed, llocation, eqL(location, 'text', {"5839"}), llocation.reversed), 
                                ("i", lstep.reversed)
                            )
                ),
                entity_location_before_label('el2', path=(
                                ("e", lentity.reversed),
                                ("e", lentity.reversed, llocation, eqL(location, 'text', {"5839"}), llocation.reversed), 
                                ("i", lstep.reversed)
                            )
                )
            )
        ), active = All_LC
    )

    ### multi-class action_label and the actual create, destroy, move label alignment
    ifL(
        action_label.move('a1'),
        action_move(path=('a1',)), active = All_LC
    )
    ifL(
        action_label.create('a1'),
        action_create(path=('a1',)), active = All_LC
    )
    ifL(
        action_label.destroy('a1'),
        action_destroy(path=('a1',)), active = All_LC
    )

    ### when-create and action-create should match
    ifL(
        andL(
            when_create('a1'),
            step('i', path=('a1', action_step)),
            entity('e', path=('a1', action_entity))
        ),
        action_create('a2', path=(('i', action_step.reversed), ('e', action_entity.reversed))), active = All_LC
    )

    ### when-destroy and action-destroy should match
    ifL(
        andL(
            when_destroy('a1'),
            step('i', path=('a1', action_step)),
            entity('e', path=('a1', action_entity))
        ),
        action_destroy('a2', path=(('i', action_step.reversed), ('e', action_entity.reversed))), active = All_LC
    )

    # ifL(
    #     input_entity('e'),
    #     notL(
    #         entity_location_before_label('el1', path=(
    #                             ("e", lentity.reversed),
    #                             ("e", lentity.reversed, lstep, eqL(step, 'index', {0})),
    #                             # ("e", lentity.reversed, llocation, eqL(location, 'text', {5839, 150, 14794, 597, 1}))
    #                             ("l", llocation.reversed)

    #         ))
    #     )
    # )
    ### if entity is output, the last state should not be `none`

    ### if entity exists, the location should not be `none`
    ifL(
        andL(
            after_existence('a11'),
            entity('e', path=('a11', action_entity)),
            step('i', path=('a11', action_step))
        ),
        notL(
            entity_location_label('el1', path=(
                                    ("e", lentity.reversed),
                                    ("e", lentity.reversed, llocation, eqL(location, 'text', {"5839", "150", "14794", "597", "1", "3"}), llocation.reversed),
                                    ("i", lstep.reversed)
                                )
            )
        ), active = All_LC
    )

    ifL(
        andL(
            before_existence('a12'),
            entity('e', path=('a12', action_entity)),
            step('i', path=('a12', action_step))
        ),
        notL(
            entity_location_before_label('el1', path=(
                                    ("e", lentity.reversed),
                                    ("e", lentity.reversed, llocation, eqL(location, 'text', {"5839", "150", "14794", "597", "1", "3"}), llocation.reversed),
                                    ("i", lstep.reversed)
                                )
            )
        ), active = All_LC
    )

    ### if location change is true, the locations after for step i-1 and i should not match
    ifL(
        andL(
            location_change('a1'),
            step('i', path=('a1', action_step)),
            entity('e', path=('a1', action_entity)),
        ),
        ifL(
            entity_location_before_label('el1', path=(
                        ("e", lentity.reversed),
                        ("i", lstep.reversed)
                    )
            ),
            notL(
                entity_location_label('el2', path=(
                        ("e", lentity.reversed),
                        ("i", lstep.reversed),
                        ("e", llocation, llocation.reversed)
                    )
                ),
            )
        ), active = All_LC
    )
    
    ### if location change is true, there should be one action either create, destroy, or move
    ifL(
        location_change('a14'),
        orL(
            action_create(path=('a14')),
            action_destroy(path=('a14')),
            action_move(path=('a14'))
        ), active = All_LC
    )

    ifL(
        orL(
            action_create(path=('a14')),
            action_destroy(path=('a14')),
            action_move(path=('a14'))
        ),
        location_change('a14'), active = All_LC
    )

    ### if the the location should not match the entity itself
    # forAllL(
    #     combinationC(step, entity)('i', 'e'),
    #     ifL(
    #         entity_location_label('el1', path=(
    #                         ("e", lentity.reversed),
    #                         ("i", lstep.reversed)
    #             )
    #         )
    #     ),
    #     notL(
    #         existsL(
    #             same_mention('sm1', path=(
    #                 ("e", same_entity.reversed),
    #                 ("el1", llocation, same_location.reversed)
    #             ))
    #         )
    #     )
    # )

    ### if entity and location match each other in same_mention, the entity_location with that location should be false
    # forAllL(
    #     combinationC(step, entity, location)('i', 'e', 'l'),
    #     ifL(
    #         existsL(
    #             same_mention('sm1', path=(
    #                 ("e", same_entity.reversed),
    #                 ("l", same_location.reversed)
    #             ))
    #         ),
    #         notL(
    #             entity_location_label('el1', path=(
    #                 ("e", lentity.reversed),
    #                 ("l", llocation.reversed),
    #                 ("i", lstep.reversed)
    #             ))
    #         )
    #     ), active = Tested_Lc
    # )

    forAllL(
        combinationC(same_mention, step)('sm1', 'i'),
        notL(
            entity_location_label('el1', path=(
                ("sm1", same_entity, lentity.reversed),
                ("sm1", same_location, llocation.reversed),
                ("i", lstep.reversed)
            ))
        ), active = Tested_Lc
    )

    ### this has a problem, for instance carbon can be inside carbon-based mixture

    ### if the location of entity `e` is `l` which matches another entity `e1`, then the entity `e1` should exist
    
    # if some location is in same_mention relation with current entity and step selected from the combination then the action for this entity in the current step has to be after_existance
    # in case the entity in the given step can associated with multiply actions then we have to use existsL in the second part of if
    forAllL( 
        combinationC(step, entity)('i', 'e'),
        # combinationC(step, entity, location('l1'), entity(path=('l1', same_location.reversed, same_entity)))('i', 'e', 'l', 'e2')
        ifL(
            entity_location_label('el1', path=("e", lentity.reversed, eqL(lstep, "i"))),
            ifL(
                existsL(
                    same_mention('sm1', path=(("el1", llocation, same_location.reversed)))
                ),
                atLeastL(
                    after_existence('a1', path=("el1", llocation, same_location.reversed, same_entity, action_entity.reversed, eqL(action_step, "i"))), 1
                )
            )  
        ), active=All_LC 
    )

    ### if entity 'e' is located in a location 'l' which corresponds to an entity 'e1' and entity 'e1' is destroyed, the entity `e` is either moved or destroyed
    # forAllL(
    #     combinationC(step, entity)('i', 'e'),
    #     ifL(
    #         andL(
    #             ### entity `e` at step `i` is located at `el1.llocation`
    #             entity_location_before_label('el1', path=(
    #                             ("e", lentity.reversed),
    #                             ("i", lstep.reversed)
    #             )),
    #             existsL(
    #                 ### if there exist an entity mention which matches `el1.llocation`, and that entity is `destroyed`
    #                 andL(
    #                     ### entity `sm1.same_entity` is the same mention as `el.llocation`
    #                     same_mention('sm1', path=(
    #                     ("el1", llocation, same_location.reversed)
    #                     )),
    #                     ### `sm1.same_entity` is destroyed at step `i`
    #                     action_destroy('a1', path=(
    #                         ("sm1", same_entity, action_entity.reversed),
    #                         ("i", action_step.reversed)
    #                     )),
    #                 ) 
    #             ),        
    #         ),
    #         orL(
    #             ### the original entity `e` should be either moved or destroyed at step `i`
    #             action_move('a2', path=(
    #                 ("e", action_entity.reversed),
    #                 ("i", action_step.reversed)
    #             )),
    #             action_destroy('a3', path=(
    #                 ("e", action_entity.reversed),
    #                 ("i", action_step.reversed)
    #             ))
    #         )
    #     ), active = Tested_Lc
    # )
        
    ### sum(x1, x2, i + j) :- image(x1, i), image(x2, j)
    # def check_func(i, j, k):
    #     return i + j == k
    
    # ifL(
    #     sum(var='x', val='k'), --> one, two, three, ..., eighteen
    #     --> (x1, k1), (x1, k2), (x1, k3)
    #     --> (x2, k1), (x2, k2), (x2, k3)
    #     andL(
    #         image(var='y = x1.arg1', val='i'), --> one, two, three, ... ,nine
    #         image(var='z = x1.arg2', val='j'), --> one, two, three, ..., nine
    #         # check_valL(check_func(i, j, k))
    #         # check_varL()
    #     )
    # )


    # ifL(
    #     c1(),
    #     c2(), 
    #     check_valL()
    # )
    



    
    # [x1, x2]
    # [[x1.arg1], [x2.arg1]]
    # [[x1.arg2], [x2.arg2]]

    # (x1, x1.arg1, x1.arg2)
    # (x2, x2.arg1, x2.arg2)

    # (x1, k1), (x1, k2), (x1, k3)
    # ([(x1.arg1, i1), ])

    # check_func(**filled_values)
    # filled_values {"i": 10, "j": 11, "k": 21}

    ### At least one input should exist
    atLeastAL(input_entity('e'), active = All_LC)

    ### At least one output should exist
    # atLeastAL(output_entity('e'), active = All_LC)

    # graph.visualize("./image")

    #from PIL import Image
    # Open an image file
    #graphImage = Image.open('image.png')
    # Display the image
    #graphImage.show()
