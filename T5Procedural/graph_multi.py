from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import orL, andL, existsL, notL, atLeastL, atMostL, ifL, nandL, V, exactL, forAllL, eqL
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
        
    #  ------------ Destroy
    # No subsequent destroy action unless there is a create action between them
    
    ### the before action of step i+1 should match the after action of step i
    ifL(
        andL(
            entity_location_label('x'),
            step('i', path=('x', lstep)),
            entity('e', path=('x', lentity))
        ),
        andL(
            step('j', path=('i', ebefore_arg2.reversed, ebefore_arg1)),
            entity_location_label('y', path=(('j', lstep.reversed), ('e', lentity.reversed), ('x', llocation, llocation.reversed)))
        )
    )

    ### if entity is input, the first state should not be `none`
    forAllL(
        combinationC(entity, location(path=eqL(location, 'text', {5839})))('e', 'l'),
        ifL(
            input_entity('e'),
            notL(
                entity_location_before_label('el1', path=(
                                    ("e", lentity.reversed),
                                    ("e", lentity.reversed, lstep, eqL(step, 'index', {0}), lstep.reversed),
                                    # ("e", lentity.reversed, llocation, eqL(location, 'text', {5839}))
                                    ("l", llocation.reversed)

                ))
            )
        )
    )

    ifL(
        andL(
            notL(before_existence('a56')),
            entity('e', path=('a56', action_entity)),
            step('i', path=('a56', action_step))
        ),
        entity_location_before_label('el1', path=(
                                ("e", lentity.reversed),
                                ("e", lentity.reversed, llocation, eqL(location, 'text', {5839}), llocation.reversed),
                                ("i", lstep.reversed)
                            )
        ), active = True, name='checking_CL'
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
            ), active = True
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
    # No subsequent move action unless there is a create action between them

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
                        action_label.destroy('a3', path=(('j1', action_step.reversed), ('e', action_entity.reversed))),
                        action_label.outside('a4', path=(('j1', action_step.reversed), ('e', action_entity.reversed)))
                    )
                    ), 
                # then exists step k associated with entity e, which is between step i and j1 associated with create action a3
                existsL(
                    andL(
                        step('k', path=(('j1', before_arg1.reversed, before_arg2), ('i', before_arg2.reversed, before_arg1))), 
                        action_label.create('a5', path=(('k', action_step.reversed), ('e', action_entity.reversed)))
                        )
                    )
                )
            ), active = True
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
                        action_label.destroy('a3', path=(('j1', action_step.reversed), ('e', action_entity.reversed))),
                        action_label.outside('a4', path=(('j1', action_step.reversed), ('e', action_entity.reversed)))
                    )
                    ), 
                # then exists step k associated with entity e, which is between step i and j1 associated with create action a3
                existsL(
                    andL(
                        step('k', path=(('j1', before_arg1.reversed, before_arg2), ('i', before_arg2.reversed, before_arg1))), 
                        action_label.create('a5', path=(('k', action_step.reversed), ('e', action_entity.reversed)))
                        )
                    )
                )
            ), active = True
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
            entity('e', path=('a1', action_entity)),
            entity_location_label('x', path=(('i', lstep.reversed), ('e', lentity.reversed)))
            ), 
        andL(
            step('j', path=('i', ebefore_arg2.reversed, ebefore_arg1)),
            notL(entity_location_label('y', path=(('j', lstep.reversed), ('e', lentity.reversed), ('x', llocation, llocation.reversed))))
        ),
        active = True
    )
    
    ### There can only be one location for each entity at each step
    forAllL(
         combinationC(step, entity)('i', 'e'), #this is the search space, cartesian product is expected between options
         exactL(
             entity_location_label('x', path=(('i', lstep.reversed), ('e', lentity.reversed))), 1
         ), # this is the condition that should hold for every assignment
     )
    
    ### for each entity, only one when_creat and when_destroy can be correct
    ifL(
        entity('e'),
        atMostL(
            when_create('x', path=('e', action_entity.reversed)),
        )
    )

    ifL(
        entity('e'),
        atMostL(
            when_destroy('x', path=('e', action_entity.reversed)),
        )
    )

    ### the input/output and alternative should match each other
    ifL(input_entity('x'), input_entity_alt(path=('x')))
    ifL(output_entity('x'), output_entity_alt(path=('x')))

    ### for each step and entity at most one action is applicable
    forAllL(
        combinationC(step, entity)('i', 'e'),
        ifL(
            action(path=(('i', action_step.reversed), ('e', action_entity.reversed))),
            atMostL(action_create, action_destroy, action_move)
        )
    )

    ### for each step and entity at most one action is applicable
    forAllL(
         combinationC(step, entity)('i', 'e'),
         ifL(
             action('x', path=(('i', action_step.reversed), ('e', action_entity.reversed))),
             atMostL(action_create(path='x'), action_destroy(path='x'), action_move(path='x'))
         )
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
                                    ("e", lentity.reversed, llocation, eqL(location, 'text', {5839}), llocation.reversed), 
                                    ("i", lstep.reversed)
                                )
                    )
                ),
                entity_location_before_label('el1', path=(
                                    ("e", lentity.reversed),
                                    ("e", lentity.reversed, llocation, eqL(location, 'text', {5839}), llocation.reversed), 
                                    ("i", lstep.reversed)
                                )
                    )
            )
        )
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
                                    ("e", lentity.reversed, llocation, eqL(location, 'text', {5839}), llocation.reversed), 
                                    ("i", lstep.reversed)
                                )
                    )
                ),
                entity_location_label('el1', path=(
                                    ("e", lentity.reversed),
                                    ("e", lentity.reversed, llocation, eqL(location, 'text', {5839}), llocation.reversed), 
                                    ("i", lstep.reversed)
                                )
                    )
            )
        )
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
                                    ("e", lentity.reversed, llocation, eqL(location, 'text', {5839}), llocation.reversed), 
                                    ("i", lstep.reversed)
                                )
                    )
                ),
                notL(entity_location_before_label('el1', path=(
                                    ("e", lentity.reversed),
                                    ("e", lentity.reversed, llocation, eqL(location, 'text', {5839}), llocation.reversed), 
                                    ("i", lstep.reversed)
                                )
                    )
                )
            )
        )
    )

    ### if action is move, the location should be different from the previous step(before/after location)
    forAllL(
        combinationC(step, entity)('i', 'e'),
        ifL(
            action_label.move('a1', path=(('i', action_step.reversed), ('e', action_entity.reversed))),
            ifL(
                entity_location_label('el1', path=(
                                ("e", lentity.reversed),
                                ("i", lstep.reversed)
                            )
                ),
                notL(
                    entity_location_before_label('el2', path=(
                                        ("e", lentity.reversed),
                                        ("i", lstep.reversed),
                                        ('e1', llocation, llocation.reversed)
                                    )
                        )
                )
            )
        )
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
                                    ("e", lentity.reversed, llocation, eqL(location, 'text', {5839}), llocation.reversed), 
                                    ("i", lstep.reversed)
                                )
                    )
                ),
                notL(entity_location_before_label('el1', path=(
                                    ("e", lentity.reversed),
                                    ("e", lentity.reversed, llocation, eqL(location, 'text', {5839}), llocation.reversed), 
                                    ("i", lstep.reversed)
                                )
                    )
                )
            )
        )
    )

    ### if action is outside, the location should  be `none` and before location should be none
    forAllL(
        combinationC(step, entity)('i', 'e'),
        ifL(
            action_label.outside('a1', path=(('i', action_step.reversed), ('e', action_entity.reversed))),
            andL(
                entity_location_label('el1', path=(
                                ("e", lentity.reversed),
                                ("e", lentity.reversed, llocation, eqL(location, 'text', {5839}), llocation.reversed), 
                                ("i", lstep.reversed)
                            )
                ),
                entity_location_before_label('el1', path=(
                                ("e", lentity.reversed),
                                ("e", lentity.reversed, llocation, eqL(location, 'text', {5839}), llocation.reversed), 
                                ("i", lstep.reversed)
                            )
                )
            )
        )
    )

    ### multi-class action_label and the actual create, destroy, move label alignment
    ifL(
        action_label.move('a1'),
        action_move('a1')
    )
    ifL(
        action_label.create('a1'),
        action_create('a1')
    )
    ifL(
        action_label.destroy('a1'),
        action_destroy('a1')
    )

    ### when-create and action-create should match
    ifL(
        andL(
            when_create('a1'),
            step('i', path=('a1', action_step)),
            entity('e', path=('a1', action_entity))
        ),
        action_create('a2', path=(('i', action_step.reversed), ('e', action_entity.reversed)))
    )

    ### when-destroy and action-destroy should match
    ifL(
        andL(
            when_destroy('a1'),
            step('i', path=('a1', action_step)),
            entity('e', path=('a1', action_entity))
        ),
        action_destroy('a2', path=(('i', action_step.reversed), ('e', action_entity.reversed)))
    )

   
    
    # ifL(
    #     input_entity('e'),
    #     notL(
    #         entity_location_before_label('el1', path=(
    #                             ("e", lentity.reversed),
    #                             ("e", lentity.reversed, lstep, eqL(step, 'index', {0})),
    #                             # ("e", lentity.reversed, llocation, eqL(location, 'text', {5839}))
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
                                    ("e", lentity.reversed, llocation, eqL(location, 'text', {5839}), llocation.reversed),
                                    ("i", lstep.reversed)
                                )
            )
        ), active = True
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
                                    ("e", lentity.reversed, llocation, eqL(location, 'text', {5839}), llocation.reversed),
                                    ("i", lstep.reversed)
                                )
            )
        ), active = True
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
                        ("el1", llocation, llocation.reversed)
                    )
                ),
            )
        )
    )
    ### if location change is true, the locations before for step i and after step i should not match
    ### if location change is true, there should be one action either create, destroy, or move
    ifL(
        andL(
            action('a14'),
            step('i', path=('a14', action_step)),
            entity('e', path=('a14', action_entity)),
        ),
        ifL(
            location_change(path=('a14')),
            orL(
                action_create(path=('a14')),
                action_destroy(path=('a14')),
                action_move(path=('a14'))
            )
        ), active = True
    )

    ### if the the location should not match the entity itself