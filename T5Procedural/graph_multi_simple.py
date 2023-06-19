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
    action_label = action(name="action_label", ConceptClass=EnumConcept, values=["create", "exists", "move", "destroy", "prior", "post"])

    same_mention = Concept(name='same_mention')
    (same_entity, same_location) = same_mention.has_a(se1=entity, se2=location)
    
    ### entity, step, and after location
    entity_location = Concept(name='entity_location')
    (lentity, lstep, llocation) = entity_location.has_a(lentity=entity, lstep=step, llocation=location)
    entity_location_label = entity_location(name='entity_location_label')
    entity_location_before_label = entity_location(name='entity_location_before_label')
     
    # LC Active status
    All_LC = False
    Tested_Lc = False
    action_level_lc = True
    location_action_lc = False
    location_level_lc = True

    ### the first action label cannot Post
    forAllL(
        combinationC(entity, step(path=(eqL(step, 'index', {0}))))('e', 'i'),
        notL(
            action_label.post('a110', path=(('i', action_step.reversed), ('e', action_entity.reversed)))
        ), active = action_level_lc
    )

    ### the alone candidate "3" is never the answer of a location
    forAllL(
        combinationC(entity, step, location(path=eqL(location, 'text', {"3"})))('e', 'i', 'l'),
        # combinationC(entity, location(path=eqL(location, 'text', {5839})))('e', 'l'),
        notL(
            entity_location_label('el1', path=(
                                ("e", lentity.reversed),
                                ("i", lstep.reversed),
                                ("l", llocation.reversed)
                )
            )
        ), active = location_level_lc
    )

    ### if action before is none and after is not none, then the action is create
    forAllL(
        combinationC(entity, step)('e', 'i'),
        # combinationC(entity, location(path=eqL(location, 'text', {5839})))('e', 'l'),
        ifL(
            andL(
                entity_location_before_label('el1', path=(
                                ("e", lentity.reversed),
                                ("i", lstep.reversed),
                                ("e", lentity.reversed, llocation, eqL(location, 'text', {"5839", "150", "14794", "597", "1", "3"}), llocation.reversed),
                )),
                notL(entity_location_label('el2', path=(
                                ("e", lentity.reversed),
                                ("i", lstep.reversed),
                                ("e", lentity.reversed, llocation, eqL(location, 'text', {"5839", "150", "14794", "597", "1", "3"}), llocation.reversed),
                ))),
            ),
            action_label.create('a', path=(('i', action_step.reversed), ('e', action_entity.reversed)))
        ), active = location_action_lc
    )

    ### the locations should match
    forAllL(
        combinationC(entity, exact_before)('e', 'step_rel'),
        ifL(
            entity_location_label('x', path=(('e', lentity.reversed), ('step_rel', ebefore_arg1, lstep.reversed))),
            entity_location_before_label('y', path=(('e', lentity.reversed), ('step_rel', ebefore_arg2, lstep.reversed), ('x', llocation, llocation.reversed))),
        ), active = location_level_lc
    )
    
    ### possible actions before destroy
    ifL(
       andL(
            action_label.destroy('a1'), 
            step('i', path=('a1', action_step)),
            entity('e', path=('a1', action_entity)),
            step('k', path=('i', ebefore_arg2.reversed, ebefore_arg1))
        ),
        orL(
            action_label.create('a2', path=(('k', action_step.reversed), ('e', action_entity.reversed))),
            action_label.move('a2', path=(('k', action_step.reversed), ('e', action_entity.reversed))),
            action_label.exists('a2', path=(('k', action_step.reversed), ('e', action_entity.reversed)))
        ), active = action_level_lc
    )

    ### possible actions before create
    ifL(
       andL(
            action_label.create('a1'), 
            step('i', path=('a1', action_step)),
            entity('e', path=('a1', action_entity)),
            step('k', path=('i', ebefore_arg2.reversed, ebefore_arg1))
        ),
        orL(
            action_label.prior('a2', path=(('k', action_step.reversed), ('e', action_entity.reversed))),
            action_label.post('a2', path=(('k', action_step.reversed), ('e', action_entity.reversed))),
            action_label.destroy('a2', path=(('k', action_step.reversed), ('e', action_entity.reversed)))
        ), active = action_level_lc
    )

    ### possible actions before move
    ifL(
       andL(
            action_label.move('a1'), 
            step('i', path=('a1', action_step)),
            entity('e', path=('a1', action_entity)),
            step('k', path=('i', ebefore_arg2.reversed, ebefore_arg1))
        ),
        orL(
            action_label.create('a2', path=(('k', action_step.reversed), ('e', action_entity.reversed))),
            action_label.exists('a2', path=(('k', action_step.reversed), ('e', action_entity.reversed))),
            action_label.move('a2', path=(('k', action_step.reversed), ('e', action_entity.reversed)))
        ), active = action_level_lc
    )
    
    ### possible actions before exists
    ifL(
       andL(
            action_label.exists('a1'), 
            step('i', path=('a1', action_step)),
            entity('e', path=('a1', action_entity)),
            step('k', path=('i', ebefore_arg2.reversed, ebefore_arg1))
        ),
        orL(
            action_label.exists('a2', path=(('k', action_step.reversed), ('e', action_entity.reversed))),
            action_label.move('a2', path=(('k', action_step.reversed), ('e', action_entity.reversed))),
            action_label.create('a2', path=(('k', action_step.reversed), ('e', action_entity.reversed)))
        ), active=action_level_lc
    )

    ### possible actions before prior
    ifL(
       andL(
            action_label.prior('a1'), 
            step('i', path=('a1', action_step)),
            entity('e', path=('a1', action_entity)),
            step('k', path=('i', ebefore_arg2.reversed, ebefore_arg1))
        ),
        action_label.prior('a2', path=(('k', action_step.reversed), ('e', action_entity.reversed)),
        ), active=action_level_lc
    )

    ### possible actions before post
    ifL(
       andL(
            action_label.post('a1'), 
            step('i', path=('a1', action_step)),
            entity('e', path=('a1', action_entity)),
            step('k', path=('i', ebefore_arg2.reversed, ebefore_arg1))
        ),
        orL(
            action_label.post('a2', path=(('k', action_step.reversed), ('e', action_entity.reversed))),
            action_label.destroy('a2', path=(('k', action_step.reversed), ('e', action_entity.reversed)))
        ), active=action_level_lc
    )

    ### before post, there should be a destroy somewhere
    ifL(
        andL(
            action_label.post('a110'),
            step('i', path=('a110', action_step)),
            entity('e', path=('a110', action_entity))
        ),
        existsL(
            andL(
                    step('j', path=(('i', before_arg2.reversed, before_arg1))),
                    action_label.destroy('a117', path=(('j', action_step.reversed), ('e', action_entity.reversed)))
                )
        ), active = action_level_lc
    )

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
            1
        ), active = All_LC
    )

    ### If the action is move, then the location from step before should be different from the current step
    ifL(
        # action a1 is move, i is a1's step and e is action entity
        andL(
            action_label.move('a1'), 
            step('i', path=('a1', action_step)),
            entity('e', path=('a1', action_entity)),
            entity_location_before_label('x', path=(('i', lstep.reversed), ('e', lentity.reversed)))
            ), 
        notL(entity_location_label('y', path=(('j', lstep.reversed), ('e', lentity.reversed), ('x', llocation, llocation.reversed))))
        , active = location_action_lc
    )
    

    ### if action is exists, the location should not change
    ifL(
        andL(
            action_label.exists('a1'),
            step('i', path=('a1', action_step)),
            entity('e', path=('a1', action_entity)),
            entity_location_before_label('x', path=(('i', lstep.reversed), ('e', lentity.reversed)))
            ),
        entity_location_label('y', path=(('i', lstep.reversed), ('e', lentity.reversed), ('x', llocation, llocation.reversed))),
        active = location_action_lc
    )
    
    ### There can only be one location for each entity at each step
    forAllL(
         combinationC(step, entity)('i', 'e'), #this is the search space, cartesian product is expected between options
         exactL(
             entity_location_label('x', path=(('i', lstep.reversed), ('e', lentity.reversed))), 1
         ), active = location_level_lc # this is the condition that should hold for every assignment
    )
    
    ### there can only be one location before for each entity at each step
    forAllL(
         combinationC(step, entity)('i', 'e'), #this is the search space, cartesian product is expected between options
         exactL(
             entity_location_before_label('x', path=(('i', lstep.reversed), ('e', lentity.reversed))), 1
         ), active = location_level_lc # this is the condition that should hold for every assignment
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
                entity_location_before_label('el1', path=(
                                    ("e", lentity.reversed),
                                    ("e", lentity.reversed, llocation, eqL(location, 'text', {"5839"}), llocation.reversed), 
                                    ("i", lstep.reversed)
                                )
                    )
            )
        ), active = location_action_lc
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
                entity_location_label('el1', path=(
                                    ("e", lentity.reversed),
                                    ("e", lentity.reversed, llocation, eqL(location, 'text', {"5839"}), llocation.reversed), 
                                    ("i", lstep.reversed)
                                )
                    )
            )
        ), active = location_action_lc
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
                notL(entity_location_before_label('el1', path=(
                                    ("e", lentity.reversed),
                                    ("e", lentity.reversed, llocation, eqL(location, 'text', {"5839", "150", "14794", "597", "1", "3"}), llocation.reversed), 
                                    ("i", lstep.reversed)
                                )
                    )
                )
            )
        ), active = location_action_lc
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
                                        ('el1', llocation, llocation.reversed)
                                    )
                        )
                )
            )
        ), active = location_action_lc
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
        ), active = location_action_lc
    )

    ### if action is prior, the location should  be `none` and before location should be none
    forAllL(
        combinationC(step, entity)('i', 'e'),
        ifL(
            action_label.prior('a1', path=(('i', action_step.reversed), ('e', action_entity.reversed))),
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
        ), active = location_action_lc
    )

    ### if action is post, the location should  be `none` and before location should be none
    forAllL(
        combinationC(step, entity)('i', 'e'),
        ifL(
            action_label.post('a1', path=(('i', action_step.reversed), ('e', action_entity.reversed))),
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
        ), active = location_action_lc
    )

    ### for same mention of entities some relations should hold
    forAllL(
        combinationC(same_mention, step)('sm1', 'i'),
        notL(
            entity_location_label('el1', path=(
                ("sm1", same_entity, lentity.reversed),
                ("sm1", same_location, llocation.reversed),
                ("i", lstep.reversed)
            ))
        ), active = False
    )

    # graph.visualize("./image")

    #from PIL import Image
    # Open an image file
    #graphImage = Image.open('image.png')
    # Display the image
    #graphImage.show()
