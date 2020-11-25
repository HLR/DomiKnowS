from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import orL, andL, existsL, notL, atLeastL, atMostL, ifL, nandL

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('global') as graph:
    procedure = Concept("procedure")
    step = Concept("step")
    (procedure_contain_step, ) = procedure.contains(step)
    word = Concept("word")
    (step_contains_word, ) = step.contains(word)
    entity = Concept("entity")
    entity_step = Concept("entity_step")
    (entity_of_step, step_of_entity) = entity_step.has_a(arg1=entity, arg2=step)
    entity_step_word = Concept("entity_step_word")
    (entity_of_step_word, step_of_entity_word, word_of_entity_step) = entity_step.has_a(arg1=entity, arg2=step, arg3=word)
    location_start = entity_step_word("start_location")
    location_end = entity_step_word("end_location")
    non_existence = entity_step("none_existence")
    unknown_loc = entity_step("unknown_location")
    known_loc = entity_step("known_location")
    before = Concept("before")
    (before_arg1, before_arg2) = before.has_a(arg1=step, arg2=step)
    action = Concept("action")
    (action_arg1, action_arg2) = action.has_a(arg1=entity_step, arg2=entity_step)
    create = action("create")
    destroy = action("create")
    other = action("create")

    nandL(create, destroy, other)
    nandL(known_loc, unknown_loc, non_existence)
    ifL(create, ("x", ), andL(non_existence, ("x", "arg1"), orL(known_loc, ("x", "arg2"), unknown_loc, ("x", "arg2"))))
    ifL(destroy, ("x", ), andL(orL(known_loc, ("x", "arg1"), unknown_loc, ("x", "arg1")), non_existence, ("x", "arg2")))
#     atMostL(1, ("x"), andL(entity, "y", create()))
    # No entity_step




    # No entity_step


