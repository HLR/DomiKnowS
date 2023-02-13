from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import (
    orL,
    andL,
    existsL,
    notL,
    atLeastL,
    atMostL,
    ifL,
    nandL,
    eqL,
)

Graph.clear()
Concept.clear()
Relation.clear()

with Graph("global") as graph:
    procedure = Concept("procedure")
    text = Concept("text")
    entity = Concept("entity")
    (procedure_text, procedure_entity) = procedure.has_a(arg1=text, arg2=entity)
    step = Concept("step")
    (text_contain_step,) = text.contains(step)

    pair = Concept("pair")
    (pair_entity, pair_step) = pair.has_a(entity, step)

    word = Concept("word")
    (pair_contains_words,) = pair.contains(word)

    word1 = Concept("word1")

    non_existence = pair("non_existence")
    unknown_loc = pair("unknown_location")
    known_loc = pair("known_location")

    triplet = Concept("triplet")
    (triplet_entity, triplet_step, triplet_word) = triplet.has_a(entity, step, word)

    before = Concept("before")
    (before_arg1, before_arg2) = before.has_a(arg1=step, arg2=step)