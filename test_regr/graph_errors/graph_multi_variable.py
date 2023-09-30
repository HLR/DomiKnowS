from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import ifL, andL, nandL, atMostL, existsL, notL
from domiknows.graph.concept import EnumConcept

def setup_graph(fix_constraint=False):
    Graph.clear()
    Concept.clear()
    Relation.clear()

    with Graph('global') as graph:
        with Graph('linguistic') as ling_graph:        
            word = Concept(name='word')
            phrase = Concept(name='phrase')
            sentence = Concept(name='sentence')
            (rel_sentence_contains_word,) = sentence.contains(word)
            (rel_sentence_contains_phrase,) = sentence.contains(phrase)
            (rel_phrase_contains_word,) = phrase.contains(word)

            pair = Concept(name='pair')
            (rel_pair_phrase1, rel_pair_phrase2, ) = pair.has_a(arg1=phrase, arg2=phrase)

        with Graph('application') as app_graph:
            entity = phrase(name='entity')
            entity_label = entity(name="entity_label", ConceptClass=EnumConcept, values=["people", "organization", 'location', 'other', 'O'])
            pair_label = pair(name="pair_label", ConceptClass=EnumConcept, values=["work_for", "located_in", 'live_in', 'orgbase_on', 'kill'])
    #       
            
            if fix_constraint:
                ifL(pair_label.work_for('x'), andL(entity_label.people(path=('x', rel_pair_phrase1)), entity_label.organization(path=('x', rel_pair_phrase2))))
            else:
                ifL(pair_label.work_for('x', 'y'), andL(entity_label.people(path=('x')), entity_label.organization(path=('y'))))
    return graph