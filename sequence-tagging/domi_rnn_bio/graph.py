import sys
sys.path.append(".")
sys.path.append("../")
sys.path.append("../../")

from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import orL, andL, existsL, notL, atLeastL, atMostL, ifL, nandL, V, exactL
from regr.graph import EnumConcept

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('sequence_tagging') as graph:
    sentence = Concept(name="sentence")
    word = Concept(name="word")
    sen_word_rel = sentence.contains(word)
    labels = word(name="labels", ConceptClass=EnumConcept,
                     values=['O', 'I_PER', 'I_ORG', 'I_LOC', 'I_MISC', 'B_MISC', 'B_ORG', 'B_LOC', 'B_PER'])

    exactL(labels.O, labels.I_PER, labels.I_ORG, labels.I_MISC, labels.B_MISC, labels.B_ORG, labels.B_LOC, labels.B_PER)

    ### BIO constraints

    before = Concept(name="before")
    (before_arg1, before_arg2) = before.has_a(word,word)
   #  # before  -> B-, I-

    ## bio constraints start
    ifL(
        before('x'), 
        ifL(
           labels.I_PER(path=('x', before_arg2)),
           labels.B_PER(path=('x', before_arg1)),
        ),
        active = True
    )
    ifL(
        before('x'), 
        ifL(
           labels.I_ORG(path=('x', before_arg2)),
           labels.B_ORG(path=('x', before_arg1)),
        ),
        active = True
    )
    ifL(
        before('x'), 
        ifL(
           labels.I_MISC(path=('x', before_arg2)),
           labels.B_MISC(path=('x', before_arg1)),
        ),
        active = True
    )
    ifL(
        before('x'), 
        ifL(
           labels.I_LOC(path=('x', before_arg2)),
           labels.B_LOC(path=('x', before_arg1)),
        ),
        active = True
    )
    ifL(
        before('x'), 
        ifL(
           labels.O(path=('x', before_arg1)),
           notL(andL(labels.I_LOC(path=('x', before_arg2)), labels.I_ORG(path=('x', before_arg2)), labels.I_MISC(path=('x', before_arg2)), labels.I_PER(path=('x', before_arg2)),)),
        ),
        active = True
    )

graph.detach()
