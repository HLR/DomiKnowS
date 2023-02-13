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
                     values=['O', 'B_MISC', 'I_MISC', 'B_PER', 'I_PER', 'B_ORG', 'I_ORG', 'B_LOC', 'I_LOC', 'X'])
    ### {'O': 0, 'B_MISC': 1, 'I_MISC': 2, 'B_PER': 3, 'I_PER': 4, 'B_ORG': 5, 'I_ORG': 6, 'B_LOC': 7, 'I_LOC': 8, '[CLS]': 9, '[SEP]': 10, 'X': 11}

    # exactL(labels.O, labels.B_MISC, labels.I_MISC, labels.B_PER, labels.I_PER, labels.B_ORG, labels.I_ORG, labels.B_LOC, labels.I_LOC, labels.CLS, labels.SEP, labels.X)
    exactL(labels.O, labels.B_MISC, labels.I_MISC, labels.B_PER, labels.I_PER, labels.B_ORG, labels.I_ORG, labels.B_LOC, labels.I_LOC, labels.X)

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
