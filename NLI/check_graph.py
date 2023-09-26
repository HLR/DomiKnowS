import sys, os
sys.path.append(".")
sys.path.append("../..")

from domiknows.graph import Graph, Concept, Relation, EnumConcept
from domiknows.graph.logicalConstrain import orL, andL, existsL, notL, atLeastL, atMostL, ifL, V, nandL, fixedL, exactL

Graph.clear()
Relation.clear()
Concept.clear()
with Graph('global') as graph:
    with Graph('linguistic') as lgraph:
        pair = Concept(name="pairs")
        sentence = Concept(name="sentences")
        (rel_pair_premise, rel_pair_hypothesis) = pair.has_a(arg1=sentence, arg2=sentence)

    with Graph('NLI_graph') as ngraph:
        # 3 classes in a multi-class setting
        nli_class = pair(name="nli_class", ConceptClass=EnumConcept,
                            values=["entailment", "neutral", "contradiction"])
        
    ### To introduce the symmetric and transitive constraints, we first define the two new concepts to related multiple pairs to each other
    ### The symmetric concept is used to relate two pairs to each other
    symmetric = Concept(name="symmetric")
    s_pair1, s_pair2 = symmetric.has_a(arg1=pair, arg2=pair)

    ### The transitive concept is used to relate three pairs to each other
    transitive = Concept("transitive")
    t_pair1, t_pair2, t_pair3 = transitive.has_a(arg11=pair, arg22=pair, arg33=pair)

    ### Symmetric constraint
    #### If the premise x, hypothesis y is entailed, then its reverse premise y, hypothesis y cannot be of type contradiction
    ### FOL: (nli_class(pair1, 'entailment') ∧ symmetric(pair1, pair2)) → ¬nli_class(pair2, 'contradiction')
    ifL(
        andL(
            nli_class.entailment('x'),
            existsL(
                symmetric('s', path=('x', s_pair1.reversed)) ### going from argument to the relation requires the use if .reversed
            )
        ),
        andL(
            notL(
                nli_class.contradiction(path=('x', s_pair1.reversed, s_pair2))
            )
        ),
        name = "symmetric_constraint_1"
    )

    #### If the premise x, hypothesis y is neural, then its reverse premise y, hypothesis y cannot be of type contradiction
    #### FOL: (nli_class(pair1, 'neutral') ∧ symmetric(pair1, pair2)) → ¬nli_class(pair2, 'contradiction')
    ifL(
        andL(
            nli_class.neutral('x'),
            existsL(
                symmetric('s', path=('x', s_pair1.reversed)) ### going from argument to the relation requires the use if .reversed
            )
        ),
        andL(
            notL(
                nli_class.contradiction(path=('x', s_pair1.reversed, s_pair2))
            )
        ),
        name = "symmetric_constraint_2"
    )

    #### If the premise x, hypothesis y is contradiction, then its reverse premise y, hypothesis y should also be contradiction
    #### FOL: (nli_class(pair1, 'contradiction') ∧ symmetric(pair1, pair2)) → nli_class(pair2, 'contradiction')
    ifL(
        andL(
            nli_class.contradiction('x'),
            existsL(
                symmetric('s', path=('x', s_pair1.reversed)) ### going from argument to the relation requires the use if .reversed
            )
        ),
        nli_class.contradiction(path=('x', s_pair1.reversed, s_pair2)),
        name = "symmetric_constraint_3"
    )

    ### Transitive constraint
    #### If entailed (x,y) and entailed (y,z), then, we should also have entailed (x, z) known as transitive property of entailment Ent(X, Y) + Ent(Y, Z) => Ent(X, Z)
    #### FOL: (nli_class(pair1, 'entailment') ∧ nli_class(pair2, 'entailment') ∧ transitive(pair1, pair2, pair3)) → nli_class(pair3, 'entailment')
    ifL(
        andL(
            nli_class.entailment('x'),
            existsL(
                transitive("t", path=('x', t_pair1.reversed)) ### going from argument to the relation requires the use if .reversed
            ),
        ),
        ifL(
            nli_class.entailment(path=('t', t_pair2)),
            nli_class.entailment(path=('t', t_pair3))
        ),
        name = "transitive_constraint_1"
    )