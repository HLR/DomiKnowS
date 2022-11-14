from regr.graph import Graph, Concept, Relation, EnumConcept
from regr.graph.logicalConstrain import orL, andL, existsL, notL, atLeastL, atMostL, ifL, V, nandL

Graph.clear()
Concept.clear()
Relation.clear()
with Graph('Useful_arg') as graph:
    # Group of sentence
    group_pairs = Concept(name="group_pairs")
    pairs = Concept(name="pairs")
    group_pair_contains,  = group_pairs.contains(pairs)

    # 3 classes
    answer_class = pairs(name="answer_class", ConceptClass=EnumConcept,
                         values=["entailment", "neutral", "contradiction"])

    # Logical Constrains
    symmetric = Concept(name="symmetric")
    s_sent1, s_sent2 = symmetric.has_a(arg1=pairs, arg2=pairs)
    # Ent(X1, X2) => !CON(X2, X1)
    ifL(andL(answer_class.entailment('x'), existsL(symmetric('s', path=('x', symmetric)))),
        notL(answer_class.contradiction(path=('s', s_sent2))))

    # Neu(X1, X2) => !CON(X2, X1)
    ifL(andL(answer_class.neutral('x'), existsL(symmetric('s', path=('x', symmetric)))),
        notL(answer_class.contradiction(path=('s', s_sent2))))

    # CON(X1, X2) => CON(X2, X1)
    ifL(andL(answer_class.contradiction('x'), existsL(symmetric('s', path=('x', symmetric)))),
        answer_class.contradiction(path=('s', s_sent2)))

    # Ent(X1, X2) + Ent(X2, X3) => Ent(X1, X3)
    transitive = Concept("transitive")
    t_sent1, t_sent2, t_sent3 = transitive.has_a(arg11=pairs, arg22=pairs, arg33=pairs)
    ifL(andL(answer_class.entailment('x'),
             existsL(transitive("t", path=('x', transitive))),
             answer_class.entailment(path=('t', t_sent2))),
        answer_class.entailment(path=('t', t_sent3)))
