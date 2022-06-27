from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import orL, andL, existsL, notL, atLeastL, atMostL, ifL, V, nandL

Graph.clear()
Concept.clear()
Relation.clear()
with Graph('Useful_arg') as graph:

    # Group of sentence
    sentence_group = Concept(name="sentence_group")
    sentence = Concept(name="sentence")
    sentence_group_contains, = sentence_group.contains(sentence)

    # 3 classes
    entailment = sentence(name="entailment")
    neutral = sentence(name="neutral")
    contradiction = sentence(name="contradiction")

    # want only one to be true
    nandL(entailment, neutral)  # if both true, fail
    nandL(entailment, contradiction)
    nandL(neutral, contradiction)
    orL(entailment, neutral, contradiction)

    # Example Constrain

    symmetric = Concept(name="symmetric")
    s_sent1, s_sent2 = symmetric.has_a(arg1=sentence, arg2=sentence)

    # Ent(X1, X2) => !CON(X2, X1)
    ifL(entailment('x'), notL(contradiction(path=('x', symmetric, s_sent2))))

    # Neu(X1, X2) => !CON(X2, X1)
    ifL(neutral('x'), notL(contradiction(path=('x', symmetric, s_sent2))))

    # CON(X1, X2) => CON(X2, X1)
    ifL(neutral('x'), contradiction(path=('x', symmetric, s_sent2)))

    # Ent(X1, X2) + Ent(X2, X3) => Ent(X1, X3)
    transitive = Concept("transitive")
    t_sent1, t_sent2, t_sent3 = transitive.has_a(arg11=sentence, arg22=sentence, arg33=sentence)
    ifL(andL(entailment('x'), entailment(path=('x', transitive, t_sent2))), entailment(path=('x', transitive, t_sent3)))

