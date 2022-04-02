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
    # T => ENT(X1, X1)
    # Ent(X1, X2) => !CON(X2, X1)
    # Neu(X1, X2) => !CON(X2, X1)
    # CON(X1, X2) => CON(X2, X1)
    # Ent(X1, X2) + Ent(X2, X3) => Ent(X1, X3)

