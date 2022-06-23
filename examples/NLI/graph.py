from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import orL, andL, existsL, notL, atLeastL, atMostL, ifL, V, nandL

Graph.clear()
Concept.clear()
Relation.clear()
with Graph('Useful_arg') as graph:

    # TODO: Change the graph to be group of sentences to enable batch implemented
    # Look at the animal and flower example
    #
    sentence = Concept(name="sentence")

    # 3 classes
    entailment = sentence(name="entailment")
    neutral = sentence(name="neutral")
    contradiction = sentence(name="contradiction")

    # want only one to be true
    nandL(entailment, neutral)  # if both true, fail
    nandL(entailment, contradiction)
    nandL(neutral, contradiction)
    orL(entailment, neutral, contradiction)
