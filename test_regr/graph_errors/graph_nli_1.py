from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import ifL, andL, nandL, atMostL, existsL, notL
from domiknows.graph.concept import EnumConcept

def setup_graph(fix_constraint=False):
    Graph.clear()
    Concept.clear()
    Relation.clear()
        
    with Graph('global') as graph:
        with Graph('linguistic') as lgraph:
            pair = Concept(name='pair')
            premise = Concept(name='premise')
            hypothesis = Concept(name='hypothesis')
            (rel_pair_premise, rel_pair_hypothesis) = pair.has_a(arg1=premise, arg2=hypothesis)

        with Graph('nli_graph') as ngraph:
            nli_class = pair(name='nli_class', ConceptClass=EnumConcept,
                            values=['entailment', 'contradiction', 'neutral'])

            if fix_constraint:
                ifL(
                    andL(
                        pair('x'),
                        pair('y'),
                        pair(path=('x', rel_pair_premise, rel_pair_hypothesis.reversed)),
                        pair(path=('y')),
                    ),
                    ifL(
                        nli_class.entailment(path=('x')),
                        notL(nli_class.contradiction(path=('y')))
                    ),
                    name="pair_symmetry_constraint"
                ) 

    return graph           