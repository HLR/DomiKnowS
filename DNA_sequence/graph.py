import sys
sys.path.append('.')
from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.concept import EnumConcept
from domiknows.graph.relation import disjoint
from domiknows.graph.logicalConstrain import ifL, orL, andL, atMostL
from domiknows.program import SolverPOIProgram
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.metric import MacroAverageTracker, PRF1Tracker
from domiknows.sensor.pytorch.sensors import ReaderSensor
from domiknows.sensor.pytorch.learners import ModuleLearner

# Clear the graph
Graph.clear()
Concept.clear()
Relation.clear()

# Create a graph to store the knowledge about DNA
with Graph('dna_and_gene_knowledge') as graph:
    # DNA sequence concepts and relations
    dna_sequence = Concept(name='dna_sequence')
    nucleotide = Concept(name='nucleotide')
    A = nucleotide(name='A')
    T = nucleotide(name='T')
    C = nucleotide(name='C')
    G = nucleotide(name='G')
    N = nucleotide(name='N')

    # A & T are complementary, C & G are complementary
    complementary = Concept("complementary")
    complementary.has_a(arg1=A, arg2=T)
    complementary.has_a(arg1=C, arg2=G)

    dna_sequence.contains(nucleotide)

    # nucleotides are disjoint
    disjoint(A, T, C, G, N)

    # DNA sequence only contains A, T, C, G nucleotides
    ifL(
        nucleotide('n'),
        orL(
            A(path=('n',)),
            T(path=('n',)),
            C(path=('n',)),
            G(path=('n',)),
            N(path=('n',))
        )
    )


    # Only allow maximum of 100 nucleotides in a DNA sequence

    # dna_sequence['label'] = ModuleLearner('sequence', module = nn.Linear(100, 1))

    gene_family = dna_sequence(name = "gene_family", ConceptClass = EnumConcept,\
                            values = ['gene_family_1', 'gene_family_2',\
                            'gene_family_3', 'gene_family_4', 'gene_family_5', 'gene_family_6'])
