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
    """nucleotide = Concept(name='nucleotide')
    A = nucleotide(name='A')
    T = nucleotide(name='T')
    C = nucleotide(name='C')
    G = nucleotide(name='G')
    N = nucleotide(name='N')

    # A & T are complementary, C & G are complementary
    two_rna = Concept("complementary")
    two_rna.has_a(arg1=dna_sequence, arg2=dna_sequence)
    dna_sequence.contains(nucleotide)"""

    gene_family = dna_sequence(name = "gene_family", ConceptClass = EnumConcept,\
                            values = ['gene_family_one', 'gene_family_two',\
                            'gene_family_three', 'gene_family_four', 'gene_family_five', 'gene_family_six'])
