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

    strand1 = Concept(name='Strand_1')
    strand2 = Concept(name='Strand_2')

    # nucleotide base concepts
    nucleotide_base = Concept(name='Nucleotide_base')
    a = nucleotide_base(name='A')
    c = nucleotide_base(name='C')
    t = nucleotide_base(name='T')
    g = nucleotide_base(name='G')
    n = nucleotide_base(name='N')

    disjoint(a, t, c, g, n)

    # strand concepts and relations
    (rel_strand1_contains_base,) = strand1.contains(nucleotide_base)
    (rel_strand2_contains_base,) = strand2.contains(nucleotide_base)
    (rel_dna_sequence_contains_strand1,) = dna_sequence.contains(strand1)
    (rel_dna_sequence_contains_strand2,) = dna_sequence.contains(strand2)
    
    # the position concept to represent the position of nucleotides in the strand
    position = Concept(name='Position')
    (rel_strand1_position, rel_strand2_position) = position.has_a(strand1, strand2)
    
    gene_family = dna_sequence(name = "gene_family", ConceptClass = EnumConcept,\
                            values = ['gene_family_one', 'gene_family_two',\
                            'gene_family_three', 'gene_family_four', 'gene_family_five', 'gene_family_six'])
    
    # Define the logical constraints
    
    # if nucleotide base on Strand 1 at any given position is 'A', 
    # then the nucleotide base on Strand 2 at that corresponding position is 'T'.
    ifL(
        a('x'),
        t(path=('x', rel_strand1_position.reversed, rel_strand2_position)),
        name="constraint_a_t_pairing"
    )

    # if nucleotide base on Strand 1 at any given position is 'T',
    # then the nucleotide base on Strand 2 at that corresponding position is 'A'.
    ifL(
        t('x'),
        a(path=('x', rel_strand1_position.reversed, rel_strand2_position)),
        name="constraint_t_a_pairing"
    )

    # if nucleotide base on Strand 1 at any given position is 'C',
    # then the nucleotide base on Strand 2 at that corresponding position is 'G'.
    ifL(
        c('x'),
        g(path=('x', rel_strand1_position.reversed, rel_strand2_position)),
        name="constraint_c_g_pairing"
    )

    # if nucleotide base on Strand 1 at any given position is 'G',
    # then the nucleotide base on Strand 2 at that corresponding position is 'C'.
    ifL(
        g('x'),
        c(path=('x', rel_strand1_position.reversed, rel_strand2_position)),
        name="constraint_g_c_pairing"
    )

    # if nucleotide base on Strand 1 at any given position is 'N',
    # then the nucleotide base on Strand 2 at that corresponding position is 'N'.
    ifL(
        n('x'),
        n(path=('x', rel_strand1_position.reversed, rel_strand2_position)),
        name="constraint_n_n_pairing"
    )