import torch, logging
from torch import nn
from transformers import AdamW
from sklearn.model_selection import train_test_split
from graph import graph, dna_sequence, nucleotide, A, T, C, G, N, rel_complementary_A_T, \
    rel_complementary_C_G, dna_sequence_contains_nucleotide, disjoint, ifL, orL, andL, atMostL, gene_family

from reader import read_domiknows_data, truncate
from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.concept import EnumConcept
from domiknows.graph.relation import disjoint
from domiknows.graph.logicalConstrain import ifL, orL, andL, atMostL
from domiknows.program import SolverPOIProgram
from domiknows.program.model.pytorch import model_helper, PoiModel
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.metric import MacroAverageTracker, PRF1Tracker
from domiknows.sensor.pytorch.sensors import ReaderSensor, FunctionalSensor
from domiknows.sensor.pytorch.learners import ModuleLearner



def preprocess_data(sequences, labels):
    """Converts sequences to a tensor of one-hot encoded sequences and labels to a tensor."""
    nucleotide_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    encoded_sequences = [[nucleotide_to_idx[nucleotide] for nucleotide in seq] for seq in sequences]
    
    sequences_tensor = torch.tensor(encoded_sequences, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return encoded_sequences


# Set up logging
logging.basicConfig(level=logging.INFO)



# Download and preprocess DNA sequence data
data_path = "human.txt"  
train, test = read_domiknows_data(data_path)

dna_sequence['input_sequence'] = ReaderSensor(keyword = 'input_sequence')
dna_sequence['output_sequence'] = ReaderSensor(keyword = 'output_sequence')

dna_sequence['encoded_sequence'] = FunctionalSensor('output_sequence', forward=preprocess_data)
dna_sequence[gene_family] = ModuleLearner('encoded_sequences', module=nn.Linear(601, 6))
dna_sequence[gene_family] = ReaderSensor(keyword = 'gene_family', label = True)

program = SolverPOIProgram(graph, poi = [dna_sequence[gene_family]], 
            loss=MacroAverageTracker(NBCrossEntropyLoss()), metric=PRF1Tracker())

program.train(train, train_epoch_num=5, Optim=torch.optim.Adam, device='auto')
program.test(test, device='auto')
print(program.model.loss)
# print("LABELS:", labels[:5])
for datanode in program.populate(dataset=test):
    print('datanode:', datanode)
