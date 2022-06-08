import sys
sys.path.append('../../')

import logging
logging.basicConfig(level=logging.INFO)

from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import andL, existsL, notL, atMostL, ifL, fixedL, eqL, exactL
from regr.graph import EnumConcept
from regr.sensor.pytorch.sensors import JointSensor, ReaderSensor, FunctionalSensor
from regr.sensor.pytorch.learners import ModuleLearner
import torch

from cnn import Net
import config
from graph import graph, sudoku, empty_entry, empty_rel, empty_entry_label, same_row, same_col, same_table

cols_indices = torch.arange(config.size).repeat((config.size, 1)).flatten()
rows_indices = torch.arange(config.size).repeat((config.size, 1)).T.flatten()
cols_indices = cols_indices.unsqueeze(-1)
rows_indices = rows_indices.unsqueeze(-1)

tables_indices = torch.empty((config.size, config.size))

for i in range(0, config.size, 3):
    for j in range(0, config.size, 3):
        tables_indices[i:i+3, j:j+3] = i + j//3

tables_indices = tables_indices.flatten().unsqueeze(-1) # 81 x 1

sudoku['images'] = ReaderSensor(keyword='images') # 1 x 81 x 784
sudoku['all_logits'] = ModuleLearner('images', module=Net(config.size)) # 1 x 81 x 9

sudoku['labels'] = ReaderSensor(keyword='labels') # 1 x 81

def unpack(logits):
    return torch.ones((config.size ** 2, 1)), rows_indices, cols_indices, tables_indices, logits[0]

empty_entry[empty_rel, 'rows', 'cols', 'tables', empty_entry_label] = JointSensor(sudoku['all_logits'], forward=unpack)
