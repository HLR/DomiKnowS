import sys
sys.path.append('../../')

import logging
logging.basicConfig(level=logging.INFO)

from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import andL, existsL, notL, atMostL, ifL, fixedL, eqL, exactL
from regr.graph import EnumConcept
from regr.sensor.pytorch.sensors import JointSensor, ReaderSensor, FunctionalSensor
from regr.sensor.pytorch.learners import ModuleLearner

import config

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('global') as graph:
    sudoku = Concept("sodoku")
    
    empty_entry = Concept(name='empty_entry')
    (empty_rel, ) = sudoku.contains(empty_entry)
    
    same_row = Concept(name="same_row")
    (same_row_arg1, same_row_arg2) = same_row.has_a(row1=empty_entry, row2=empty_entry)
    
    same_col = Concept(name="same_col")
    (same_col_arg1, same_col_arg2) = same_col.has_a(col1=empty_entry, col2=empty_entry)
    
    same_table = Concept(name="same_table")
    (same_table_arg1, same_table_arg2) = same_table.has_a(table1=empty_entry, table2=empty_entry)
    
    empty_entry_label = empty_entry(name="empty_entry_label", ConceptClass=EnumConcept, 
                                    values=[f'v{d}' for d in range(config.size)])
    v = [getattr(empty_entry_label, a) for a in ('', *empty_entry_label.enum)]

    for row_num in range(config.size):
        for j in range(1, config.size + 1):
            exactL(v[j](path = (eqL(empty_entry, "rows", {row_num}))))
            exactL(v[j](path = (eqL(empty_entry, "cols", {row_num}))))
            exactL(v[j](path = (eqL(empty_entry, "tables", {row_num}))))
